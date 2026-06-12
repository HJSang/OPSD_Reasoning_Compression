"""Full-vocab (top-K truncated) reverse KL distillation loss for slime.

Implements CRISP milestone 2 (FULL_KL_PLAN.md): a distribution-level per-token
reverse KL between the student and the conciseness-conditioned teacher,
replacing the sampled-token OPD advantage penalty.

For each response position t, with teacher top-K ids x_1..x_K and log-probs
log p_k (fetched by crisp_opd.reward_func via sglang ``top_logprobs_num``) and
exact student probabilities q_k from the training logits:

    q_tail = 1 - sum_k q_k          p_tail = 1 - sum_k p_k
    KL_K(q_t || p_t) = sum_k q_k (log q_k - log p_k)
                       + q_tail (log q_tail - log p_tail)

This is the reverse KL of the coarsened (K+1)-bucket distributions: >= 0, zero
iff the coarsened distributions match, a lower bound of the exact reverse KL,
and equal to it (verl's ``opsd_worker._compute_reverse_kl_loss`` objective) at
K=V.

Approximation fidelity is governed by ``q_tail`` — the STUDENT's probability
mass outside the teacher's top-K — not by teacher coverage alone (reverse KL
is an expectation under q, so a diffuse student hides large log-ratios in the
tail bucket). In CRISP's self-distillation regime q ~ p and q_tail is tiny;
the tail term's gradient also actively pushes student mass back onto the
teacher's support. Both ``q_tail`` and ``teacher_topk_coverage`` are logged
per step — treat rising q_tail as the signal to raise K.

Wiring (full-KL mode, see run-qwen3-8b-crisp.sh):

    --loss-type custom_loss
    --custom-loss-function-path slime_crisp.crisp_full_kl_loss.full_kl_loss_function
    --disable-compute-advantages-and-returns
    --recompute-loss-function          # frees the per-chunk softmax graph
    --calculate-per-token-loss         # global token mean == verl normalization

Supports TP (vocab-parallel logits) via two small autograd functions modeled
on slime's ``_VocabParallelEntropy``; degrades to plain ops when no process
group is active, so the math is unit-testable on CPU. CP must be 1 (asserted
upstream in the plumbing patch) and qkv_format must be "thd".
"""

import logging

import torch

logger = logging.getLogger(__name__)

_warned_temperature = False


def _tp_world_size(tp_group) -> int:
    import torch.distributed as dist

    if tp_group is None or not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(group=tp_group)


class _AllReduceSumKeepGrad(torch.autograd.Function):
    """all_reduce(SUM) whose backward passes the (TP-replicated) grad through.

    Correct because the downstream loss is identical on every TP rank, so the
    upstream gradient is already the full dL/d(sum); each rank then applies its
    local Jacobian (here: identity onto its own summand).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, tp_group) -> torch.Tensor:
        import torch.distributed as dist

        x = x.clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=tp_group)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class _VocabParallelLogSumExp(torch.autograd.Function):
    """logsumexp over the (TP-sharded) vocab dim. [N, V_local] -> [N, 1]."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, tp_group) -> torch.Tensor:
        import torch.distributed as dist

        logits_max = logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)
        sum_exp = (logits - logits_max).exp().sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)
        lse = logits_max + sum_exp.log()
        ctx.save_for_backward(logits, lse)
        return lse

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logits, lse = ctx.saved_tensors
        # d lse / d logits_local = softmax_local; recomputed here so the
        # forward never stores an [N, V] probability tensor.
        return grad_output * (logits - lse).exp(), None


def vocab_parallel_topk_log_probs(
    logits: torch.Tensor, ids: torch.Tensor, tp_group=None
) -> torch.Tensor:
    """Student log-probs at arbitrary token ids under vocab parallelism.

    Args:
        logits: [N, V_local] (full V when tp world size is 1). Differentiable.
        ids: [N, K] global token ids.
        tp_group: tensor-parallel process group (None -> single shard).

    Returns:
        [N, K] log q(ids), differentiable in ``logits``.
    """
    tp_size = _tp_world_size(tp_group)
    v_local = logits.size(-1)

    if tp_size == 1:
        gathered = logits.gather(-1, ids)
        lse = logits.logsumexp(dim=-1, keepdim=True)
        return gathered - lse

    import torch.distributed as dist

    shard_start = dist.get_rank(group=tp_group) * v_local
    in_shard = (ids >= shard_start) & (ids < shard_start + v_local)
    local_ids = (ids - shard_start).clamp(0, v_local - 1)
    # Out-of-shard gathers are masked to 0 and contributed by the owning rank
    # via the SUM all-reduce; their grads are killed by the same mask.
    gathered = logits.gather(-1, local_ids) * in_shard
    gathered = _AllReduceSumKeepGrad.apply(gathered, tp_group)
    lse = _VocabParallelLogSumExp.apply(logits, tp_group)
    return gathered - lse


def bucketed_reverse_kl(
    log_q_topk: torch.Tensor,
    teacher_top_logprobs: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token reverse KL over teacher-top-K buckets + one tail bucket.

    Args:
        log_q_topk: [N, K] student log-probs at the teacher's top-K ids (grad).
        teacher_top_logprobs: [N, K] teacher log-probs at the same ids (no grad).
        eps: tail-mass clamp for numerical safety.

    Returns:
        (kl [N], teacher_coverage [N], q_tail [N]) — kl is differentiable;
        the diagnostics are detached.
    """
    q_k = log_q_topk.exp()
    p_coverage = teacher_top_logprobs.exp().sum(dim=-1)
    q_tail = (1.0 - q_k.sum(dim=-1)).clamp(min=eps)
    p_tail = (1.0 - p_coverage).clamp(min=eps)

    kl_topk = (q_k * (log_q_topk - teacher_top_logprobs)).sum(dim=-1)
    kl_tail = q_tail * (q_tail.log() - p_tail.log())
    kl = kl_topk + kl_tail
    return kl, p_coverage.detach(), q_tail.detach()


def full_kl_loss_function(args, batch, logits: torch.Tensor, sum_of_sample_mean):
    """slime ``custom_loss`` entry point: mean per-token KL_K(q_t || p_t).

    Contract (slime loss.py::loss_function): receives the full vocab-parallel
    student logits [1, T, V] for the packed micro-batch and returns
    ``(loss, metrics)``; reduction/rescaling is handled by the caller through
    ``sum_of_sample_mean`` and ``--calculate-per-token-loss``.
    """
    from megatron.core import mpu

    assert args.qkv_format == "thd", (
        f"full_kl_loss_function supports qkv_format='thd' only, got {args.qkv_format!r}"
    )
    teacher_top_ids = batch.get("teacher_top_ids")
    teacher_top_logprobs = batch.get("teacher_top_logprobs")
    assert teacher_top_ids is not None and teacher_top_logprobs is not None, (
        "batch is missing teacher_top_ids/teacher_top_logprobs. Set crisp_kl_mode: full "
        "in the custom config so crisp_opd.reward_func fetches the teacher top-K, and "
        "run with the patched slime (crisp branch) that plumbs these fields."
    )

    assert logits.size(0) == 1, f"{logits.shape}"
    raw_logits = logits  # unscaled [1, T, V], for the kl_sampled diagnostic
    logits = logits.squeeze(0)

    # Mirror get_log_probs_and_entropy: train-time log-probs are defined at the
    # rollout temperature. The teacher's sglang input logprobs are raw
    # (temperature-free), so the objective only matches the paper's at 1.0.
    rollout_temperature = getattr(args, "rollout_temperature", 1.0)
    if rollout_temperature != 1.0:
        global _warned_temperature
        if not _warned_temperature:
            logger.warning(
                "full_kl_loss_function: rollout_temperature=%s scales student logits but "
                "teacher top-K log-probs from sglang are raw; the KL mixes temperatures.",
                rollout_temperature,
            )
            _warned_temperature = True
        logits = logits / rollout_temperature

    tp_group = mpu.get_tensor_model_parallel_group()
    chunk_size = getattr(args, "crisp_kl_chunk_size", None) or 128

    kl_list = []
    coverage_sum = torch.tensor(0.0, device=logits.device)
    q_tail_sum = torch.tensor(0.0, device=logits.device)
    n_kl_tokens = 0

    offset = 0
    for total_length, response_length, top_ids, top_lps in zip(
        batch["total_lengths"], batch["response_lengths"], teacher_top_ids, teacher_top_logprobs, strict=True
    ):
        end = offset + total_length
        start = end - response_length
        offset = end
        if response_length == 0:
            kl_list.append(logits.new_zeros((0,)))
            continue

        assert top_ids.size(0) == response_length, (
            f"teacher_top_ids rows ({top_ids.size(0)}) != response_length ({response_length}); "
            "teacher scoring must cover exactly the response span (see crisp_opd.reward_func)."
        )
        # Position i of the response is predicted by logits at sequence
        # position (start - 1 + i): same shift as _extract_per_sample.
        resp_logits = logits[start - 1 : end - 1]

        sample_kl = []
        for c in range(0, response_length, chunk_size):
            d = min(c + chunk_size, response_length)
            log_q_k = vocab_parallel_topk_log_probs(
                resp_logits[c:d].float(), top_ids[c:d], tp_group
            )
            kl, coverage, q_tail = bucketed_reverse_kl(log_q_k, top_lps[c:d])
            sample_kl.append(kl)
            coverage_sum += coverage.sum()
            q_tail_sum += q_tail.sum()
            n_kl_tokens += d - c
        kl_list.append(torch.cat(sample_kl, dim=0))

    kl_per_token = torch.cat(kl_list, dim=0)
    loss = sum_of_sample_mean(kl_per_token)
    # Keep the graph connected when every sample in the micro-batch is empty
    # (same guard as sft_loss_function).
    if kl_per_token.numel() == 0:
        loss = loss + 0 * logits.sum()

    metrics = {
        "loss": loss.clone().detach(),
        "teacher_topk_coverage": coverage_sum / max(1, n_kl_tokens),
        "q_tail": q_tail_sum / max(1, n_kl_tokens),
    }

    # Estimator-comparison diagnostic: the milestone-1 sampled-token reverse KL
    # (log q(y_t) - log p(y_t)) on the same batch, when sampled teacher
    # log-probs were collected alongside the top-K.
    sampled_teacher = batch.get("teacher_log_probs")
    if sampled_teacher:
        from slime.backends.megatron_utils.loss import get_log_probs_and_entropy

        with torch.no_grad():
            # raw_logits: get_log_probs_and_entropy applies the rollout
            # temperature itself, so passing the scaled logits would double-scale.
            _, lp = get_log_probs_and_entropy(
                raw_logits,
                args=args,
                unconcat_tokens=batch["unconcat_tokens"],
                total_lengths=batch["total_lengths"],
                response_lengths=batch["response_lengths"],
                with_entropy=False,
                max_seq_lens=batch.get("max_seq_lens", None),
            )
            sampled_kl = torch.cat(
                [q - t for q, t in zip(lp["log_probs"], sampled_teacher, strict=True)], dim=0
            )
            metrics["kl_sampled"] = sum_of_sample_mean(sampled_kl).detach()

    return loss, metrics
