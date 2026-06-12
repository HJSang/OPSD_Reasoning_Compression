"""CRISP on slime: conciseness-prompted teacher scoring for on-policy distillation.

Implements the CRISP method (teacher = same model conditioned on a conciseness
instruction; per-token reverse KL on the student's own rollouts) on top of
slime's sglang-mode OPD machinery (``--use-opd --opd-type sglang``).

Three entry points, wired via slime CLI args:

  --custom-rm-path                  slime_crisp.crisp_opd.reward_func
  --custom-reward-post-process-path slime_crisp.crisp_opd.post_process_rewards
  --rollout-function-path           slime_crisp.crisp_opd.generate_rollout

``reward_func`` scores each student rollout under the conciseness-conditioned
teacher: it builds ``teacher_prompt(question) + response_tokens`` and requests
a prefill-only forward (``max_new_tokens=0, return_logprob=True``) from the
teacher sglang server (``--rm-url``). The returned per-token log-probs are
stored on ``sample.teacher_log_probs``; slime's ``apply_opd_kl_to_advantages``
then turns them into the sampled-token reverse-KL penalty
``adv_t -= opd_kl_coef * (log pi_student(y_t) - log pi_teacher(y_t))``.

The scalar return value of ``reward_func`` is math correctness (metrics only,
mirroring CRISP's metrics-only verification); ``post_process_rewards`` zeroes
it for training so the learning signal is pure distillation.

``generate_rollout`` wraps slime's default rollout and implements CRISP's
periodic teacher refresh (theta_tilde <- theta every M rollouts) by telling the
teacher server to reload weights from the actor's latest HF dump
(``/update_weights_from_disk``). Requires ``--save-hf`` pointing at a fixed
path and ``--save-interval`` == the refresh interval.

CRISP-specific knobs (set via ``--custom-config-path`` YAML; all land on args):

  crisp_teacher_update_interval: 50            # M; 0 = frozen teacher
  crisp_teacher_hf_path: /path/to/hf_dump      # must equal --save-hf
  crisp_teacher_url: http://ip:port/generate   # optional; defaults to --rm-url
  crisp_teacher_prompt_prefix / _suffix        # optional template overrides
  crisp_kl_mode: sampled | full                # full also fetches teacher top-K
  crisp_teacher_topk: 256                      # K for crisp_kl_mode: full

Alignment invariant (see METHOD.md): teacher and student share the exact same
response token ids after different prompts, so per-position log-probs align by
construction. The token ids echoed back by the sglang server are checked
against the student's response tokens to enforce this.
"""

import logging

logger = logging.getLogger(__name__)

# Must match workspace/config/prompts.json -> "length_prune_teacher"
# (the paper's Figure-3 conciseness instruction).
DEFAULT_TEACHER_PREFIX = (
    "Solve the following math problem concisely and correctly. Be direct — avoid "
    "unnecessary elaboration, redundant steps, or restating the problem. Focus only "
    "on the key reasoning steps needed to reach the answer.\n\n"
    "The last line of your response should be of the form Answer: $Answer (without "
    "quotes) where $Answer is the answer to the problem.\n\n"
)
DEFAULT_TEACHER_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'

_TOKENIZER = None


def _get_tokenizer(args):
    """Lazily load and cache the tokenizer (shared with the student model)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from slime.utils.processing_utils import load_tokenizer

        _TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    return _TOKENIZER


async def _post(url, payload):
    """Indirection over slime's retrying HTTP helper (patchable in tests)."""
    from slime.utils.http_utils import post

    return await post(url, payload)


def _compute_correct(response: str, label: str) -> float:
    """Metrics-only math correctness via slime's math_dapo scorer (paper footnote 1)."""
    from slime.rollout.rm_hub.math_dapo_utils import compute_score

    return 1.0 if compute_score(response, label) > 0 else 0.0


def build_teacher_prompt_ids(args, question: str) -> list[int]:
    """Tokenize the conciseness-conditioned teacher prompt for ``question``."""
    prefix = getattr(args, "crisp_teacher_prompt_prefix", None) or DEFAULT_TEACHER_PREFIX
    suffix = getattr(args, "crisp_teacher_prompt_suffix", None) or DEFAULT_TEACHER_SUFFIX
    messages = [{"role": "user", "content": prefix + question + suffix}]
    tokenizer = _get_tokenizer(args)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        **(getattr(args, "apply_chat_template_kwargs", None) or {}),
    )


async def reward_func(args, sample, **kwargs):
    """Score one student rollout under the conciseness-conditioned teacher.

    Sets ``sample.teacher_log_probs`` (len == ``sample.response_length``) and
    returns math correctness in {0.0, 1.0} — metrics only, zeroed for training
    by ``post_process_rewards``.
    """
    if sample.response_length == 0:
        sample.teacher_log_probs = []
        if getattr(args, "crisp_kl_mode", "sampled") == "full":
            sample.teacher_top_ids = []
            sample.teacher_top_logprobs = []
        return 0.0

    question = (sample.metadata or {}).get("question")
    assert question, (
        "CRISP reward_func requires sample.metadata['question'] (the bare problem "
        "text). Prepare data with slime_crisp/prepare_crisp_slime_data.py."
    )

    teacher_prompt_ids = build_teacher_prompt_ids(args, question)
    response_tokens = list(sample.tokens[-sample.response_length :])

    url = getattr(args, "crisp_teacher_url", None) or args.rm_url
    assert url, "CRISP requires --rm-url (or crisp_teacher_url) pointing at the teacher /generate endpoint"

    # Prefill-only scoring pass: teacher prompt + the student's exact response
    # token ids. logprob_start_len=0 mirrors slime's OPD example; we trim to
    # the response span below.
    payload = {
        "input_ids": teacher_prompt_ids + response_tokens,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    # Full-KL mode (milestone 2): also fetch the teacher's top-K distribution
    # per position, consumed by crisp_full_kl_loss.full_kl_loss_function.
    full_kl = getattr(args, "crisp_kl_mode", "sampled") == "full"
    if full_kl:
        payload["top_logprobs_num"] = int(getattr(args, "crisp_teacher_topk", 256))

    output = await _post(url, payload)

    # input_token_logprobs entries are [logprob, token_id, ...]; the first
    # entry of the sequence has logprob None, but it can never fall inside the
    # response span because the teacher prompt is non-empty.
    entries = output["meta_info"]["input_token_logprobs"][-sample.response_length :]
    echoed_ids = [e[1] for e in entries]
    assert echoed_ids == response_tokens, (
        "CRISP alignment violation: token ids echoed by the teacher server do not "
        "match the student's response tokens. Teacher and student must share the "
        f"exact response token ids (got {len(echoed_ids)} vs {len(response_tokens)} "
        "tokens; first mismatch at index "
        f"{next((i for i, (a, b) in enumerate(zip(echoed_ids, response_tokens)) if a != b), -1)})."
    )
    teacher_log_probs = [e[0] for e in entries]
    assert all(lp is not None for lp in teacher_log_probs), (
        "Teacher server returned None log-probs inside the response span; "
        "check logprob_start_len handling."
    )
    sample.teacher_log_probs = teacher_log_probs

    if full_kl:
        # input_top_logprobs: per input position, a list of [logprob, token_id, ...]
        # entries (length top_logprobs_num). Trim to the response span like
        # input_token_logprobs.
        top_entries = output["meta_info"]["input_top_logprobs"][-sample.response_length :]
        k = payload["top_logprobs_num"]
        assert all(pos is not None and len(pos) == k for pos in top_entries), (
            "input_top_logprobs malformed inside the response span (None or wrong K); "
            "check the deployed sglang version's top_logprobs_num support."
        )
        sample.teacher_top_logprobs = [[e[0] for e in pos] for pos in top_entries]
        sample.teacher_top_ids = [[e[1] for e in pos] for pos in top_entries]

    if sample.label is None:
        return 0.0
    return _compute_correct(sample.response, sample.label)


def post_process_rewards(args, samples, **kwargs):
    """Zero task rewards for pure distillation; keep correctness for logging.

    Returns ``(raw_rewards, train_rewards)``: raw correctness goes into the
    logged ``raw_reward`` (training-accuracy curve, paper Fig. 2); training
    rewards are all 0.0 so advantages come solely from the OPD KL penalty.
    """
    missing = [
        i for i, s in enumerate(samples) if s.teacher_log_probs is None and s.response_length > 0
    ]
    assert not missing, (
        f"{len(missing)} samples are missing teacher_log_probs (indices {missing[:5]}...). "
        "Was reward_func wired via --custom-rm-path?"
    )
    if getattr(args, "crisp_kl_mode", "sampled") == "full":
        missing_topk = [
            i for i, s in enumerate(samples) if s.teacher_top_ids is None and s.response_length > 0
        ]
        assert not missing_topk, (
            f"crisp_kl_mode=full but {len(missing_topk)} samples are missing teacher_top_ids "
            f"(indices {missing_topk[:5]}...). reward_func must run with the same custom config."
        )
    raw_rewards = [float(s.reward) if s.reward is not None else 0.0 for s in samples]
    return raw_rewards, [0.0] * len(samples)


def _refresh_teacher_weights(args, rollout_id):
    """POST /update_weights_from_disk to the teacher server (theta_tilde <- theta)."""
    import os

    import requests

    hf_path = getattr(args, "crisp_teacher_hf_path", None)
    assert hf_path, (
        "crisp_teacher_update_interval is set but crisp_teacher_hf_path is not. "
        "Set both in the --custom-config-path YAML (and point --save-hf at the same path)."
    )
    if not os.path.exists(os.path.join(hf_path, "config.json")):
        raise FileNotFoundError(
            f"Teacher refresh at rollout {rollout_id}: no HF dump at {hf_path}. "
            "Ensure --save-hf points there and --save-interval == crisp_teacher_update_interval."
        )

    url = getattr(args, "crisp_teacher_url", None) or args.rm_url
    base = url.rsplit("/", 1)[0]
    logger.info(f"CRISP teacher refresh at rollout {rollout_id}: loading {hf_path} into {base}")
    resp = requests.post(f"{base}/update_weights_from_disk", json={"model_path": hf_path}, timeout=1200)
    resp.raise_for_status()
    body = resp.json()
    if not body.get("success", True):
        raise RuntimeError(f"Teacher refresh failed: {body}")
    logger.info(f"CRISP teacher refresh complete at rollout {rollout_id}")


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    """slime's default sglang rollout + CRISP periodic teacher refresh.

    Refreshes the teacher *before* generating rollout ``M, 2M, ...`` — by then
    the actor has saved its HF dump at rollout ``M-1`` (slime saves when
    ``(rollout_id + 1) % save_interval == 0``), so the teacher becomes the
    student after exactly M training steps: progressive compression.
    """
    interval = getattr(args, "crisp_teacher_update_interval", 0) or 0
    if not evaluation and interval > 0 and rollout_id > 0 and rollout_id % interval == 0:
        _refresh_teacher_weights(args, rollout_id)

    from slime.rollout.sglang_rollout import generate_rollout as default_generate_rollout

    return default_generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
