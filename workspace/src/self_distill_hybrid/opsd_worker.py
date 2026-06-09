"""
OPSD (On-Policy Self-Distillation) worker.

Extends verl's AsyncActorRolloutRefWorker with two registered methods:
  - ``update_opsd``    -- JSD / reverse-KL training step using the frozen ref
    model (``ref_module_fsdp``) as teacher and the trainable
    ``actor_module_fsdp`` as student.
  - ``update_teacher`` -- optional hard-copy of current student weights into
    the ref model (triggered every ``opsd.teacher_update_freq`` steps).

The training step does two forward passes per micro-batch (teacher no-grad +
student with-grad) and trains on ALL rollouts, not just correct ones.
"""

import logging
import math

import psutil
import torch
import torch.distributed
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.protocol import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.attention_utils import index_first_axis, rearrange, unpad_input
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

logger = logging.getLogger(__name__)


class OPSDWorker(AsyncActorRolloutRefWorker):
    """Hybrid-engine actor+rollout+ref worker with OPSD JSD/reverse-KL training.

    Inherits init_model, wake_up, sleep, generate_sequences, save_checkpoint,
    and the FSDP model/optimizer/scheduler plumbing from
    AsyncActorRolloutRefWorker.

    Adds:
      - ``update_opsd``    -- JSD / reverse-KL update against the frozen ref.
      - ``update_teacher`` -- hard-copy student weights into the ref module.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def update_teacher(self) -> DataProto:
        """Hard-copy student (actor) weights to teacher (ref) model.

        Iterates the FSDP parameter lists in lock-step and copies *shard*
        contents directly. The actor and ref modules are wrapped with the
        same FSDP policy on the same DP mesh during init, so their local
        parameter shards correspond element-wise — no unsharded
        materialisation is needed.

        The previous implementation wrapped the copy in nested
        ``FSDP.summon_full_params(...)``, which transiently materialises
        the FULL (un-sharded) actor and ref params on every rank — for
        Qwen3-14B that's ~30 GiB × 2 modules = ~60 GiB on top of the
        sharded params already on-GPU, and it OOMs on H100 80 GB when
        ``teacher_update_freq>0`` fires right after a training step that
        already loaded the actor for fwd/bwd.

        Uses ``Dispatch.ONE_TO_ALL`` because every FSDP rank needs to
        fire this op (each holds its own shards) and the call carries no
        data — verl's analogous control ops (init_model, save_checkpoint)
        use the same dispatch.

        Returns:
            DataProto with meta_info confirming the update.
        """
        assert self._is_actor, "update_teacher requires actor role"
        assert hasattr(self, "ref_module_fsdp") and self.ref_module_fsdp is not None, (
            "update_teacher requires ref_module_fsdp (ref model)"
        )

        # Load both models to GPU if offloaded
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
            load_fsdp_model_to_gpu(self.ref_module_fsdp)

        # Direct shard-level copy. Each rank's actor.parameters() iterator
        # yields its local shard; same for ref.parameters(). FSDP wraps both
        # with identical policy and world layout, so the i-th local shard
        # of actor maps to the i-th local shard of ref.
        with torch.no_grad():
            n_params = 0
            for p_student, p_teacher in zip(
                self.actor_module_fsdp.parameters(),
                self.ref_module_fsdp.parameters(),
            ):
                if p_student.shape != p_teacher.shape:
                    raise RuntimeError(
                        f"update_teacher shard mismatch at param {n_params}: "
                        f"actor shard {tuple(p_student.shape)} vs ref shard "
                        f"{tuple(p_teacher.shape)}. Both FSDP modules must "
                        "be wrapped with the same policy on the same mesh."
                    )
                p_teacher.data.copy_(p_student.data)
                n_params += 1

        logger.info("Teacher weights updated from student (shard-level, %d params)", n_params)

        # Offload back to CPU if needed
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.ref_module_fsdp)
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        return DataProto(meta_info={"teacher_updated": True})

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_opsd(self, data: DataProto) -> DataProto:
        """Perform one OPSD training step using JSD between teacher and student.

        The input DataProto should contain:
          - teacher_input_ids, teacher_attention_mask, teacher_position_ids, teacher_loss_mask
          - student_input_ids, student_attention_mask, student_position_ids, student_loss_mask

        The method:
          1. Loads actor FSDP model + optimizer + ref model to GPU (if offloaded)
          2. Splits data into micro-batches for gradient accumulation
          3. For each micro-batch:
             a. Forward teacher (ref_module_fsdp) with no_grad -> teacher logits
             b. Forward student (actor_module_fsdp) with grad -> student logits
             c. Compute JSD loss over response positions
          4. Clips gradients and steps optimizer + scheduler
          5. Offloads everything back to CPU (if offloaded)

        Returns:
            DataProto with meta_info containing training metrics.
        """
        assert self._is_actor, "update_opsd requires actor role"

        # --- Load actor model & optimizer to GPU if offloaded ---
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        # --- Load ref model to GPU if offloaded ---
        ref_offloaded = False
        if hasattr(self, "ref_module_fsdp") and self.ref_module_fsdp is not None:
            if self._is_offload_param:
                load_fsdp_model_to_gpu(self.ref_module_fsdp)
                ref_offloaded = True

        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # will be moved per micro-batch
            beta = data.meta_info.get("opsd_beta", 0.5)
            loss_type = data.meta_info.get("opsd_loss_type", "jsd")
            metrics = self._opsd_training_step(data, beta=beta, loss_type=loss_type)

            # LR scheduler step
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["opsd/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.actor_lr_scheduler.step()

            metrics["perf/max_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated() / (1024**3)
            )
            metrics["perf/max_memory_reserved_gb"] = (
                torch.cuda.max_memory_reserved() / (1024**3)
            )
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            output = DataProto(meta_info={"metrics": metrics})
            output = output.to("cpu")

        # --- Offload ref model back to CPU ---
        if ref_offloaded:
            offload_fsdp_model_to_cpu(self.ref_module_fsdp)
            log_gpu_memory_usage("After offload ref model during update_opsd", logger=logger)

        # --- Offload actor model & optimizer back to CPU ---
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_opsd", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_opsd", logger=logger)

        return output

    # ------------------------------------------------------------------
    # Core OPSD training logic
    # ------------------------------------------------------------------

    def _opsd_training_step(
        self, data: DataProto, beta: float = 0.5, loss_type: str = "jsd"
    ) -> dict:
        """Divergence-based training with gradient accumulation.

        For each micro-batch:
          1. Forward teacher (frozen ref model) on teacher_input_ids -> teacher logits
          2. Forward student (trainable actor) on student_input_ids -> student logits
          3. Extract response logits using respective loss_masks
          4. Compute divergence loss (JSD or reverse KL)
          5. Backward with gradient accumulation scaling

        Args:
            data: DataProto with teacher_* and student_* tensors.
            beta: JSD interpolation parameter (0.5 = symmetric JSD).
                  Unused for reverse_kl but passed through for API consistency.
            loss_type: "jsd" or "reverse_kl".

        Returns:
            Dictionary of training metrics.
        """
        self.actor_module_fsdp.train()
        self.ref_module_fsdp.eval()

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_liger = self.config.model.get("use_liger", False)

        micro_batch_size = self.config.actor.get(
            "ppo_micro_batch_size_per_gpu",
            self.config.actor.get("micro_batch_size_per_gpu", 2),
        )

        batch_size = data.batch["student_input_ids"].shape[0]
        if batch_size == 0:
            return {"opsd/loss": 0.0, "opsd/num_tokens": 0, "opsd/batch_size": 0}

        micro_batches = data.split(micro_batch_size)
        n_micro_batches = len(micro_batches)
        grad_accum = max(1, n_micro_batches)

        device = get_device_id()

        self.actor_optimizer.zero_grad()
        total_loss = 0.0
        total_tokens = 0
        total_student_entropy = 0.0
        total_teacher_entropy = 0.0
        total_entropy_tokens = 0

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(device)

            # Extract teacher tensors
            t_input_ids = micro_batch.batch["teacher_input_ids"]
            t_attention_mask = micro_batch.batch["teacher_attention_mask"]
            t_position_ids = micro_batch.batch["teacher_position_ids"]
            t_loss_mask = micro_batch.batch["teacher_loss_mask"]

            # Extract student tensors
            s_input_ids = micro_batch.batch["student_input_ids"]
            s_attention_mask = micro_batch.batch["student_attention_mask"]
            s_position_ids = micro_batch.batch["student_position_ids"]
            s_loss_mask = micro_batch.batch["student_loss_mask"]

            # Teacher forward (frozen, no grad)
            with torch.no_grad():
                if use_remove_padding:
                    teacher_logits = self._forward_logits_unpadded(
                        self.ref_module_fsdp, t_input_ids, t_attention_mask, t_position_ids,
                        t_loss_mask,
                    )
                else:
                    teacher_logits = self._forward_logits_padded(
                        self.ref_module_fsdp, t_input_ids, t_attention_mask, t_position_ids,
                        t_loss_mask,
                    )

            # Student forward (trainable, with grad)
            if use_remove_padding:
                student_logits = self._forward_logits_unpadded(
                    self.actor_module_fsdp, s_input_ids, s_attention_mask, s_position_ids,
                    s_loss_mask,
                )
            else:
                student_logits = self._forward_logits_padded(
                    self.actor_module_fsdp, s_input_ids, s_attention_mask, s_position_ids,
                    s_loss_mask,
                )

            # Teacher and student tokenize the SAME response_text after their
            # respective prompts. Teacher prompt is longer (question + teacher
            # solution) than student prompt (question only), but response_ids
            # are identical, so the shifted loss_mask selects exactly K = len(
            # response_ids) logits on each side. The invariant is enforced
            # upstream in two places:
            #   (1) sd_dataset.SelfDistillDataset.filter_overlong_prompts drops
            #       rows whose teacher prompt exceeds data.max_prompt_length.
            #   (2) sd_verifier.build_opsd_batch drops a pair if either side's
            #       prompt+response would exceed max_length in _tokenize_sequence
            #       (returns None), preventing one-sided truncation.
            # This assert is a defensive cross-check, not the primary guarantee.
            assert teacher_logits.shape[0] == student_logits.shape[0], (
                f"teacher/student response-token count mismatch: "
                f"{teacher_logits.shape[0]} vs {student_logits.shape[0]}. "
                "Upstream invariant (filter_overlong_prompts + build_opsd_batch) "
                "was violated; check data.max_prompt_length and opsd.sft_max_length."
            )
            n_resp = teacher_logits.shape[0]
            if n_resp == 0:
                continue

            t_logits_aligned = teacher_logits
            s_logits_aligned = student_logits
            del teacher_logits  # free memory

            # Compute divergence loss
            loss_fn_map = {
                "jsd": (self._compute_jsd_loss, self._compute_jsd_loss_liger),
                "reverse_kl": (self._compute_reverse_kl_loss, self._compute_reverse_kl_loss_liger),
            }
            if loss_type not in loss_fn_map:
                raise ValueError(f"Unknown loss_type: {loss_type!r}. Expected one of {list(loss_fn_map)}")
            fn_standard, fn_liger = loss_fn_map[loss_type]

            if use_liger:
                loss, n_tokens = fn_liger(t_logits_aligned, s_logits_aligned, beta=beta)
            else:
                loss, n_tokens = fn_standard(t_logits_aligned, s_logits_aligned, beta=beta)

            # Compute entropy for both teacher and student (no grad needed)
            with torch.no_grad():
                if use_liger:
                    s_ent, t_ent = self._compute_entropy_liger(
                        s_logits_aligned, t_logits_aligned
                    )
                else:
                    s_ent, t_ent = self._compute_entropy(
                        s_logits_aligned, t_logits_aligned
                    )
                total_student_entropy += s_ent * n_resp
                total_teacher_entropy += t_ent * n_resp
                total_entropy_tokens += n_resp

            del t_logits_aligned

            # Scale for gradient accumulation
            scaled_loss = loss / grad_accum
            scaled_loss.backward()

            total_loss += loss.detach().item()
            total_tokens += n_tokens

        # --- Gradient clipping and optimizer step ---
        grad_clip = self.config.actor.get("grad_clip", 1.0)
        if isinstance(self.actor_module_fsdp, FSDP):
            grad_norm = self.actor_module_fsdp.clip_grad_norm_(max_norm=grad_clip)
        else:
            from torch.distributed._composable.fsdp import FSDPModule
            if isinstance(self.actor_module_fsdp, FSDPModule):
                from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_
                grad_norm = fsdp2_clip_grad_norm_(
                    self.actor_module_fsdp.parameters(), max_norm=grad_clip
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor_module_fsdp.parameters(), max_norm=grad_clip
                )

        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()

        if torch.isfinite(grad_norm):
            self.actor_optimizer.step()
        else:
            logger.warning("Non-finite grad_norm (%.4f), skipping optimizer step", grad_norm.item())
            self.actor_optimizer.zero_grad()

        avg_loss = total_loss / max(1, n_micro_batches)

        avg_student_entropy = total_student_entropy / max(1, total_entropy_tokens)
        avg_teacher_entropy = total_teacher_entropy / max(1, total_entropy_tokens)

        return {
            "opsd/loss": avg_loss,
            "opsd/grad_norm": grad_norm.detach().item(),
            "opsd/num_tokens": int(total_tokens),
            "opsd/batch_size": batch_size,
            "opsd/beta": beta,
            "opsd/use_liger": int(use_liger),
            "opsd/student_entropy": avg_student_entropy,
            "opsd/teacher_entropy": avg_teacher_entropy,
            "opsd/entropy_diff": avg_student_entropy - avg_teacher_entropy,
        }

    # ------------------------------------------------------------------
    # Forward logits: Padded path
    # ------------------------------------------------------------------

    def _forward_logits_padded(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning response-position logits (padded path).

        Args:
            model: FSDP model (ref_module_fsdp or actor_module_fsdp).
            input_ids: (B, max_len)
            attention_mask: (B, max_len)
            position_ids: (B, max_len)
            loss_mask: (B, max_len) — 1 for response tokens

        Returns:
            Flattened response logits: (N_response_tokens, vocab_size)
            where N_response_tokens = sum of loss_mask shifted positions.
        """
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = outputs.logits  # (B, max_len, V)
            del outputs

        # Shift for next-token prediction: logits[t] predicts token[t+1]
        shift_logits = logits[:, :-1, :]  # (B, max_len-1, V)
        shift_loss_mask = loss_mask[:, 1:]  # (B, max_len-1)
        del logits

        # Flatten and select response positions
        B, S, V = shift_logits.shape
        flat_logits = shift_logits.reshape(B * S, V)
        flat_mask = shift_loss_mask.reshape(B * S)
        del shift_logits

        response_indices = flat_mask.nonzero(as_tuple=True)[0]
        response_logits = flat_logits[response_indices]  # (N, V)

        return response_logits

    # ------------------------------------------------------------------
    # Forward logits: Unpadded path
    # ------------------------------------------------------------------

    def _forward_logits_unpadded(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning response-position logits (unpadded path).

        Same as _forward_logits_padded but uses flash_attn_varlen for memory
        efficiency on long sequences.

        Returns:
            Flattened response logits: (N_response_tokens, vocab_size)
        """
        # Unpad inputs
        input_ids_rmpad, indices, *_ = unpad_input(
            input_ids.unsqueeze(-1), attention_mask
        )
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)  # (1, total_nnz)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids_rmpad,
                attention_mask=None,  # triggers flash_attn_varlen
                position_ids=position_ids_rmpad,
                use_cache=False,
            )
            logits_rmpad = outputs.logits.squeeze(0)  # (total_nnz, V)
            del outputs

        # Unpad loss_mask using the same indices, then shift
        loss_mask_flat = loss_mask.reshape(-1)
        loss_mask_rmpad = loss_mask_flat[indices]  # (total_nnz,)

        # Shifted loss mask: position i predicts token i+1
        shifted_loss_mask = torch.zeros_like(loss_mask_rmpad)
        shifted_loss_mask[:-1] = loss_mask_rmpad[1:]

        # Select response-position logits
        response_indices = shifted_loss_mask.nonzero(as_tuple=True)[0]
        response_logits = logits_rmpad[response_indices]  # (N, V)

        return response_logits

    # ------------------------------------------------------------------
    # JSD loss computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_jsd_loss(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        beta: float = 0.5,
        chunk_size: int = 128,
    ) -> tuple[torch.Tensor, int]:
        """Compute Jensen-Shannon divergence between teacher and student logits.

        JSD_beta(p_T || p_S) = beta * KL(p_T || m) + (1-beta) * KL(p_S || m)
        where m = beta * p_T + (1-beta) * p_S

        Processes tokens in chunks to bound the peak size of any single
        intermediate (N, V) probability tensor. Teacher-side intermediates
        get freed each chunk (no grad). Student-side intermediates are
        retained in the autograd graph until backward, so per-step peak is
        still proportional to N — but in-flight transient peaks are bounded
        by ~chunk_size * V * 4 bytes per fp32 buffer.

        Args:
            teacher_logits: (N, V) — logits from frozen teacher (no grad).
            student_logits: (N, V) — logits from trainable student (with grad).
            beta: Interpolation parameter (0.5 = symmetric JSD).
            chunk_size: Number of tokens to process at a time.

        Returns:
            (loss, n_tokens) — scalar loss and number of tokens.
        """
        n_tokens = teacher_logits.shape[0]
        if n_tokens == 0:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0

        # Accumulate JSD sum over chunks of tokens.
        # student_logits[start:end] is a view; .float() below allocates a
        # new fp32 chunk but stays differentiable (grad flows back to bf16).
        jsd_sum = torch.tensor(0.0, device=student_logits.device)

        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)

            # Convert chunks to float32 for numerical stability
            t_chunk = teacher_logits[start:end].float()
            s_chunk = student_logits[start:end].float()

            t_log_probs = F.log_softmax(t_chunk, dim=-1)
            s_log_probs = F.log_softmax(s_chunk, dim=-1)
            del t_chunk, s_chunk

            t_probs = t_log_probs.exp()
            s_probs = s_log_probs.exp()

            # Mixture: m = beta * p_T + (1-beta) * p_S
            m_log_probs = (beta * t_probs + (1.0 - beta) * s_probs).clamp(min=1e-8).log()

            # KL(p_T || m) per token
            kl_t = (t_probs * (t_log_probs - m_log_probs)).sum(dim=-1)
            del t_probs, t_log_probs

            # KL(p_S || m) per token
            kl_s = (s_probs * (s_log_probs - m_log_probs)).sum(dim=-1)
            del s_probs, s_log_probs, m_log_probs

            jsd_chunk = beta * kl_t + (1.0 - beta) * kl_s
            jsd_sum = jsd_sum + jsd_chunk.sum()
            del kl_t, kl_s, jsd_chunk

        loss = jsd_sum / n_tokens
        return loss, n_tokens

    @staticmethod
    def _compute_entropy(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        chunk_size: int = 128,
    ) -> tuple[float, float]:
        """Compute average per-token entropy for student and teacher distributions.

        H(p) = -sum(p * log(p))

        Uses chunked processing to avoid OOM, same as _compute_jsd_loss.

        Args:
            student_logits: (N, V) — student logits.
            teacher_logits: (N, V) — teacher logits.
            chunk_size: Number of tokens to process at a time.

        Returns:
            (student_entropy, teacher_entropy) — average per-token entropy (nats).
        """
        n_tokens = student_logits.shape[0]
        if n_tokens == 0:
            return 0.0, 0.0

        s_entropy_sum = 0.0
        t_entropy_sum = 0.0

        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)

            s_log_probs = F.log_softmax(student_logits[start:end].float(), dim=-1)
            s_probs = s_log_probs.exp()
            s_entropy_sum += -(s_probs * s_log_probs).sum(dim=-1).sum().item()
            del s_probs, s_log_probs

            t_log_probs = F.log_softmax(teacher_logits[start:end].float(), dim=-1)
            t_probs = t_log_probs.exp()
            t_entropy_sum += -(t_probs * t_log_probs).sum(dim=-1).sum().item()
            del t_probs, t_log_probs

        return s_entropy_sum / n_tokens, t_entropy_sum / n_tokens

    # ------------------------------------------------------------------
    # Liger-style JSD loss: logsumexp mixture + progressive teacher freeing
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_jsd_loss_liger(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        beta: float = 0.5,
        chunk_size: int = 64,
    ) -> tuple[torch.Tensor, int]:
        """Memory-efficient JSD using logsumexp for the mixture distribution.

        Improvements over ``_compute_jsd_loss`` (per-chunk transient peak;
        student-side intermediates are still pinned in the autograd graph):
          1. **logsumexp mixture**: Computes log(beta*p_T + (1-beta)*p_S) via
             logsumexp in log-space, avoiding explicit probability tensors.
          2. **Progressive teacher freeing**: Clones teacher logits into chunks
             and frees the original, so only one teacher chunk is alive at a time.
          3. **Smaller default chunk_size** (256 vs 512) for lower per-chunk peak.

        Args:
            teacher_logits: (N, V) — logits from frozen teacher (no grad).
            student_logits: (N, V) — logits from trainable student (with grad).
            beta: JSD interpolation parameter (0.5 = symmetric JSD).
            chunk_size: Number of tokens per chunk.

        Returns:
            (loss, n_tokens) — scalar loss and number of response tokens.
        """
        n_tokens = teacher_logits.shape[0]
        if n_tokens == 0:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0

        # Pre-compute log(beta) and log(1-beta) for logsumexp mixture
        log_beta = math.log(beta) if beta > 0 else float("-inf")
        log_1m_beta = math.log(1.0 - beta) if beta < 1 else float("-inf")

        # Clone teacher chunks and free the original contiguous tensor.
        # Teacher has no grad, so cloning is cheap and frees ~N*V*2 bytes.
        teacher_chunks = [c.clone() for c in teacher_logits.split(chunk_size, dim=0)]
        del teacher_logits

        jsd_sum = torch.tensor(0.0, device=student_logits.device)

        for i, t_chunk in enumerate(teacher_chunks):
            start = i * chunk_size
            end = start + t_chunk.shape[0]

            # Convert to float32 for numerical stability
            t_lp = F.log_softmax(t_chunk.float(), dim=-1)
            s_lp = F.log_softmax(student_logits[start:end].float(), dim=-1)
            del t_chunk
            teacher_chunks[i] = None  # allow GC

            # logsumexp mixture: log_m = log(beta * exp(t_lp) + (1-beta) * exp(s_lp))
            #                         = logsumexp([t_lp + log_beta, s_lp + log_1m_beta])
            log_m = torch.logsumexp(
                torch.stack([t_lp + log_beta, s_lp + log_1m_beta], dim=0),
                dim=0,
            )

            # KL(p_T || m) and KL(p_S || m) via F.kl_div with log_target=True
            kl_t = F.kl_div(log_m, t_lp, reduction="none", log_target=True).sum(dim=-1)
            del t_lp
            kl_s = F.kl_div(log_m, s_lp, reduction="none", log_target=True).sum(dim=-1)
            del s_lp, log_m

            jsd_chunk = beta * kl_t + (1.0 - beta) * kl_s
            jsd_sum = jsd_sum + jsd_chunk.sum()
            del kl_t, kl_s, jsd_chunk

        loss = jsd_sum / n_tokens
        return loss, n_tokens

    # ------------------------------------------------------------------
    # Reverse KL loss: KL(student || teacher)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_reverse_kl_loss(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        beta: float = 0.5,
        chunk_size: int = 128,
    ) -> tuple[torch.Tensor, int]:
        """Compute reverse KL divergence: KL(student || teacher).

        KL(p_S || p_T) = sum_x p_S(x) * [log p_S(x) - log p_T(x)]

        This is "mode-seeking": the student concentrates probability mass on
        the teacher's high-probability tokens, which encourages the student
        to adopt the teacher's concise reasoning style without spreading
        probability over tokens the teacher assigns low weight to.

        The ``beta`` parameter is accepted for API compatibility with JSD
        callers but is unused (reverse KL has no mixture parameter).

        Args:
            teacher_logits: (N, V) — logits from frozen teacher (no grad).
            student_logits: (N, V) — logits from trainable student (with grad).
            beta: Unused, kept for API compatibility with JSD.
            chunk_size: Number of tokens to process at a time.

        Returns:
            (loss, n_tokens) — scalar mean reverse-KL loss and token count.
        """
        n_tokens = teacher_logits.shape[0]
        if n_tokens == 0:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0

        kl_sum = torch.tensor(0.0, device=student_logits.device)

        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)

            t_log_probs = F.log_softmax(teacher_logits[start:end].float(), dim=-1)
            s_log_probs = F.log_softmax(student_logits[start:end].float(), dim=-1)

            # KL(p_S || p_T) = sum p_S * (log p_S - log p_T)
            s_probs = s_log_probs.exp()
            kl_chunk = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)
            del t_log_probs, s_log_probs, s_probs

            kl_sum = kl_sum + kl_chunk.sum()
            del kl_chunk

        loss = kl_sum / n_tokens
        return loss, n_tokens

    @staticmethod
    def _compute_reverse_kl_loss_liger(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        beta: float = 0.5,
        chunk_size: int = 64,
    ) -> tuple[torch.Tensor, int]:
        """Memory-efficient reverse KL: KL(student || teacher).

        Same as ``_compute_reverse_kl_loss`` but with progressive teacher
        freeing (clone teacher chunks, delete original) for lower peak memory.

        Args:
            teacher_logits: (N, V) — logits from frozen teacher (no grad).
            student_logits: (N, V) — logits from trainable student (with grad).
            beta: Unused, kept for API compatibility with JSD.
            chunk_size: Number of tokens per chunk.

        Returns:
            (loss, n_tokens) — scalar mean reverse-KL loss and token count.
        """
        n_tokens = teacher_logits.shape[0]
        if n_tokens == 0:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0

        teacher_chunks = [c.clone() for c in teacher_logits.split(chunk_size, dim=0)]
        del teacher_logits

        kl_sum = torch.tensor(0.0, device=student_logits.device)

        for i, t_chunk in enumerate(teacher_chunks):
            start = i * chunk_size
            end = start + t_chunk.shape[0]

            t_lp = F.log_softmax(t_chunk.float(), dim=-1)
            s_lp = F.log_softmax(student_logits[start:end].float(), dim=-1)
            del t_chunk
            teacher_chunks[i] = None

            # KL(p_S || p_T) via F.kl_div(input=log_p_T, target=log_p_S, log_target=True)
            # F.kl_div with log_target=True computes: exp(target) * (target - input)
            # i.e. sum p_S * (log p_S - log p_T) = KL(p_S || p_T)
            kl_chunk = F.kl_div(t_lp, s_lp, reduction="none", log_target=True).sum(dim=-1)
            del t_lp, s_lp

            kl_sum = kl_sum + kl_chunk.sum()
            del kl_chunk

        loss = kl_sum / n_tokens
        return loss, n_tokens

    # ------------------------------------------------------------------
    # Liger-style entropy: logsumexp-based, smaller chunks
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_entropy_liger(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        chunk_size: int = 64,
    ) -> tuple[float, float]:
        """Memory-efficient per-token entropy using smaller chunks.

        H(p) = -sum(p * log p), computed in log-space via log_softmax.

        Args:
            student_logits: (N, V) — student logits.
            teacher_logits: (N, V) — teacher logits.
            chunk_size: Number of tokens per chunk.

        Returns:
            (student_entropy, teacher_entropy) — average per-token entropy (nats).
        """
        n_tokens = student_logits.shape[0]
        if n_tokens == 0:
            return 0.0, 0.0

        s_entropy_sum = 0.0
        t_entropy_sum = 0.0

        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)

            # Student entropy: H = -sum(p * log_p)
            s_lp = F.log_softmax(student_logits[start:end].float(), dim=-1)
            s_entropy_sum += -(s_lp.exp() * s_lp).sum(dim=-1).sum().item()
            del s_lp

            # Teacher entropy
            t_lp = F.log_softmax(teacher_logits[start:end].float(), dim=-1)
            t_entropy_sum += -(t_lp.exp() * t_lp).sum(dim=-1).sum().item()
            del t_lp

        return s_entropy_sum / n_tokens, t_entropy_sum / n_tokens
