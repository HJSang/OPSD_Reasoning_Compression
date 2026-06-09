"""
OPSD (On-Policy Self-Distillation) Trainer using VERL's HybridEngine.

Orchestrates the OPSD training loop:
  1. Generate: Use sft_prompt (question-only) for student rollout generation
  2. Verify: Check math correctness for metrics (but do NOT filter)
  3. Train: JSD update on ALL rollouts using frozen teacher (ref model)

Key differences from SelfDistillTrainer (sd_trainer.py):
  - Generation uses sft_prompt (question-only), not sd_prompt (with teacher solution)
  - Training uses JSD loss between teacher and student logit distributions
  - ALL rollouts are trained on, not just correct ones
  - Teacher model is the frozen ref_module_fsdp (initial policy weights)
"""

import json
import logging
import os
import time
import uuid
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.checkpoint_engine import CheckpointEngineManager
from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics

from .sd_verifier import build_opsd_batch, verify_batch

py_logger = logging.getLogger(__name__)


class OPSDTrainer:
    """OPSD trainer: JSD-based on-policy self-distillation.

    The student generates from question-only prompts, and training minimizes
    the JSD divergence between teacher (frozen ref model) and student
    distributions on ALL student rollouts (not just correct ones).

    Args:
        config: OmegaConf config (based on opsd_trainer.yaml).
        tokenizer: HuggingFace tokenizer.
        role_worker_mapping: Mapping from roles to worker classes.
        resource_pool_manager: Manager for Ray resource pools.
        ray_worker_group_cls: Class for Ray worker groups.
        processor: Optional multimodal processor.
        train_dataset: SD prompt dataset.
        collate_fn: Batch collation function.
        device_name: Device name for training.
        val_reward_fn: Reward manager for generation-based validation.
        val_dataset: RL-format dataset for generation-based validation.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, type],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        collate_fn=None,
        device_name=None,
        val_reward_fn=None,
        val_dataset: Optional[Dataset] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "OPSDTrainer requires hybrid_engine=True"
        assert Role.ActorRolloutRef in role_worker_mapping, (
            f"OPSDTrainer requires ActorRolloutRef in role_worker_mapping (for ref model), "
            f"got {list(role_worker_mapping.keys())}"
        )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        # Actor-side DP world size = total GPUs / Ulysses-SP. Used as the
        # divisor when padding batches before the nd-compute dispatcher slices
        # along the actor mesh's DP axis.
        #
        # TP is rollout-only (sglang's tensor_model_parallel_size) — it does
        # NOT collapse the actor's DP axis. The actor's FSDP+Ulysses mesh has
        # dimensions (dp, ulysses_sp), so dp_world = total_gpus / ulysses_sp.
        # Earlier we divided by TP*SP, which under-padded (or skipped padding
        # entirely) when TP>1: e.g. TP=2 SP=4 8GPUs gave dp_world=1, but the
        # dispatcher actually requested chunks=2 and the batch crashed with
        # AssertionError("only support equal chunk").
        total_gpus = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        sp = self.config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
        self.dp_world = max(1, total_gpus // int(sp))
        py_logger.info(
            "DP world = %d (total_gpus=%d, ulysses_sp=%d)",
            self.dp_world, total_gpus, sp,
        )

        # OPSD-specific config
        self.opsd_config = self.config.get("opsd", {})
        self.beta = self.opsd_config.get("beta", 0.5)
        self.loss_type = self.opsd_config.get("loss_type", "jsd")  # "jsd" or "reverse_kl"
        self.sft_max_length = self.opsd_config.get("sft_max_length", 32768)
        self.check_structure = self.opsd_config.get("check_structure", True)
        self.log_sample_count = self.opsd_config.get("log_sample_count", 5)
        self.test_freq = self.opsd_config.get("test_freq", 10)
        self.log_freq = self.opsd_config.get("log_freq", 5)
        self.teacher_update_freq = self.opsd_config.get("teacher_update_freq", 0) or 0

        # Detailed logging
        self.log_dir = self.opsd_config.get(
            "detailed_log_dir",
            os.path.join(
                self.config.trainer.get("default_local_dir", "checkpoints"),
                "detailed_logs",
            ),
        )
        self.rollout_log_dir = os.path.join(self.log_dir, "rollout")
        self.opsd_log_dir = os.path.join(self.log_dir, "opsd")
        self.val_log_dir = os.path.join(self.log_dir, "val_generations")
        os.makedirs(self.rollout_log_dir, exist_ok=True)
        os.makedirs(self.opsd_log_dir, exist_ok=True)
        os.makedirs(self.val_log_dir, exist_ok=True)
        py_logger.info("Rollout logs -> %s", self.rollout_log_dir)
        py_logger.info("OPSD logs    -> %s", self.opsd_log_dir)
        py_logger.info("Val gen logs -> %s", self.val_log_dir)

        # Create dataloader
        self._create_dataloader(train_dataset, collate_fn)

        # Generation-based validation
        self.val_reward_fn = val_reward_fn
        self._build_val_dataloader(val_dataset)

    # ------------------------------------------------------------------
    # Initialization helpers (same patterns as SelfDistillTrainer)
    # ------------------------------------------------------------------

    def _create_dataloader(self, train_dataset: Optional[Dataset], collate_fn):
        if train_dataset is None:
            raise ValueError("train_dataset must be provided for OPSDTrainer")

        self.train_dataset = train_dataset

        if collate_fn is None:
            from .sd_dataset import collate_fn as sd_collate_fn
            collate_fn = sd_collate_fn

        batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
        num_workers = self.config.data.get("dataloader_num_workers", 0)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            shuffle=self.config.data.get("shuffle", True),
        )
        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.get("total_training_steps") is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        py_logger.info("Dataloader batches: %d", len(self.train_dataloader))
        py_logger.info("Total training steps: %d", self.total_training_steps)

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
        except Exception as e:
            py_logger.warning("Could not set total_training_steps in config: %s", e)

    def _build_val_dataloader(self, val_dataset: Optional[Dataset]):
        self.val_dataloader = None
        if val_dataset is None or self.val_reward_fn is None:
            return

        from verl.utils.dataset.rl_dataset import collate_fn as rl_collate_fn

        val_batch_size = self.config.data.get("val_batch_size", None)
        if val_batch_size is None:
            val_batch_size = len(val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            num_workers=0,
            shuffle=False,
            drop_last=False,
            collate_fn=rl_collate_fn,
        )

        py_logger.info(
            "Generation-based val dataloader ready: %d samples, %d batch(es)",
            len(val_dataset), len(self.val_dataloader),
        )

    def _dump_val_generations(
        self, inputs, outputs, scores, reward_extra_infos_dict,
        data_sources=None, token_counts=None,
    ):
        """Dump validation samples as JSONL, one file per global step."""
        filename = os.path.join(
            self.val_log_dir, f"step_{self.global_steps:06d}.jsonl"
        )

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        if data_sources is not None and len(data_sources) == n:
            base_data["data_source"] = list(data_sources)
        if token_counts is not None and len(token_counts) == n:
            base_data["response_tokens"] = token_counts

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        py_logger.info(
            "Dumped %d val generations to %s", n, filename
        )

    def _validate(self) -> dict:
        """Generate responses on val prompts and evaluate with reward function."""
        if self.val_dataloader is None or self.val_reward_fn is None:
            return {}

        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_uids = []
        sample_token_counts = []

        val_kwargs = self.config.actor_rollout_ref.rollout.val_kwargs
        val_n = val_kwargs.get("n", 1)
        val_do_sample = val_kwargs.get("do_sample", False)
        val_temperature = val_kwargs.get("temperature", None)
        val_top_p = val_kwargs.get("top_p", None)
        val_top_k = val_kwargs.get("top_k", None)
        val_max_tokens = self.config.opsd.get("val_max_tokens", None)

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                    dtype=object,
                )

            test_batch = test_batch.repeat(repeat_times=val_n, interleave=True)

            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch.get("reward_model", {}).get("style") == "model"
            ):
                return {}

            reward_model_keys = {"data_source", "reward_model", "extra_info", "uid"} & test_batch.non_tensor_batch.keys()
            non_tensor_keys_to_pop = set(test_batch.non_tensor_batch.keys()) - reward_model_keys
            test_gen_batch = test_batch.pop(
                batch_keys=[],
                non_tensor_batch_keys=list(non_tensor_keys_to_pop),
            )
            test_gen_batch.non_tensor_batch.update(test_batch.non_tensor_batch)

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": val_do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            if val_temperature is not None:
                test_gen_batch.meta_info["temperature"] = val_temperature
            if val_top_p is not None:
                test_gen_batch.meta_info["top_p"] = val_top_p
            if val_top_k is not None:
                test_gen_batch.meta_info["top_k"] = val_top_k
            if val_max_tokens is not None:
                test_gen_batch.meta_info["max_new_tokens"] = val_max_tokens

            size_divisor = self.config.actor_rollout_ref.rollout.get("agent", {}).get(
                "num_workers", self.actor_rollout_wg.world_size
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(
                test_gen_batch_padded
            )
            # Do NOT sleep replicas here. Mirrors verl PPO's _validate, which
            # only sleeps when a colocated reward model needs the GPU, and
            # then immediately wakes it back up. Sleeping unconditionally
            # would leave replicas asleep when _validate returns, causing the
            # next training-iteration generate_sequences to hit dead workers.
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids
            ]
            sample_outputs.extend(output_texts)
            # Count non-pad response tokens
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            for ids in output_ids:
                sample_token_counts.append(int((ids != pad_id).sum().item()))

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            input_ids = test_batch.batch["prompts"]
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids
            ]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            reward_result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = reward_result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_info = reward_result.get("reward_extra_info", {})
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(
                        values if isinstance(values, list) else [values]
                    )

            data_source_lst.append(
                test_batch.non_tensor_batch.get(
                    "data_source", ["unknown"] * reward_tensor.shape[0]
                )
            )

        if not sample_scores:
            return {}

        # Sanity check: print first prompt/response pair
        py_logger.info(
            "Step %d: val sample[0] score=%.2f\n  PROMPT: %.300s\n  RESPONSE: %.500s",
            self.global_steps,
            sample_scores[0],
            sample_inputs[0],
            sample_outputs[0],
        )

        data_sources = np.concatenate(data_source_lst, axis=0)
        data_src2var2metric2val = process_validation_metrics(
            data_sources, sample_uids, reward_extra_infos_dict
        )
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max(
                    int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()
                )
                for metric_name, metric_val in metric2val.items():
                    if (
                        var_name == core_var
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and f"@{n_max}" in metric_name
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # Response length metrics (overall + per data_source)
        metric_dict["val/avg_response_tokens"] = sum(sample_token_counts) / max(1, len(sample_token_counts))
        ds_token_counts: dict[str, list] = defaultdict(list)
        for ds, tc in zip(data_sources, sample_token_counts):
            ds_token_counts[ds].append(tc)
        for ds, counts in ds_token_counts.items():
            metric_dict[f"val/{ds}/avg_response_tokens"] = sum(counts) / len(counts)

        # Dump val generations to JSONL for inspection
        self._dump_val_generations(
            sample_inputs, sample_outputs, sample_scores,
            reward_extra_infos_dict,
            data_sources=data_sources,
            token_counts=sample_token_counts,
        )

        py_logger.info(
            "Step %d: _validate() complete -- %d samples, %d metrics",
            self.global_steps, len(sample_scores), len(metric_dict),
        )
        return metric_dict

    # ------------------------------------------------------------------
    # Worker initialization
    # ------------------------------------------------------------------

    def init_workers(self):
        """Initialize distributed workers using Ray backend.

        Creates the actor+rollout+ref worker group, the AgentLoopManager, and
        the CheckpointEngineManager that syncs trainer weights into the
        sglang rollout replicas. Requires ActorRolloutRef role for the frozen
        teacher (ref model).
        """
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(
            Role.ActorRolloutRef
        )
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRolloutRef],
            config=self.config.actor_rollout_ref,
            role=str(Role.ActorRolloutRef),
        )
        self.resource_pool_to_cls[actor_rollout_resource_pool][str(Role.ActorRolloutRef)] = actor_rollout_cls

        all_wg = {}
        wg_kwargs = {"device_name": self.device_name}

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        self.actor_rollout_wg = all_wg[str(Role.ActorRolloutRef)]
        self.actor_rollout_wg.init_model()

        # Create AgentLoopManager for async generation
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get(
            "agent_loop_manager_class"
        )
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # New verl 0.7.x AgentLoopManager exposes a `.create(...)` classmethod
        # that drives the async _initialize_llm_servers + _init_agent_loop_workers
        # init phases. The bare constructor leaves agent_loop_workers unset, which
        # crashes generate_sequences() on the first val/train step.
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
        )

        # CheckpointEngineManager bridges the trainer FSDP model and the sglang
        # rollout replicas. Without update_weights() after each actor step,
        # generation would keep using the initial weights — breaking the
        # on-policy assumption of OPSD.
        checkpoint_engine_config = omega_conf_to_dataclass(
            self.config.actor_rollout_ref.rollout.checkpoint_engine
        )
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )
        # Start with rollout asleep; fit() will wake + sync before first gen.
        self.checkpoint_manager.sleep_replicas()

        py_logger.info("OPSD workers initialized successfully")

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
            if self.config.trainer.get("default_hdfs_dir")
            else None
        )

        local_mkdir_safe(actor_local_path)

        max_ckpt = self.config.trainer.get("max_actor_ckpt_to_keep", None)

        py_logger.info("Saving checkpoint to %s", actor_local_path)
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
            max_ckpt_to_keep=max_ckpt,
        )
        py_logger.info("Checkpoint saved: step %d", self.global_steps)

    # ------------------------------------------------------------------
    # Teacher weight update
    # ------------------------------------------------------------------

    def _update_teacher_weights(self):
        """Hard-copy student weights to teacher (ref) model via workers.

        Dispatched as Dispatch.ONE_TO_ALL on the worker side, so no input
        payload is needed.
        """
        self.actor_rollout_wg.update_teacher()
        py_logger.info(
            "Step %d: Teacher weights updated from student (teacher_update_freq=%d)",
            self.global_steps, self.teacher_update_freq,
        )

    # ------------------------------------------------------------------
    # Core Training Loop
    # ------------------------------------------------------------------

    def fit(self):
        """Main training loop: epoch-based OPSD.

        For each epoch, iterates over the SD prompt dataset:
          1. Swap raw_prompt to sft_prompt (question-only) for generation
          2. Generate student responses via AgentLoopManager
          3. Verify correctness for metrics (but do NOT filter)
          4. Train on ALL responses using JSD between teacher and student
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        progress_bar = tqdm(total=self.total_training_steps, desc="OPSD Training")

        # Wake rollout replicas and push initial trainer weights so the very
        # first generation reflects the actual init checkpoint, not whatever
        # sglang loaded internally.
        self.checkpoint_manager.update_weights()

        self.global_steps += 1

        # Optional: run generation-based validation before training starts
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", False):
            py_logger.info("Running val_before_train...")
            val_metrics = self._validate()
            if val_metrics:
                logger.log(data=val_metrics, step=self.global_steps)

        for epoch in range(self.config.trainer.total_epochs):
            epoch_metrics = {
                "epoch/total_generated": 0,
                "epoch/total_correct": 0,
                "epoch/total_trained": 0,
                "epoch/steps": 0,
            }

            for batch_dict in self.train_dataloader:
                step_t0 = time.time()
                metrics = {}

                # ---- Phase 1: Generate (question-only prompt) ----
                batch = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    dtype=object,
                )
                batch.meta_info["global_steps"] = self.global_steps
                # Cap training generation to sd_max_tokens so we don't generate
                # up to the full response_length (which may be larger for val).
                sd_max_tokens = self.config.opsd.get("sd_max_tokens", None)
                if sd_max_tokens is not None:
                    batch.meta_info["max_new_tokens"] = int(sd_max_tokens)

                # OPSD: save original teacher prompts as JSON strings before swapping
                # raw_prompt contains parsed sd_prompt (list of dicts); convert to JSON
                # so _opsd_update() can pass them to build_opsd_batch().
                sd_prompts_json = np.empty(len(batch.non_tensor_batch["raw_prompt"]), dtype=object)
                for i in range(len(batch.non_tensor_batch["raw_prompt"])):
                    sd_prompts_json[i] = json.dumps(batch.non_tensor_batch["raw_prompt"][i])
                batch.non_tensor_batch["sd_prompt"] = sd_prompts_json

                # Swap raw_prompt to sft_prompt (question-only) for student generation
                for i in range(len(batch.non_tensor_batch["raw_prompt"])):
                    batch.non_tensor_batch["raw_prompt"][i] = json.loads(
                        batch.non_tensor_batch["sft_prompt"][i]
                    )

                gen_t0 = time.time()
                gen_output = self.async_rollout_manager.generate_sequences(batch)
                # Release rollout GPU memory so the training forward/backward
                # passes don't OOM. Matches the verl PPO loop.
                self.checkpoint_manager.sleep_replicas()
                gen_time = time.time() - gen_t0

                gen_output.meta_info.pop("timing", None)

                # ---- Phase 2: Verify (metrics only, no filtering) ----
                verify_t0 = time.time()
                responses, correct_mask, predictions = self._verify_responses(
                    gen_output, batch
                )
                verify_time = time.time() - verify_t0

                batch_size = len(responses)
                n_correct = sum(correct_mask)

                metrics["sd/batch_size"] = batch_size
                metrics["sd/n_correct"] = n_correct
                metrics["sd/accuracy"] = n_correct / max(1, batch_size)
                metrics["timing/generate_s"] = gen_time
                metrics["timing/verify_s"] = verify_time

                # Response length metrics (tokens)
                _prompt_len = gen_output.batch["prompts"].shape[1]
                _max_resp_tokens = gen_output.batch["responses"].shape[1]
                student_token_counts = []
                for i in range(batch_size):
                    resp_attn = gen_output.batch["attention_mask"][i, _prompt_len:]
                    student_token_counts.append(resp_attn.sum().item())
                metrics["sd/avg_student_token_count"] = sum(student_token_counts) / max(1, len(student_token_counts))
                metrics["sd/max_student_token_count"] = max(student_token_counts) if student_token_counts else 0
                metrics["sd/min_student_token_count"] = min(student_token_counts) if student_token_counts else 0

                # Truncation rate: response filled the entire allocated token buffer
                _n_truncated = sum(
                    1
                    for tc in student_token_counts
                    if tc == _max_resp_tokens
                )
                metrics["sd/truncation_rate"] = _n_truncated / max(1, batch_size)
                metrics["sd/n_truncated"] = _n_truncated

                teacher_solutions = list(batch.non_tensor_batch.get("teacher_solution", []))
                if teacher_solutions:
                    teacher_lens = [len(t) for t in teacher_solutions]
                    metrics["sd/avg_teacher_response_len"] = sum(teacher_lens) / max(1, len(teacher_lens))

                # ---- Phase 3: Train (OPSD JSD on ALL responses) ----
                train_t0 = time.time()
                opsd_metrics = self._opsd_update(batch, responses)
                train_time = time.time() - train_t0

                metrics.update(opsd_metrics)
                metrics["timing/train_s"] = train_time

                # ---- Teacher weight update (if configured) ----
                if self.teacher_update_freq > 0 and self.global_steps % self.teacher_update_freq == 0:
                    teacher_t0 = time.time()
                    self._update_teacher_weights()
                    metrics["timing/teacher_update_s"] = time.time() - teacher_t0
                    metrics["opsd/teacher_updated"] = 1.0

                # Sync the freshly-updated student weights into the rollout
                # replicas so the next training/val generation is on-policy.
                weight_sync_t0 = time.time()
                self.checkpoint_manager.update_weights()
                metrics["timing/weight_sync_s"] = time.time() - weight_sync_t0

                epoch_metrics["epoch/total_generated"] += batch_size
                epoch_metrics["epoch/total_correct"] += n_correct
                epoch_metrics["epoch/total_trained"] += batch_size  # ALL responses trained
                epoch_metrics["epoch/steps"] += 1

                # ---- Log samples ----
                if self.global_steps % self.log_freq == 0:
                    self._log_rollout_samples(batch, responses, correct_mask, predictions)

                # ---- Phase 4: Validation ----
                is_last_step = self.global_steps >= self.total_training_steps
                is_val_step = self.test_freq > 0 and self.global_steps % self.test_freq == 0

                if self.val_reward_fn is not None and (is_val_step or is_last_step):
                    val_gen_t0 = time.time()
                    val_gen_metrics = self._validate()
                    val_gen_time = time.time() - val_gen_t0
                    metrics.update(val_gen_metrics)
                    metrics["timing/val_gen_s"] = val_gen_time

                # ---- Logging ----
                step_time = time.time() - step_t0
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                    "timing/step_s": step_time,
                })

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)

                # ---- Checkpoint ----
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    self._save_checkpoint()

                self.global_steps += 1

                if is_last_step:
                    py_logger.info(
                        "Epoch %d summary -- Generated: %d, Correct: %d, Trained: %d",
                        epoch, epoch_metrics["epoch/total_generated"],
                        epoch_metrics["epoch/total_correct"],
                        epoch_metrics["epoch/total_trained"],
                    )
                    logger.log(data=epoch_metrics, step=self.global_steps - 1)
                    progress_bar.close()
                    return

            # End-of-epoch summary
            total_gen = epoch_metrics["epoch/total_generated"]
            total_cor = epoch_metrics["epoch/total_correct"]
            py_logger.info(
                "Epoch %d complete -- Generated: %d, Correct: %d (%.1f%%), Trained: %d (ALL)",
                epoch, total_gen, total_cor,
                100 * total_cor / max(1, total_gen),
                epoch_metrics["epoch/total_trained"],
            )
            logger.log(data=epoch_metrics, step=self.global_steps - 1)

            epoch_metrics = {
                "epoch/total_generated": 0,
                "epoch/total_correct": 0,
                "epoch/total_trained": 0,
                "epoch/steps": 0,
            }

        progress_bar.close()
        py_logger.info("OPSD training complete!")

    # ------------------------------------------------------------------
    # Phase 2: Verification (metrics only)
    # ------------------------------------------------------------------

    def _verify_responses(self, gen_output: DataProto, original_batch: DataProto):
        """Decode generated responses and verify correctness (for metrics only)."""
        prompt_length = gen_output.batch["prompts"].shape[1]
        batch_size = gen_output.batch["responses"].shape[0]

        responses = []
        for i in range(batch_size):
            response_ids = gen_output.batch["responses"][i]
            resp_attn_mask = gen_output.batch["attention_mask"][i, prompt_length:]
            valid_ids = response_ids[resp_attn_mask.bool()]
            text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
            responses.append(text)

        ground_truths = list(original_batch.non_tensor_batch["ground_truth"])
        correct_mask, predictions = verify_batch(
            responses, ground_truths, check_structure=self.check_structure
        )

        return responses, correct_mask, predictions

    # ------------------------------------------------------------------
    # Phase 3: OPSD Update
    # ------------------------------------------------------------------

    def _opsd_update(self, original_batch: DataProto, responses: list[str]) -> dict:
        """Build OPSD batch and dispatch JSD training to workers.

        Trains on ALL responses (no correctness filtering).

        Args:
            original_batch: Original batch with sd_prompt and sft_prompt.
            responses: ALL student-generated response strings.

        Returns:
            Dictionary of training metrics.
        """
        teacher_prompts = list(original_batch.non_tensor_batch["sd_prompt"])
        student_prompts = list(original_batch.non_tensor_batch["sft_prompt"])

        opsd_batch = build_opsd_batch(
            teacher_prompts=teacher_prompts,
            student_prompts=student_prompts,
            responses=responses,
            tokenizer=self.tokenizer,
            max_length=self.sft_max_length,
        )

        if opsd_batch is None:
            return {"opsd/loss": 0.0, "opsd/skipped": 1.0}

        n_samples = opsd_batch.batch["student_input_ids"].shape[0]

        # DP-pad: ensure batch is divisible by DP world size (not total GPUs).
        # The nd-compute dispatcher slices along the actor mesh's DP axis,
        # so divisibility by dp_world is what's required.
        n_dp = self.dp_world
        if n_samples % n_dp != 0:
            pad_to = ((n_samples // n_dp) + 1) * n_dp
            pad_count = pad_to - n_samples
            padded_dict = {}
            for key in opsd_batch.batch.keys():
                tensor = opsd_batch.batch[key]
                last = tensor[-1:].expand(pad_count, *tensor.shape[1:]).clone()
                padded_dict[key] = torch.cat([tensor, last], dim=0)
            opsd_batch = DataProto.from_single_dict(padded_dict)
            py_logger.debug(
                "Step %d: Padded OPSD batch from %d to %d for %d DP workers",
                self.global_steps, n_samples, pad_to, n_dp,
            )

        # Pass config via meta_info so the worker can read it
        opsd_batch.meta_info["opsd_beta"] = self.beta
        opsd_batch.meta_info["opsd_loss_type"] = self.loss_type

        py_logger.info(
            "Step %d: OPSD update with %d samples (beta=%.2f, loss=%s)",
            self.global_steps, n_samples, self.beta, self.loss_type,
        )

        # Dispatch to workers
        opsd_output = self.actor_rollout_wg.update_opsd(opsd_batch)

        # Reduce metrics across DP workers
        opsd_metrics = reduce_metrics(opsd_output.meta_info["metrics"])
        opsd_metrics["opsd/n_samples"] = n_samples
        return opsd_metrics

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_rollout_samples(
        self,
        batch: DataProto,
        responses: list[str],
        correct_mask: list[bool],
        predictions: list[str],
    ):
        """Log student generation samples."""
        n_log = min(self.log_sample_count, len(responses))
        if n_log == 0:
            return

        questions = list(batch.non_tensor_batch["question"])
        ground_truths = list(batch.non_tensor_batch["ground_truth"])
        teacher_solutions = list(batch.non_tensor_batch.get("teacher_solution", []))
        sft_prompts = list(batch.non_tensor_batch.get("sft_prompt", []))
        sd_prompts = list(batch.non_tensor_batch.get("sd_prompt", []))

        n_correct = sum(correct_mask)
        avg_resp_len = sum(len(r) for r in responses) / max(1, len(responses))
        py_logger.info(
            "Step %d rollout: %d/%d correct, avg response len: %.0f chars",
            self.global_steps, n_correct, len(correct_mask), avg_resp_len,
        )
        for i in range(n_log):
            status = "correct" if correct_mask[i] else "wrong"
            q_preview = str(questions[i])[:80] if i < len(questions) else "?"
            resp_preview = responses[i][:120].replace("\n", "\\n") if responses[i] else "(empty)"
            py_logger.info("  [%s] Q: %s...", status, q_preview)
            py_logger.info("       Pred: %s | GT: %s", predictions[i], ground_truths[i])
            py_logger.info("       Resp: %s...", resp_preview)

        samples = []
        for i in range(n_log):
            # sft_prompt and sd_prompt are stored as JSON strings; parse for readability
            sft_prompt_parsed = None
            if i < len(sft_prompts):
                try:
                    sft_prompt_parsed = json.loads(sft_prompts[i])
                except (json.JSONDecodeError, TypeError):
                    sft_prompt_parsed = str(sft_prompts[i])

            sd_prompt_parsed = None
            if i < len(sd_prompts):
                try:
                    sd_prompt_parsed = json.loads(sd_prompts[i])
                except (json.JSONDecodeError, TypeError):
                    sd_prompt_parsed = str(sd_prompts[i])

            samples.append({
                "sample_idx": i,
                "question": str(questions[i]) if i < len(questions) else "",
                "student_generation_prompt": sft_prompt_parsed,
                "teacher_logit_prompt": sd_prompt_parsed,
                "student_response": responses[i],
                "teacher_response": str(teacher_solutions[i]) if i < len(teacher_solutions) else "",
                "ground_truth": str(ground_truths[i]) if i < len(ground_truths) else "",
                "prediction": str(predictions[i]),
                "is_correct": bool(correct_mask[i]),
            })

        step_file = os.path.join(self.rollout_log_dir, f"step_{self.global_steps:06d}.json")
        with open(step_file, "w", encoding="utf-8") as f:
            json.dump(
                {"step": self.global_steps, "n_samples": n_log, "samples": samples},
                f, indent=2, ensure_ascii=False,
            )
