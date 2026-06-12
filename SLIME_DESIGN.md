# slime Design Notes & CRISP Integration Plan

Notes on the [THUDM/slime](https://github.com/THUDM/slime) framework (submodule at `slime/`, pinned `ee72ab5`, v0.3.0-25), focused on the three subsystems that matter for porting CRISP: **rollout generation**, the **training engine**, and **weight update** — plus a concrete plan for running CRISP on slime.

**One-liner:** Megatron-Core for training + SGLang for inference, orchestrated by Ray. Nearly everything is pluggable via `--*-function-path` hooks (rollout fn, reward fn, data source, advantage fn, loss hooks).

---

## 0. Top-level control flow

Two drivers at the repo root:

```
train.py        — synchronous loop; supports colocate (train+rollout time-slice the same GPUs)
train_async.py  — disaggregated only; one-step pipelining (rollout k+1 generates while step k trains)
```

Per `rollout_id`, `train.py` does:

```python
rollout_data_ref = rollout_manager.generate(rollout_id)   # sglang rollouts → per-DP-rank ray Boxes
actor_model.async_train(rollout_id, rollout_data_ref)     # Megatron fwd/bwd
actor_model.update_weights()                              # push new weights into sglang engines
# (+ optional offload/onload dance in colocate mode, periodic eval/save)
```

`train_async.py` overlaps generation and training, syncing weights every `--update-weights-interval` rollouts (always after the in-flight generation finishes — never mid-generation).

GPU layout (`slime/ray/placement_group.py`): one Ray placement group; bundles sorted by node-IP + GPU id. **Colocate** = rollout engines share the actor's GPUs (offset 0); **disaggregated** = rollout GPUs appended after actor GPUs.

---

## 1. Rollout generation

### Actors and topology

- **`RolloutManager`** (`slime/ray/rollout.py`, CPU-only Ray actor) owns:
  - the **`DataSource`** (`slime/rollout/data_source.py`) — prompt dataset with epoch/offset/shuffle state, checkpointable; `RolloutDataSourceWithBuffer` adds a buffer for partial/aborted rollouts;
  - the **rollout function** (`--rollout-function-path`, default `slime/rollout/sglang_rollout.py::generate_rollout`) and eval function;
  - the **SGLang fleet**.
- **Fleet topology:** `RolloutServer` (one per *model* — multi-model via `--sglang-config` YAML, each with its own router) → `ServerGroup`s (homogeneous engines; supports prefill/decode **PD-disaggregation**, encoder groups, placeholders) → **`SGLangEngine`** Ray actors (`slime/backends/sglang_utils/sglang_engine.py`).
- Each `SGLangEngine` is a thin wrapper that **spawns an sglang HTTP server subprocess** and registers it with an `sglang_router`. All generation goes over HTTP through the router (load balancing / consistent-hash sessions). This is server-based, unlike verl's in-process engines.
- Health monitors restart dead engines; `RolloutServer.recover()` re-creates them and flags `num_new_engines` so the next weight update reconnects + re-pushes weights.

### Generation flow (`sglang_rollout.py`)

1. `RolloutManager.generate(rollout_id)` → rollout fn with the data source.
2. Asyncio fan-out of **groups** (`n_samples_per_prompt` samples per prompt; semaphore = `sglang_server_concurrency × num_engines`).
3. Each sample POSTs `input_ids` to the router `/generate` with `return_logprob=True`. Response **tokens and logprobs are appended directly to the `Sample`** (token-in/token-out — no retokenization, exact tokens for training).
4. Reward via `rm_hub` (per-sample or whole-group `--group-rm`).
5. **Over-sampling + dynamic-filter loop**: keeps submitting `over_sampling_batch_size` prompts and filtering groups (e.g. drop zero-std GRPO groups) until `rollout_batch_size` good groups are collected; then **aborts** stragglers. With `--partial-rollout`, aborted partial generations return to the buffer and resume next rollout (their off-policy prefix can be loss-masked).
6. `Sample` (`slime/utils/types.py`) is the universal currency: `tokens`, `response_length`, `loss_mask`, `reward`, `rollout_log_probs`, `teacher_log_probs`, `metadata`, status.

### Conversion to train data (in the manager, not the trainer)

`_convert_samples_to_train_data`:
- reward post-processing **including GRPO group normalization** (mean-center ± std-normalize per group);
- loss masks; `rollout_mask_sums` (per-rollout mask totals, so token-weighted loss stays correct when first-fit packing splits a rollout across micro-batches);
- optional passthroughs: `rollout_log_probs` (for off-policy correction), `teacher_log_probs` (OPD), routed experts (R3: rollout routing replay for MoE).

`_split_train_data_by_dp` → `build_dp_schedule`: sequence-length-balanced first-fit packing into micro-batches, one `ray.put` **Box per DP rank**.

---

## 2. Training engine

- **`RayTrainGroup`** (`slime/ray/actor_group.py`) spawns one **`MegatronTrainRayActor`** (`slime/backends/megatron_utils/actor.py`) per GPU; `torch.distributed` + full Megatron-Core parallelism (TP / PP / EP / CP / VPP, distributed optimizer).

### The key design trick: weight tags (`TensorBackuper`)

There is **one set of live Megatron GPU buffers**. CPU-backed snapshots live under tags:

| tag | role |
|---|---|
| `actor` | the trainable policy (re-backed-up after every train step) |
| `ref` | frozen reference for KL (`--ref-load`); refreshable via `--ref-update-interval` |
| `teacher` | OPD teacher (`--opd-teacher-load`, megatron mode) |
| `old_actor` / `rollout_actor` | behavior policy queue for off-policy correction (`--keep-old-actor`) |

`_switch_model(tag)` restores a tag into the live model. Ref/teacher forward passes are **weight swaps, not second resident models** — unlike verl's `ActorRolloutRef` (separately materialized FSDP ref model). Memory cost is CPU RAM + swap time, not GPU.

### `train_actor` sequence (per rollout)

1. swap→`ref`, `forward_only` → `ref_log_probs` (if KL enabled)
2. swap→`teacher`, `forward_only` → `teacher_log_probs` (OPD megatron mode)
3. swap→`actor` (or `old_actor`), `forward_only` → `log_probs` (old/behavior log-probs)
4. `compute_advantages_and_returns` (`loss.py`) — estimator: grpo / gspo / ppo / reinforce++ / custom
5. `train()` (`model.py`) — Megatron pipeline-scheduled fwd/bwd over micro-batches; `loss_type` ∈ {policy_loss, sft_loss, value_loss, custom_loss}
6. re-backup `actor`; optionally refresh `ref` every `--ref-update-interval` rollouts

Colocate offload: `torch_memory_saver` pause/resume + destroy/reload of NCCL process groups (`sleep`/`wake_up`).

---

## 3. Weight update (train → rollout)

Common protocol (all transports): `weight_version += 1` → `pause_generation` + `flush_cache` on engines → stream weights in **bounded buckets** (`--update-weight-buffer-size`) → `continue_generation`. Per bucket: TP all-gather → EP all-gather (experts) → **Megatron→HF conversion on the fly** (`megatron_to_hf/` per-arch converters; optional fp8/int4 quantization) → send.

| Class | Mode | Transport |
|---|---|---|
| `UpdateWeightFromTensor` | colocate | flatten bucket → CUDA-IPC serialize → gloo-gather to the engine-owning rank → `engine.update_weights_from_tensor` (zero-copy, same GPUs) |
| `UpdateWeightFromDistributed` | disaggregated | dedicated NCCL group per PP-source rank (`slime-pp_{k}` = train rank + all engine GPUs); broadcast; a Ray `Lock` serializes buckets to avoid NCCL deadlock |
| `UpdateWeightFromDistributedDelta` | disaggregated | only changed params, delta-encoded — for fast frequent sync |
| `UpdateWeightFromDisk` | fallback | save HF checkpoint; engines `update_weights_from_disk` |

`--check-weight-update-equal` snapshots engine weights and bit-compares after the first push. CI also verifies `weight_version` propagated to engines.

---

## 4. Native on-policy distillation (OPD) in slime

`examples/on_policy_distillation/` + `slime/rollout/on_policy_distillation.py` + `loss.py::apply_opd_kl_to_advantages`.

- **Two teacher modes** (`--opd-type`):
  - **`sglang`** — teacher on an external SGLang server. During rollout, a "reward func" POSTs the student's `sample.tokens` with `max_new_tokens=0, return_logprob=True, logprob_start_len=0` — a **prefill-only scoring pass** — and `post_process_rewards` trims the returned logprobs to the response span → `sample.teacher_log_probs`. Scalar reward = 0.
  - **`megatron`** — teacher checkpoint loaded as the `teacher` weight tag; `teacher_log_probs` computed by a training-side forward pass. Requires same architecture as the policy.
- **Loss formulation** — *not* a separate loss; an additive advantage penalty, orthogonal to the estimator:

  ```
  adv_t -= opd_kl_coef · (log π_student(y_t) − log π_teacher(y_t))
  ```

  i.e. the Tinker-cookbook **sampled-token reverse-KL** pushed through the policy-gradient machinery (reward 0 for pure distillation).

### Crucial difference vs. our verl CRISP implementation

| | CRISP on verl (`workspace/`) | slime OPD |
|---|---|---|
| KL estimator | **full-vocabulary** per-token `KL(q_t‖p_t)` from teacher+student logits (exact, dense gradient) | **sampled-token** `log q(y_t) − log p(y_t)` via advantage (unbiased policy-gradient estimate, higher variance) |
| Teacher | same model, **conciseness prompt**, periodic hard-copy from student | separate (usually larger) model, static |
| Teacher pass | training-side forward with swapped *prompt* (`sd_prompt`) | sglang prefill-only scoring of `sample.tokens` (same prompt) or megatron forward |
| GT answers | never in loss | reward = 0 in pure distillation (same property) |

---

## 5. CRISP-on-slime integration plan

Goal: reproduce CRISP (teacher = same model + conciseness instruction, reverse KL on student rollouts, teacher refresh every M steps) on slime's infra.

### What's free

- On-policy rollouts + per-step weight sync: core loop.
- Teacher log-probs on student tokens + reverse-KL training signal: `--use-opd --opd-kl-coef`.
- No-GT training: OPD already uses reward 0.
- DAPO-math-17k data, Qwen3-8B recipes: `examples/on_policy_distillation/run-qwen3-8B-opd.sh` is the template.

### What needs building

1. **Conciseness-prompt teacher scoring (the core change).** A custom CRISP reward func (variant of `on_policy_distillation.reward_func`): instead of scoring `sample.tokens` verbatim, build
   `teacher_input_ids = tokenize(conciseness_prompt(question)) + response_tokens`
   and request logprobs with `logprob_start_len = len(teacher_prompt_ids)`; store the response-span logprobs as `sample.teacher_log_probs`. This is exactly our `sd_prompt`/`sft_prompt` pair, expressed as a prefill-only scoring call. Alignment invariant from `METHOD.md` carries over: same response tokens after different prompts → positions align by construction (no padding/truncation games needed since sglang scores exact token ids).
2. **Teacher refresh every M steps (progressive compression).**
   - *Megatron mode*: mirror `--ref-update-interval` — add `--opd-teacher-update-interval` that re-backups `actor → teacher` (one `weights_backuper.backup("teacher")` call in `train_actor`). Cleanest.
   - *sglang mode*: point the teacher scoring at **the same updatable engines** (teacher = current policy + conciseness prompt). That is M=1 — which the paper showed is catastrophically unstable — so for M>1 either keep a second non-updatable model entry in `--sglang-config` refreshed manually, or use megatron mode.
3. **Choice of KL estimator.**
   - *Option A (use slime as-is)*: sampled-token reverse KL via `apply_opd_kl_to_advantages`. Cheap; different gradient variance than the paper; needs empirical validation that compression/accuracy match Table 2.
   - *Option B (faithful)*: add a `full_kl` distillation loss type — teacher forward keeps full logits (or top-k) and computes exact per-token reverse KL like `opsd_worker._compute_reverse_kl_loss`. Fully specified in `workspace/slime_crisp/FULL_KL_PLAN.md`: sglang `top_logprobs_num` on the existing scoring call + bucketed reverse KL as a `custom_loss` (`--custom-loss-function-path`), ~5-file plumbing patch to the slime fork.
   - Recommendation: start with A (a few hundred lines total, mostly config), benchmark on MATH-500 against the verl numbers, fall back to B only if the sampled estimator can't reproduce the compression–accuracy trade-off.
4. **Config**: student prompt = original DAPO prompt (already the OPD example's data path); temperature 1.0; `n_samples_per_prompt=1` (CRISP uses single rollouts; note GRPO normalization is degenerate at n=1 — use reward 0 + pure OPD penalty, estimator effectively REINFORCE with zero advantage base); batch 32; lr 1e-6; `rollout_max_response_len=8192`.
5. **Eval**: MATH-500/AIME via slime's `--eval-datasets` + a `rm_hub` scorer mirroring `dual_path_math_verify` (slime ships `math_dapo_utils`/`math_utils` which match the paper's footnote-1 grading).

### Suggested first milestone

Qwen3-8B, megatron-mode teacher initialized from the same checkpoint, custom conciseness-prompt scoring func, `--opd-teacher-update-interval 50`, sampled-KL estimator (Option A), 100 steps on DAPO-17k → compare MATH-500 acc/len against `workspace/` Table-2 numbers (86.6% / 1,921 tok for 8B).
