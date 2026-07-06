# CRISP — Method & Algorithm (with code-vs-paper verification)

This document describes the CRISP / OPSD reasoning-compression algorithm **as implemented in `workspace/`** and verifies it against the paper *CRISP: Compressed Reasoning via Iterative Self-Policy Distillation* (`crisp_compressed_reasoning_via_iterative_self_policy_distillation.pdf`).

**Verdict up front:** the implementation faithfully matches the paper. The default loss, the on-policy training loop, the teacher-refresh mechanism, the prompt templates, and the training hyperparameters all correspond exactly. The code additionally implements a few options the paper does not center on (JSD loss, memory-efficient "liger" loss variants, a word-limit teacher template, and an OPSD-with-reference-solution teacher) — these are supersets, not contradictions. Details and the exact correspondence are below.

---

## 1. The idea in one paragraph

A reasoning model already knows how to be concise; it just needs permission. CRISP takes **one model** and conditions it two ways via the prompt:

- **Student** `π_θ(· | x)` — the original math prompt `x`, no special instruction.
- **Teacher** `π_θ̃(· | x, c)` — the same problem prefixed with a **conciseness instruction** `c` ("Solve concisely, be direct…").

Training generates **student** rollouts and minimizes the **per-token reverse KL** between the student and the (stop-gradient) teacher distribution on the student's own tokens. No ground-truth answers, no token budgets, no reward model, no difficulty estimator. The conciseness signal emerges from the KL objective and adapts to problem difficulty automatically.

---

## 2. Problem formulation (paper §3.1)

A reasoning model `π_θ` maps input `x` to output `y = (r, a)` — a reasoning trace `r` inside `<think>…</think>` followed by an answer `a`. Goal: learn `θ*` that produces **shorter traces while maintaining accuracy**. The student gets the original DAPO-17K prompt `x`; the teacher gets the same prompt prefixed with conciseness instruction `c`.

---

## 3. Training objective (paper §3.2, Eq. 1)

CRISP minimizes per-token **reverse KL** between student and a **stop-gradient** teacher, on **student-generated** rollouts:

```
L(θ) = E_{x~D, y~π_θ(·|x)} [ Σ_t  D_KL( π_θ(· | x, y_<t)  ‖  π_θ̃(· | x, c, y_<t) ) ]
```

- `θ̃` = teacher weights (periodically synced with the student; stop-gradient — no gradient flows through the teacher's forward pass).
- `y ~ π_θ(·|x)` makes training **on-policy** — the student is optimized on its own generations, which avoids the distribution shift of off-policy SFT.
- **Reverse** KL `D_KL(student ‖ teacher)` is mode-seeking: the student is penalized for placing mass where the teacher assigns low probability, but *not* for being uncertain where the teacher is uncertain. The paper argues this is what makes iterative teacher refreshes stable (forward KL over-reacts to refreshes and collapses accuracy — Appendix I).
- In practice the loss is **normalized by the number of response tokens `|y|`**.

### Code correspondence ✅

`workspace/src/self_distill_hybrid/opsd_worker.py::_compute_reverse_kl_loss`:

```python
# KL(p_S || p_T) = sum p_S * (log p_S - log p_T)
s_probs = s_log_probs.exp()
kl_chunk = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)   # reverse KL, student‖teacher
...
loss = kl_sum / n_tokens                                          # normalized by |y|
```

- Student logits computed **with grad** (`actor_module_fsdp`), teacher logits **with `torch.no_grad()`** (`ref_module_fsdp`) → stop-gradient teacher. ✅
- Division by `n_tokens` → the "normalized by `|y_i|`" note in Algorithm 1. ✅
- Loss direction is exactly `KL(student ‖ teacher)` = reverse KL. ✅

---

## 4. Teacher parameterization (paper §3.3, Eq. 2)

Two regimes, both supported in code:

1. **Fully frozen teacher** `θ̃ = θ_0` (the Zhao et al. 2026 OPSD-style baseline). Simple and stable, but becomes a progressively weaker compression target as the student internalizes conciseness.
2. **Periodic teacher update** (CRISP default): every `M` steps, hard-copy student weights into the teacher,
   ```
   θ̃ ← θ   every M steps
   ```
   Each refresh makes a stronger compression target → *progressive compression*. The paper uses **M = 50** for all main experiments; the ablation (Appendix C.2, Fig. 5) shows **M ∈ {40, 50, 60}** is a stable plateau, **M = 10** degrades, and **M = 1** is catastrophic (entropy explodes, accuracy collapses to ~2%).

### Code correspondence ✅

`opsd_worker.py::update_teacher` — hard-copies student FSDP shards into the ref module in lock-step (no `summon_full_params`, to avoid OOM on Qwen3-14B). Driven by `opsd.teacher_update_freq` (`TEACHER_UPDATE_FREQ` env var):

- `TEACHER_UPDATE_FREQ=0` → never refresh = **fully frozen teacher** (regime 1). ✅
- `TEACHER_UPDATE_FREQ=50` → **periodic refresh every 50 steps** = CRISP default (regime 2). ✅
- `execution-configs/*-tu{1,10,20,40,60,80,100}.json` reproduce the exact `M`-sweep ablation from Fig. 5. ✅

> Note: the YAML default `opsd.teacher_update_freq: 0` is the frozen baseline; the README launch examples and execution-configs set `M=50` for the paper's main results.

---

## 5. Training algorithm (paper Algorithm 1)

```
Input:  model π_θ, dataset D = {x_i}, conciseness instruction c,
        learning rate η, teacher update interval M
Output: compressed model π_θ*

Initialize teacher: θ̃ ← θ_0
for each training step k = 1, 2, …:
    if k mod M == 0:
        θ̃ ← θ                                   # periodic refresh
    sample batch {x_1, …, x_B} ~ D
    for each x_i in batch:
        y_i ~ π_θ(· | x_i)                       # student rollout (on-policy)
        for each token position t = 1 … |y_i|:
            q_t ← π_θ (· | x_i,    y_{i,<t})      # student logits (grad)
            p_t ← π_θ̃(· | x_i, c, y_{i,<t})      # teacher logits (NO grad)
            D_KL(q_t ‖ p_t)
        L_i ← Σ_t D_KL(q_t ‖ p_t)
    θ ← θ − η ∇_θ (1/B) Σ_i L_i                  # normalized by |y_i| in practice
return π_θ*
```

### Code correspondence — where each step lives ✅

| Algorithm 1 step | Implementation |
|---|---|
| `θ̃ ← θ_0` init teacher | `main_opsd.py` maps worker to `Role.ActorRolloutRef` so the **ref model = frozen teacher** is materialized |
| `if k mod M == 0: θ̃ ← θ` | `opsd_trainer.fit()` calls `_update_teacher_weights()` → `update_teacher()` when `global_steps % teacher_update_freq == 0` |
| sample batch | `SelfDistillDataset` + `StatefulDataLoader` (`sd_dataset.py`) |
| `y_i ~ π_θ(·\|x_i)` student rollout | `fit()` swaps the prompt to **`sft_prompt` (question-only)** and calls `async_rollout_manager.generate_sequences` (sglang) |
| `q_t` student logits (grad) | `opsd_worker._forward_logits_*(actor_module_fsdp, student_*…)` |
| `p_t` teacher logits (no grad) | `opsd_worker._forward_logits_*(ref_module_fsdp, teacher_*…)` inside `torch.no_grad()` |
| `Σ_t D_KL(q_t‖p_t)`, normalize | `_compute_reverse_kl_loss` → `kl_sum / n_tokens` |
| `θ ← θ − η∇L` | grad-accum over micro-batches, `clip_grad_norm_`, `actor_optimizer.step()` in `_opsd_training_step` |
| weight sync to rollout | `CheckpointEngineManager.update_weights()` after each step → next generation is on-policy |

**Trains on ALL rollouts.** The loop verifies correctness (`verify_batch`) but uses it **only for metrics — never to filter the batch**, matching the paper's "no ground-truth answers" claim (GT touches logging, never the loss). ✅

---

## 6. Prompt templates (paper Appendix B, Fig. 3 & 4) — exact match ✅

The paper's prompt boxes are **byte-identical** to `workspace/config/prompts.json`.

**Student** `π_θ(·|x)` — the original DAPO-17K prompt, used verbatim (`prepare_length_prune_data.get_original_prompt_json`):

> Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer … Remember to put your answer on its own line after "Answer:".

**Teacher** `π_θ̃(·|x,c)` — conciseness instruction `c` (`prompts.json → length_prune_teacher`):

> **Solve the following math problem concisely and correctly. Be direct — avoid unnecessary elaboration, redundant steps, or restating the problem. Focus only on the key reasoning steps needed to reach the answer.** The last line of your response should be of the form Answer: $Answer … Remember to put your answer on its own line after "Answer:".

**Soft-budget teacher** `π_θ̃(·|x,c_p)` (ablation, `prompts.json → length_prune_teacher_percent_reduce`):

> Solve the following math problem correctly **using {p}% fewer tokens than you normally would.** Be more concise — cut unnecessary elaboration, redundant steps, and verbose explanations while preserving correctness. …

The ablation (paper Table 4: qualitative "be concise" beats explicit `p ∈ {20,50,80}%` targets) is reproduced by the `length_prune_concise` vs `length_prune_{20,50,80}pct` data variants and the matching `execution-configs/*-{20,50,80}pct.json`. ✅

---

## 7. The load-bearing implementation invariant

Teacher and student sequences share the **same response tokens** but have **different-length prompts** (teacher prompt is longer). The reverse-KL loss aligns teacher vs. student response logits **by position**, so per-sample response-token counts must match exactly. The code enforces this by **dropping** (never truncating) any over-length pair:

- `sd_dataset.SelfDistillDataset` drops rows whose teacher prompt exceeds `data.max_prompt_length`.
- `sd_verifier.build_opsd_batch` / `_tokenize_sequence` return `None` (drop the pair) if either side exceeds `opsd.sft_max_length`.
- `opsd_worker._opsd_training_step` asserts `teacher_logits.shape[0] == student_logits.shape[0]` as a defensive cross-check.

This is an engineering detail not in the paper's math, but it is exactly what makes the per-token KL of Eq. 1 well-defined across the two differently-prompted sequences.

---

## 8. Training hyperparameters (paper §5.1 & Appendix F) — match ✅

| Setting | Paper | Code (README launch / execution-configs) |
|---|---|---|
| Models | Qwen3-8B, Qwen3-14B | `MODEL_PATH=…/Qwen3-8B` / `…/Qwen3-14B` |
| Data | ~13,600 DAPO-Math-17k, **no GT in loss** | `DAPO-Math-17k-dedup`, train on all rollouts |
| Epochs | 1 (~100 steps to converge; step 100 = default ckpt) | `TOTAL_EPOCHS=1` |
| Learning rate | 1e-6 | `LEARNING_RATE=1e-6` |
| Batch size | 32 | `TRAIN_BATCH_SIZE=32` |
| Teacher update `M` | 50 | `TEACHER_UPDATE_FREQ=50` |
| Rollout | single (n=1), temperature 1.0, max 8,192 tokens | `rollout.n=1`, `SD_TEMPERATURE=1.0`, `SD_MAX_TOKENS=8192` |
| Eval token budgets | 8,192 and 30,000 | `VAL_MAX_TOKENS=30000` (and 8K variant) |
| Eval sampling | mean@8 | `val_kwargs.n=8` |
| Hardware | 1 node × 8 H200 | `N_GPUS=8`, `nnodes=1` |
| Framework | verl HybridEngine + sglang | verl `deec5d02`, `rollout.name=sglang` |
| Parallelism | FSDP + Ulysses-SP deg 4 (train), TP deg 2 (infer) | `ULYSSES_SP_SIZE=4`, `TP_SIZE=2` |
| Precision | bfloat16, gradient checkpointing, CPU offload | `autocast(bfloat16)`, `enable_gradient_checkpointing=true`, `param_offload`/`optimizer_offload=true` |

Benchmarks (MATH-500, AIME 2024, AIME 2025) and the dual-path math grading are implemented in `process_eval_data.py` and `rewards/dual_path_math_verify.py` (mirrors veRL's `math_dapo` grading, with an added `Answer:`-line extraction path). ✅

---

## 9. Emergent properties (paper §5.3–5.5) — observable in code

These are not enforced by the loss; they emerge. The code logs the signals needed to reproduce the paper's findings:

- **Difficulty-adaptive compression** (~1.6× more on easy vs. hard): emerges from the KL objective; the trainer logs per-`data_source` response-token counts (`val/{ds}/avg_response_tokens`).
- **Entropy preservation** (Finding 3, the central contrast with RL length penalties): `opsd_worker` computes and logs `opsd/student_entropy`, `opsd/teacher_entropy`, `opsd/entropy_diff` every step.
- **Training-time accuracy rises with no correctness reward** (Fig. 2): `sd/accuracy` is logged (metrics-only verification).

---

## 10. Where the code goes beyond the paper (supersets, not conflicts)

| Code capability | Status vs. paper |
|---|---|
| `OPSD_LOSS_TYPE=jsd` (+ `OPSD_BETA`) — symmetric Jensen-Shannon divergence | Extra option. The paper's CRISP method is **reverse KL**; it discusses forward KL only as a negative comparison (Appendix I). JSD is an available alternative, not the paper's method. |
| `_compute_*_liger` loss variants (logsumexp mixture + progressive teacher-chunk freeing) | Memory-engineering only; numerically equivalent (verified in `test_opsd_jsd.py`). Toggled by `USE_LIGER`. |
| `prompts.json → length_prune_teacher_word_limit` | A word-cap teacher variant; paper only reports the percentage soft-budget. |
| `prompts.json → opsd_qwen3_teacher` (includes a **Reference Solution**) and `self_distill*` templates | The **OPSD-with-ground-truth** teacher family (Zhao et al. 2026 baseline / classic self-distillation), where the teacher gets privileged info. **CRISP uses the conciseness teacher only** (`length_prune_*`) and is explicitly the no-GT variant. The repo is historically named "OPSD" but the paper's method is the `length_prune_concise` config. |
| DP-padding to `total_gpus // ulysses_sp`, `expandable_segments` ban, chunked losses | Pure infra correctness/memory; orthogonal to the algorithm. |

---

## 11. Summary

| Claim in paper | Verified in code? |
|---|---|
| One model, two prompt conditionings (student `x`, teacher `x,c`) | ✅ `sft_prompt` vs `sd_prompt` |
| Per-token **reverse** KL `D_KL(student‖teacher)` | ✅ `_compute_reverse_kl_loss` |
| Stop-gradient teacher (no grad through teacher) | ✅ `torch.no_grad()` on `ref_module_fsdp` |
| On-policy (train on student's own rollouts) | ✅ generation from `sft_prompt`, then train on those tokens |
| No ground-truth answers / no reward in loss | ✅ verification is metrics-only, never filters |
| Periodic teacher refresh `θ̃ ← θ` every `M=50` | ✅ `update_teacher` @ `TEACHER_UPDATE_FREQ` |
| Loss normalized by `|y|` | ✅ `kl_sum / n_tokens` |
| Exact student/teacher/soft-budget prompts | ✅ byte-identical in `prompts.json` |
| Hyperparameters (lr, batch, rollout, hardware, parallelism) | ✅ README / execution-configs |

**The `workspace/` implementation is consistent with the paper's described method and Algorithm 1.** The only things to keep in mind: the algorithm's "default" in the paper is *reverse KL + M=50*, whereas the bare YAML default is *reverse KL + M=0 (frozen teacher)*; use the README launch block or `execution-configs/` to reproduce the paper's numbers.
