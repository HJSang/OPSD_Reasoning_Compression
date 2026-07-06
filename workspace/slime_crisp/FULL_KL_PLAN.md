# Plan: full-vocab reverse KL `KL(q_t‖p_t)` on slime (milestone 2)

Closes the one fundamental estimator gap between this port and the paper/verl implementation
(see README "What differs"): replace the sampled-token advantage penalty with a per-token
**distribution-level** reverse KL, computed from full student logits and (top-K-truncated)
teacher log-probs — recovering verl's `_compute_reverse_kl_loss` objective as K→V.

Status: **implemented** (loss: `crisp_full_kl_loss.py`; rollout top-K: `crisp_opd.py`;
plumbing: slime submodule branch `crisp`, commit `e0ef620`; 23 unit tests passing).
Not yet run on GPUs — the A/B in §6 needs both arms on the cluster.

**Correction found during testing** (encoded in `test_full_kl_loss.py`): approximation
fidelity is governed by **`q_tail`** (the *student's* mass outside teacher top-K), not by
teacher coverage alone — reverse KL is an expectation under `q`, so a diffuse student hides
large log-ratios in the tail bucket (the bucketed KL stays a lower bound). In CRISP's
self-distillation regime q≈p so q_tail is tiny, and the tail term's gradient actively shrinks
it; treat rising `q_tail` (logged per step) as the signal to raise K.

---

## 1. Why and what

| | milestone 1 (sampled) | milestone 2 (this plan) | verl reference |
|---|---|---|---|
| objective | `adv_t = −β(log q(y_t) − log p(y_t))` through PG loss | `(1/N) Σ_t KL_K(q_t‖p_t)` direct loss | `(1/N) Σ_t KL(q_t‖p_t)` full vocab |
| gradient | unbiased, **one-sample** estimate per token | dense over teacher top-K + tail bucket | dense over full vocab |
| teacher info needed | `log p(y_t)` (1 float/token) | top-K `(id, log p)` pairs/token | full teacher logits |

The training-side student logits are already full-vocab (the training forward); the only thing
the sampled estimator throws away is the **teacher's** distribution. We recover it via sglang's
`top_logprobs_num` on the existing prefill-only scoring call — no second training-side model,
no change to the teacher-refresh machinery.

### The truncated objective: bucketed reverse KL

For each response position `t`, with teacher top-K ids `x_1..x_K` (from sglang), teacher
log-probs `log p_k`, and student probabilities `q_k = q_t(x_k)` (exact, from training logits):

```
q_tail = 1 − Σ_k q_k          p_tail = 1 − Σ_k p_k
KL_K(q_t‖p_t) = Σ_k q_k (log q_k − log p_k)  +  q_tail (log q_tail − log p_tail)
```

Properties: a true KL on the coarsened (K+1)-bucket space — non-negative, zero iff the
coarsened distributions match, **exactly the verl objective at K=V**. Reasoning teachers are
low-entropy (paper Fig. 8: ~0.3 nats), so top-K coverage `Σ_k p_k` should be ≥99.9% at K=256;
we log it every step to quantify the truncation (§5). Differentiable in the student logits
everywhere (including the tail term, which pushes mass off non-teacher tokens).

---

## 2. Architecture decision

**Chosen: B1 — sglang teacher top-K + custom Megatron loss.**

Rejected alternatives:
- **B2: Megatron-mode teacher** (`teacher` weight tag) — the teacher must see a *different,
  longer prompt* than the student, so the teacher forward needs a second token sequence per
  sample plumbed through the data pipeline; and since slime's weight tags time-share one set of
  GPU buffers, teacher logits would have to be *stored* between the teacher phase and the train
  phase — full logits are `T×V≈8192×150k×2B ≈ 2.4 GB/sample`, so storage forces top-K anyway.
  Same approximation, far more plumbing. Only advantage: bit-identical teacher precision.
- **B3: resident second model** (verl-style) — contradicts slime's core memory design; a deep
  fork we don't want to maintain.

B1 keeps everything from milestone 1 (teacher server, conciseness prompt, `/update_weights_from_disk`
refresh, alignment check) and adds one payload field + one custom loss.

---

## 3. Implementation steps

### Step 1 — rollout side: fetch teacher top-K (ours, `crisp_opd.py`)

In `reward_func`, when `crisp_kl_mode == "full"` (new knob in `crisp_config.yaml`):

- add `"top_logprobs_num": K` (`crisp_teacher_topk`, default 256) to the scoring payload;
- parse `meta_info["input_top_logprobs"]` (per-position list of `[logprob, token_id, ...]`),
  trim to the response span exactly like `input_token_logprobs`;
- store `sample.teacher_top_ids` (T×K ints) and `sample.teacher_top_logprobs` (T×K floats);
- keep storing sampled `teacher_log_probs` too (cheap; enables logging both estimators).

⚠️ Verify the sglang field name/shape on the deployed version first (one curl); it has shifted
across sglang releases.

### Step 2 — plumbing: carry T×K fields to the loss (small slime fork patch)

The pipeline whitelists keys at four places; new fields need ~5 mechanical edits. Fork the
submodule to `HJSang/slime` branch `crisp` (candidate upstream PR: "generic top-K teacher
log-probs for OPD"):

1. `slime/utils/types.py` — `Sample.teacher_top_ids`, `Sample.teacher_top_logprobs`.
2. `slime/ray/rollout.py::_convert_samples_to_train_data` — pass-through (mirror the existing
   `teacher_log_probs` lines).
3. `slime/ray/rollout.py::_split_train_data_by_dp` — add both keys to the per-rank key list.
4. `slime/backends/megatron_utils/actor.py::_get_rollout_data` — move to GPU; **assert CP==1
   for now** (the `slice_log_prob_with_cp` helper is 1-D/per-token; our milestone runs
   `context-parallel-size 1`, so defer 2-D CP slicing).
5. `slime/backends/megatron_utils/model.py` — add both keys to the `get_batch([...])` list in
   the train `forward_step`.

### Step 3 — the loss (ours, new `crisp_full_kl_loss.py`)

Wire: `--loss-type custom_loss --custom-loss-function-path slime_crisp.crisp_full_kl_loss.full_kl_loss_function`.
slime's contract (`loss.py::loss_function`): `func(args, batch, logits, sum_of_sample_mean) → (loss, metrics)`,
with `logits` = full student `[1, T, V]` (vocab-parallel under TP).

```
full_kl_loss_function(args, batch, logits, sum_of_sample_mean):
  1. slice response-position logits          # reuse get_log_probs_and_entropy's slicing path
  2. per token-chunk (chunk_size≈128, fp32 upcast — mirrors verl's chunked loss):
       lse        = vocab_parallel_logsumexp(logits_chunk)          # max + sum-exp all-reduce over TP
       log_q_K    = vocab_parallel_gather(logits_chunk, top_ids) − lse   # local mask+gather, all-reduce(SUM)
       q_K        = exp(log_q_K);  q_tail = clamp(1 − Σ q_K, min=eps)
       p_tail     = clamp(1 − Σ exp(teacher_top_logprobs), min=eps)
       kl_chunk   = Σ_k q_K (log_q_K − log p_K) + q_tail (log q_tail − log p_tail)
  3. loss = sum_of_sample_mean(kl_per_token)  # --calculate-per-token-loss ⇒ global token mean = verl parity
  4. zero-token guard: loss += 0 * logits.sum()        # same trick as sft_loss_function
  5. metrics: kl, teacher_topk_coverage (Σ p_K), q_tail, student_entropy
     (reuse compute_entropy_from_logits under no_grad), sampled-KL for comparison
```

New primitive `vocab_parallel_topk_log_probs(logits, ids_TK, tp_group)` (~40 lines): can't reuse
`compute_log_probs` directly (it gathers 1 id/position); implement K-id gather as
local-shard-range mask + `torch.gather` + `all_reduce(SUM)`, sharing one logsumexp per position.
Degrade to plain torch when TP world size == 1 → CPU-unit-testable.

### Step 4 — config & mode switching

- `crisp_config.yaml`: `crisp_kl_mode: full|sampled`, `crisp_teacher_topk: 256`.
- Launch script (full mode): drop `--use-opd`/`--opd-*`, add `--loss-type custom_loss
  --custom-loss-function-path ...` and `--disable-compute-advantages-and-returns` — skips the
  whole advantage pipeline *and the extra old-log-prob forward pass* (the custom loss needs only
  the training forward; per-step cost roughly matches verl's teacher-fwd + student-fwd/bwd,
  with the teacher fwd amortized into rollout-time scoring).
- Everything else (data, prompts, refresh, reward zeroing) unchanged from milestone 1.

---

## 4. Tests (extend `test_crisp_opd.py` + new `test_full_kl_loss.py`)

1. **Math**: bucketed `KL_K` vs exact `KL` on a toy vocab (V=50): equal at K=V; lower bound and
   monotone non-decreasing in K; zero iff distributions equal; gradient matches
   `torch.autograd.gradcheck` on the K=V case against a direct full-KL implementation
   (this is the verl-equivalence proof at small scale).
2. **Primitive**: `vocab_parallel_topk_log_probs` TP=1 path vs plain `log_softmax + gather`.
3. **Parsing**: `input_top_logprobs` trimming + id-echo alignment check (extend the existing
   alignment test to the top-K field).
4. **Plumbing smoke** (GPU, 1 node): `--num-rollout 2`, assert `teacher_top_*` reach the loss
   with the right shapes and `teacher_topk_coverage > 0.99`.

---

## 5. Observability

Per-step logs (all cheap, from the loss metrics): `loss` (= bucketed KL), `kl_sampled` (same
batch — directly measures estimator variance/bias), `teacher_topk_coverage`, `q_tail`, plus
the existing `raw_reward` accuracy curve. **`q_tail` is the fidelity diagnostic** (see status
note above); if it rises above ~1e-2, raise K before trusting the run.

---

## 6. Validation: the A/B that motivates all of this

Same data, prompts, hyperparameters, M=50, 100 steps, Qwen3-8B:

| arm | loss | expectation |
|---|---|---|
| A (milestone 1) | sampled-token KL via OPD penalty | ? — the open question |
| B (this plan) | bucketed full KL, K=256 | ≈ verl reference |
| verl reference | `workspace/` Table-2 run | MATH-500 86.6% / 1,921 tok |

Acceptance for B: MATH-500 accuracy within ~1 pt and response length within ~10% of the verl
run at step 100. Ablate K ∈ {64, 256, 1024} only if coverage logging shows it matters.
If A ≈ B, the sampled estimator is vindicated (cheaper, zero fork); if A ≪ B, milestone 2
becomes the default recipe.

---

## 7. Risks & costs

- **sglang `top_logprobs_num` overhead**: K floats+ids per scored token serialized over HTTP
  (≈ T×256×8B ≈ 8 MB per 4k-token response). Measure scoring throughput; drop to K=64 or gzip
  if the teacher GPU becomes the bottleneck.
- **API drift**: `input_top_logprobs` name/shape varies across sglang versions — verify first.
- **CP > 1 unsupported** (asserted) until 2-D CP slicing is added; irrelevant to the 8B recipe.
- **Numerics**: `q_tail`/`p_tail` need `clamp(min=1e-8)`; upcast chunks to fp32 like verl.
- **Fork maintenance**: ~5-file mechanical patch; pin to our fork branch and propose upstream.

Estimated effort: Step 1 ~0.5 d, Step 2 ~0.5 d, Step 3+4 ~1.5 d, A/B runs = cluster time.
