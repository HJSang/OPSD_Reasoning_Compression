# CRISP on slime

Runs CRISP — teacher = the **same model** conditioned on a conciseness prompt, reverse-KL
distillation on the student's own rollouts, teacher refresh every M steps — on
[slime](../../slime). Two KL estimators, switched via `CRISP_KL_MODE`:

- **`sampled`** (milestone 1): sampled-token reverse KL through slime's OPD advantage penalty
  (`--use-opd --opd-type sglang`). Stock slime.
- **`full`** (milestone 2, [`FULL_KL_PLAN.md`](FULL_KL_PLAN.md)): bucketed full-vocab reverse KL
  over the teacher's top-K (`crisp_full_kl_loss.py`, wired as a Megatron `custom_loss`).
  Equals the verl objective exactly at K=V (gradient-verified in tests). Requires the patched
  slime submodule (branch `crisp`: plumbs `teacher_top_ids/logprobs`, ~30 lines).

Design rationale: [`SLIME_DESIGN.md`](../../SLIME_DESIGN.md) §5; algorithm: [`METHOD.md`](../../METHOD.md).

## What differs from the verl implementation (`workspace/src/`)

| | verl (`self_distill_hybrid/`) | this (slime) |
|---|---|---|
| KL estimator | full-vocabulary per-token `KL(q_t‖p_t)` from logits | **sampled-token** `log q(y_t) − log p(y_t)` as an advantage penalty (slime OPD) |
| Teacher pass | training-side forward, swapped prompt | **prefill-only scoring call** to a teacher sglang server |
| Teacher refresh | `update_teacher` FSDP shard copy | actor saves HF dump (`--save-hf`) → teacher server `/update_weights_from_disk` |
| Response tokens in training | decoded text **re-tokenized** (+EOS appended) | **exact rollout token ids** (no re-tokenization; cleaner) |
| In-training accuracy metric | dual-path `math_verify` (sympy) | `math_dapo` minerva (`Answer:` extraction) — what the paper's benchmarks used (footnote 1); curves may differ a few points from verl logs |
| Teacher logit precision | trainer bf16 forward | sglang inference kernels (numerically close, not bit-equal) |

Verified identical (audited, with regression tests): teacher prompt strings (byte-equal to
`config/prompts.json`), question/GT extraction (0 mismatches on 2,000 real rows), train split
(row-for-row equal to verl's seed-42 80% split, 13,918 rows), teacher-refresh window semantics
(steps 1–50 use θ₀, 51–100 use θ₅₀ in both), sampling params, optimizer settings, no
correctness filtering. Whether the sampled-token estimator reproduces the paper's
compression–accuracy trade-off is exactly what this milestone tests.

## Files

- `crisp_opd.py` — the three slime hooks:
  - `reward_func` (`--custom-rm-path`): builds `teacher_prompt(question) + response_tokens`,
    requests a prefill-only forward from the teacher server, stores per-token log-probs on
    `sample.teacher_log_probs` (with a token-id alignment check), returns math correctness
    (metrics only, logged as the training-accuracy curve).
  - `post_process_rewards` (`--custom-reward-post-process-path`): zeroes training rewards —
    pure distillation; the signal is slime's `adv_t -= opd_kl_coef·(logπ_s(y_t) − logπ_t(y_t))`.
  - `generate_rollout` (`--rollout-function-path`): default rollout + teacher refresh
    (θ̃←θ) every `crisp_teacher_update_interval` rollouts via `/update_weights_from_disk`.
- `crisp_full_kl_loss.py` — milestone-2 loss: `vocab_parallel_topk_log_probs` (TP-aware,
  autograd-correct), `bucketed_reverse_kl` (top-K + tail bucket), and the slime `custom_loss`
  entry point `full_kl_loss_function` (logs `teacher_topk_coverage`, `q_tail`, and the
  same-batch `kl_sampled` estimator-comparison diagnostic).
- `crisp_config.yaml` / `crisp_config_full_kl.yaml` — per-mode knobs, merged onto args via
  `--custom-config-path`.
- `prepare_crisp_slime_data.py` — DAPO parquet → slime jsonl
  (`prompt` = original DAPO content, `label` = GT, `metadata.question` = bare question).
- `run-qwen3-8b-crisp.sh` — 8-GPU launch: actor 4 | rollout 3 | teacher 1; paper recipe
  (batch 32, lr 1e-6, temp 1.0, 8192-token rollouts, n=1, 100 steps, M=50).
  `CRISP_KL_MODE=full bash run-qwen3-8b-crisp.sh` for milestone 2.
- `test_crisp_opd.py`, `test_full_kl_loss.py` — 23 unit tests (no GPU/slime runtime needed).

## Invariants to keep

1. **Same response token ids** under both prompts — enforced by the token-id echo check in
   `reward_func` (the slime analogue of verl's drop-don't-truncate rule, see `METHOD.md` §7).
2. `--save-interval` == `crisp_teacher_update_interval`, `--save-hf` == `crisp_teacher_hf_path`
   (fixed path, overwritten each save).
3. Synchronous `train.py` only — `train_async.py` would refresh the teacher mid-pipeline.
4. Full-KL mode: CP must be 1 and `qkv_format` must be `thd` (asserted); the slime submodule
   must be on the `crisp` branch (the plumbing patch).

## Success criterion (vs paper Table 2, Qwen3-8B @30K budget)

MATH-500: base 77.7% / 4,661 tok → CRISP 86.6% / 1,921 tok (58.8% reduction).
Run both estimators (`CRISP_KL_MODE=sampled` vs `full`) and compare against the verl
reference; the `kl_sampled` diagnostic logged by the full-KL loss measures the estimator gap
on identical batches. See [`FULL_KL_PLAN.md`](FULL_KL_PLAN.md) §6 for acceptance criteria.
