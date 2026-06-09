"""
Response verification and SFT batch construction for self-distillation.

Two responsibilities:
1. Verify student-generated responses (structure + math correctness)
2. Build tokenized SFT training batches from verified (correct) responses
"""

import json
import logging
import re
from typing import Optional

import torch
from transformers import PreTrainedTokenizer

from verl.protocol import DataProto

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def validate_response_structure(text: str) -> bool:
    """Check that a student response has the required <think>…</think> structure.

    Accepts either ``Answer: X`` (Qwen2.5-style) or ``\\boxed{X}`` (Qwen3-style)
    as the answer indicator.  Mirrors the logic from
    ``self_distill_gen.py._validate_response_structure``.
    """
    if not text or len(text.strip()) < 50:
        return False

    has_think_open = "<think>" in text
    has_think_close = "</think>" in text
    has_answer = "answer:" in text.lower() or "\\boxed{" in text

    if not (has_think_open and has_think_close and has_answer):
        return False

    # The <think> block must contain substantive content
    try:
        think_start = text.index("<think>") + len("<think>")
        think_end = text.index("</think>")
        think_content = text[think_start:think_end].strip()
        if len(think_content) < 20:
            return False
    except ValueError:
        return False

    return True


_MATH_VERIFY_FUNC = None


def _get_math_verify_func():
    """Lazy-init the math_verify metric function (sympy-backed symbolic eq)."""
    global _MATH_VERIFY_FUNC
    if _MATH_VERIFY_FUNC is None:
        from math_verify.metric import math_metric
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

        _MATH_VERIFY_FUNC = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
    return _MATH_VERIFY_FUNC


_ANSWER_RE = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")


def _extract_answer_field(response: str) -> Optional[str]:
    matches = _ANSWER_RE.findall(response)
    if not matches:
        return None
    candidate = matches[-1].strip().rstrip(".,;:?\"' \t")
    return candidate if candidate else None


def _math_verify_score(pred_text: str, gold: str) -> bool:
    """Return True iff math_verify finds pred_text symbolically equivalent to gold."""
    if not pred_text or not pred_text.strip():
        return False
    try:
        from math_verify.errors import TimeoutException
    except ImportError:
        TimeoutException = Exception  # type: ignore
    try:
        gold_boxed = "\\boxed{" + str(gold) + "}"
        score, _ = _get_math_verify_func()([gold_boxed], [pred_text])
        return float(score) > 0
    except TimeoutException:
        return False
    except Exception:
        logger.exception("Unexpected math_verify failure; counting response as incorrect")
        return False


def verify_response(
    response_text: str,
    ground_truth: str,
    check_structure: bool = True,
) -> tuple[bool, str]:
    """Verify a single student response for correctness and structure.

    Scoring uses HuggingFace's ``math_verify`` (sympy symbolic equivalence) on
    two extraction paths, matching ``workspace/src/eval/scoring.py``:
      1. Primary: ``Answer: X`` line (matches the student prompt instruction).
      2. Fallback: any ``\\boxed{...}`` in the full response (Qwen3 thinking
         mode often emits boxed without the ``Answer:`` line).

    A response is correct iff EITHER path yields a math_verify positive match.

    Args:
        response_text: The student's generated response.
        ground_truth: Expected answer string.
        check_structure: Whether to also require <think> + answer structure.

    Returns:
        (is_correct, extracted_prediction)
    """
    if not response_text or not response_text.strip():
        return False, ""

    # Primary path: "Answer: X" extraction, wrapped in \boxed{...} so the
    # LatexExtractionConfig reliably picks it up.
    answer_payload = _extract_answer_field(response_text)
    primary_correct = False
    if answer_payload is not None:
        primary_correct = _math_verify_score(
            "\\boxed{" + answer_payload + "}", ground_truth
        )

    # Fallback path: math_verify on the full response finds any \boxed{...}
    # via LatexExtractionConfig.
    fallback_correct = _math_verify_score(response_text, ground_truth)

    is_correct = primary_correct or fallback_correct
    pred = answer_payload if answer_payload is not None else "[BOXED]"

    if is_correct and check_structure:
        if not validate_response_structure(response_text):
            return False, pred

    return is_correct, pred


def verify_batch(
    responses: list[str],
    ground_truths: list[str],
    check_structure: bool = True,
) -> tuple[list[bool], list[str]]:
    """Verify a batch of responses.

    Returns:
        (correct_mask, predictions) — both lists of length len(responses).
    """
    correct_mask = []
    predictions = []
    for resp, gt in zip(responses, ground_truths):
        ok, pred = verify_response(resp, gt, check_structure=check_structure)
        correct_mask.append(ok)
        predictions.append(pred)
    return correct_mask, predictions


# ---------------------------------------------------------------------------
# SFT Batch Construction
# ---------------------------------------------------------------------------


def build_sft_batch(
    sft_prompts: list[str],
    responses: list[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 32768,
) -> Optional[DataProto]:
    """Build a tokenized SFT training batch from verified correct responses.

    For each (sft_prompt, response) pair:
      1. Apply chat template to sft_prompt to get prompt token IDs
      2. Tokenize response text (+ EOS)
      3. Concatenate and create loss_mask (0 for prompt, 1 for response)
      4. Right-pad to max_length

    Args:
        sft_prompts: List of JSON-string chat-format messages (SFT prompts).
        responses: List of verified response strings.
        tokenizer: HuggingFace tokenizer.
        max_length: Max total sequence length (prompt + response).

    Returns:
        DataProto with batch keys: input_ids, attention_mask, position_ids, loss_mask.
        Returns None if no valid samples after filtering.
    """
    if not sft_prompts:
        return None

    all_input_ids = []
    all_attention_mask = []
    all_position_ids = []
    all_loss_mask = []

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    skipped = 0
    for sft_prompt_str, response_text in zip(sft_prompts, responses):
        # Parse prompt and apply chat template
        messages = json.loads(sft_prompt_str)
        prompt_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )

        # Tokenize response (no special tokens — we add EOS manually)
        response_ids = tokenizer.encode(response_text, add_special_tokens=False)

        # Ensure EOS at the end
        if not response_ids or response_ids[-1] != tokenizer.eos_token_id:
            response_ids = response_ids + [tokenizer.eos_token_id]

        full_ids = prompt_ids + response_ids
        loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

        # Truncate if exceeds max_length
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
            loss_mask = loss_mask[:max_length]
            # Make sure we don't lose the last few tokens that matter
            # If truncation cuts into response, that's ok — we still train on
            # the partial response up to max_length

        seq_len = len(full_ids)
        if seq_len < 2:
            skipped += 1
            continue

        # Right-pad to max_length
        pad_len = max_length - seq_len
        input_ids = full_ids + [pad_token_id] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len
        loss_mask_padded = loss_mask + [0] * pad_len
        position_ids = list(range(seq_len)) + [0] * pad_len

        all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
        all_attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))
        all_position_ids.append(torch.tensor(position_ids, dtype=torch.long))
        all_loss_mask.append(torch.tensor(loss_mask_padded, dtype=torch.float32))

    if skipped:
        logger.warning("Skipped %d samples during SFT batch construction (too short)", skipped)

    if not all_input_ids:
        return None

    batch_dict = {
        "input_ids": torch.stack(all_input_ids),
        "attention_mask": torch.stack(all_attention_mask),
        "position_ids": torch.stack(all_position_ids),
        "loss_mask": torch.stack(all_loss_mask),
    }
    return DataProto.from_single_dict(batch_dict)


# ---------------------------------------------------------------------------
# OPSD Batch Construction
# ---------------------------------------------------------------------------


def _tokenize_sequence(
    prompt_str: str,
    response_text: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    pad_token_id: int,
) -> Optional[dict]:
    """Tokenize a single (prompt, response) pair into padded tensors.

    Returns dict with input_ids, attention_mask, position_ids, loss_mask, or
    None if the sequence is too short OR would exceed max_length. Refusing
    over-length sequences (rather than truncating) is load-bearing: OPSD's
    JSD/reverse-KL loss aligns teacher and student response logits by
    position, so per-sample response-token counts must match between the
    two prompts. Truncating just the teacher (because its prompt is longer)
    would silently misalign all subsequent samples in the flattened batch.
    """
    messages = json.loads(prompt_str)
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    if not response_ids or response_ids[-1] != tokenizer.eos_token_id:
        response_ids = response_ids + [tokenizer.eos_token_id]

    full_ids = prompt_ids + response_ids
    loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

    if len(full_ids) > max_length:
        return None

    seq_len = len(full_ids)
    if seq_len < 2:
        return None

    pad_len = max_length - seq_len
    return {
        "input_ids": torch.tensor(full_ids + [pad_token_id] * pad_len, dtype=torch.long),
        "attention_mask": torch.tensor([1] * seq_len + [0] * pad_len, dtype=torch.long),
        "position_ids": torch.tensor(list(range(seq_len)) + [0] * pad_len, dtype=torch.long),
        "loss_mask": torch.tensor(loss_mask + [0] * pad_len, dtype=torch.float32),
    }


def _log_first_opsd_pair(
    t_prompt: str, s_prompt: str, response_text: str, max_length: int
) -> None:
    """Log the first (teacher, student, response) triple of each batch for sanity-checking.

    Decodes the JSON-encoded chat prompts so they're readable in logs, and
    truncates the response to a preview so noisy logs stay bounded.
    """
    try:
        t_msgs = json.loads(t_prompt)
        s_msgs = json.loads(s_prompt)
    except (json.JSONDecodeError, TypeError):
        t_msgs, s_msgs = t_prompt, s_prompt

    response_preview = response_text[:500] + ("…" if len(response_text) > 500 else "")
    # Emitted at WARNING so the preview survives the cluster's default
    # log level (Hydra raises __main__ to INFO but not the root logger,
    # so logger.info(...) here would be filtered).
    logger.warning(
        "OPSD batch[0] sample preview (max_length=%d):\n"
        "  teacher_prompt (sd_prompt) messages: %s\n"
        "  student_prompt (sft_prompt) messages: %s\n"
        "  response (%d chars): %s",
        max_length, t_msgs, s_msgs, len(response_text), response_preview,
    )


def build_opsd_batch(
    teacher_prompts: list[str],
    student_prompts: list[str],
    responses: list[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 32768,
) -> Optional[DataProto]:
    """Build paired teacher/student tokenized sequences for OPSD JSD training.

    For each sample, creates two sequences with the same response tokens but
    different prompts:
      - Teacher: sd_prompt (question + teacher solution) + student response
      - Student: sft_prompt (question only) + student response

    The loss_mask marks response positions where JSD should be computed.

    Args:
        teacher_prompts: JSON-string chat messages with teacher solution (sd_prompt).
        student_prompts: JSON-string chat messages with question only (sft_prompt).
        responses: Student-generated response strings.
        tokenizer: HuggingFace tokenizer.
        max_length: Max total sequence length.

    Returns:
        DataProto with keys: teacher_input_ids, teacher_attention_mask,
        teacher_position_ids, teacher_loss_mask, student_input_ids,
        student_attention_mask, student_position_ids, student_loss_mask.
        Returns None if no valid samples.
    """
    if not teacher_prompts:
        return None

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    teacher_seqs = []
    student_seqs = []
    skipped = 0
    for idx, (t_prompt, s_prompt, response_text) in enumerate(
        zip(teacher_prompts, student_prompts, responses)
    ):
        if idx == 0:
            _log_first_opsd_pair(t_prompt, s_prompt, response_text, max_length)

        t_seq = _tokenize_sequence(t_prompt, response_text, tokenizer, max_length, pad_token_id)
        s_seq = _tokenize_sequence(s_prompt, response_text, tokenizer, max_length, pad_token_id)

        if t_seq is None or s_seq is None:
            skipped += 1
            continue

        teacher_seqs.append(t_seq)
        student_seqs.append(s_seq)

    if skipped:
        total = len(teacher_prompts)
        logger.warning(
            "OPSD batch construction skipped %d/%d samples (too short or longer than max_length=%d). "
            "Bump opsd.sft_max_length if this fraction is large — silently dropped samples are "
            "the only way to keep teacher/student response logits aligned per-sample.",
            skipped, total, max_length,
        )

    if not teacher_seqs:
        return None

    batch_dict = {}
    for prefix, seqs in [("teacher_", teacher_seqs), ("student_", student_seqs)]:
        for key in ["input_ids", "attention_mask", "position_ids", "loss_mask"]:
            batch_dict[f"{prefix}{key}"] = torch.stack([s[key] for s in seqs])

    return DataProto.from_single_dict(batch_dict)
