"""Unit tests for the CRISP teacher-scoring logic (no GPU / no slime runtime).

Heavy imports in crisp_opd are lazy, so these tests run with only the module's
pure-python logic: payload construction, response-span trimming, alignment
checks, and reward post-processing.

Run:  pytest workspace/slime_crisp/test_crisp_opd.py
"""

import asyncio
from argparse import Namespace
from dataclasses import dataclass, field

import pytest

import crisp_opd


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """apply_chat_template -> deterministic ids: [100, len(content)]."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert tokenize and add_generation_prompt
        assert messages[0]["role"] == "user"
        self.last_content = messages[0]["content"]
        self.last_kwargs = kwargs
        return [100, len(messages[0]["content"]) % 1000]


@dataclass
class FakeSample:
    tokens: list = field(default_factory=list)
    response_length: int = 0
    response: str = ""
    label: str | None = None
    reward: float | None = None
    metadata: dict = field(default_factory=dict)
    teacher_log_probs: list | None = None


def make_args(**overrides):
    base = dict(
        hf_checkpoint="fake",
        rm_url="http://teacher/generate",
        apply_chat_template_kwargs={"enable_thinking": True},
    )
    base.update(overrides)
    return Namespace(**base)


def fake_post_factory(captured, prompt_len, response_tokens, logprob_value=-0.5, corrupt_index=None):
    """Return an async fake for crisp_opd._post echoing sglang's logprob format."""

    async def fake_post(url, payload):
        captured["url"] = url
        captured["payload"] = payload
        ids = list(response_tokens)
        if corrupt_index is not None:
            ids[corrupt_index] += 1
        entries = [[None, payload["input_ids"][0]]]
        for tok in payload["input_ids"][1 : prompt_len]:
            entries.append([-1.0, tok])  # prompt span (discarded by trimming)
        for tok in ids:
            entries.append([logprob_value, tok])  # response span
        return {"meta_info": {"input_token_logprobs": entries}}

    return fake_post


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch):
    tok = FakeTokenizer()
    monkeypatch.setattr(crisp_opd, "_TOKENIZER", tok)
    yield tok
    crisp_opd._TOKENIZER = None


# ---------------------------------------------------------------------------
# reward_func
# ---------------------------------------------------------------------------


def test_reward_func_scores_response_span(monkeypatch, fake_tokenizer):
    response_tokens = [11, 12, 13]
    sample = FakeSample(
        tokens=[1, 2, 3] + response_tokens,  # student prompt ids + response ids
        response_length=3,
        response="... Answer: 42",
        label="42",
        metadata={"question": "What is 6*7?"},
    )
    captured = {}
    monkeypatch.setattr(crisp_opd, "_post", fake_post_factory(captured, prompt_len=2, response_tokens=response_tokens))
    monkeypatch.setattr(crisp_opd, "_compute_correct", lambda response, label: 1.0)

    reward = asyncio.run(crisp_opd.reward_func(make_args(), sample))

    # teacher logprobs trimmed to exactly the response span
    assert sample.teacher_log_probs == [-0.5, -0.5, -0.5]
    assert reward == 1.0
    # payload: teacher prompt ids (FakeTokenizer -> 2 ids) + student response ids
    assert captured["payload"]["input_ids"][-3:] == response_tokens
    assert len(captured["payload"]["input_ids"]) == 2 + 3
    assert captured["payload"]["sampling_params"]["max_new_tokens"] == 0
    assert captured["payload"]["return_logprob"] is True
    assert captured["url"] == "http://teacher/generate"
    # teacher prompt uses the conciseness instruction around the bare question
    assert fake_tokenizer.last_content.startswith(crisp_opd.DEFAULT_TEACHER_PREFIX)
    assert "What is 6*7?" in fake_tokenizer.last_content
    assert fake_tokenizer.last_content.endswith(crisp_opd.DEFAULT_TEACHER_SUFFIX)
    # chat-template kwargs (e.g. enable_thinking) forwarded to the teacher side
    assert fake_tokenizer.last_kwargs == {"enable_thinking": True}


def test_reward_func_alignment_violation_raises(monkeypatch):
    response_tokens = [11, 12, 13]
    sample = FakeSample(
        tokens=[1, 2] + response_tokens,
        response_length=3,
        metadata={"question": "q"},
        label="1",
    )
    monkeypatch.setattr(
        crisp_opd,
        "_post",
        fake_post_factory({}, prompt_len=2, response_tokens=response_tokens, corrupt_index=1),
    )

    with pytest.raises(AssertionError, match="alignment violation"):
        asyncio.run(crisp_opd.reward_func(make_args(), sample))


def test_reward_func_requires_question_metadata(monkeypatch):
    sample = FakeSample(tokens=[1, 11], response_length=1, metadata={})
    with pytest.raises(AssertionError, match="question"):
        asyncio.run(crisp_opd.reward_func(make_args(), sample))


def test_reward_func_empty_response(monkeypatch):
    sample = FakeSample(tokens=[1, 2], response_length=0, metadata={"question": "q"})
    reward = asyncio.run(crisp_opd.reward_func(make_args(), sample))
    assert reward == 0.0
    assert sample.teacher_log_probs == []


def test_reward_func_prompt_override(monkeypatch, fake_tokenizer):
    response_tokens = [11]
    sample = FakeSample(
        tokens=[1, 2] + response_tokens, response_length=1, metadata={"question": "q"}, label=None
    )
    monkeypatch.setattr(crisp_opd, "_post", fake_post_factory({}, prompt_len=2, response_tokens=response_tokens))
    args = make_args(crisp_teacher_prompt_prefix="BE BRIEF: ", crisp_teacher_prompt_suffix=" END")

    reward = asyncio.run(crisp_opd.reward_func(args, sample))

    assert fake_tokenizer.last_content == "BE BRIEF: q END"
    assert reward == 0.0  # label None -> metrics skipped


# ---------------------------------------------------------------------------
# post_process_rewards
# ---------------------------------------------------------------------------


def test_post_process_zeroes_training_rewards():
    samples = [
        FakeSample(reward=1.0, response_length=3, teacher_log_probs=[-0.1] * 3),
        FakeSample(reward=0.0, response_length=2, teacher_log_probs=[-0.2] * 2),
        FakeSample(reward=None, response_length=0, teacher_log_probs=[]),
    ]
    raw, train = crisp_opd.post_process_rewards(make_args(), samples)
    assert raw == [1.0, 0.0, 0.0]  # correctness preserved for logging
    assert train == [0.0, 0.0, 0.0]  # pure distillation


def test_post_process_detects_missing_teacher_logprobs():
    samples = [FakeSample(reward=1.0, response_length=3, teacher_log_probs=None)]
    with pytest.raises(AssertionError, match="teacher_log_probs"):
        crisp_opd.post_process_rewards(make_args(), samples)


# ---------------------------------------------------------------------------
# teacher refresh gating
# ---------------------------------------------------------------------------


def test_refresh_gating(monkeypatch):
    calls = []
    monkeypatch.setattr(crisp_opd, "_refresh_teacher_weights", lambda args, rid: calls.append(rid))
    inner_calls = []

    def fake_default(args, rollout_id, data_source, evaluation=False):
        inner_calls.append(rollout_id)
        return "rollout-output"

    import sys
    import types

    fake_mod = types.ModuleType("slime.rollout.sglang_rollout")
    fake_mod.generate_rollout = fake_default
    fake_pkg_rollout = types.ModuleType("slime.rollout")
    fake_pkg = types.ModuleType("slime")
    monkeypatch.setitem(sys.modules, "slime", fake_pkg)
    monkeypatch.setitem(sys.modules, "slime.rollout", fake_pkg_rollout)
    monkeypatch.setitem(sys.modules, "slime.rollout.sglang_rollout", fake_mod)

    args = make_args(crisp_teacher_update_interval=50)
    for rid in [0, 1, 49, 50, 99, 100]:
        assert crisp_opd.generate_rollout(args, rid, data_source=None) == "rollout-output"
    assert calls == [50, 100]  # not at 0; only at multiples of M
    assert inner_calls == [0, 1, 49, 50, 99, 100]

    # eval never refreshes; interval 0 disables
    crisp_opd.generate_rollout(args, 50, data_source=None, evaluation=True)
    crisp_opd.generate_rollout(make_args(crisp_teacher_update_interval=0), 50, data_source=None)
    assert calls == [50, 100]


# ---------------------------------------------------------------------------
# Parity with the verl pipeline (workspace/src, workspace/config)
# ---------------------------------------------------------------------------


def test_teacher_prompt_matches_prompts_json():
    """The conciseness instruction must stay byte-identical to the verl pipeline's."""
    import json
    import pathlib

    prompts_path = pathlib.Path(__file__).resolve().parents[1] / "config" / "prompts.json"
    cfg = json.loads(prompts_path.read_text())["length_prune_teacher"]
    assert crisp_opd.DEFAULT_TEACHER_PREFIX == cfg["prefix"]
    assert crisp_opd.DEFAULT_TEACHER_SUFFIX == cfg["suffix"]


def test_dapo_header_matches_verl_prep():
    """Question extraction must strip the same DAPO header as the verl pipeline."""
    import pathlib
    import sys

    verl_data_dir = str(pathlib.Path(__file__).resolve().parents[1] / "src" / "data")
    sys.path.insert(0, verl_data_dir)
    try:
        import prepare_length_prune_data as verl_prep

        import prepare_crisp_slime_data as slime_prep

        dapo_prompt = [{"role": "user", "content": slime_prep.DAPO_PREFIX + "What is 2+2?"}]
        assert verl_prep.extract_question_from_dapo(dapo_prompt) == "What is 2+2?"
        assert slime_prep.extract_question(dapo_prompt[0]["content"]) == "What is 2+2?"
    finally:
        sys.path.remove(verl_data_dir)
