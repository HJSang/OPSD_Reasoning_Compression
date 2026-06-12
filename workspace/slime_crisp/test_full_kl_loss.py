"""Unit tests for the bucketed full-vocab reverse KL (milestone 2).

Pure-torch tests of the loss math and the TP=1 path of the vocab-parallel
primitives — no GPU, no megatron, no distributed init.

Run:  pytest workspace/slime_crisp/test_full_kl_loss.py
"""

import pytest
import torch

from crisp_full_kl_loss import bucketed_reverse_kl, vocab_parallel_topk_log_probs

V = 13  # toy vocab


def make_dists(n_tokens=5, seed=0, vocab=V):
    g = torch.Generator().manual_seed(seed)
    student_logits = torch.randn(n_tokens, vocab, generator=g, dtype=torch.float64)
    teacher_logits = torch.randn(n_tokens, vocab, generator=g, dtype=torch.float64)
    return student_logits, teacher_logits


def exact_reverse_kl(student_logits, teacher_logits):
    """Direct full-vocab KL(q||p) — the verl objective (opsd_worker)."""
    log_q = student_logits.log_softmax(-1)
    log_p = teacher_logits.log_softmax(-1)
    return (log_q.exp() * (log_q - log_p)).sum(-1)


def bucketed_from_logits(student_logits, teacher_logits, k):
    """Run the production pipeline: teacher top-K -> gather -> bucketed KL."""
    log_p = teacher_logits.log_softmax(-1)
    top_lps, top_ids = log_p.topk(k, dim=-1)
    log_q_k = vocab_parallel_topk_log_probs(student_logits, top_ids, tp_group=None)
    return bucketed_reverse_kl(log_q_k, top_lps)


# ---------------------------------------------------------------------------
# vocab_parallel_topk_log_probs (TP=1 path)
# ---------------------------------------------------------------------------


def test_topk_log_probs_matches_log_softmax_gather():
    student_logits, _ = make_dists()
    ids = torch.randint(0, V, (5, 4))
    out = vocab_parallel_topk_log_probs(student_logits, ids, tp_group=None)
    expected = student_logits.log_softmax(-1).gather(-1, ids)
    torch.testing.assert_close(out, expected)


def test_topk_log_probs_gradient():
    student_logits, _ = make_dists()
    leaf = student_logits.clone().requires_grad_(True)
    ids = torch.randint(0, V, (5, 4))

    def fn(x):
        return vocab_parallel_topk_log_probs(x, ids, tp_group=None)

    assert torch.autograd.gradcheck(fn, (leaf,), raise_exception=True)


# ---------------------------------------------------------------------------
# bucketed_reverse_kl: math properties
# ---------------------------------------------------------------------------


def test_equals_exact_kl_at_k_equals_v():
    student_logits, teacher_logits = make_dists()
    kl, coverage, q_tail = bucketed_from_logits(student_logits, teacher_logits, k=V)
    exact = exact_reverse_kl(student_logits, teacher_logits)
    # At K=V the tail buckets carry ~0 mass (clamped at eps), so the bucketed
    # KL reduces to the exact full-vocab reverse KL.
    torch.testing.assert_close(kl, exact, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(coverage, torch.ones_like(coverage), atol=1e-6, rtol=0)


def test_gradient_matches_exact_kl_at_k_equals_v():
    """The verl-equivalence proof: at K=V, gradients match the direct loss."""
    student_logits, teacher_logits = make_dists()

    leaf_bucketed = student_logits.clone().requires_grad_(True)
    kl, _, _ = bucketed_from_logits(leaf_bucketed, teacher_logits, k=V)
    kl.mean().backward()

    leaf_exact = student_logits.clone().requires_grad_(True)
    exact_reverse_kl(leaf_exact, teacher_logits).mean().backward()

    torch.testing.assert_close(leaf_bucketed.grad, leaf_exact.grad, atol=1e-6, rtol=1e-5)


def test_nonnegative_and_zero_iff_equal():
    student_logits, teacher_logits = make_dists(n_tokens=20, seed=3)
    kl, _, _ = bucketed_from_logits(student_logits, teacher_logits, k=6)
    assert (kl >= -1e-9).all()

    # identical distributions -> zero at any K
    kl_same, _, _ = bucketed_from_logits(student_logits, student_logits, k=6)
    torch.testing.assert_close(kl_same, torch.zeros_like(kl_same), atol=1e-9, rtol=0)


def test_monotone_nondecreasing_in_k():
    """Coarsening loses information: KL_K is non-decreasing in K (up to eps)."""
    student_logits, teacher_logits = make_dists(n_tokens=10, seed=7)
    means = []
    for k in (2, 4, 8, V):
        kl, _, _ = bucketed_from_logits(student_logits, teacher_logits, k=k)
        means.append(kl.mean().item())
    for lo, hi in zip(means, means[1:], strict=False):
        assert hi >= lo - 1e-7, f"KL_K not monotone: {means}"


def test_lower_bounds_exact_kl():
    student_logits, teacher_logits = make_dists(n_tokens=10, seed=11)
    exact = exact_reverse_kl(student_logits, teacher_logits)
    for k in (2, 4, 8):
        kl, _, _ = bucketed_from_logits(student_logits, teacher_logits, k=k)
        assert (kl <= exact + 1e-7).all()


def test_self_distillation_regime_small_k_is_near_exact():
    """Student ~ teacher on a shared peaked support (the CRISP regime).

    The bucketed-KL gap is governed by q_tail — the STUDENT's mass outside the
    teacher top-K — not by teacher coverage alone (reverse KL is an expectation
    under q, so a diffuse student hides arbitrarily large per-token log-ratios
    in the tail bucket). In CRISP, student and teacher are the same base model
    (training loss ~1e-2), so q_tail is tiny and small K is near exact. The
    loss logs q_tail per step as the fidelity diagnostic.
    """
    g = torch.Generator().manual_seed(5)
    teacher_logits = torch.randn(8, V, generator=g, dtype=torch.float64) * 12.0  # peaked
    student_logits = teacher_logits + 0.3 * torch.randn(8, V, generator=g, dtype=torch.float64)

    kl, coverage, q_tail = bucketed_from_logits(student_logits, teacher_logits, k=4)
    exact = exact_reverse_kl(student_logits, teacher_logits)

    assert coverage.min() > 0.99
    assert q_tail.max() < 0.01
    torch.testing.assert_close(kl, exact, atol=5e-3, rtol=0.05)


def test_diffuse_student_gap_is_bounded_by_tail():
    """Counterpart: a diffuse student under-counts via the tail bucket, but the
    bucketed KL stays a lower bound and q_tail flags the regime."""
    g = torch.Generator().manual_seed(5)
    student_logits = torch.randn(8, V, generator=g, dtype=torch.float64)  # diffuse
    teacher_logits = torch.randn(8, V, generator=g, dtype=torch.float64) * 12.0  # peaked

    kl, coverage, q_tail = bucketed_from_logits(student_logits, teacher_logits, k=4)
    exact = exact_reverse_kl(student_logits, teacher_logits)

    assert coverage.min() > 0.99  # teacher coverage alone does NOT imply small gap...
    assert (kl <= exact + 1e-7).all()  # ...but the bucketed KL never overshoots
    assert q_tail.max() > 0.3  # and q_tail exposes the diffuse-student regime


def test_tail_term_pushes_mass_toward_teacher_support():
    """Gradient sanity: student mass outside teacher top-K must be penalized."""
    teacher_logits = torch.full((1, V), -10.0, dtype=torch.float64)
    teacher_logits[0, :2] = 5.0  # teacher concentrated on tokens {0, 1}
    student_logits = torch.zeros(1, V, dtype=torch.float64).requires_grad_(True)  # uniform student

    kl, _, q_tail = bucketed_from_logits(student_logits, teacher_logits, k=2)
    kl.sum().backward()

    # Increasing logits of teacher-supported tokens decreases the KL...
    assert (student_logits.grad[0, :2] < 0).all()
    # ...and increasing tail-token logits increases it.
    assert (student_logits.grad[0, 2:] > 0).all()
    assert q_tail.item() > 0.5  # uniform student leaves most mass in the tail
