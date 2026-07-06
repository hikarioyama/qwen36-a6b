"""Tests for the SWE-RL (SEARCH/REPLACE) reward module. Run: pytest -q."""

from __future__ import annotations

import difflib
import time

import pytest

from reward import (
    FormatError,
    apply_code_change,
    calculate_diff_fallback_reward,
    calculate_reward,
    calculate_reward_with_fallback,
    calculate_search_replace_reward,
    compute_reward,
    compute_rewards,
    diff_to_search_replace,
    extract_patch,
    oracle_new_from_patch,
    extract_thought_solution,
    generate_unified_diff,
    get_normalized_patch,
    parse_search_replace,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

CALC = (
    "def add(a, b):\n"
    "    return a + b\n"
    "\n"
    "\n"
    "def div(a, b):\n"
    "    return a / b\n"
    "\n"
    "\n"
    "def mul(a, b):\n"
    "    return a * b\n"
)
UTIL = "def helper():\n    return 42\n"

CODE_CONTEXT = {"pkg/calc.py": CALC, "pkg/util.py": UTIL}

# Oracle edit: guard division by zero.
ORACLE_SR = {"pkg/calc.py": [("    return a / b", "    return a / b if b else 0")]}
ORACLE_NEW = apply_code_change(CODE_CONTEXT, ORACLE_SR)


def sr_block(path: str, search: str, replace: str, lang: str = "python") -> str:
    return (
        f"```{lang}\n### {path}\n<<<<<<< SEARCH\n{search}\n=======\n"
        f"{replace}\n>>>>>>> REPLACE\n```"
    )


def wrap(solution_body: str, thought: str = "I will guard the divisor.") -> str:
    return f"<think>\n{thought}\n</think>\n<solution>\n{solution_body}\n</solution>"


PERFECT = wrap(sr_block("pkg/calc.py", "    return a / b", "    return a / b if b else 0"))


# --------------------------------------------------------------------------- #
# extract_thought_solution
# --------------------------------------------------------------------------- #

def test_extract_valid():
    thought, answer = extract_thought_solution(PERFECT)
    assert thought == "I will guard the divisor."
    assert "SEARCH" in answer


@pytest.mark.parametrize(
    "bad",
    [
        "<solution>x</solution>",                         # no think
        "<think>t</think>",                               # no solution
        "<think>a</think><think>b</think><solution>s</solution>",  # dup think
        "<think></think><solution>s</solution>",          # empty thought
    ],
)
def test_extract_format_errors(bad):
    with pytest.raises(FormatError):
        extract_thought_solution(bad)


# --------------------------------------------------------------------------- #
# parse_search_replace
# --------------------------------------------------------------------------- #

def test_parse_single():
    d = parse_search_replace(sr_block("a.py", "old", "new"))
    assert d == {"a.py": [("old", "new")]}


def test_parse_multiple_files_and_blocks():
    text = (
        sr_block("a.py", "o1", "n1")
        + "\nsome prose\n"
        + sr_block("a.py", "o2", "n2")
        + "\n"
        + sr_block("b.py", "o3", "n3")
    )
    d = parse_search_replace(text)
    assert d["a.py"] == [("o1", "n1"), ("o2", "n2")]
    assert d["b.py"] == [("o3", "n3")]


def test_parse_none_on_prose():
    assert parse_search_replace("no blocks here at all") == {}


# --------------------------------------------------------------------------- #
# apply_code_change
# --------------------------------------------------------------------------- #

def test_apply_ok():
    out = apply_code_change(CODE_CONTEXT, ORACLE_SR)
    assert "return a / b if b else 0" in out["pkg/calc.py"]


def test_apply_search_not_found():
    with pytest.raises(FormatError):
        apply_code_change(CODE_CONTEXT, {"pkg/calc.py": [("nonexistent line", "x")]})


def test_apply_identical_search_replace():
    with pytest.raises(FormatError):
        apply_code_change(CODE_CONTEXT, {"pkg/calc.py": [("    return a / b", "    return a / b")]})


# --------------------------------------------------------------------------- #
# diff helpers
# --------------------------------------------------------------------------- #

def test_generate_unified_diff_empty_when_identical():
    assert generate_unified_diff("x\ny\n", "x\ny\n") == ""


def test_get_normalized_patch_skips_unchanged():
    patch = get_normalized_patch(CODE_CONTEXT, ORACLE_NEW)
    assert set(patch) == {"pkg/calc.py"}  # util.py unchanged -> skipped


# --------------------------------------------------------------------------- #
# calculate_reward (general)
# --------------------------------------------------------------------------- #

def test_calculate_reward_perfect():
    r, _ = calculate_reward(CODE_CONTEXT, ORACLE_NEW, ORACLE_NEW)
    assert r == pytest.approx(1.0)


def test_calculate_reward_no_change_both_empty_is_one():
    r, meta = calculate_reward(CODE_CONTEXT, CODE_CONTEXT, CODE_CONTEXT)
    assert r == 1.0 and meta["similarities"] == []


def test_calculate_reward_wrong_file_is_zero():
    pred = apply_code_change(CODE_CONTEXT, {"pkg/util.py": [("    return 42", "    return 43")]})
    r, _ = calculate_reward(CODE_CONTEXT, ORACLE_NEW, pred)
    # union = {calc.py (oracle-only), util.py (pred-only)} -> both 0.0 -> mean 0.0
    assert r == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# calculate_search_replace_reward (end to end)
# --------------------------------------------------------------------------- #

def test_e2e_perfect_reward_one():
    r, _ = calculate_search_replace_reward(CODE_CONTEXT, ORACLE_NEW, PERFECT)
    assert r == pytest.approx(1.0)


def test_e2e_close_but_not_exact_between_zero_and_one():
    close = wrap(sr_block("pkg/calc.py", "    return a / b", "    return a / b if b != 0 else None"))
    r, _ = calculate_search_replace_reward(CODE_CONTEXT, ORACLE_NEW, close)
    assert 0.0 < r < 1.0


def test_e2e_format_fail_no_tags():
    r, meta = calculate_search_replace_reward(CODE_CONTEXT, ORACLE_NEW, "just prose, no tags")
    assert r == -1.0 and "error" in meta


def test_e2e_no_blocks_is_format_fail():
    r, _ = calculate_search_replace_reward(CODE_CONTEXT, ORACLE_NEW, wrap("no search replace blocks"))
    assert r == -1.0


def test_e2e_search_not_found_is_format_fail():
    bad = wrap(sr_block("pkg/calc.py", "    return a / ZZZ", "x"))
    r, _ = calculate_search_replace_reward(CODE_CONTEXT, ORACLE_NEW, bad)
    assert r == -1.0


# --------------------------------------------------------------------------- #
# Reward hacking
# --------------------------------------------------------------------------- #

def test_hack_empty_edit_is_format_fail():
    # search == replace -> FormatError -> -1 (reference's built-in guard).
    noop = wrap(sr_block("pkg/calc.py", "    return a / b", "    return a / b"))
    r, _ = calculate_search_replace_reward(CODE_CONTEXT, ORACLE_NEW, noop)
    assert r == -1.0


def test_hack_verbatim_oracle_scores_high_by_design():
    # Emitting the exact oracle edit scores 1.0. Intended: in real training the
    # oracle new-content is held out, so it cannot be copied. Documented limit.
    r, _ = calculate_search_replace_reward(CODE_CONTEXT, ORACLE_NEW, PERFECT)
    assert r == pytest.approx(1.0)


def test_hack_touch_extra_wrong_file_dilutes_reward():
    # Correct edit to calc.py PLUS a spurious edit to util.py. util.py is not in
    # the oracle patch -> that file scores 0.0 -> mean drops below 1.0.
    body = (
        sr_block("pkg/calc.py", "    return a / b", "    return a / b if b else 0")
        + "\n"
        + sr_block("pkg/util.py", "    return 42", "    return 42  # noise")
    )
    r, _ = calculate_search_replace_reward(CODE_CONTEXT, ORACLE_NEW, wrap(body))
    assert r == pytest.approx(0.5)  # (1.0 + 0.0) / 2


# --------------------------------------------------------------------------- #
# Wrapper dict API
# --------------------------------------------------------------------------- #

def test_compute_reward_shape():
    r = compute_reward(PERFECT, CODE_CONTEXT, ORACLE_NEW)
    assert r["reward"] == pytest.approx(1.0)
    assert r["format_valid"] is True
    assert r["error"] is None
    assert r["n_oracle_files"] == 1
    assert r["n_pred_files"] == 1
    assert r["per_file"][0]["path"] == "pkg/calc.py"


def test_compute_reward_format_fail_shape():
    r = compute_reward("garbage", CODE_CONTEXT, ORACLE_NEW)
    assert r["reward"] == -1.0
    assert r["format_valid"] is False
    assert r["error"]
    assert r["n_oracle_files"] == 1  # still reports oracle scope
    assert r["per_file"] == []


def test_compute_rewards_batch():
    close = wrap(sr_block("pkg/calc.py", "    return a / b", "    return a / b if b != 0 else None"))
    out = compute_rewards([PERFECT, close, "garbage"], CODE_CONTEXT, ORACLE_NEW)
    assert len(out) == 3
    assert out[0]["reward"] == pytest.approx(1.0)
    assert 0.0 < out[1]["reward"] < 1.0
    assert out[2]["reward"] == -1.0


# --------------------------------------------------------------------------- #
# Speed
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Diff fallback + flags
# --------------------------------------------------------------------------- #

def _git_diff(path: str, old: str, new: str) -> str:
    body = difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="", n=3,
    )
    return f"diff --git a/{path} b/{path}\n" + "\n".join(body)


CALC_DIFF = _git_diff("pkg/calc.py", CALC, ORACLE_NEW["pkg/calc.py"])
DIFF_ONLY = f"Here is the fix:\n```diff\n{CALC_DIFF}\n```"  # no think/solution, no SR


def test_extract_patch_from_fenced():
    assert extract_patch(DIFF_ONLY).startswith("diff --git")


def test_extract_patch_none_on_prose():
    assert extract_patch("no diff here") is None


def test_diff_to_search_replace_roundtrip():
    sr = diff_to_search_replace(CALC_DIFF)
    pred = apply_code_change(CODE_CONTEXT, sr)
    assert pred["pkg/calc.py"] == ORACLE_NEW["pkg/calc.py"]


def test_diff_to_search_replace_skips_identical_hunks():
    # Gold patches often end with a spurious "-line / +line" (identical text, an
    # EOF-newline artifact). It must be dropped, not turned into a no-op edit
    # that trips apply_code_change's identical guard.
    diff = (
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
        "@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2\n"
        "@@ -9,2 +9,2 @@\n ctx\n-    return z\n+    return z\n"  # identical -> skip
    )
    srp = diff_to_search_replace(diff)
    pairs = srp["x.py"]
    assert all(s != r for s, r in pairs)
    assert any("return 1" in s and "return 2" in r for s, r in pairs)


def test_fallback_scores_raw_diff():
    # Model emitted a raw diff (no SEARCH/REPLACE). Fallback should recover 1.0.
    r, meta = calculate_diff_fallback_reward(CODE_CONTEXT, ORACLE_NEW, DIFF_ONLY)
    assert r == pytest.approx(1.0)
    assert meta["method"] == "diff_fallback"


def test_with_fallback_recovers_when_sr_missing():
    r = compute_reward(DIFF_ONLY, CODE_CONTEXT, ORACLE_NEW)  # fallback on by default
    assert r["reward"] == pytest.approx(1.0)
    assert r["method"] == "diff_fallback"
    assert r["format_valid"] is True


def test_pure_compat_disables_fallback():
    # allow_diff_fallback=False -> a diff-only completion is a format fail (-1),
    # which is the faithful SWE-RL behavior for A/B.
    r = compute_reward(DIFF_ONLY, CODE_CONTEXT, ORACLE_NEW, allow_diff_fallback=False)
    assert r["reward"] == -1.0
    assert r["format_valid"] is False


def test_sr_success_does_not_use_fallback():
    r = compute_reward(PERFECT, CODE_CONTEXT, ORACLE_NEW)
    assert r["method"] == "search_replace"
    assert r["reward"] == pytest.approx(1.0)


def test_both_paths_fail_is_minus_one():
    r = compute_reward(wrap("prose only, no blocks and no diff"), CODE_CONTEXT, ORACLE_NEW)
    assert r["reward"] == -1.0
    assert r["method"] == "format_fail"


def test_truncated_diff_fallback_fails_gracefully():
    # A diff cut off mid-hunk: the reconstructed search block won't match the
    # file (or yields no usable hunk) -> -1, no crash.
    truncated = "```diff\n" + CALC_DIFF[: len(CALC_DIFF) // 2]
    r = compute_reward(truncated, CODE_CONTEXT, ORACLE_NEW)
    assert r["reward"] == -1.0


def test_oracle_new_from_patch_matches_direct_apply():
    # Deriving oracle_new_content from (base + oracle patch) equals the direct
    # SEARCH/REPLACE application -> reward vs it is 1.0 for the perfect completion.
    derived = oracle_new_from_patch(CODE_CONTEXT, CALC_DIFF)
    assert derived["pkg/calc.py"] == ORACLE_NEW["pkg/calc.py"]
    r = compute_reward(PERFECT, CODE_CONTEXT, derived)
    assert r["reward"] == pytest.approx(1.0)


def test_normalize_absorbs_trailing_whitespace_symmetrically():
    # Model's replacement carries trailing whitespace the oracle lacks.
    dirty = wrap(sr_block("pkg/calc.py", "    return a / b", "    return a / b if b else 0   "))
    r_norm = calculate_reward_with_fallback(CODE_CONTEXT, ORACLE_NEW, dirty, normalize=True)[0]
    r_raw = calculate_reward_with_fallback(CODE_CONTEXT, ORACLE_NEW, dirty, normalize=False)[0]
    assert r_norm == pytest.approx(1.0)
    assert r_raw < 1.0


# --------------------------------------------------------------------------- #
# Speed
# --------------------------------------------------------------------------- #

def test_speed_1000_batch_under_10ms_each():
    comps = [PERFECT] * 1000
    t0 = time.perf_counter()
    out = compute_rewards(comps, CODE_CONTEXT, ORACLE_NEW)
    dt = time.perf_counter() - t0
    assert len(out) == 1000
    assert dt / 1000 < 0.010, f"{dt/1000*1e3:.2f} ms/item exceeds 10 ms"
