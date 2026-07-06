"""End-to-end integration tests against the GRPO data contract.

Contract (ratified by team-lead):
  record['repo_files']       -> code_context      (base file contents)
  record['oracle_new_files'] -> oracle_new_content (preferred), else derived from
  record['oracle_patch']     applied to repo_files.

Two layers:
  * Synthetic-contract tests exercise ``score_record`` and its error handling --
    always run.
  * Real-data tests read a committed fixture of actual grpo_prompts oracle
    patches; a live-file hook runs against ~ data if scp'd locally.
"""

from __future__ import annotations

import json
import os

import pytest

from reward import (
    FormatError,
    _normalize_change,
    compute_reward,
    diff_to_search_replace,
    iter_joined,
    oracle_new_from_patch,
    score_record,
)


def _all_search_blocks_unique(repo_files: dict, oracle_patch: str) -> bool:
    """True if every hunk's SEARCH block occurs exactly once in its base file.

    When a block repeats, ``apply_code_change`` (str.replace) is ambiguous and
    over-applies -- a known limitation, orthogonal to patch *direction*. We only
    assert the direction cross-check on records where derivation is unambiguous.
    """
    try:
        srp = diff_to_search_replace(oracle_patch)
    except FormatError:
        return False
    for path, pairs in srp.items():
        base = "\n" + repo_files.get(path, "")
        for search, _replace in pairs:
            if base.count("\n" + search) != 1:
                return False
    return True


def _same_modulo_ws(a: str, b: str) -> bool:
    """Equal ignoring per-line trailing whitespace and a trailing newline.

    Matches the reward's own tolerance (normalize=True). ``apply_code_change``
    can drop the file's final ``\\n`` when the edit is near EOF -- a benign
    artifact the reward absorbs; only a true content/direction difference should
    fail the guard.
    """
    return _normalize_change(a).rstrip("\n") == _normalize_change(b).rstrip("\n")

HERE = os.path.dirname(__file__)
FIXTURE = os.path.join(HERE, "testdata", "real_records_sample.jsonl")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def sr_block(path: str, search: str, replace: str) -> str:
    return (
        f"```python\n### {path}\n<<<<<<< SEARCH\n{search}\n=======\n"
        f"{replace}\n>>>>>>> REPLACE\n```"
    )


def wrap(body: str) -> str:
    return f"<think>\nfix\n</think>\n<solution>\n{body}\n</solution>"


def reconstruct_from_patch(oracle_patch: str) -> tuple[dict[str, str], dict[str, str]]:
    """Build (buggy_base, fixed_new) from a patch's hunks alone.

    Forward direction: '-' + context = buggy base (pre), '+' + context = fixed
    (post). Returns (code_context=buggy, oracle_new=fixed).

    NOTE: this yields a *standalone hunk-region* file, not the real full file --
    valid only for a self-match smoke on real oracle content, never for scoring a
    free-form model completion. Production must use record['repo_files'].
    """
    files: dict[str, tuple[list[str], list[str]]] = {}
    path = None
    in_hunk = False
    for ln in oracle_patch.split("\n"):
        if ln.startswith("+++ "):
            p = ln[4:].strip()
            p = p[2:] if p.startswith("b/") else p
            path = None if p == "/dev/null" else p
            if path:
                files.setdefault(path, ([], []))
            in_hunk = False
        elif ln.startswith("@@"):
            in_hunk = True
        elif in_hunk and path:
            if ln.startswith("-"):
                files[path][0].append(ln[1:])
            elif ln.startswith("+"):
                files[path][1].append(ln[1:])
            elif ln.startswith(" "):
                files[path][0].append(ln[1:])
                files[path][1].append(ln[1:])
    cc = {p: "\n".join(o) for p, (o, n) in files.items()}
    on = {p: "\n".join(n) for p, (o, n) in files.items()}
    return cc, on


def load_fixture() -> list[dict]:
    if not os.path.exists(FIXTURE):
        return []
    return [json.loads(l) for l in open(FIXTURE) if l.strip()]


# --------------------------------------------------------------------------- #
# Synthetic contract (always runs)
# --------------------------------------------------------------------------- #

# Contract direction: repo_files = BUGGY (pre), oracle_new_files = FIXED (post),
# oracle_patch = buggy->fixed (forward-applies to repo_files).
BUGGY_FILES = {"pkg/m.py": "def f(x):\n    return x + 1\n\n\ndef g(x):\n    return x\n"}
ORACLE_PATCH = (  # buggy -> fixed
    "diff --git a/pkg/m.py b/pkg/m.py\n"
    "--- a/pkg/m.py\n"
    "+++ b/pkg/m.py\n"
    "@@ -1,3 +1,3 @@\n"
    " def f(x):\n"
    "-    return x + 1\n"
    "+    return x + 2\n"
    " \n"
)
PERFECT = wrap(sr_block("pkg/m.py", "    return x + 1", "    return x + 2"))


def test_score_record_derives_oracle_from_patch():
    record = {"repo_files": BUGGY_FILES, "oracle_patch": ORACLE_PATCH}
    r = score_record(record, PERFECT)
    assert r["reward"] == pytest.approx(1.0)
    assert r["method"] == "search_replace"


def test_score_record_uses_oracle_new_files_when_present():
    oracle_new = oracle_new_from_patch(BUGGY_FILES, ORACLE_PATCH)
    record = {"repo_files": BUGGY_FILES, "oracle_new_files": oracle_new}
    r = score_record(record, PERFECT)
    assert r["reward"] == pytest.approx(1.0)


def test_score_record_missing_repo_files_raises():
    with pytest.raises(ValueError, match="repo_files"):
        score_record({"oracle_patch": ORACLE_PATCH}, PERFECT)


def test_score_record_missing_oracle_raises():
    with pytest.raises(ValueError, match="oracle"):
        score_record({"repo_files": BUGGY_FILES}, PERFECT)


def test_score_record_wrong_edit_scored_low():
    record = {"repo_files": BUGGY_FILES, "oracle_patch": ORACLE_PATCH}
    wrong = wrap(sr_block("pkg/m.py", "    return x", "    return x * 2"))  # edits g, not f
    r = score_record(record, wrong)
    assert r["reward"] < 1.0


# --------------------------------------------------------------------------- #
# Real data: committed fixture (reconstructed self-match)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not load_fixture(), reason="no committed real_records fixture")
def test_real_oracle_patches_self_match():
    records = load_fixture()
    scored = 0
    for rec in records:
        cc, on = reconstruct_from_patch(rec["oracle_patch"])
        changed = {p for p in cc if cc[p] != on[p]}
        if not changed:
            continue
        body = "\n".join(sr_block(p, cc[p], on[p]) for p in changed)
        r = compute_reward(wrap(body), cc, on)
        assert r["reward"] == pytest.approx(1.0), rec.get("instance_id")
        assert r["format_valid"]
        scored += 1
    assert scored > 0


@pytest.mark.skipif(not load_fixture(), reason="no committed real_records fixture")
def test_real_oracle_patch_malformed_is_minus_one():
    rec = load_fixture()[0]
    cc, on = reconstruct_from_patch(rec["oracle_patch"])
    r = compute_reward("no tags, no blocks, no diff", cc, on)
    assert r["reward"] == -1.0


# --------------------------------------------------------------------------- #
# Live full-contract e2e: real v1 data (main + sidecar, joined on instance_id)
# --------------------------------------------------------------------------- #
# Defaults to the local full-run files; override with env vars. The main path may
# also be a single pre-joined jsonl (repo_files inline), which we detect.

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LIVE_MAIN = os.environ.get("GRPO_PROMPTS_LOCAL") or os.path.join(_DATA_DIR, "grpo_prompts.jsonl")
LIVE_FILES = os.environ.get("GRPO_PROMPTS_FILES_LOCAL") or os.path.join(_DATA_DIR, "grpo_prompts_files.jsonl")
_LIVE_N = int(os.environ.get("GRPO_LIVE_N", "15"))


def _load_live(n: int) -> list[dict]:
    """First ``n`` joined records from the live data (main + sidecar), or []."""
    if not os.path.exists(LIVE_MAIN):
        return []
    if os.path.exists(LIVE_FILES):
        out = []
        for rec in iter_joined(LIVE_MAIN, LIVE_FILES):
            out.append(rec)
            if len(out) >= n:
                break
        return out
    # single pre-joined file fallback
    out = []
    for line in open(LIVE_MAIN):
        if line.strip():
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


@pytest.mark.skipif(not _load_live(1), reason="no local v1 data (main+sidecar)")
def test_live_records_full_contract():
    records = _load_live(_LIVE_N)
    have_base = [r for r in records if isinstance(r.get("repo_files"), dict) and r["repo_files"]]
    assert have_base, "joined live records must carry repo_files from the sidecar"
    for rec in have_base:
        cc = rec["repo_files"]  # buggy (pre)
        oracle_new = oracle_new_from_patch(cc, rec["oracle_patch"])  # fixed (post)
        # A perfect completion = the oracle edit as one SEARCH/REPLACE per file.
        body = "\n".join(sr_block(p, cc[p], new) for p, new in oracle_new.items())
        r = score_record(rec, wrap(body))
        assert r["reward"] == pytest.approx(1.0), rec.get("instance_id")
        # malformed on the same real record -> -1
        assert score_record(rec, "no tags no blocks no diff")["reward"] == -1.0


@pytest.mark.skipif(not _load_live(1), reason="no local v1 data (main+sidecar)")
def test_live_patch_direction_matches_oracle_new_files():
    """Direction guard (catches a reversed patch that self-match cannot).

    ``oracle_new_from_patch(repo_files, oracle_patch)`` (patch applied FORWARD to
    the BUGGY base) must equal the pipeline's FIXED ``oracle_new_files`` (git
    apply). A reversal would make buggy != fixed and fail here, while self-match
    would stay 1.0 -- the silent bug flagged in review. Compared modulo trailing
    whitespace / final newline (a benign apply artifact the reward also absorbs).
    """
    records = _load_live(_LIVE_N)
    both = [
        r for r in records
        if isinstance(r.get("repo_files"), dict) and r["repo_files"]
        and isinstance(r.get("oracle_new_files"), dict) and r["oracle_new_files"]
    ]
    assert both, "joined live records must carry repo_files + oracle_new_files"
    checked = 0
    for rec in both:
        # Only records where the content-anchored derivation is unambiguous -- a
        # repeated SEARCH block makes str.replace over-apply (limitation #3), not
        # a direction bug. On the clean records a reversal would still fail here.
        if not _all_search_blocks_unique(rec["repo_files"], rec["oracle_patch"]):
            continue
        derived = oracle_new_from_patch(rec["repo_files"], rec["oracle_patch"])
        for path, target in rec["oracle_new_files"].items():
            assert _same_modulo_ws(derived.get(path, ""), target), (
                f"{rec.get('instance_id')} {path}: forward-applied oracle_patch != "
                "oracle_new_files (reversed/broken patch?)"
            )
        checked += 1
    assert checked > 0, "no unambiguous records to cross-check direction"
