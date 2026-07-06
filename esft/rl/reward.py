"""SWE-RL rule-based reward for GRPO / rejection-sampling patch rollouts.

Reference: "SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open
Software Evolution" (arXiv:2502.18449) and the official implementation
``facebookresearch/swe-rl`` (``src/swerl/core/reward.py``).

Main path -- SEARCH/REPLACE (faithful to the reference)
-------------------------------------------------------
The policy emits SEARCH/REPLACE edits (writing correct @@ line numbers is
infeasible; SWE-RL chose this format for exactly that reason -- it lifts the
format pass-rate so a GRPO group is not flooded with format-fail -1 that
collapse the advantage):

    <think> ...reasoning... </think>
    <solution>
    ```python
    ### path/to/file.py
    <<<<<<< SEARCH
    original code (verbatim, indentation-exact)
    =======
    replacement code
    >>>>>>> REPLACE
    ```
    </solution>

Reward: parse -> apply edits to the *original* file contents (``code_context``)
-> regenerate a per-file unified diff with ``difflib.unified_diff`` for both the
prediction and the oracle -> per-file ``SequenceMatcher(autojunk=False).ratio()``
-> arithmetic mean over the union of touched files, in [0, 1]. Any format/parse/
apply violation -> -1.0 (``FormatError``). This is the reference algorithm.

Diff fallback (our extension, gated by ``allow_diff_fallback``)
--------------------------------------------------------------
If SEARCH/REPLACE parsing fails, we try to extract a raw unified diff from the
completion (robust 5-stage extractor), convert each hunk to a (search, replace)
pair by string content (ignoring the model's unreliable @@ line numbers), and run
it through the *same* apply+compare arena. Only if that also fails -> -1.0.
Set ``allow_diff_fallback=False`` for pure SWE-RL paper-compat A/B (no fallback).

Normalization (approved enhancement over the paper)
---------------------------------------------------
Before ``SequenceMatcher`` we optionally normalize BOTH sides symmetrically
(CRLF->LF, strip trailing whitespace) so the reward measures the edit, not
incidental byte-formatting. Default ``normalize=True``; both sides get the same
transform so the ratio is not systematically biased. ``normalize=False`` gives
the byte-exact paper behavior. (In the difflib-generated arena there are no git
``index``/sha lines, so normalization only affects CRLF / trailing whitespace.)

Similarity strings are capped at ``MAX_SIM_CHARS`` to bound the O(n*m) worst case
and blunt giant-patch reward hacking (ratio = 2M/T already shrinks with size).

Pure-Python / CPU. ``compute_reward`` / ``compute_rewards`` are the GRPO-facing
wrappers; the lower-level functions mirror the reference signatures.
"""

from __future__ import annotations

import difflib
import re
from typing import Iterable, Optional, TypedDict

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

FORMAT_FAIL_REWARD = -1.0
MAX_SIM_CHARS = 20_000

THINK_START = "<think>"
THINK_END = "</think>"
ANSWER_START = "<solution>"
ANSWER_END = "</solution>"

SEARCH_REPLACE_REGEX = (
    r"```.*?\n### (.*)\n<<<<<<< SEARCH\n([\s\S]*?)\n=======\n"
    r"([\s\S]*?)\n>>>>>>> REPLACE\n```"
)

# Fallback diff extractor
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_DIFF_RE = re.compile(r"```(?:diff|patch)[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)
_DIFF_GIT_RE = re.compile(r"^diff --git ", re.MULTILINE)
_HUNK_HDR_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.MULTILINE)


class FormatError(Exception):
    """Raised for any parse/apply violation. Maps to reward -1.0."""


class ChangeSimilarity(TypedDict):
    path: str
    pred_change: str
    oracle_change: str
    similarity: float


# --------------------------------------------------------------------------- #
# Faithful reference core (facebookresearch/swe-rl)
# --------------------------------------------------------------------------- #

def extract_thought_solution(output: str) -> tuple[str, str]:
    """Split a completion into (thought, solution). Faithful to the reference.

    Requires exactly one of each of ``<think>``/``</think>``/``<solution>``/
    ``</solution>`` and a non-empty thought; otherwise ``FormatError``.
    """
    for tag in (THINK_START, THINK_END, ANSWER_START, ANSWER_END):
        if output.count(tag) != 1:
            raise FormatError(f"count of {tag} is not 1")
    thought = output.split(THINK_START)[1].split(THINK_END)[0].strip()
    answer = output.split(ANSWER_START)[1].split(ANSWER_END)[0].strip()
    if len(thought) == 0:
        raise FormatError("Thought is empty")
    return thought, answer


def parse_search_replace(text: str) -> dict[str, list[tuple[str, str]]]:
    """Parse SEARCH/REPLACE blocks into ``{path: [(search, replace), ...]}``."""
    pairs: list[tuple[str, str, str]] = re.findall(SEARCH_REPLACE_REGEX, text)
    out: dict[str, list[tuple[str, str]]] = {}
    for path, search, replace in pairs:
        out.setdefault(path, []).append((search, replace))
    return out


def apply_code_change(
    code_context: dict[str, str],
    search_replace_dict: dict[str, list[tuple[str, str]]],
    silent: bool = False,
) -> dict[str, str]:
    """Apply SEARCH/REPLACE edits to file contents. Faithful to the reference.

    A leading ``\\n`` is prepended to both the file content and each search/
    replace string so a search anchored at column 0 matches correct indentation.
    ``FormatError`` if a search equals its replace (empty edit / reward hack) or
    a search block does not occur in the file.
    """
    new_content_dict: dict[str, str] = {}
    for path, search_replaces in search_replace_dict.items():
        new_content = "\n" + code_context.get(path, "")
        for search, replace in search_replaces:
            if not silent and len(search) == len(replace) and search == replace:
                raise FormatError("Search and replace blocks are identical")
            search = "\n" + search
            replace = "\n" + replace
            if not silent and search not in new_content:
                raise FormatError(f"Search block not found in the code: {search}")
            new_content = new_content.replace(search, replace)
        new_content_dict[path] = new_content[1:]  # drop the leading "\n"
    return new_content_dict


def generate_unified_diff(old_code: str, new_code: str, n_context: int = 3) -> str:
    """Per-file unified diff body (the two ``---/+++`` header lines dropped)."""
    diff = difflib.unified_diff(
        old_code.splitlines(),
        new_code.splitlines(),
        fromfile="old",
        tofile="new",
        lineterm="",
        n=n_context,
    )
    try:
        next(diff)
        next(diff)
        return "\n".join(diff)
    except StopIteration:
        return ""


def get_normalized_patch(
    code_context: dict[str, str],
    new_content_dict: dict[str, str],
) -> dict[str, str]:
    """``{path: unified_diff}`` for every file that actually changed."""
    patch_dict: dict[str, str] = {}
    for path, new_content in new_content_dict.items():
        patch = generate_unified_diff(code_context.get(path, ""), new_content)
        if patch:
            patch_dict[path] = patch
    return patch_dict


def _normalize_change(s: str) -> str:
    """Symmetric, side-agnostic normalization: CRLF->LF, strip trailing ws."""
    return "\n".join(line.rstrip() for line in s.replace("\r\n", "\n").split("\n"))


def compute_change_similarities(
    pred_patch: dict[str, str],
    oracle_patch: dict[str, str],
    normalize: bool = True,
) -> list[ChangeSimilarity]:
    """Per-file diff-string similarity over the union of touched files.

    A file that only one side touches scores 0.0 (this penalizes "touched the
    wrong file" and "missed a file" -- no separate files_match needed). When
    ``normalize`` is set, both sides get the SAME transform before comparison so
    the ratio is not systematically biased. Strings are capped at MAX_SIM_CHARS.
    """
    sims: list[ChangeSimilarity] = []
    for path in set(oracle_patch) | set(pred_patch):
        pred_change = pred_patch.get(path, "")
        oracle_change = oracle_patch.get(path, "")
        if oracle_change == "" or pred_change == "":
            similarity = 0.0
        else:
            a = _normalize_change(pred_change) if normalize else pred_change
            b = _normalize_change(oracle_change) if normalize else oracle_change
            similarity = difflib.SequenceMatcher(
                None, a[:MAX_SIM_CHARS], b[:MAX_SIM_CHARS], autojunk=False
            ).ratio()
        sims.append(
            ChangeSimilarity(
                path=path,
                pred_change=pred_change,
                oracle_change=oracle_change,
                similarity=similarity,
            )
        )
    return sims


def calculate_reward(
    code_context: dict[str, str],
    oracle_new_content: dict[str, str],
    pred_new_content: dict[str, str],
    normalize: bool = True,
    _oracle_patch: Optional[dict[str, str]] = None,
) -> tuple[float, dict]:
    """General reward: mean per-file diff similarity in [0, 1].

    Both oracle and prediction are given as *new file contents*; each is turned
    into a normalized diff against ``code_context`` and compared. Both-empty
    (identical no-ops) rewards 1.0.

    ``_oracle_patch`` is an internal cache: the oracle diff is identical for every
    completion in a GRPO group, so callers may precompute it once and thread it in
    to avoid recomputation. When None it is computed here (reference behavior).
    """
    oracle_patch = (
        _oracle_patch if _oracle_patch is not None
        else get_normalized_patch(code_context, oracle_new_content)
    )
    pred_patch = get_normalized_patch(code_context, pred_new_content)
    similarities = compute_change_similarities(pred_patch, oracle_patch, normalize)
    if len(similarities) == 0:
        return 1.0, {"similarities": []}
    reward = sum(s["similarity"] for s in similarities) / len(similarities)
    return reward, {"similarities": similarities}


def calculate_search_replace_reward(
    code_context: dict[str, str],
    oracle_new_content: dict[str, str],
    output: str,
    normalize: bool = True,
    _oracle_patch: Optional[dict[str, str]] = None,
) -> tuple[float, dict]:
    """SEARCH/REPLACE reward. -1.0 on any ``FormatError``. Faithful wrapper."""
    try:
        thought, answer = extract_thought_solution(output)
        pred_search_replaces = parse_search_replace(answer)
        if len(pred_search_replaces) == 0:
            raise FormatError("No valid search blocks found")
        pred_new_content = apply_code_change(code_context, pred_search_replaces)
        reward, metadata = calculate_reward(
            code_context, oracle_new_content, pred_new_content, normalize, _oracle_patch
        )
        metadata["thought"] = thought
        metadata["answer"] = answer
        metadata["method"] = "search_replace"
        return reward, metadata
    except FormatError as e:
        return FORMAT_FAIL_REWARD, {"error": str(e), "method": "format_fail"}


# --------------------------------------------------------------------------- #
# Diff fallback (our extension)
# --------------------------------------------------------------------------- #

def extract_patch(text: str) -> Optional[str]:
    """Pull a raw unified diff out of a completion (fallback extractor).

    Strategy: strip ``<think>`` spans; prefer the LAST fenced ```diff/```patch
    block (final-answer convention, tolerant of a missing closing fence for
    truncation); else raw ``diff --git`` region to EOF; else a headerless
    ``--- / +++`` diff. Returns the diff text or None.
    """
    if not text:
        return None
    cleaned = _THINK_BLOCK_RE.sub("", text)
    blocks = [m.group(1) for m in _FENCE_DIFF_RE.finditer(cleaned)]
    for block in reversed(blocks):
        if _DIFF_GIT_RE.search(block) or "@@" in block:
            return block.strip("\n")
    m = _DIFF_GIT_RE.search(cleaned)
    if m:
        return cleaned[m.start():].strip("\n")
    if "@@" in cleaned:
        lines = cleaned.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
                return "\n".join(lines[i:]).strip("\n")
    return None


def _strip_ab_prefix(path: str) -> str:
    path = path.split("\t", 1)[0]
    return path[2:] if path.startswith(("a/", "b/")) else path


def diff_to_search_replace(diff_text: str) -> dict[str, list[tuple[str, str]]]:
    """Convert a unified diff into ``{path: [(search, replace), ...]}`` by hunk.

    For each hunk: search = context + removed lines, replace = context + added
    lines (the model's @@ line numbers are ignored -- we anchor by content, like
    SEARCH/REPLACE). Raises ``FormatError`` if no usable hunk is found.
    """
    out: dict[str, list[tuple[str, str]]] = {}
    path: Optional[str] = None
    search: list[str] = []
    replace: list[str] = []
    in_hunk = False

    def flush():
        nonlocal search, replace
        s, r = "\n".join(search), "\n".join(replace)
        # Skip no-op hunks (s == r). Gold patches often carry a spurious trailing
        # "-line / +line" with identical text -- an EOF-newline artifact, not a
        # real edit. Applying it would trip apply_code_change's identical guard;
        # dropping it is exact (a search==replace edit changes nothing).
        if in_hunk and path is not None and s != r:
            out.setdefault(path, []).append((s, r))
        search, replace = [], []

    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            flush()
            in_hunk = False
            p = _strip_ab_prefix(line[4:].strip())
            path = None if p == "/dev/null" else p
        elif line.startswith("--- ") or line.startswith("diff --git "):
            flush()
            in_hunk = False
        elif line.startswith("@@"):
            flush()
            in_hunk = True
        elif in_hunk:
            tag = line[:1]
            if tag == "+":
                replace.append(line[1:])
            elif tag == "-":
                search.append(line[1:])
            elif tag == " " or line == "":
                search.append(line[1:] if tag == " " else "")
                replace.append(line[1:] if tag == " " else "")
            elif tag == "\\":
                pass  # "\ No newline at end of file"
            else:
                flush()
                in_hunk = False
    flush()
    out = {p: v for p, v in out.items() if p and v}
    if not out:
        raise FormatError("No usable hunks in diff fallback")
    return out


def oracle_new_from_patch(
    code_context: dict[str, str],
    oracle_patch: str,
) -> dict[str, str]:
    """Derive oracle post-edit contents from a git/unified oracle patch + base.

    Convenience for data prep: SWE-smith records carry the oracle as a unified
    diff (``patch``) plus the base file contents (``code_context``). This applies
    the patch by content-anchored hunks (same path as the diff fallback) to yield
    the ``oracle_new_content`` that ``compute_reward`` needs. Raises
    ``FormatError`` if the patch does not cleanly anchor to the base files.

    Direction: the patch is applied FORWARD -- ``-`` lines are removed from the
    base and ``+`` lines are the desired target (fix). For SWE-smith the stored
    ``oracle_patch`` is already the fix direction (its ``+`` side = corrected
    code). When ``oracle_new_files`` is supplied by the data pipeline, prefer it
    and cross-check ``oracle_new_from_patch(...) == oracle_new_files`` -- a
    mismatch reveals a reversed/broken patch that self-match alone cannot catch.
    """
    return apply_code_change(code_context, diff_to_search_replace(oracle_patch))


def calculate_diff_fallback_reward(
    code_context: dict[str, str],
    oracle_new_content: dict[str, str],
    output: str,
    normalize: bool = True,
    _oracle_patch: Optional[dict[str, str]] = None,
) -> tuple[float, dict]:
    """Fallback: extract a raw diff, convert to edits, run the same arena."""
    try:
        diff = extract_patch(output)
        if diff is None:
            raise FormatError("No diff found in fallback")
        search_replaces = diff_to_search_replace(diff)
        pred_new_content = apply_code_change(code_context, search_replaces)
        reward, metadata = calculate_reward(
            code_context, oracle_new_content, pred_new_content, normalize, _oracle_patch
        )
        metadata["method"] = "diff_fallback"
        return reward, metadata
    except FormatError as e:
        return FORMAT_FAIL_REWARD, {"error": str(e), "method": "format_fail"}


def calculate_reward_with_fallback(
    code_context: dict[str, str],
    oracle_new_content: dict[str, str],
    output: str,
    allow_diff_fallback: bool = True,
    normalize: bool = True,
    _oracle_patch: Optional[dict[str, str]] = None,
) -> tuple[float, dict]:
    """SEARCH/REPLACE primary; on format failure try the diff fallback (unless
    disabled for pure SWE-RL compat). Both fail -> -1.0."""
    reward, meta = calculate_search_replace_reward(
        code_context, oracle_new_content, output, normalize, _oracle_patch
    )
    if reward == FORMAT_FAIL_REWARD and allow_diff_fallback:
        fb_reward, fb_meta = calculate_diff_fallback_reward(
            code_context, oracle_new_content, output, normalize, _oracle_patch
        )
        if fb_meta.get("method") == "diff_fallback":
            fb_meta["search_replace_error"] = meta.get("error")
            return fb_reward, fb_meta
        # fallback also failed: keep the original SR error, note the fb error too
        meta["fallback_error"] = fb_meta.get("error")
    return reward, meta


# --------------------------------------------------------------------------- #
# GRPO / rejection-sampling wrapper
# --------------------------------------------------------------------------- #

class RewardResult(TypedDict, total=False):
    reward: float          # scalar for RL; -1.0 on format failure, else [0,1]
    format_valid: bool     # scored (not a format failure)
    method: str            # "search_replace" | "diff_fallback" | "format_fail"
    error: Optional[str]   # error message on failure, else None
    n_pred_files: int
    n_oracle_files: int
    per_file: list[dict]   # [{path, similarity}, ...] diagnostics


def compute_reward(
    completion: str,
    code_context: dict[str, str],
    oracle_new_content: dict[str, str],
    allow_diff_fallback: bool = True,
    normalize: bool = True,
    _oracle_patch: Optional[dict[str, str]] = None,
) -> RewardResult:
    """Score one completion for one instance.

    Args:
      completion: raw model output (SEARCH/REPLACE inside <think>/<solution>).
      code_context: ``{path: original file content}`` for oracle-touched files.
      oracle_new_content: ``{path: oracle post-edit file content}``.
      allow_diff_fallback: if the SEARCH/REPLACE parse fails, try to score a raw
        unified diff extracted from the output. Set False for pure SWE-RL compat.
      normalize: symmetric CRLF/trailing-ws normalization before similarity.
      _oracle_patch: internal per-group cache of the oracle diff (see
        ``compute_rewards``); computed here when None.

    Returns a ``RewardResult``. ``reward`` is the RL scalar (mean per-file diff
    similarity, or -1.0). Sub-fields are diagnostics -- GRPO uses ``reward``
    directly; rejection-sampling can filter on ``reward``/``format_valid``.
    """
    oracle_patch = (
        _oracle_patch if _oracle_patch is not None
        else get_normalized_patch(code_context, oracle_new_content)
    )
    reward, meta = calculate_reward_with_fallback(
        code_context, oracle_new_content, completion,
        allow_diff_fallback, normalize, oracle_patch,
    )
    n_oracle = len(oracle_patch)
    if meta.get("method") == "format_fail":
        return {
            "reward": reward,
            "format_valid": False,
            "method": "format_fail",
            "error": meta.get("error"),
            "n_pred_files": 0,
            "n_oracle_files": n_oracle,
            "per_file": [],
        }
    sims = meta.get("similarities", [])
    per_file = [{"path": s["path"], "similarity": s["similarity"]} for s in sims]
    n_pred = sum(1 for s in sims if s["pred_change"])
    return {
        "reward": reward,
        "format_valid": True,
        "method": meta.get("method", "search_replace"),
        "error": None,
        "n_pred_files": n_pred,
        "n_oracle_files": n_oracle,
        "per_file": per_file,
    }


def _record_code_context(record: dict) -> dict[str, str]:
    """Base (pre-edit) file contents for a data record, per the data contract."""
    repo_files = record.get("repo_files")
    if not isinstance(repo_files, dict) or not repo_files:
        raise ValueError(
            "record is missing 'repo_files' (base contents of oracle-touched "
            "files). The reward cannot reconstruct patches without it. Ask the "
            "data pipeline to include 'repo_files' (and ideally 'oracle_new_files')."
        )
    return repo_files


def _record_oracle_new(record: dict, code_context: dict[str, str]) -> dict[str, str]:
    """Oracle post-edit contents: prefer the field, else derive from the patch."""
    onf = record.get("oracle_new_files")
    if isinstance(onf, dict) and onf:
        return onf
    patch = record.get("oracle_patch")
    if isinstance(patch, str) and patch.strip():
        return oracle_new_from_patch(code_context, patch)
    raise ValueError(
        "record has neither 'oracle_new_files' nor a usable 'oracle_patch'; "
        "cannot determine the oracle target."
    )


def score_record(
    record: dict,
    completion: str,
    allow_diff_fallback: bool = True,
    normalize: bool = True,
) -> RewardResult:
    """Trainer-facing adapter: score one completion against one data record.

    Encodes the data contract so the RL loop has a single entry point:
      code_context      <- record['repo_files']         (base file contents)
      oracle_new_content <- record['oracle_new_files']  (preferred) else derived
                            from record['oracle_patch'] applied to repo_files.

    Raises ``ValueError`` (loudly) if the record lacks the base contents -- there
    is no correct silent fallback (reconstructing a standalone file from the patch
    hunks only covers the changed region, not the whole file the model may edit).
    """
    code_context = _record_code_context(record)
    oracle_new_content = _record_oracle_new(record, code_context)
    return compute_reward(
        completion, code_context, oracle_new_content, allow_diff_fallback, normalize
    )


# --------------------------------------------------------------------------- #
# Data loading: main + sidecar join (v1 contract)
# --------------------------------------------------------------------------- #
# v1 ships two files joined on instance_id:
#   grpo_prompts.jsonl        (main):   prompt_messages, oracle_patch, oracle_files, ...
#   grpo_prompts_files.jsonl  (sidecar): instance_id, repo_files (buggy/pre),
#                                        oracle_new_files (fixed/post)
# ``score_record`` consumes the joined dict; these helpers produce it.

def join_main_and_files(main_record: dict, files_record: dict) -> dict:
    """Merge a sidecar files row into its main record (same ``instance_id``)."""
    mid, fid = main_record.get("instance_id"), files_record.get("instance_id")
    if mid != fid:
        raise ValueError(f"instance_id mismatch in join: {mid!r} != {fid!r}")
    merged = dict(main_record)
    merged["repo_files"] = files_record.get("repo_files")
    merged["oracle_new_files"] = files_record.get("oracle_new_files")
    return merged


def iter_joined(main_path: str, files_path: str) -> "Iterable[dict]":
    """Stream joined records from the main + sidecar jsonl files.

    The sidecar is indexed by ``instance_id`` in memory (it holds file contents,
    so it is the larger file, but one dict of references is affordable); the main
    file is streamed. Yields dicts ready for ``score_record``. Main rows without a
    sidecar match are skipped.
    """
    import json

    sidecar: dict[str, dict] = {}
    with open(files_path) as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                sidecar[rec["instance_id"]] = rec
    with open(main_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            main_record = json.loads(line)
            files_record = sidecar.get(main_record.get("instance_id"))
            if files_record is not None:
                yield join_main_and_files(main_record, files_record)


def compute_rewards(
    completions: Iterable[str],
    code_context: dict[str, str],
    oracle_new_content: dict[str, str],
    allow_diff_fallback: bool = True,
    normalize: bool = True,
) -> list[RewardResult]:
    """Batch API for a GRPO group: shared instance (``code_context`` + oracle),
    many sampled completions. Returns one ``RewardResult`` per completion.

    The oracle diff is identical for every completion in the group, so it is
    computed once here and threaded into each ``compute_reward`` call (instead of
    being rebuilt per completion). Faithful: the reward value is unchanged.
    """
    oracle_patch = get_normalized_patch(code_context, oracle_new_content)
    return [
        compute_reward(
            c, code_context, oracle_new_content,
            allow_diff_fallback, normalize, _oracle_patch=oracle_patch,
        )
        for c in completions
    ]
