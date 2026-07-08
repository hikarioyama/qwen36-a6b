#!/usr/bin/env python
"""Unified evaluation harness for the ESFT G2/G3 comparison chain.

One harness measures every subject model in the ESFT chain on a pluggable set of
benchmarks, reusing the machinery proven by ``topk_sweep.py`` (2-GPU data
parallelism, top-k override, greedy GSM8K scoring, atomic per-result flush).

Subject models (``--model``)
----------------------------
- ``base``    : the plain Qwen3.6-35B-A3B (BF16). ``--topk`` overrides
                ``gate.top_k`` on every routed-MoE block, byte-identically to
                ``topk_sweep.py`` (softmax over all 256 experts -> topk(K) ->
                unconditional renorm). Default model path = the local 35B.
- ``patched`` : base + an ESFT-trained expert patch, applied in-place by
                ``esft_qwen.esft_patch.load_expert_patch``. Run it at the top-k it
                was TRAINED with (K*, recorded as ``_meta.top_k`` in the expert
                config, = 8 by default). Pass ``--patch <patch.safetensors>``.
- ``nvfp4``   : the trained experts baked to NVFP4 as a standalone model dir (G3
                precision gate). Only the load path is wired here -- pass
                ``--nvfp4-model-path <dir>``; producing the baked model is out of
                scope. Run at K* as well.
- ``dense``   : the 27B dense model. Not a MoE, so ``--topk`` is ignored (there is
                no ``gate.top_k`` to override). Default path = the local 27B.

Benchmarks (``--benchmark``, comma-separated; pluggable)
--------------------------------------------------------
- ``gsm8k`` : GSM8K test, 0-shot chat + "Let's think step by step.", greedy,
              answer = number after the last ``####`` else last number in the
              post-``</think>`` segment, exact numeric match (identical protocol
              to ``topk_sweep.py``). ``--n 600`` supported, deterministic first-N.
- ``mmlu``  : ``cais/mmlu`` all/test, 4-way multiple choice, greedy, answer = the
              A-D letter parsed from the post-``</think>`` segment. Used for the
              ESFT "general non-regression, within -1pt" side check.
- ``humaneval`` : OpenAI HumanEval (``openai/openai_humaneval``), greedy pass@1.
              The ```python block after the model's ``</think>`` is executed against
              the task's ``check`` asserts inside a locked-down throwaway subprocess.
- ``mbpp``      : MBPP (``google-research-datasets/mbpp`` test), greedy pass@1; the
              extracted function is run against the row's ``test_list`` asserts in the
              same sandbox (the prompt shows the tests, pinning the function name).
- ``jmmlu`` : ``nlp-waseda/JMMLU`` (Japanese MMLU), 4-way MC. Same protocol as
              ``mmlu`` -- inherits its prompt / letter-extraction / ``--choice-logprob``
              path unchanged, only the data source differs. The Japanese-side
              non-regression / k-expansion check.
- ``bfcl``  : registered stub; ``.load()`` raises ``NotImplementedError``.

A benchmark is a class implementing ``load / format_prompt / extract_answer /
score`` (see :class:`Benchmark`). Raw items produced by ``load`` are picklable
dicts (loaded in the parent); prompt rendering happens per-GPU worker.

Output
------
Per (model, topk, benchmark): a record
``{"model","topk","benchmark","n","acc","ci95","correct","truncated_n","tok_s",
"tok_s_parallel",...}`` (CI95 = normal approximation, to match ``topk_sweep.py``)
atomically flushed to ``~/esft/reports/eval/{tag}.json`` after each benchmark
completes on both GPUs. Every scored item is also dumped to
``reports/eval/{tag}_items.json`` as ``{"id","pred","gold","correct","truncated",
"gen_len"}`` so two runs over the same items can be compared PAIRED:

    eval_harness.py --paired-verdict A_items.json B_items.json

(exact McNemar on the discordant pairs; see :func:`paired_verdict`). That paired
test is the G2/G3 decision rule -- "two independent CIs don't overlap" is NOT
(it over-requires ~+5pt at n=600). ``truncated`` marks items whose generation
hit ``max_new`` before ``</think>`` closed (their parsed answer is unreliable).

Run (real eval needs both GPUs free; the orchestrator launches these):
    ~/esft-work/venv/bin/python ~/esft/eval_harness.py \
        --model base --benchmark gsm8k --topk 8 --n 600 --tag base_k8

CPU checks only (no GPU): ``import eval_harness`` + ``tests/test_eval_harness.py``.
"""
from __future__ import annotations

import os
import sys
import re
import json
import math
import time
import argparse
import multiprocessing as mp

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_35B = os.path.expanduser("~/esft-work/models/Qwen3.6-35B-A3B")
MODEL_27B = os.path.expanduser("~/esft-work/models/Qwen3.6-27B")
REPORT_DIR = os.path.expanduser("~/esft/reports/eval")
EOS_IDS = [248046, 248044]        # <|im_end|>, <|endoftext|>
TOP_K_DEFAULT = 8                 # Qwen3.6-35B-A3B ships top-8; also the ESFT K*.


# ============================ benchmark interface ============================

class Benchmark:
    """Pluggable benchmark contract.

    Subclasses implement four methods so a new benchmark = one class, no changes
    to the harness. ``load`` runs in the parent process and must return *picklable*
    items (plain dicts/tuples) because they are shipped to the GPU workers;
    ``format_prompt`` runs per-worker (it needs the tokenizer).
    """

    name = "base"
    default_max_new = 1536
    enable_thinking = True   # False => render prompts in no-think mode
    supports_logprob = False  # True => a choice-logprob path exists (see --choice-logprob)

    def _ct_kwargs(self):
        """Extra apply_chat_template kwargs. Adds enable_thinking=False only in
        no-think mode, so think-mode prompts stay byte-identical to before."""
        return {} if self.enable_thinking else {"enable_thinking": False}

    def load(self, n: int, seed: int, shuffle: bool) -> list:
        """Return up to ``n`` raw, picklable items (deterministic first-N unless
        ``shuffle``)."""
        raise NotImplementedError

    def format_prompt(self, item: dict, tok) -> str:
        """Render one item to a generation-ready prompt string (chat template)."""
        raise NotImplementedError

    def extract_answer(self, text: str):
        """Parse the model's decoded completion into a comparable prediction."""
        raise NotImplementedError

    def score(self, pred, item: dict) -> bool:
        """True iff ``pred`` matches the item's gold answer."""
        raise NotImplementedError


# ------------------------------- GSM8K --------------------------------------

def _last_number(text):
    """Number after the last ``####`` in the post-</think> segment, else the last
    number there. Identical to topk_sweep._last_number."""
    seg = text.split("</think>")[-1] if "</think>" in text else text
    m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", seg)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", seg)
    return nums[-1].replace(",", "") if nums else None


def _gold_gsm8k(answer):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer)
    return m.group(1).replace(",", "") if m else None


def _num_eq(a, b):
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-4
    except ValueError:
        return a.strip() == b.strip()


class Gsm8k(Benchmark):
    name = "gsm8k"
    default_max_new = 1536

    def load(self, n, seed, shuffle):
        from datasets import load_dataset
        d = load_dataset("openai/gsm8k", "main", split="test")
        if shuffle:
            d = d.shuffle(seed=seed)
        d = d.select(range(min(n, len(d))))
        return [{"question": r["question"], "gold": _gold_gsm8k(r["answer"])} for r in d]

    def format_prompt(self, item, tok):
        return tok.apply_chat_template(
            [{"role": "user",
              "content": item["question"] + "\nLet's think step by step."}],
            add_generation_prompt=True, tokenize=False, **self._ct_kwargs())

    def extract_answer(self, text):
        return _last_number(text)

    def score(self, pred, item):
        return _num_eq(pred, item["gold"])


# -------------------------------- MMLU --------------------------------------

_LETTERS = ["A", "B", "C", "D"]

# Keyword-anchored letter. The keyword is case-insensitive ((?i:...)) but the
# captured letter must be a STANDALONE UPPERCASE A-D: guarded by non-alnum on both
# sides so a lowercase a-d inside a word ("answer is clearly B" must not yield the
# 'c' of "clearly") or a letter glued to other characters never counts. The gap
# stays case-sensitive (blocks only uppercase A-D and newlines) so ordinary words
# like "is"/"clearly" may sit between the keyword and the letter.
_MMLU_KEYWORD_RE = re.compile(
    r"(?i:correct answer|correct option|answer|option)\b"
    r"[^A-D\n]{0,30}?"
    r"(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])")
# Fallback: last standalone uppercase A-D, same delimiter rule as the keyword form
# (the two patterns must never disagree on what counts as a letter).
_MMLU_BARE_RE = re.compile(r"(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])")


class Mmlu(Benchmark):
    """cais/mmlu all/test, 4-way MC. General non-regression side check for ESFT."""

    name = "mmlu"
    default_max_new = 1536

    def load(self, n, seed, shuffle):
        from datasets import load_dataset
        d = load_dataset("cais/mmlu", "all", split="test")
        # Deterministic first-N by default; shuffle only with an explicit seed so
        # the subset is reproducible either way (matches topk_sweep semantics).
        if shuffle:
            d = d.shuffle(seed=seed)
        d = d.select(range(min(n, len(d))))
        items = []
        for r in d:
            items.append({
                "question": r["question"],
                "choices": list(r["choices"]),
                "subject": r["subject"],
                "gold": _LETTERS[int(r["answer"])],
            })
        return items

    def format_prompt(self, item, tok):
        opts = "\n".join(f"{L}. {c}" for L, c in zip(_LETTERS, item["choices"]))
        content = (f"{item['question']}\n\n{opts}\n\n"
                   "Answer with the letter (A, B, C, or D) of the correct option.")
        return tok.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False, **self._ct_kwargs())

    def extract_answer(self, text):
        seg = text.split("</think>")[-1] if "</think>" in text else text
        # Prefer an explicit "answer/option is (X)"; fall back to the last
        # standalone uppercase A-D.
        m = _MMLU_KEYWORD_RE.findall(seg)
        if m:
            return m[-1]
        m = _MMLU_BARE_RE.findall(seg)
        return m[-1] if m else None

    def score(self, pred, item):
        return pred is not None and pred.upper() == item["gold"]

    # ---- choice-logprob path (no generation; forward-once, argmax over A-D) ----
    #
    # Reasoning models keep explaining even with enable_thinking=False, so a
    # generative MMLU truncates before emitting a letter. Instead render the
    # no-think prompt, append a cue whose next token is the answer letter, run one
    # forward pass, and pick argmax over the four letter token ids. Same-condition
    # across arms; O(1) forward => minutes/arm for 600 items.
    supports_logprob = True
    LOGPROB_CUE = "The best answer is ("

    def format_prompt_logprob(self, item, tok):
        opts = "\n".join(f"{L}. {c}" for L, c in zip(_LETTERS, item["choices"]))
        content = (f"{item['question']}\n\n{opts}\n\n"
                   "Answer with the letter (A, B, C, or D) of the correct option.")
        base = tok.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False)
        return base + self.LOGPROB_CUE

    def logprob_choice_ids(self, tok, sample_prompt):
        """Token id emitted for each letter right after the cue, computed from an
        actual rendered prompt so the cue/letter tokenization boundary is exact."""
        base = tok.encode(sample_prompt, add_special_tokens=False)
        ids = []
        for L in _LETTERS:
            full = tok.encode(sample_prompt + L, add_special_tokens=False)
            ids.append(full[len(base)] if len(full) > len(base) else full[-1])
        return ids


# -------------------------------- JMMLU -------------------------------------

class Jmmlu(Mmlu):
    """JMMLU (``nlp-waseda/JMMLU``): the Japanese-translated + Japanese-native
    counterpart to MMLU. 7,536 four-way MC questions over 56 subjects, same
    ``question, A, B, C, D, answer`` shape as MMLU (``answer`` is the gold LETTER).

    Only the data source differs from :class:`Mmlu`; ``format_prompt`` /
    ``extract_answer`` / ``score`` and the whole ``--choice-logprob`` path
    (``format_prompt_logprob`` / ``logprob_choice_ids`` / ``LOGPROB_CUE``) are
    inherited byte-for-byte, so a JMMLU arm is measured identically to the English
    MMLU side check -- A/B/C/D labels, the Japanese question inserted verbatim.

    The canonical repo ships a datasets *loading script* (unsupported on current
    ``datasets``), so rows are read straight from its ``JMMLU.zip`` -- BOM-encoded
    per-subject CSVs -- through the packaged ``csv`` builder. That yields the same
    HF ``Dataset`` type ``Mmlu.load`` gets from ``cais/mmlu``, hence identical
    ``.shuffle(seed)`` and deterministic first-N ``.select`` semantics.
    """

    name = "jmmlu"

    def load(self, n, seed, shuffle):
        import zipfile
        from huggingface_hub import hf_hub_download
        from datasets import load_dataset

        zp = hf_hub_download("nlp-waseda/JMMLU", "JMMLU.zip", repo_type="dataset")
        extract_dir = zp + ".extracted"
        with zipfile.ZipFile(zp) as z:
            # test-split subject CSVs only; drop the __MACOSX/ resource forks and
            # the ._-prefixed AppleDouble sidecars the archive also carries.
            members = [m for m in z.namelist()
                       if m.startswith("JMMLU/test/") and m.endswith(".csv")
                       and not os.path.basename(m).startswith("._")]
            todo = [m for m in members
                    if not os.path.exists(os.path.join(extract_dir, m))]
            if todo:
                z.extractall(extract_dir, members=todo)
        # sorted() pins the subject concatenation order, so the deterministic
        # first-N subset (and any --seed shuffle) is reproducible run-to-run,
        # exactly like cais/mmlu's fixed row order.
        paths = sorted(os.path.join(extract_dir, m) for m in members)
        d = load_dataset("csv", data_files=paths, split="train",
                         encoding="utf-8-sig")   # utf-8-sig: strip the CSV BOM
        if shuffle:
            d = d.shuffle(seed=seed)
        d = d.select(range(min(n, len(d))))
        items = []
        for r in d:
            items.append({
                "question": r["question"],
                "choices": [r["A"], r["B"], r["C"], r["D"]],
                "gold": str(r["answer"]).strip().upper(),
            })
        return items


# ------------------------ coding (HumanEval / MBPP) -------------------------
#
# pass@1: greedy-decode a solution, take the ```python block that follows the
# reasoning model's ``</think>``, and run it against the task's own asserts.
# Untrusted model output is NEVER exec'd in-process -- every candidate runs as a
# throwaway subprocess (fresh interpreter, isolated tmp cwd, its own session so
# the whole process group can be killed, CPU + address-space rlimits, stdin
# closed) under a hard wall-clock timeout, so an infinite loop or a runaway
# allocation can neither outlive nor OOM the eval. This mirrors the standard
# HumanEval execution harness, hardened with real process isolation in place of
# in-process ``exec``.

_CODE_BLOCK_RE = re.compile(
    r"```[ \t]*(?:python|py|python3)?[ \t]*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE)
_DEF_RE = re.compile(r"(?m)^[ \t]*(?:async[ \t]+)?def[ \t]+\w+")


def _extract_code_block(text):
    """Last ```python block after the model's ``</think>`` (preferring one that
    defines a function). Returns the code string, or ``None`` if there is none."""
    seg = text.split("</think>")[-1] if "</think>" in text else text
    blocks = [b for b in _CODE_BLOCK_RE.findall(seg) if b.strip()]
    if not blocks:
        return None
    with_def = [b for b in blocks if _DEF_RE.search(b)]
    return (with_def or blocks)[-1].strip("\n")


def _sandbox_header(timeout, mem_mb):
    """Preamble prepended to every executed program: cap the child's CPU time and
    address space from the inside (belt-and-suspenders with the parent's wall-clock
    kill), then neuter destructive filesystem/process calls before any candidate
    code runs. ``resource`` is blanked afterwards so the candidate cannot lift the
    caps."""
    mem_bytes = int(mem_mb) * 1024 * 1024 if mem_mb else 0
    cpu_s = int(timeout) + 1
    return (
        "import os as _os, sys as _sys, builtins as _bi\n"
        "try:\n"
        "    import resource as _r\n"
        f"    _MEM = {mem_bytes}\n"
        "    if _MEM:\n"
        "        _r.setrlimit(_r.RLIMIT_AS, (_MEM, _MEM))\n"
        f"    _r.setrlimit(_r.RLIMIT_CPU, ({cpu_s}, {cpu_s}))\n"
        "    _r.setrlimit(_r.RLIMIT_CORE, (0, 0))\n"
        "except Exception:\n"
        "    pass\n"
        "_os.environ['OMP_NUM_THREADS'] = '1'\n"
        "for _n in ('system','remove','unlink','rmdir','removedirs','rename',\n"
        "           'renames','replace','truncate','kill','killpg','fork',\n"
        "           'forkpty','chmod','chown','chroot','setuid','fchmod','fchown'):\n"
        "    if hasattr(_os, _n):\n"
        "        setattr(_os, _n, None)\n"
        "try:\n"
        "    import shutil as _sh\n"
        "    _sh.rmtree = _sh.move = _sh.chown = None\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    import subprocess as _sp\n"
        "    _sp.Popen = _sp.run = _sp.call = _sp.check_call = _sp.check_output = None\n"
        "except Exception:\n"
        "    pass\n"
        "_bi.exit = _bi.quit = None\n"
        "_sys.modules['resource'] = None\n"
        "del _os, _sys, _bi\n")


def _run_sandboxed(body, timeout=10.0, mem_mb=4096):
    """Run ``body`` (candidate code + asserts) as an isolated subprocess.

    Returns ``(passed, detail)`` with ``passed`` True iff the program exits 0 (all
    asserts held). On timeout the entire process group is SIGKILLed, so a hung or
    looping candidate -- and anything it spawned -- is guaranteed to die."""
    import subprocess
    import tempfile
    import signal

    program = _sandbox_header(timeout, mem_mb) + "\n" + body
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})

    with tempfile.TemporaryDirectory(prefix="esft_sbx_") as td:
        prog_path = os.path.join(td, "prog.py")
        with open(prog_path, "w") as f:
            f.write(program)
        try:
            proc = subprocess.Popen(
                [sys.executable, prog_path], cwd=td,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=env, text=True, start_new_session=True)
        except Exception as e:  # noqa: BLE001
            return False, f"spawn-failed: {e}"
        try:
            _out, err = proc.communicate(timeout=timeout)
            passed = (proc.returncode == 0)
            return passed, ("" if passed else (err or "")[-800:])
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            try:
                proc.communicate(timeout=5)
            except Exception:  # noqa: BLE001
                pass
            return False, f"timeout>{timeout}s"


class _CodingBenchmark(Benchmark):
    """Shared pass@1 machinery: pull the ```python block, execute it against the
    task's asserts in the locked-down subprocess. Subclasses supply ``load`` /
    ``format_prompt`` / ``_build_program``."""

    default_max_new = 2048          # room for <think> + a full function
    timeout = 10.0                  # wall-clock seconds per candidate
    mem_mb = 4096                   # address-space cap per candidate

    def extract_answer(self, text):
        return _extract_code_block(text)

    def _build_program(self, code: str, item: dict) -> str:
        raise NotImplementedError

    def score(self, pred, item):
        if not pred or not pred.strip():
            return False
        ok, _ = _run_sandboxed(self._build_program(pred, item),
                               timeout=self.timeout, mem_mb=self.mem_mb)
        return ok


class HumanEval(_CodingBenchmark):
    """OpenAI HumanEval (``openai/openai_humaneval``), greedy pass@1."""

    name = "humaneval"

    def load(self, n, seed, shuffle):
        from datasets import load_dataset
        d = load_dataset("openai/openai_humaneval", split="test")
        if shuffle:
            d = d.shuffle(seed=seed)
        d = d.select(range(min(n, len(d))))
        return [{"task_id": r["task_id"], "prompt": r["prompt"],
                 "test": r["test"], "entry_point": r["entry_point"]} for r in d]

    def format_prompt(self, item, tok):
        content = (
            "Complete the following Python function. Reason step by step, then give "
            "the COMPLETE function definition -- signature, any imports it needs, and "
            "body -- inside a single ```python code block.\n\n"
            f"```python\n{item['prompt'].rstrip()}\n```")
        return tok.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False, **self._ct_kwargs())

    def _build_program(self, code, item):
        # candidate defines entry_point; item["test"] defines check(candidate).
        return "\n".join([code, "", item["test"], "",
                          f"check({item['entry_point']})", ""])


class Mbpp(_CodingBenchmark):
    """MBPP (``google-research-datasets/mbpp`` test split), greedy pass@1."""

    name = "mbpp"

    def load(self, n, seed, shuffle):
        from datasets import load_dataset
        d = load_dataset("google-research-datasets/mbpp", split="test")
        if shuffle:
            d = d.shuffle(seed=seed)
        d = d.select(range(min(n, len(d))))
        items = []
        for r in d:
            items.append({
                "task_id": r["task_id"],
                "text": r.get("text") or r.get("prompt") or "",
                "test_list": list(r["test_list"]),
                "test_setup_code": r.get("test_setup_code") or "",
            })
        return items

    def format_prompt(self, item, tok):
        tests = "\n".join(item["test_list"])
        content = (
            "Write a Python function that solves the task below. Reason step by step, "
            "then give the COMPLETE function definition inside a single ```python code "
            "block. It must pass these tests, so use the exact function name and "
            "signature they imply:\n\n"
            f"Task: {item['text']}\n\nTests:\n{tests}")
        return tok.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False, **self._ct_kwargs())

    def _build_program(self, code, item):
        parts = [code, ""]
        setup = item.get("test_setup_code") or ""
        if setup.strip():
            parts += [setup, ""]
        parts += list(item["test_list"])
        parts.append("")
        return "\n".join(parts)


# ------------------------------- stubs --------------------------------------

class _Stub(Benchmark):
    """Registered-but-unimplemented benchmark: interface is fixed, body pending."""

    def load(self, n, seed, shuffle):
        raise NotImplementedError(
            f"benchmark {self.name!r} is a stub; implement load/format_prompt/"
            "extract_answer/score to enable it")

    def format_prompt(self, item, tok):
        raise NotImplementedError(self.name)

    def extract_answer(self, text):
        raise NotImplementedError(self.name)

    def score(self, pred, item):
        raise NotImplementedError(self.name)


class Bfcl(_Stub):
    name = "bfcl"


BENCHMARKS = {
    "gsm8k": Gsm8k,
    "mmlu": Mmlu,
    "humaneval": HumanEval,
    "mbpp": Mbpp,
    "jmmlu": Jmmlu,
    "bfcl": Bfcl,
}


def get_benchmark(name: str) -> Benchmark:
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; known: {sorted(BENCHMARKS)}")
    return BENCHMARKS[name]()


# ============================ subject-model spec ============================

def resolve_model_spec(kind: str, *, model_path=None, patch=None,
                       nvfp4_model_path=None, topk=TOP_K_DEFAULT) -> dict:
    """Resolve ``--model`` into a picklable spec the worker consumes.

    Raises ``ValueError`` on missing required paths (the CLI turns that into a
    clean exit). ``topk`` is the routed-MoE ``gate.top_k`` override; it is forced
    to ``None`` for ``dense`` (the 27B has no MoE block to override).
    """
    if kind == "base":
        return {"kind": "base", "model_path": model_path or MODEL_35B,
                "patch": None, "topk": topk}
    if kind == "patched":
        if not patch:
            raise ValueError("--patch <patch.safetensors> is required for --model patched")
        return {"kind": "patched", "model_path": model_path or MODEL_35B,
                "patch": patch, "topk": topk}
    if kind == "nvfp4":
        if not nvfp4_model_path:
            raise ValueError("--nvfp4-model-path <dir> is required for --model nvfp4")
        return {"kind": "nvfp4", "model_path": nvfp4_model_path,
                "patch": None, "topk": topk}
    if kind == "dense":
        return {"kind": "dense", "model_path": model_path or MODEL_27B,
                "patch": None, "topk": None}
    raise ValueError(f"unknown --model {kind!r}; choose base|patched|nvfp4|dense")


def load_subject_model(spec: dict, gpu_id: int):
    """Load + configure a subject model on ``gpu_id``. Returns (tok, model, is_moe).

    Applies the top-k override on every routed-MoE block (skipped for a dense model
    that has none), then, for ``patched``, writes the ESFT expert patch in place via
    ``esft_qwen.esft_patch.load_expert_patch``. This is exactly the G2/G3 load path
    the orchestrator will exercise on real weights.
    """
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM)
    sys.path.insert(0, PROJECT_ROOT)
    from esft_qwen.common import find_moe_blocks
    from esft_qwen.esft_patch import load_expert_patch

    tok = AutoTokenizer.from_pretrained(spec["model_path"])
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            spec["model_path"], dtype=torch.bfloat16, device_map={"": gpu_id})
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            spec["model_path"], dtype=torch.bfloat16, device_map={"": gpu_id})
    model.eval()

    try:
        refs = find_moe_blocks(model)
        is_moe = True
    except ValueError:
        refs, is_moe = [], False   # dense model: nothing to override.

    if is_moe and spec.get("topk") is not None:
        k = int(spec["topk"])
        for r in refs:
            r.gate.top_k = k
        assert all(int(r.gate.top_k) == k for r in refs)

    if spec["kind"] == "patched":
        info = load_expert_patch(model, spec["patch"])
        print(f"[gpu{gpu_id}] loaded ESFT patch: {info['num_written']} expert "
              f"slices from {spec['patch']}", flush=True)

    print(f"[gpu{gpu_id}] {spec['kind']} loaded  moe={is_moe}  "
          f"moe_layers={len(refs)}  topk={spec.get('topk')}", flush=True)
    return tok, model, is_moe


# ============================ generation / scoring ==========================

def _count_gen_tokens(row_ids):
    """Number of generated tokens up to and including the first EOS."""
    hit = [row_ids.index(e) for e in EOS_IDS if e in row_ids]
    return (min(hit) + 1) if hit else len(row_ids)


def run_one_benchmark(gpu_id, tok, model, bench, items, batch_size, max_new):
    import torch
    dev = f"cuda:{gpu_id}"
    rendered = [bench.format_prompt(it, tok) for it in items]

    correct = 0
    gen_tokens = 0
    truncated = 0
    n = 0
    per_item = []
    t0 = time.time()
    for i in range(0, len(rendered), batch_size):
        chunk = rendered[i:i + batch_size]
        ichunk = items[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(dev)
        in_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new, do_sample=False,
                eos_token_id=EOS_IDS, pad_token_id=tok.pad_token_id)
        gen = out[:, in_len:]
        for row, item in zip(gen, ichunk):
            gt = row.tolist()
            n_gen = _count_gen_tokens(gt)
            gen_tokens += n_gen
            text = tok.decode(row, skip_special_tokens=True)
            pred = bench.extract_answer(text)
            ok = bool(bench.score(pred, item))
            # Generation hit the cap before the reasoning block closed: the
            # extractor then reads raw <think> text, so this item's answer is
            # unreliable and must be visible in the paired comparison.
            trunc = (n_gen >= max_new) and ("</think>" not in text)
            correct += int(ok)
            truncated += int(trunc)
            n += 1
            rec = {"id": item.get("_id"), "pred": pred,
                   "gold": item.get("gold"), "correct": ok,
                   "truncated": trunc, "gen_len": n_gen}
            if "task_id" in item:
                rec["task_id"] = item["task_id"]
            per_item.append(rec)
        print(f"[gpu{gpu_id} {bench.name}] {n}/{len(rendered)} "
              f"acc={correct/max(n,1):.3f} tok={gen_tokens} trunc={truncated}",
              flush=True)
    dt = time.time() - t0
    return {"gpu": gpu_id, "benchmark": bench.name, "correct": correct,
            "n": n, "gen_tokens": gen_tokens, "gen_time": dt,
            "truncated_n": truncated, "items": per_item}


def run_logprob_benchmark(gpu_id, tok, model, bench, items, batch_size):
    """Forward-once choice-logprob scoring (no generation). Predicts the answer
    letter as argmax over the four choice token ids at the last (left-padded)
    position. Returns the same message shape as ``run_one_benchmark`` with
    ``gen_tokens=0`` / ``truncated_n=0`` so aggregation is unchanged."""
    import torch
    dev = f"cuda:{gpu_id}"
    prompts = [bench.format_prompt_logprob(it, tok) for it in items]
    cand_ids = bench.logprob_choice_ids(tok, prompts[0]) if prompts else []
    assert len(set(cand_ids)) == len(cand_ids), \
        f"choice token ids not distinct: {cand_ids}"
    cand = torch.tensor(cand_ids, device=dev)

    correct = 0; n = 0; per_item = []
    t0 = time.time()
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        ichunk = items[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(dev)
        with torch.no_grad():
            # NEVER materialize (B, T, V) full logits: bs16 x ~4k pos x 248320
            # vocab fp32 is ~64 GiB (OOM'd 2026-07-06; the logits_to_keep kwarg
            # raises TypeError on this VL wrapper and the old fallback silently
            # took the full-logits path). Run the base model for hidden states
            # only, then apply lm_head to the last position: (B,H) -> (B,V).
            base = getattr(model, "model", model)
            head = model.get_output_embeddings()
            out = base(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"])
            h_last = out.last_hidden_state[:, -1, :]        # left-pad => real last
            last = head(h_last).float()                     # (B, V)
        picks = last[:, cand].argmax(dim=-1).tolist()
        for j, item in enumerate(ichunk):
            pred = _LETTERS[picks[j]]
            ok = (pred == item.get("gold"))
            correct += int(ok); n += 1
            per_item.append({"id": item.get("_id"), "pred": pred,
                             "gold": item.get("gold"), "correct": ok,
                             "truncated": False, "gen_len": 0})
        print(f"[gpu{gpu_id} {bench.name}(logprob)] {n}/{len(prompts)} "
              f"acc={correct/max(n,1):.3f}", flush=True)
    dt = time.time() - t0
    return {"gpu": gpu_id, "benchmark": bench.name, "correct": correct,
            "n": n, "gen_tokens": 0, "gen_time": dt,
            "truncated_n": 0, "items": per_item}


def worker(gpu_id, spec, bench_items, batch_size, max_new_map, no_think,
           choice_logprob, q):
    import torch
    torch.cuda.set_device(gpu_id)
    tok, model, _ = load_subject_model(spec, gpu_id)
    for bench_name, items in bench_items:
        bench = get_benchmark(bench_name)
        bench.enable_thinking = not no_think
        if choice_logprob and getattr(bench, "supports_logprob", False):
            msg = run_logprob_benchmark(gpu_id, tok, model, bench, items, batch_size)
        else:
            max_new = max_new_map.get(bench_name) or bench.default_max_new
            msg = run_one_benchmark(gpu_id, tok, model, bench, items, batch_size, max_new)
        q.put(msg)
        # Free this benchmark's cached activations before the next one so a heavy
        # forward (e.g. mmlu logprob) can't fragment memory out from under the
        # generative benchmark that follows it in the same process.
        torch.cuda.empty_cache()
    q.put({"gpu": gpu_id, "done": True})


# ============================ aggregation / flush ===========================

def ci95_normal(acc, n):
    """95% CI half-width, normal approximation (matches topk_sweep's style)."""
    if not n:
        return None
    return round(1.96 * math.sqrt(max(acc * (1 - acc), 0.0) / n), 4)


# --------------------------- paired significance -----------------------------
#
# base and patched are evaluated on the SAME deterministic item set, so the G2/G3
# decision is a PAIRED comparison. Requiring two independent CIs to not overlap
# demands ~+5pt at n=600 and would NO-GO a real +2-3pt ESFT gain; the decision
# rule is McNemar on the discordant pairs instead (independent CIs may still be
# reported for context, never for the verdict).

def mcnemar_exact_p(n10, n01):
    """Exact two-sided McNemar p-value.

    ``n10`` = items A got right and B got wrong; ``n01`` = the reverse. Under H0
    the discordant pairs are Binomial(n10+n01, 1/2); p = doubled small tail
    (capped at 1). Exact, so it stays valid for the small discordant counts a
    600-item eval produces."""
    nd = n10 + n01
    if nd == 0:
        return 1.0
    k = min(n10, n01)
    tail = sum(math.comb(nd, i) for i in range(k + 1)) / 2.0 ** nd
    return min(1.0, 2.0 * tail)


def paired_verdict(items_a, items_b, key="correct", alpha=0.05):
    """Paired A-vs-B verdict over per-item records evaluated on the same items.

    ``items_*``: lists of dicts carrying ``id`` and a boolean ``key`` field, as
    written to ``reports/eval/{tag}_items.json`` (records where ``key`` is absent
    or None are ignored, e.g. toolcall negatives when key="arg_match"; the same
    records drop from both sides because the item sets are identical).

    Returns ``{"n","key","acc_a","acc_b","delta","n10","n01","paired_ci95",
    "mcnemar_p","significant"}`` where ``delta`` = acc_b - acc_a, ``paired_ci95``
    is the Wald CI on delta from the discordant counts, and ``significant`` is
    the decision (exact McNemar p < alpha).
    """
    amap = {it["id"]: bool(it[key]) for it in items_a if it.get(key) is not None}
    bmap = {it["id"]: bool(it[key]) for it in items_b if it.get(key) is not None}
    if set(amap) != set(bmap):
        raise ValueError(
            f"paired_verdict: item id sets differ (A={len(amap)}, B={len(bmap)}, "
            f"common={len(set(amap) & set(bmap))}); both models must be run on "
            "the same items (same benchmark, --n, --seed, --shuffle)")
    n = len(amap)
    if n == 0:
        raise ValueError(f"paired_verdict: no items carry the field {key!r}")
    n10 = sum(1 for i in amap if amap[i] and not bmap[i])
    n01 = sum(1 for i in amap if not amap[i] and bmap[i])
    acc_a = sum(amap.values()) / n
    acc_b = sum(bmap.values()) / n
    delta = acc_b - acc_a
    se = math.sqrt(max(n10 + n01 - (n01 - n10) ** 2 / n, 0.0)) / n
    p = mcnemar_exact_p(n10, n01)
    return {"n": n, "key": key,
            "acc_a": round(acc_a, 4), "acc_b": round(acc_b, 4),
            "delta": round(delta, 4), "n10": n10, "n01": n01,
            "paired_ci95": [round(delta - 1.96 * se, 4),
                            round(delta + 1.96 * se, 4)],
            "mcnemar_p": p, "significant": p < alpha}


def _load_items_file(path):
    """Read a ``*_items.json``: {"items": {bench: [...]}} (this harness) or
    {"items": [...]} / a bare list (eval_toolcall) -> {bench_name: [records]}."""
    with open(path) as f:
        data = json.load(f)
    items = data.get("items", data) if isinstance(data, dict) else data
    if isinstance(items, list):
        return {"all": items}
    return items


def run_paired_verdict(path_a, path_b, key="correct"):
    a, b = _load_items_file(path_a), _load_items_file(path_b)
    common = [name for name in a if name in b]
    if not common:
        raise SystemExit(f"no common benchmark between {path_a} and {path_b}")
    for name in common:
        v = paired_verdict(a[name], b[name], key=key)
        print(f"[{name}] n={v['n']}  acc A={v['acc_a']:.4f} B={v['acc_b']:.4f}  "
              f"delta={v['delta']:+.4f}  paired_CI95=[{v['paired_ci95'][0]:+.4f},"
              f"{v['paired_ci95'][1]:+.4f}]  A-only={v['n10']} B-only={v['n01']}  "
              f"McNemar p={v['mcnemar_p']:.4g}  significant={v['significant']}")


def aggregate(partial, spec, max_new_map):
    """partial: {bench_name: {gpu_id: msg}} -> {bench_name: record} for benchmarks
    with BOTH gpus reported."""
    out = {}
    for bench_name in sorted(partial):
        parts = partial[bench_name]
        if len(parts) < 2:
            continue
        c = sum(p["correct"] for p in parts.values())
        n = sum(p["n"] for p in parts.values())
        g = sum(p["gen_tokens"] for p in parts.values())
        tsum = sum(p["gen_time"] for p in parts.values())
        tmax = max(p["gen_time"] for p in parts.values())
        acc = (c / n) if n else None
        out[bench_name] = {
            "model": spec["kind"],
            "topk": spec.get("topk"),
            "benchmark": bench_name,
            "n": n,
            "acc": round(acc, 4) if acc is not None else None,
            "ci95": ci95_normal(acc, n) if acc is not None else None,
            "correct": c,
            "truncated_n": sum(p.get("truncated_n", 0) for p in parts.values()),
            "gen_tokens": g,
            "tok_s": round(g / tsum, 2) if tsum else None,            # per-GPU-equiv
            "tok_s_parallel": round(g / tmax, 2) if tmax else None,   # 2-GPU wall
            "gen_time_sum_s": round(tsum, 1),
            "max_new": max_new_map.get(bench_name),
        }
    return out


def _atomic_json_dump(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def items_path_for(out_path):
    """reports/eval/{tag}.json -> reports/eval/{tag}_items.json"""
    base = out_path[:-len(".json")] if out_path.endswith(".json") else out_path
    return base + "_items.json"


def flush(out_path, partial, meta, spec, max_new_map):
    data = {"_meta": meta, "results": aggregate(partial, spec, max_new_map)}
    _atomic_json_dump(data, out_path)
    # Per-item dump (paired-analysis input): every scored item reported so far,
    # both GPUs merged, sorted by id. Written even for half-finished benchmarks
    # so a mid-run crash keeps the raw data.
    items = {}
    for bench_name, parts in partial.items():
        merged = [it for p in parts.values() for it in p.get("items", [])]
        merged.sort(key=lambda it: (it["id"] is None, it["id"]))
        if merged:
            items[bench_name] = merged
    _atomic_json_dump({"_meta": meta, "items": items}, items_path_for(out_path))


def print_table(partial, spec, max_new_map):
    agg = aggregate(partial, spec, max_new_map)
    print("\n" + "=" * 68)
    print(f"model={spec['kind']}  topk={spec.get('topk')}")
    print(f"{'benchmark':>12} | {'acc':>7} | {'ci95':>7} | {'n':>5} | {'tok/s':>8}")
    print("-" * 68)
    for name in sorted(agg):
        r = agg[name]
        print(f"{name:>12} | {r['acc']:>7} | {r['ci95']:>7} | {r['n']:>5} | "
              f"{r['tok_s']:>8}")
    print("=" * 68)


# ================================== main ====================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",
                    choices=["base", "patched", "nvfp4", "dense"],
                    help="subject model (required unless --paired-verdict)")
    ap.add_argument("--paired-verdict", nargs=2, metavar=("A_ITEMS", "B_ITEMS"),
                    help="CPU only: McNemar paired verdict between two "
                         "*_items.json files, then exit (no model, no GPU)")
    ap.add_argument("--verdict-key", default="correct",
                    help="per-item boolean field --paired-verdict compares on")
    ap.add_argument("--benchmark", default="gsm8k",
                    help="comma-separated: gsm8k,mmlu,jmmlu,humaneval,mbpp (stub: bfcl)")
    ap.add_argument("--n", type=int, default=600, help="items per benchmark")
    ap.add_argument("--topk", type=int, default=TOP_K_DEFAULT,
                    help="gate.top_k override (ignored for dense). Patched/nvfp4: use K*")
    ap.add_argument("--patch", default=None, help="ESFT expert patch (required for patched)")
    ap.add_argument("--nvfp4-model-path", default=None,
                    help="baked NVFP4 model dir (required for nvfp4)")
    ap.add_argument("--model-path", default=None,
                    help="override the default checkpoint path (base/patched/dense)")
    ap.add_argument("--tag", default=None, help="report basename -> reports/eval/{tag}.json")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=None,
                    help="cap new tokens for ALL benchmarks (default: per-benchmark)")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle each dataset with --seed (default: deterministic first-N)")
    ap.add_argument("--no-think", action="store_true",
                    help="render prompts with enable_thinking=False (empty reasoning "
                         "block, direct answer). Use for MMLU/GSM8K non-regression at "
                         "low --max-new; removes arm-dependent truncation bias.")
    ap.add_argument("--choice-logprob", action="store_true",
                    help="for benchmarks that support it (mmlu), score by forward-once "
                         "argmax over choice-letter logits instead of generating "
                         "(no truncation; fast). Other benchmarks still generate.")
    ap.add_argument("--report-dir", default=REPORT_DIR)
    args = ap.parse_args()

    if args.paired_verdict:
        run_paired_verdict(args.paired_verdict[0], args.paired_verdict[1],
                           key=args.verdict_key)
        return
    if not args.model:
        ap.error("--model is required (or use --paired-verdict)")

    try:
        spec = resolve_model_spec(
            args.model, model_path=args.model_path, patch=args.patch,
            nvfp4_model_path=args.nvfp4_model_path, topk=args.topk)
    except ValueError as e:
        ap.error(str(e))

    gpus = [int(x) for x in args.gpus.split(",")]
    assert len(gpus) == 2, "this harness is wired for exactly 2 GPUs"

    bench_names = [b.strip() for b in args.benchmark.split(",") if b.strip()]
    for b in bench_names:
        if b not in BENCHMARKS:
            ap.error(f"unknown benchmark {b!r}; known: {sorted(BENCHMARKS)}")

    tag = args.tag or f"{spec['kind']}_k{spec.get('topk')}_{'_'.join(bench_names)}"
    out_path = os.path.join(args.report_dir, f"{tag}.json")

    # Load every benchmark's items in the parent (picklable) and split even/odd.
    max_new_map = {b: args.max_new for b in bench_names}  # None => per-bench default
    items_g0, items_g1 = [], []
    for b in bench_names:
        bench = get_benchmark(b)
        items = bench.load(args.n, args.seed, args.shuffle)
        # Stable per-item id = position in the deterministic load order, attached
        # BEFORE the even/odd GPU split so two runs on the same (--n, --seed,
        # --shuffle) can be joined item-by-item for the paired verdict.
        for j, it in enumerate(items):
            it["_id"] = j
        items_g0.append((b, items[0::2]))
        items_g1.append((b, items[1::2]))
        print(f"loaded {len(items)} {b} items", flush=True)

    meta = {
        "model": spec["kind"], "model_path": spec["model_path"],
        "patch": spec.get("patch"), "topk": spec.get("topk"),
        "benchmarks": bench_names, "n_per_benchmark": args.n,
        "batch_size": args.batch_size, "max_new": args.max_new,
        "seed": args.seed, "shuffle": args.shuffle, "gpus": gpus,
        "no_think": args.no_think, "choice_logprob": args.choice_logprob,
        "split": "first-N" + (" shuffled" if args.shuffle else ""),
        "note": "reasoning model; per-benchmark max_new may truncate <think>",
    }

    q = mp.Queue()
    subsets = {gpus[0]: items_g0, gpus[1]: items_g1}
    procs = [mp.Process(target=worker,
                        args=(g, spec, subsets[g], args.batch_size, max_new_map,
                              args.no_think, args.choice_logprob, q))
             for g in gpus]
    for p in procs:
        p.start()

    partial = {}
    done = set()
    while len(done) < len(procs):
        msg = q.get()
        if msg.get("done"):
            done.add(msg["gpu"])
            continue
        partial.setdefault(msg["benchmark"], {})[msg["gpu"]] = msg
        flush(out_path, partial, meta, spec, max_new_map)
        agg = aggregate(partial, spec, max_new_map)
        if msg["benchmark"] in agg:
            r = agg[msg["benchmark"]]
            print(f">>> {msg['benchmark']} COMPLETE  acc={r['acc']}±{r['ci95']}  "
                  f"tok/s={r['tok_s']}  (flushed to {out_path})", flush=True)

    for p in procs:
        p.join()

    flush(out_path, partial, meta, spec, max_new_map)
    print_table(partial, spec, max_new_map)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
