#!/usr/bin/env python3
"""v3 の en 系 eval 転写除去 (run>=6 基準)。

背景 (DEVLOG 2026-07-11): v4 プリフライトの 5k 監査で code レーンに
HumanEval/BFCL 転写 (連続 run 11-18 のクラスタ) を検出。ja 系のヒットは
CJK 8 文字 gram の低エントロピーによる偽陽性と判定 (run<=11、実物は一般表現)。
→ 本パスは en 系 eval (jmmlu/mifeval 以外) に限り、連続 run>=6 の行だけ除去する。
標準 matcher (min_hits=2, run 不問) より緩いが、偽陽性の機構を除いた上での
転写検出としては保守的十分 (run 6 = 13 単語連続一致)。

出力: 除去行の sha256 リスト + 統計 + 除去行サンプル (目視用)。
"""
import json, os, sys, hashlib, collections
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))  # deploy copy lives one level down
import corpus_decontam_v1 as dc

V3 = os.environ.get('V3_INPUT', 'v3.jsonl')
OUT = os.environ.get('V3_EN_FILTER_OUT', 'v3_en_filter')
RUN_MIN = 6

print('[build] eval index...', flush=True)
IDX = dc.build_eval_index()
N = dc.NGRAM_SIZE
EN_MASK = 0
for bit, name in enumerate(dc.EVAL_NAMES):
    if name not in ('jmmlu', 'mifeval'):
        EN_MASK |= (1 << bit)
print(f'[build] done. en sets = {[n for n in dc.EVAL_NAMES if n not in ("jmmlu","mifeval")]}', flush=True)

def check(args):
    lineno, line = args
    row = json.loads(line)
    best = 0; best_sets = 0; best_win = ''
    for text in dc.walk_text(row):
        toks = dc.normalize_tokens(text)
        run = 0; run_sets = 0
        for i in range(len(toks) - N + 1):
            mask = IDX.masks.get(dc.gram_digest(toks, i), 0) & EN_MASK
            if mask:
                run += 1; run_sets |= mask
                if run > best:
                    best = run; best_sets = run_sets
                    best_win = ' '.join(toks[max(0, i - run + 1):i + N])[:200]
            else:
                run = 0; run_sets = 0
    if best >= RUN_MIN:
        sets = [dc.EVAL_NAMES[b] for b in range(len(dc.EVAL_NAMES)) if best_sets & (1 << b)]
        return (lineno, hashlib.sha256(line.strip().encode()).hexdigest(),
                row.get('_source', '?'), best, sets, best_win)
    return None

if __name__ == '__main__':
    import os
    os.makedirs(OUT, exist_ok=True)
    with open(V3) as f:
        lines = [(i, l) for i, l in enumerate(f) if l.strip()]
    print(f'[scan] {len(lines)} rows, workers=12', flush=True)
    removals = []
    with Pool(12) as pool:
        for n, res in enumerate(pool.imap_unordered(check, lines, chunksize=200)):
            if res: removals.append(res)
            if n % 40000 == 0:
                print(f'[scan] {n}/{len(lines)} flagged={len(removals)}', flush=True)
    removals.sort()
    by_src = collections.Counter(r[2] for r in removals)
    by_run = collections.Counter(min(r[3], 20) for r in removals)
    with open(f'{OUT}/removal_sha256.txt', 'w') as f:
        for r in removals: f.write(r[1] + '\n')
    with open(f'{OUT}/removals_detail.jsonl', 'w') as f:
        for r in removals:
            f.write(json.dumps({'lineno': r[0], 'sha256': r[1], 'source': r[2],
                                'run': r[3], 'sets': r[4], 'window': r[5]}, ensure_ascii=False) + '\n')
    stats = {'total_rows': len(lines), 'removed': len(removals), 'run_min': RUN_MIN,
             'by_source': dict(by_src), 'by_run_capped20': dict(sorted(by_run.items()))}
    with open(f'{OUT}/stats.json', 'w') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    print('V3_EN_FILTER_DONE', flush=True)
