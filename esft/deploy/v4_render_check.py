#!/usr/bin/env python3
"""G2-③: v4 全量を trainer と同一 tokenizer/chat template でレンダリング検証。

各行に apply_chat_template を適用し、例外ゼロ + トークン長分布を出す。
訓練炉と同居するため nice 19 / 8 workers 前提 (呼び出し側で nice)。
"""
import json, os, collections
from multiprocessing import Pool

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# Paths are host-specific; override via env. Defaults document the layout used
# for the recorded run (base is the pinned Qwen3.6-35B-A3B snapshot revision).
MODEL = os.environ.get('BASE_MODEL', 'Qwen/Qwen3.6-35B-A3B')
V4 = os.environ.get('V4_DATA', 'data/v4_20260711.jsonl')
OUT = os.environ.get('V4_RENDER_OUT', 'data/v4_render_check.json')
SEQ = 7168

tok = None

def init():
    global tok
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)

def check(args):
    i, l = args
    try:
        row = json.loads(l)
        ids = tok.apply_chat_template(row['messages'], tokenize=True)
        if hasattr(ids, 'keys'):  # dict / BatchEncoding (UserDict 派生で isinstance(dict) が False)
            ids = ids['input_ids']
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return (len(ids), None)
    except Exception as e:
        return (0, f'line {i}: {type(e).__name__}: {str(e)[:200]}')

if __name__ == '__main__':
    with open(V4) as f:
        lines = [(i, l) for i, l in enumerate(f) if l.strip()]
    errors = []
    lens = []
    with Pool(8, initializer=init) as pool:
        for n, (ln, err) in enumerate(pool.imap_unordered(check, lines, chunksize=256)):
            if err: errors.append(err)
            else: lens.append(ln)
            if n % 50000 == 0:
                print(f'[render] {n}/{len(lines)} errors={len(errors)}', flush=True)
    lens.sort()
    stats = {
        'rows': len(lines), 'ok': len(lens), 'errors': len(errors),
        'error_samples': errors[:20],
        'tokens_p50': lens[len(lens)//2] if lens else 0,
        'tokens_p90': lens[len(lens)*9//10] if lens else 0,
        'tokens_p99': lens[len(lens)*99//100] if lens else 0,
        'tokens_max': lens[-1] if lens else 0,
        'over_seq_7168': sum(1 for x in lens if x > SEQ),
        'total_tokens': sum(lens),
    }
    with open(OUT, 'w') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats)[:800], flush=True)
    print('V4_RENDER_CHECK_DONE', flush=True)
