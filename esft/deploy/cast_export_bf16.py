#!/usr/bin/env python3
"""HF export (fp32, ~128GB) を bf16 (~64GB) に鋳直して転送量を半減する。

trainer の最終 save_pretrained は fp32 master weights をそのまま書くため
export が bf16 の 2 倍になる (2026-07-11 実測 128.06 GiB)。eval/trainer は
どのみち bf16 でロードするので、転送前に鋳直しても数値上は等価。
shard 単位でストリーム処理 (RAM ピーク ≈ 出力 shard 1 枚分 + 最大テンソル)。

usage: cast_export_bf16.py <export_dir> <out_dir>
"""
import json, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)

index_path = os.path.join(src, 'model.safetensors.index.json')
index = json.load(open(index_path))
shards = sorted(set(index['weight_map'].values()))
total = 0
for s, shard in enumerate(shards):
    out = {}
    with safe_open(os.path.join(src, shard), framework='pt') as f:
        for name in f.keys():
            t = f.get_tensor(name)
            if t.dtype == torch.float32:
                t = t.to(torch.bfloat16)
            out[name] = t
            total += t.numel() * t.element_size()
    save_file(out, os.path.join(dst, shard), metadata={'format': 'pt'})
    print(f'[cast] {shard} done ({s+1}/{len(shards)})', flush=True)
    del out

index['metadata']['total_size'] = total
with open(os.path.join(dst, 'model.safetensors.index.json'), 'w') as f:
    json.dump(index, f, indent=2)
for name in os.listdir(src):
    if name.endswith(('.json', '.jinja', '.txt')) and name != 'model.safetensors.index.json':
        shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
    if name.startswith('tokenizer') or name in ('vocab.json', 'merges.txt'):
        shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
print(f'[cast] total {total/1073741824:.2f} GiB -> {dst}', flush=True)
print('CAST_BF16_DONE', flush=True)
