# Toucan/selfgen → trainer converter と Qwen tool-template 検証 (2026-07-11)

## 結論

**Measured (CPU only):** 指定した stock tokenizer snapshot
`995ad96eacd98c81ed38be0c5b274b04031597b0` の chat template は、`tools=` を
受け取ると tool schema と固定 instruction を **先頭 system message として合成**し、
assistant の `tool_calls` と role `tool` をそれぞれ XML の `<tool_call>` と
`<tool_response>` に展開する。`tools=` を渡さなくてもこの二つの role/field は展開するが、
利用可能な tool 定義は一切出力されない。

しかし現在の `train_fullffn_dcp.py:render_and_tokenize` は tool schema を渡せないだけでなく、
各 message prefix を別々に render する。そのため data-only で system に schema を焼き込むと、
最初の `[{role: system}]` prefix を template が `No user query found in messages.` として拒否する。
さらに、文字列としては prefix 性を保つが token ID では assistant 境界で prefix 性を保たない。

従って、**現在の trainer と完全等価な `tools=` 学習をする正しい方法は trainer 改修である**。
このタスクでは trainer 本体を変更しない制約に従い、converter は実行可能な暫定形式として、同一
tool-schema 本文を最初の user content の先頭に置く `user-preamble` を出力する。これは tool call
surface の教師信号を保存するが、inference-time `tools=` の system context と同一ではない。全量 SFT
をこの暫定形式で開始することは、この差分を受容する明示的な判断なしには推奨しない。

## Template の実測

対象はローカル snapshot の `chat_template.jinja` (SHA-256
`e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`) と
`transformers 5.8.0` の `Qwen2Tokenizer`。モデル重みはロードしていない。

template の該当箇所は次の通りである（ローカル snapshot `chat_template.jinja`）。

- L45--60: `tools` が iterable の時、`<|im_start|>system`、`# Tools`、各
  `tool | tojson`、固定の function-call instruction を出力する。既存先頭 system content は
  L54--58 でその末尾に追加される。
- L67--80: user query を見つけられなければ `No user query found in messages.` を raise する。
  したがって incremental trainer の system-only 一件目は render 不可能である。
- L105--129: assistant `tool_calls[].function.{name,arguments}` を
  `<tool_call><function=…><parameter=…>` に出力する。arguments は mapping が必要である。
- L131--142: role `tool` は user channel の `<tool_response>` に出力される。`name` は template
  には出力されない。
- L147--150: generation prompt は `<|im_start|>assistant\n<think>\n`（thinking 有効時）を付与する。

合成会話（user → assistant `weather(city=Tokyo)` → tool → assistant、tool schema 1 件）での
CPU 実測は n=1 conversation / 4 message turns、same tokenizer/template 条件である。

| 呼び出し | 実測 |
|---|---|
| `tools=` なし | `# Tools` / schema は出ない。assistant call は `<tool_call>`、tool は `<tool_response>` として出る。|
| `tools=[…]` | 上記の system block が先頭に出て、schema と固定 instruction が出る。assistant/tool の展開は同じ。|
| `tools=` branch の system block と converter の `_tool_system_content` | 文字列完全一致（unit test）。|

### Prefix 性

同じ n=1 / 4 turns を trainer と同じ
`messages[:i+1]`, `add_generation_prompt=(role != "assistant")` で検査した。

| 境界 | render text prefix | token-ID prefix |
|---|---:|---:|
| user → assistant(tool call) | true | **false** |
| assistant → tool | true | true |
| tool → assistant | true | **false** |

具体的には user-only prefix の末尾は `<think>\n` だが、assistant を含めた render は
`<think>\n\n</think>\n\n…` となる。文字列の前者は後者の接頭辞だが、fast tokenizer が newline
を結合するので token 列の前者は後者の token 列の接頭辞ではない。このため現在の
`prev_len` による token index 差分は、完全 render token 列の分割ではない。これは tool data
固有ではなく、通常の user→assistant 境界にも存在する template/trainer の相互作用である。

`esft/tests/test_corpus_to_trainer_v1.py` はこの二つの性質（text=true、ID=false）を将来の
template 更新に対する CPU regression test として固定した。

## Converter 仕様

追加: `esft/corpus_to_trainer_v1.py`

- JSONL は一行ずつ、Parquet は `pyarrow.ParquetFile.iter_batches`（default 128 rows）で読む。
  corpus table を materialize しないので、converter 自身の追加ピークメモリはおおむね batch と
  一 record に支配され、8 GB 制約を十分下回る設計である（全量 peak は未測定）。
- Toucan string columns は最大二回 JSON parse する。`available_tools`（Kimi-K2/OSS/Qwen3）と
  `tools`（SFT）を処理する。
- legacy `assistant.function_call` は native
  `assistant.tool_calls=[{type:function,function:{name,arguments:<mapping>}}]` にする。`function` は
  `tool` に、SFT の `tool_call` / `tool_response` はそれぞれ assistant tool call / `tool` にする。
- assistant `tool_calls` の arguments JSON string は object に parse する。parse できない tool call
  や未知 role は stderr に記録して当該 record を skip し、黙って text 化しない。
- trainer に `tools=` を渡す場所がないため、template と同じ tool-schema 本文を最初の user turn の
  先頭へ注入する。Toucan legacy system の `<|im_system|>tool_declare…` は二重化せず schema として
  吸収する。普通の source system text も同じ preamble に続ける。
- 出力は `{"messages": [...], "_source": SOURCE_TAG, "_domain": DOMAIN}` の JSONL。
  `--source-tag` は必須、`--domain` の default は `toolcall`。source column, raw tools, metadata は
  trainer output に持ち込まない。
- Parquet は同名 `.done` / suffix `.done`、shard directory の `manifest.json`、または mtime が
  default 300 秒以上古い場合にしか読む。書込み中 shard は stderr に `SKIP active parquet` と出す。

### CPU smoke / tests

すべて CPU only、network/SSH/GPU は未使用。

- `python3 esft/tests/test_corpus_to_trainer_v1.py`: **PASS**, n=7 tests。
- selfgen 実データ smoke: n=3 records, same converter/settings, invalid skip=0。
  `user, assistant(tool_calls), tool...` に変換され、tool-call arguments は mapping になった。
- 完成済み Toucan Kimi-K2 clean shard smoke: n=3 records, same converter/settings,
  invalid skip=0。legacy `function_call` / `function` を round-trip した。
- SFT `tool_call/tool_response` は clean output がまだ利用可能でなかったため synthetic fixture
  n=1 で正規化を検証した。Kimi-K2/OSS/Qwen3 は legacy fixture と clean Kimi-K2 n=3 で確認した。
- 変換済み selfgen 実 record n=1 を実際の `render_and_tokenize` に通し、assistant tool-call turn
  には labels があり、non-assistant (user/tool) turns の labels はすべて `-100` であることを検証した。
  ただし上記 token-ID prefix failure のため、この pass は「現在の関数が role に従い label を置く」
  ことの検査であり、full-render tokenization との同一性の主張ではない。

実測時点で clean tree に存在した Parquet は Kimi-K2 40 shards（eligible 40）と OSS 7 shards
（eligible 3）のみ。Qwen3/SFT は 0 shards で、対応する done/manifest は未確認だった。これは
writer 進行中の状態であり、全量数や schema coverage の完了主張ではない。

## trainer を正しく直す場合の提案（未適用）

以下は方向を示す最小 diff であり、この task では適用していない。実装時は worker payload、cache
key、テストを同時に更新する必要がある。

```diff
diff --git a/esft/deploy/train_fullffn_dcp.py b/esft/deploy/train_fullffn_dcp.py
@@
-def render_and_tokenize(tokenizer, messages, mask_prompt=True, ignore_id=-100):
+def render_and_tokenize(tokenizer, messages, tools=None, mask_prompt=True, ignore_id=-100):
@@
-        text = tokenizer.apply_chat_template(
-            convo, tokenize=False,
-            add_generation_prompt=(msg["role"] != "assistant"),
-        )
+        text = tokenizer.apply_chat_template(
+            convo, tools=tools, tokenize=False,
+            add_generation_prompt=(msg["role"] != "assistant"),
+        )
@@
-render_and_tokenize(tokenizer, rec["messages"])
+render_and_tokenize(tokenizer, rec["messages"], tools=rec.get("tools"))
```

その上で converter は user-preamble を止め、native `tools` を record に保存する必要がある。
ただしこの diff **だけ**では token-ID prefix failure は解消しない。`render_and_tokenize` は最終
render を一度だけ tokenise し、fast tokenizer の offset mapping を用いて assistant の character
span に完全に含まれる token だけ label する方式へ置き換える必要がある。境界をまたぐ token は
mask する。これで input IDs は final template render と一致し、assistant-only supervision を保てる。

## Fable 全量変換手順（未起動）

Qwen3/SFT を含む clean 完成状態（各 shard の `.done`、または全体 manifest）を先に確認する。
writer 進行中の root に対して下の command を実行して部分データを「全量」と呼ばない。

```bash
cd ~/projects/qwen36-a6b

RUN_ROOT=/mnt/vault/corpora/derived/qwen36-a6b-intake-20260711-v1
OUT="$RUN_ROOT/trainer/toucan_selfgen_trainer_v1.jsonl"
mkdir -p "$(dirname "$OUT")"

python3 esft/corpus_to_trainer_v1.py \
  --input esft/data/selfgen_toolcall_v1/20260711_toolcall_v1_pilot500_r3/train.jsonl \
  --input "$RUN_ROOT/clean/toucan-1.5m" \
  --output "$OUT" \
  --source-tag toucan-selfgen-20260711-v1 \
  --domain toolcall \
  --batch-size 128 \
  --min-parquet-age-seconds 300
```

完了時の stderr JSON (`written`, `skipped_invalid`) と入力 shard inventory を保存する。出力は user-preamble
暫定形式なので、training launch 前に上記 trainer 改修方針を採用するか、意図的に暫定方式を受容するかを
決める必要がある。
