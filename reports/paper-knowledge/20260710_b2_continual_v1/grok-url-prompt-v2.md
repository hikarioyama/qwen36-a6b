Return only direct primary-paper URLs, one URL per line. Do not output any other
text. No title, bullet, numbering, markdown, explanation, status line, or summary.

Use at most eight web-search/fetch tool calls, then stop and return the URLs even
if coverage is incomplete. Return 12-18 deduplicated URLs in arbitrary order.

Find the most directly useful primary papers for interpreting this running
experiment: sparse-MoE expert-specific delta training of a Qwen-style 35B-A3B
model, frozen router top-k 32, packed 7168-token continual-pretraining data,
Adafactor, and CE plus beta=0.5 forward KL from cached top-64 teacher logits.
The decision concern is target specialization versus broad reasoning/coding
forgetting.

Cover only these four themes:

1. sparse MoE routing, expert specialization, and expert fine-tuning;
2. continual pretraining and catastrophic forgetting in language models;
3. language-model knowledge distillation using forward KL or truncated teacher
   distributions;
4. replay/data mixing or parameter-efficient methods that preserve transfer.

Prefer canonical and directly applicable original papers. Use arxiv.org/abs,
openreview.net/forum, aclanthology.org, or DOI paper pages. Exclude surveys, blogs,
documentation, GitHub, model cards, leaderboards, and news.
