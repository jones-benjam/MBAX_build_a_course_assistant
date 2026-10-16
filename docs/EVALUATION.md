# Evaluation: answer quality and reranker ON vs OFF

Run on 2026-10-15 against the class endpoints (Qwen3.6-35B vision LLM, Nemotron-3
text embeddings, Qwen3-VL-Embedding-2B, Qwen3-VL-Reranker-2B) with the syllabus and
the Week 2–5 decks loaded (129 pages/slides, 160 chunks). Every question was asked
once with the reranker on and once with it off, on the same index. Raw results:
`data/results/eval_20261015-145853.json`; readable report with every answer:
`data/results/eval_latest.md`. Reproduce with:

```bash
MOCK_SERVICES=0 python scripts/evaluate.py --materials course_materials
```

## Question set (`data/results/questions.json`)

| Id | Type | Question | Expected source |
|----|------|----------|-----------------|
| syl-grading | syllabus | How is the final grade calculated in this course? | Syllabus p.2 |
| syl-office-hours | syllabus | When are the professor's office hours and how are they held? | Syllabus p.1 |
| syl-late-policy | syllabus | What is the late policy for assignments and quizzes? | Syllabus p.2–3 |
| w2-quantization | slide text | What is quantization, and roughly how large is Qwen3-Coder-30B-A3B-Instruct at 16, 8 and 4 bits? | Week 2 slide 15 |
| w3-chatml | slide text | What is ChatML and which special tokens does it use to mark messages? | Week 3 slide 8 |
| w4-branching | slide text | According to the Week 4 slides, why would you create a branch in a repository, and what is the default branch usually called? | Week 4 slide 25 |
| w5-reranking | slide text | In a hybrid RAG pipeline, what does the reranking step do? | Week 5 slide 18 |
| w2-meme-visual | **visual** | Find the meme about Vibe Coding on Prod in the Week 2 slides and explain what its image and text show. | Week 2 slide 33 (title only; content is in the image) |
| w2-trends-visual | **visual** | What does the chart on the 'Vibe Coding' slide in Week 2 show, and what happened around 2025? | Week 2 slide 29 (no body text; a Google Trends chart) |
| unanswerable | missing info | What was the class average score on Assignment 1? | none |

Correctness is judged automatically: an answer is correct when every expected
keyword group appears (alternatives allowed) and the assistant did not declare the
information missing; the unanswerable question is correct when the assistant declares
the information missing and cites nothing. "Supported" is the app's own source-support
check. We read every answer as well; the two judgements agree except where noted below.

## Results

| Metric | Reranker ON | Reranker OFF |
|---|---|---|
| Correct answers | 8 / 10 | 9 / 10 |
| Answers passing the support check | 10 / 10 | 10 / 10 |
| Expected page cited (9 answerable) | 9 / 9 | 9 / 9 |
| Expected page in the retrieved top-8 | 9 / 9 | 9 / 9 |
| Mean rank of the expected page in retrieval | **1.0** | 1.67 |
| Mean retrieval latency | 3.9 s | **0.4 s** |
| Mean end-to-end latency per answer | 5.1 s | **1.9 s** |
| Median end-to-end latency per answer | 4.2 s | **1.7 s** |
| Unanswerable question handled | yes | yes |

Per question (rank = position of the expected page among the evidence given to the LLM):

| Id | ON correct | ON rank | ON s | OFF correct | OFF rank | OFF s |
|---|---|---|---|---|---|---|
| syl-grading | ✅ | 1 | 4.2 | ✅ | 2 | 2.5 |
| syl-office-hours | ✅ | 1 | 4.9 | ✅ | 1 | 1.7 |
| syl-late-policy | ❌ (3/4 facts) | 1 | 4.3 | ✅ | 1 | 1.9 |
| w2-quantization | ✅ | 1 | 3.4 | ✅ | 1 | 1.8 |
| w3-chatml | ❌ (see limitation) | 1 | 4.2 | ❌ (see limitation) | 3 | 1.6 |
| w4-branching | ✅ | 1 | 14.8 | ✅ | 1 | 1.6 |
| w5-reranking | ✅ | 1 | 3.3 | ✅ | 1 | 1.6 |
| w2-meme-visual | ✅ | 1 | 4.7 | ✅ | 3 | 2.4 |
| w2-trends-visual | ✅ | 1 | 4.4 | ✅ | 2 | 2.2 |
| unanswerable | ✅ | – | 2.9 | ✅ | – | 1.6 |

Both visual questions were answered from the image: the meme answer names the "One
Does Not Simply" Boromir image and quotes its caption; the trends answer reads the
chart title, the two lines and the January 2025 crossover, none of which is in the
slide text.

## Interpretation

- **Retrieval precision:** the reranker moved the expected page to rank 1 in all nine
  answerable questions; without it the expected page was still always in the top 8
  but sat at rank 2–3 for four questions, including both visual ones (the meme slide
  dropped to rank 3). With the reranker the LLM is handed the right slide first and
  usually cites only that slide; without it the LLM sees more neighbours and the
  answers cite more pages, e.g. the grading answer without reranking had six
  citations spread over three syllabus pages (p.2, p.3 and p.7) instead of one.
- **Answer correctness** was effectively the same (8 vs 9 of 10). The one
  difference is `syl-late-policy`: with reranking the answer omitted that the lowest
  assignment/quiz score is dropped; without it the answer included that detail
  because the extra neighbouring chunk was in the evidence. This is a wording
  difference of the LLM, not a retrieval failure; the expected page was rank 1 both
  times.
- **Latency:** the reranker costs about 3.5 s per question (3.9 s vs 0.4 s retrieval),
  because every candidate's slide image is sent to the multimodal reranker. End to
  end that is 5.1 s vs 1.9 s per answer on average; the ON mean is pulled up by one
  14.8 s outlier (`w4-branching`), so the medians (4.2 s vs 1.7 s) are the fairer
  comparison. Note that `scripts/evaluate.py` retrieves twice per question (once to
  measure the expected page's rank, once inside the answer), so the reranker runs
  twice per question during evaluation; only the second run is in the end-to-end time.
  (Since this run, the script retrieves once and passes that evidence to the
  answerer, and the reranker scores each page once instead of each chunk; see
  "Changes since this run" below.)
- **Decision: keep the reranker ON by default** and keep the toggle in the UI. In
  both modes the cited (and therefore displayed) slide was the expected one for all
  nine answerable questions, and reranker OFF was as accurate and about 2.5 times
  faster. What the reranker adds is precision: the expected slide is always the top
  evidence and answers carry fewer extra citations, so the first image shown is the
  right one. That matters most for visual questions, where the LLM cannot compensate
  with text. The few extra seconds are acceptable for a study tool; users who want
  fast text-only lookups can switch it off. Reranker OFF would also be a defensible
  default on this evidence: the trade-off is precision of the first slide shown
  against speed, not answer correctness.

## Limitation investigated: special tokens disappear from answers

`w3-chatml` failed in both modes for the same reason. The correct answer must mention
`<|im_start|>` and `<|im_end|>`. The model's reply reads "uses the special tokens
`<|im_start|>` to mark the beginning of a message and `` to mark the end": the
`<|im_end|>` is gone. A direct probe confirmed the cause: asked to repeat the three
tokens `<|im_start|>`, `<|im_end|>` and `<|endoftext|>` verbatim, the endpoint
returns "the tokens are  and  and " with and without `skip_special_tokens: false`. These
strings are the model's own control tokens, and the vLLM server strips them from
generated text. The retrieved slide is correct and is shown to the user with its
excerpt, so the information is still on screen, but the composed answer cannot
contain it. A fix would need the model to write such tokens in an escaped form (for
example with spaces inside the brackets). That workaround has since been added (see
"Changes since this run"); it has not yet been checked on the live endpoint.

Two smaller findings from the run, both already handled by the app:

- Citation ids vs. slide numbers: the LLM occasionally cites a slide number where an
  evidence-tag number is expected (seen once in this run, "[S7]" for slide 7). The
  app's citation rescue remapped it to the block from that slide and recorded a
  warning.
- The LLM once cited an evidence id beyond the eight blocks supplied; the app dropped
  it and warned.

## What was not tested

- Only one run per question (temperature 0); variance across runs was not measured.
- Chunking parameters and `top_k` were held fixed; only the reranker was compared.
- Weeks 6+ and the "Local Inference" deck were not loaded.

## Changes since this run (results above not yet re-run)

These changes affect speed and the ChatML failure; the numbers above predate them.
Re-run `MOCK_SERVICES=0 python scripts/evaluate.py --materials course_materials`
to refresh them.

- **Page-level reranking:** candidates are grouped by page or slide and each page is
  scored once, so a slide image is sent to the reranker once rather than once per
  chunk. At most `RERANK_CANDIDATES` pages (default 10) are scored; the rest keep
  their fused order behind them.
- **Parallel retrieval:** keyword, text-embedding and visual-embedding searches run
  concurrently; the fused ranking is unchanged.
- **Query-embedding cache:** repeated questions skip the two embedding calls.
- **Single retrieval in the evaluation:** the rank recorded is now exactly the
  evidence the LLM saw, the reranker runs once per question, and end-to-end time is
  retrieval time plus answering time.
- **Retrieval metrics:** the report adds mean reciprocal rank (MRR), recall@1 and
  recall@3 over the answerable questions. From the ranks above, reranker ON has
  MRR 1.0, recall@1 9/9 and recall@3 9/9; reranker OFF has MRR 0.74,
  recall@1 5/9 and recall@3 9/9.
- **Control-token workaround:** see the limitation above.
