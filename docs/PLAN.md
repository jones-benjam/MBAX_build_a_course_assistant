# MBAX 6418 Assignment 2 — Build Plan

Goal: a Python course assistant (Gradio UI) that answers questions and generates
practice quizzes from Canvas course materials using hybrid RAG (keyword + text
embeddings + visual embeddings, fused and reranked), with every answer grounded in
cited sources that include the original page/slide image.

## Steps

| # | Step | Deliverable | Done when |
|---|------|-------------|-----------|
| 0 | Outline | `docs/PLAN.md` | Team agrees on the plan |
| 1 | Project scaffold | Package layout, `requirements.txt`, `.env.example`, `.gitignore`, config loader, service-client interface with an offline **mock mode**, pytest harness, smoke test | `pytest` passes; app imports with no keys present |
| 2 | Document ingestion | Parse PDF / PPTX / DOCX / TXT / MD; PPTX → PDF via LibreOffice; render page/slide images; chunk text with source metadata (doc, page/slide); hash-based dedup; add/remove documents and purge their indexed content | Unit tests for parsing, dedup, remove-cleans-index |
| 3 | Indexing and retrieval | BM25 keyword index (bm25s); text-embedding index (ChromaDB); separate visual-embedding index (ChromaDB); reciprocal-rank fusion; optional multimodal reranking | Unit tests with mock services; toggle for rerank on/off |
| 4 | Answer generation | Evidence packaging (text + images) → vision LLM; structured `{answer, sources}` output validated with pydantic; source-support check; explicit "not in materials" behaviour | Tests for format validation, unsupported-source rejection, missing-info answer |
| 5 | Quiz generation | Material + topic filter → multiple-choice questions with fixed answer key, hidden solutions, scoring, explanations and sources | Tests: key fixed across reveals; score equals key matches; every question cites a source |
| 6 | Gradio interface | Tabs: Materials (add/remove, formats accepted), Ask (answer + slide images + sources), Quiz (take, score, reveal). Light/dark readability check | Manual walkthrough; screenshots saved |
| 7 | Testing and bug fixing | Unit + end-to-end tests incl. duplicate upload, removal, missing information, services unavailable; fix findings | `pytest` green; failures documented |
| 8 | Evaluation | 5–10 question set (syllabus, slide text, ≥2 visual incl. the "Vibe Coding on Prod" meme, 1 unanswerable); compare rerank ON vs OFF on identical questions/files; record correctness, source support, latency to `data/results/` | Results file committed; interpretation written |
| 9 | Documentation | `README.md`: setup, endpoints config (dummy values), formats and conversion, usage, tests, results, limitations; `docs/architecture.svg` embedded; ≥2 screenshots | Teammate reproduces from a fresh clone |
| 10 | Team workflow and submission | GitHub issues with completion checks per substantial step, branches/PRs, review; final fresh-clone verification; repo link on Canvas | Issues closed; latest commit on `main` is the submission |

## Design decisions (one chosen alternative per row, to revisit if needed)

| Area | Choice | Alternative considered |
|------|--------|------------------------|
| UI | Gradio | Streamlit |
| Keyword search | bm25s | SQLite FTS5 |
| Vector store | ChromaDB (persistent, two collections: `text`, `visual`) | FAISS |
| Chunking | RecursiveCharacterTextSplitter, page-bounded so every chunk keeps its page/slide | Fixed-size token windows |
| Fusion | Reciprocal rank fusion across BM25 / text / visual, then optional reranker | Weighted score sum |
| PPTX | Convert to PDF with LibreOffice headless, then treat as PDF (text + images) | Manual export |
| Services | Class endpoints at dobolyi.com (ports ≥ 9001) via env vars; `MOCK_SERVICES=1` runs deterministic local stand-ins for tests | Public APIs |
| Secrets | `.env` (gitignored), `.env.example` with dummy values | Config file in repo |

## Inputs needed from the team

1. Class endpoint connection details (host:port, path, auth header, request/response format) for: vision LLM, text embeddings, visual embeddings, multimodal reranker, document parser.
2. The Canvas course files (syllabus, weekly slide decks incl. Week 2).
3. GitHub usernames of teammates (for issue assignment and PR review).
4. Branching preference: feature branches + PRs into `main` (recommended).
