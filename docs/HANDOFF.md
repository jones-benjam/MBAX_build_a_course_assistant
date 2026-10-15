# Handoff: continuing this project in Claude Code on a local machine

This note lets a fresh Claude Code session (or a teammate) pick up exactly where the
cloud session stopped. Read `docs/PLAN.md` first for the 10-step plan.

## Why the work moved to a local machine

The class model services at `dobolyi.com` (ports 9001–9005) are plain HTTP. The cloud
sandbox's egress proxy refuses to forward an `Authorization` header over an unencrypted
connection, so the real services could not be called from the cloud. On a laptop the
plain-HTTP calls work directly and **no code changes are needed** for that.

## Status at handoff (2026-10-15)

| Step | Status | Branch / PR |
|------|--------|-------------|
| 0 Outline | done | `docs/PLAN.md` |
| 1 Scaffold, config, service clients (mock + HTTP) | done, 6 tests | `step-1-scaffold` → PR #1 (base `main`) |
| 2 Ingestion (PDF/PPTX/DOCX/TXT/MD), library with dedup + clean removal | done, 10 tests | `step-2-ingestion` → PR #2 |
| 3 Hybrid retrieval (BM25 + text + visual, RRF, reranker, filters, degradation) | done, 7 tests | `step-3-retrieval` → PR #3 |
| 4 Grounded answers (JSON `{answer, sources, missing_information}`, support check) | done, 12 tests | `step-4-answering` → PR #4 |
| 5 Quizzes (fixed key, hidden solutions, grading, sources) | done, 8 tests | `step-5-quiz` → PR #5 |
| 6 Gradio UI (Materials / Ask / Quiz, light+dark, screenshot script) | done, 5 tests | `step-6-ui` → PR #6 |
| 7 Thorough testing incl. **real services** | done: endpoint formats verified (`docs/ENDPOINTS.md`), HTTP clients fixed, 10 live tests pass on the real endpoints + course files, UI walked through on real data; fixes: LibreOffice app-bundle lookup, block-tag vs slide-number citation rescue | `step-7-testing` → PR #7 |
| 8 Evaluation (5–10 questions, rerank on vs off) | done: `scripts/evaluate.py`, 10 questions, results in `data/results/`, write-up `docs/EVALUATION.md` | `step-8-evaluation` → PR #8 |
| 9 README, SVG architecture diagram, real screenshots | done in PRs: #18 AGENTS.md/CONTRIBUTING, #19 `docs/architecture.svg`, #20 real screenshots, #22 README | `step-9-*` |
| 10 Issues, reviews, fresh-clone check, submit | todo | |

PRs are **stacked**: each PR's base is the previous step's branch. Merge in order and
retarget the next PR to `main` after each merge (or merge the whole stack from
`step-6-ui` once reviewed). `step-6-ui` contains everything.

`./run_tests.sh` → 71 tests pass offline (`MOCK_SERVICES=1`; 2 skip without LibreOffice, 10 live tests skip without `LIVE_SERVICES=1`), `pyflakes` clean.

## Everything that has only been checked with mock services

- Retrieval **quality** (mock embeddings are hashed bag-of-words; rankings on real
  files were poor, as expected).
- Whether the real LLM honours the JSON schema and the "missing information" rule.
- The exact request/response format of each class endpoint (see below).
- `docs/screenshots/*` were captured on mock services and must be re-captured.

## Class endpoints: verified 2026-10-15 — see `docs/ENDPOINTS.md`

The table below is the pre-verification guess, kept for history. The verified formats are in `docs/ENDPOINTS.md`.

### Original notes (superseded)

`course_assistant/services/http_clients.py` assumes OpenAI-compatible vLLM routes.
Check each server's `/docs` or `/openapi.json` and adjust `_payload` / parsing:

| Service | Port | Note |
|---------|------|------|
| Vision LLM (`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`) | 9001 | `POST /v1/chat/completions` with `image_url` data URLs; `response_format: json_object` is assumed. Qwen3 may need `chat_template_kwargs: {"enable_thinking": false}` or stripping `<think>` blocks. Port 9000 is reserved for Hermes; do not use it. |
| Text embeddings (`nvidia/Nemotron-3-Embed-1B-BF16`) | 9002 | Class list gives base URL **without** `/v1`; try `/v1/embeddings` then `/embeddings`. Nemotron embed models may expect an instruction prefix for queries vs. passages. |
| Multimodal embeddings (`Qwen/Qwen3-VL-Embedding-2B`) | 9003 | Format for image inputs is unverified; vLLM multimodal embeddings may use a chat-style `messages` payload rather than `input: [{"image": ...}]`. |
| Multimodal reranker (`Qwen/Qwen3-VL-Reranker-2B`) | 9004 | Base URL without `/v1`; vLLM exposes `/rerank` and `/v1/rerank`; `documents` may need to be strings or multimodal objects. |
| Document parser (`dots.mocr`) | 9005 | Optional. Local PyMuPDF + LibreOffice parsing is the implemented path. |

API key for all: the value in the class list (store in `.env` only; never commit).
Keys are sent as `Authorization: Bearer <key>`.

## Step 7 findings on the real services (2026-10-15)

- All four services accepted the corrected request shapes on the first live run; the
  10 live tests (`tests/test_live_services.py`) pass in about 70 s.
- Week 2 deck: 42 slides ingest in ~45 s (one visual-embedding request per slide).
  The meme on slide 33 ranks first for the meme question through all three channels
  (keyword on the slide title, text, visual) with and without the reranker, and the
  LLM describes the image ("One Does Not Simply" / Boromir) correctly.
- Latency: answers ~2–3 s without the reranker, ~5–9 s with it (images are sent to
  the reranker). Quiz generation ~3–10 s.
- Bug found and fixed: the LLM sometimes cited a *slide number* where a *block tag*
  was expected (both are small integers), so a question about slide 11 cited block
  [S11] (slide 17). Prompts now say tags are not slide numbers, and
  `qa.rescue_citations` remaps such citations to the block from that slide when it
  clearly supports the text, recording a warning.
- Bug found and fixed: LibreOffice installed as a macOS app bundle was not found
  (`soffice` not on PATH); `ingest.find_libreoffice` now checks known locations and
  `SOFFICE_PATH`.
- Course files live in `course_materials/` inside the repo on this machine
  (gitignored); the live tests look there and in `../course_materials`.

## Step 9 findings (2026-10-15)

- Contact-sheet check of all 122 converted slides: correct except three animated
  GIFs (Week 2 s37, Week 4 s5, Week 5 s8) that LibreOffice renders as a blank first
  frame. Follow-up: issue #21.
- Week 4 slides 12 and 13 print the class endpoint key in their text. Course files
  are gitignored; never screenshot, quote or save answers from those slides.

## Course files

Not committed (copyright / size). The team has them from Canvas: syllabus PDF and the
Week 2–5 decks. Ingestion was verified on all of them (Week 2 = 42 slides; the
"Vibe Coding on 'Prod'" meme is slide 33 and its text is image-only).

## Known gaps / ideas noted during the build

- PDF text from PPTX keeps hard line breaks inside bullets; harmless for BM25 and
  embeddings, slightly ugly in excerpts.
- The quiz "topic matched nothing" fallback only triggers when retrieval returns
  nothing; dense search always returns neighbours, so a relevance threshold could be
  added once real scores are known.
- `KeywordIndex` is rebuilt in memory on every add/remove (fine at course scale).
- Reranker/answer latency with real images is unmeasured.

## Suggested first prompt for the local session

> Read docs/PLAN.md and docs/HANDOFF.md. We are at Step 7. First verify each class
> endpoint's request format against its /docs and fix course_assistant/services/http_clients.py,
> then run the app on the real course files (in ../course_materials), add end-to-end tests,
> and fix what breaks. Check in after each step, bug-check after each step.
