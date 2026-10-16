# Handoff: project status and continuation notes

This note lets a teammate or a coding agent (Hermes, Claude Code, others) pick up the
project. Read `AGENTS.md` and `docs/PLAN.md` first; team workflow is in `CONTRIBUTING.md`.

## Why the work moved to a local machine

The class model services at `dobolyi.com` (ports 9001–9005) are plain HTTP. The cloud
sandbox's egress proxy refuses to forward an `Authorization` header over an unencrypted
connection, so the real services could not be called from the cloud. On a laptop the
plain-HTTP calls work directly and **no code changes are needed** for that.

## Status (updated 2026-10-15, after PR #32)

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
| 9+ Team follow-ups | merged: #23 + #29 animated-GIF repair (malu1954-wq, Mackdug876; closes #21), #25 setup-doc gaps from the fresh-clone review (Mackdug876), #27 evaluation write-up corrections (Mackdug876), #31 key redaction before truncation, lint, README count (Mackdug876), #32 redaction tests for JSON and plain text, doc corrections (vanessabagtas). Open: #33 stop evidence tag numbers leaking into prose | various |
| 10 Issues, reviews, fresh-clone check, submit | in progress, see "Reviews" and "Open items" below | issues #16, #17 |

All step PRs are merged into `main`; the stacked `step-*` branches are history. New
work branches from `main`, one branch and PR per issue (see `CONTRIBUTING.md`).
Current test counts are in the README "Tests" section; the offline suite and
`pyflakes` are clean on `main`, and the 10 live tests pass against the class endpoints.

### Reviews

- PRs #1–#8 and #18–#22 were merged without a prior review, at the project owner's
  request, and reviewed afterwards: Mackdug876 left review comments on each,
  vanessabagtas on #7.
- #23 was merged while its review said "changes requested" (transparent GIFs turned
  black); the requested change landed in #29.
- #25, #27, #29 and #32 were merged without a review; #31 was reviewed after merge
  (vanessabagtas).
- Going forward, use the reviewer checklist in `CONTRIBUTING.md` and review before merge.

### Open items

- #33 (open PR): evidence headers and prompts no longer invite "tag" numbers into
  prose; needs a review.
- Issues #9 and #10 (reviews of PRs #7 and #8): reviews are posted; close when agreed.
- Issue #16 (fresh-clone check): offline part done by malu1954-wq and vanessabagtas. On
  2026-10-15 a fresh clone of `main` at 0990f28 also passed the 10 live tests against
  the class endpoints (run during the repository review); a teammate's own live run
  is still worth recording on the issue.
- Issue #17: final submission on Canvas.
- Optional: the quiz screenshots and `data/results/` predate the GIF repair and the
  prompt change in #33; recapture or re-run if time allows.

## Real-service verification (history)

Before Step 7 everything had only run on mock services. Since then retrieval quality,
the LLM's JSON and missing-information behaviour, every endpoint's request format, and
the screenshots have all been checked on the class endpoints (see the Step 7 findings
below, `docs/ENDPOINTS.md` and `docs/EVALUATION.md`).

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
  frame. Fixed in #23 and #29: a blank GIF area is replaced by the GIF's most detailed
  frame. Verified on the real slides during the 2026-10-15 review. Indexes built
  before the fix keep the old page images until the deck is removed and re-added.
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
- Latency is measured in `docs/EVALUATION.md` (median 4.2 s per answer with the
  reranker, 1.7 s without).
- Saved chunks store page-image paths relative to the project folder, so the app must
  be started from the repository root. If a page image is missing, the reranker raises
  a file error that stops the request instead of degrading like a service outage.

## Suggested first prompt for a coding agent

> Read AGENTS.md and docs/HANDOFF.md. Take open issue #<n>: work on a branch from
> main, keep ./run_tests.sh and pyflakes green, and open a pull request with the
> issue's checklist for a teammate to review. Never put the API key or course files
> in the repository.
