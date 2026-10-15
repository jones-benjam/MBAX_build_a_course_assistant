# Instructions for coding agents (Hermes, Claude Code, Codex, others)

This repository is the MBAX 6418 Assignment 2 course assistant: a Python app that
answers questions and generates practice quizzes from course materials with hybrid
RAG (keyword + text embeddings + visual embeddings, fused and reranked, grounded
answers with cited slide images). Read this file first, then:

1. `docs/PLAN.md` for the 10-step plan and what "done" means for each step.
2. `docs/HANDOFF.md` for current status, findings from the real services, and known gaps.
3. `docs/ENDPOINTS.md` for the verified request formats of the class model endpoints.
4. The GitHub issue you were given; its checklist defines the task.

## How to work in this repo

- **Start from `main`.** Create a branch named after the task (`step-9-readme`,
  `fix-quiz-grading`, `issue-14-screenshots`). Never commit directly to `main`.
- **Small, reviewable pull requests.** One PR per issue. The PR description
  carries a completion checklist mirroring the issue, and the body references the
  issue (`Closes #14`). A teammate other than the author reviews before merge.
- **Keep the checks green before every push:**
  ```bash
  ./run_tests.sh                              # offline: mock services, no key needed
  python -m pyflakes course_assistant tests scripts app.py
  ```
  Live checks against the real services (need a filled `.env`):
  ```bash
  LIVE_SERVICES=1 MOCK_SERVICES=0 python -m pytest tests/test_live_services.py -v -s
  MOCK_SERVICES=0 python scripts/evaluate.py --materials course_materials
  ```
- **Add tests with code.** Unit tests for new functions; end-to-end tests
  through `course_assistant.app_core.Assistant` for new workflows. Tests must run
  offline with `MOCK_SERVICES=1` (the default in `tests/conftest.py`).
- **Update the docs you touch.** `docs/HANDOFF.md` status table when a step
  changes; `README.md` when setup, usage, or results change; `docs/ENDPOINTS.md`
  if a request format changes.

## Secrets and private files (hard rules)

- The API key and real endpoint URLs live only in `.env` (gitignored). Copy
  `.env.example` and fill it in locally. Never write the key into code, docs,
  tests, logs, error messages, screenshots, commit messages, or issue comments.
- Course files from Canvas stay outside git. `course_materials/` is gitignored for
  a local copy; the class endpoint note belongs there too, never in the repo.
- Before pushing, scan your diff for anything key-shaped. The real key is a short
  number; a `git grep` for `API_KEY=` must only find dummy values.

## Layout

| Path | What it is |
|------|------------|
| `app.py` | Launches the Gradio UI (`python app.py`) |
| `course_assistant/config.py` | Settings from env/`.env`; `MOCK_SERVICES` switch |
| `course_assistant/services/` | Service interfaces, deterministic mocks, HTTP clients for the class endpoints |
| `course_assistant/ingest.py` | PDF/PPTX/DOCX/TXT/MD parsing, LibreOffice conversion, page images, chunking |
| `course_assistant/library.py` | Document library: content-hash dedup, add/remove with index cleanup |
| `course_assistant/index.py` | BM25, text and visual ChromaDB indexes, RRF fusion, reranking |
| `course_assistant/qa.py` | Grounded answers: evidence prompt, JSON validation, source support, citation rescue |
| `course_assistant/quiz.py` | Quiz generation with fixed answer key, grading, explanations |
| `course_assistant/ui.py` | Gradio tabs: Materials, Ask, Quiz |
| `scripts/evaluate.py` | Evaluation run (reranker on vs off) → `data/results/` |
| `scripts/screenshot_ui.py` | Playwright screenshots of the running app |
| `tests/` | pytest suite; `test_live_services.py` is opt-in |

## Class model endpoints

Plain-HTTP vLLM servers; base URLs in `.env`. Port 9001 vision LLM (port 9000 is
reserved for Hermes agent use; do not call it from the app), 9002 text embeddings,
9003 visual embeddings, 9004 multimodal reranker, 9005 document parser (unused; local
parsing is the implemented path). Request shapes and gotchas: `docs/ENDPOINTS.md`.

## Conventions worth knowing

- Everything degrades gracefully: a service that is down becomes a warning in the
  UI, never a crash. Keep it that way (`ServiceUnavailable` is the signal).
- Answers are JSON `{answer, sources, missing_information}`; only supplied evidence
  ids may be cited, and the app checks the sources support the answer.
- Quiz answer keys are fixed at generation time and never sent to the UI until the
  student submits or reveals.
- Python 3.12; dependencies in `requirements.txt`; LibreOffice is an external
  dependency for PPTX/DOCX (see `course_assistant/ingest.py:find_libreoffice`).
