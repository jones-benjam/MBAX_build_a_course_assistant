# Course Assistant (MBAX 6418 Assignment 2)

Python hybrid-RAG course assistant: Gradio UI, BM25 + text + visual embeddings in
separate indexes, RRF fusion, multimodal reranking, grounded answers and quizzes.

- Plan and step status: `docs/PLAN.md`; continuation notes: `docs/HANDOFF.md`.
- Run tests: `./run_tests.sh` (offline, mock services). Lint: `python -m pyflakes course_assistant tests`.
- Run app: `python app.py` (reads `.env`; copy from `.env.example`).
- Secrets: only in `.env` (gitignored). Never put the API key in code, docs, tests, logs, screenshots or commits.
- Workflow: one branch per step (`step-N-name`), PR with a completion checklist, keep tests green before pushing.
- Class endpoints: port 9001 for the vision LLM (9000 is reserved for Hermes), 9002 text embeddings, 9003 visual embeddings, 9004 reranker, 9005 parser. Verified request formats: `docs/ENDPOINTS.md`.
- Live tests (need `.env` with the real key): `LIVE_SERVICES=1 MOCK_SERVICES=0 python -m pytest tests/test_live_services.py -v -s`.
- Evaluation (real services): `MOCK_SERVICES=0 python scripts/evaluate.py --materials course_materials`; results in `data/results/`, write-up in `docs/EVALUATION.md`.
