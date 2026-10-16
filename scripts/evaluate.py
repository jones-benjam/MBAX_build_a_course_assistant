"""Evaluate answer quality on a fixed question set, comparing reranker ON vs OFF.

Usage (real services, reads .env):
    MOCK_SERVICES=0 python scripts/evaluate.py --materials course_materials
    python scripts/evaluate.py --questions data/results/questions.json --out data/results

The script ingests the course files into DATA_DIR (duplicates are skipped), then runs
every question once per mode on the same index. For each run it records whether the
answer is correct (expected keyword groups all present), whether the expected
page/slide was cited, whether the support check passed, whether a slide image was
shown, retrieval and end-to-end latency, and the answer text for manual review.
Results go to <out>/eval_<timestamp>.json and <out>/eval_latest.md.
`--rescore <results.json>` re-judges a saved run after editing the expected keywords.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_FILES = [
    "MBAX 6418 Syllabus Fall 2026.pdf",
    "MBAX 6418 - Week 2 - LLM Fundamentals v2.pptx",
    "MBAX 6418 - Week 3 - Prompt Engineering v1.pptx",
    "MBAX 6418 - Week 4 - Serving and Debugging v6.pptx",
    "MBAX 6418 - Week 5 - Context Engineering and RAG v1.pptx",
]


# ---------------------------------------------------------------- scoring helpers

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def keyword_groups_hit(answer: str, groups: list[list[str]]) -> tuple[int, int]:
    """Each group is a list of alternatives; a group is hit when any alternative appears."""
    a = _norm(answer)
    hits = sum(1 for g in groups if any(_norm(alt) in a for alt in g))
    return hits, len(groups)


def judge(q: dict, answer: str, missing: bool, error, cited_pages: list[tuple[str, int]]) -> dict:
    """Automatic correctness + source checks for one answer. Manual review may override."""
    if q["type"] == "unanswerable":
        return {"correct": bool(missing) and not cited_pages, "keyword_hits": "n/a", "expected_page_cited": None}
    hits, total = keyword_groups_hit(answer, q.get("expected_keywords", []))
    exp_doc = (q.get("expected_doc") or "").lower()
    expected_cited = any(exp_doc in d.lower() and p in q.get("expected_pages", []) for d, p in cited_pages)
    correct = hits == total and not missing and not error
    return {"correct": correct, "keyword_hits": f"{hits}/{total}", "expected_page_cited": expected_cited}


_SOURCE_RE = re.compile(r"^(.*) p\.(\d+)$")


def rescore(path: Path, questions: list[dict]) -> tuple[dict, dict, list[dict]]:
    """Re-judge a saved results file against (possibly edited) expected keywords."""
    data = json.loads(path.read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in questions}
    rows = data["results"]
    for r in rows:
        cited = []
        for s in r["sources"]:
            m = _SOURCE_RE.match(s)
            if m:
                cited.append((m.group(1), int(m.group(2))))
        r.update(judge(by_id[r["id"]], r["answer"], r["missing_information"], r["error"], cited))
    data["summary"] = summarise(rows)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data["meta"], data["summary"], rows


def expected_rank(q: dict, evidence) -> int | None:
    """1-based rank of the first evidence item from an expected page, or None."""
    exp_doc = (q.get("expected_doc") or "").lower()
    for i, ev in enumerate(evidence, start=1):
        if exp_doc in ev.chunk.doc_name.lower() and ev.chunk.page_number in q.get("expected_pages", []):
            return i
    return None


# ---------------------------------------------------------------- running

def run_question(assistant, q: dict, use_reranker: bool) -> dict:
    # Retrieve once and hand the same evidence to the answerer, so the rank we record
    # is exactly the evidence the LLM saw and the reranker runs once per question.
    # End-to-end time = retrieval time + answering time.
    t0 = time.time()
    retrieval = assistant.retrieve(q["question"], use_reranker=use_reranker)
    retrieval_s = time.time() - t0
    ans = assistant.ask(q["question"], use_reranker=use_reranker, retrieval=retrieval)
    verdict = judge(q, ans.answer, ans.missing_information, ans.error, [(s.doc_name, s.page_number) for s in ans.sources])
    return {
        "id": q["id"], "type": q["type"], "question": q["question"], "rerank": use_reranker,
        **verdict,
        "supported": ans.supported, "support_score": round(ans.support_score, 3),
        "missing_information": ans.missing_information,
        "image_shown": any(s.image_path is not None for s in ans.sources),
        "sources": [f"{s.doc_name} p.{s.page_number}" for s in ans.sources],
        "expected_rank_in_retrieval": expected_rank(q, retrieval.evidence),
        "retrieval_s": round(retrieval_s, 2), "answer_s": round(retrieval_s + ans.latency_s, 2),
        "reranked": ans.reranked, "warnings": ans.warnings, "error": ans.error,
        "answer": ans.answer,
    }


def retrieval_metrics(ranks: list[int | None]) -> dict:
    """Standard retrieval metrics over answerable questions, from the 1-based rank of
    the expected page in the evidence (None = not retrieved).

    recall@k: share of questions whose expected page is in the top k.
    MRR (mean reciprocal rank): mean of 1/rank, counting a miss as 0; 1.0 means the
    expected page always came first."""
    n = len(ranks)
    if n == 0:
        return {"mrr": 0.0, "recall_at_1": 0.0, "recall_at_3": 0.0}
    return {
        "mrr": round(sum(1.0 / r for r in ranks if r) / n, 3),
        "recall_at_1": round(sum(1 for r in ranks if r and r <= 1) / n, 3),
        "recall_at_3": round(sum(1 for r in ranks if r and r <= 3) / n, 3),
    }


def summarise(rows: list[dict]) -> dict:
    modes = {}
    for mode in (True, False):
        rs = [r for r in rows if r["rerank"] is mode]
        answerable = [r for r in rs if r["type"] != "unanswerable"]
        modes["rerank_on" if mode else "rerank_off"] = {
            **retrieval_metrics([r["expected_rank_in_retrieval"] for r in answerable]),
            "n": len(rs),
            "correct": sum(r["correct"] for r in rs),
            "supported": sum(r["supported"] for r in rs),
            "expected_page_cited": sum(bool(r["expected_page_cited"]) for r in answerable),
            "expected_in_top_k": sum(r["expected_rank_in_retrieval"] is not None for r in answerable),
            "mean_expected_rank": round(sum(r["expected_rank_in_retrieval"] for r in answerable
                                            if r["expected_rank_in_retrieval"]) /
                                        max(1, sum(1 for r in answerable if r["expected_rank_in_retrieval"])), 2),
            "mean_retrieval_s": round(sum(r["retrieval_s"] for r in rs) / max(1, len(rs)), 2),
            "mean_answer_s": round(sum(r["answer_s"] for r in rs) / max(1, len(rs)), 2),
        }
    return modes


def to_markdown(meta: dict, summary: dict, rows: list[dict]) -> str:
    out = [f"# Evaluation results ({meta['timestamp']})", "",
           f"Files: {', '.join(meta['files'])}  ", f"Questions: {meta['n_questions']}, top_k={meta['top_k']}, "
           f"services: {'mock' if meta['mock'] else 'class endpoints'}", "",
           "## Reranker ON vs OFF", "",
           "| Metric | Rerank ON | Rerank OFF |", "|---|---|---|"]
    on, off = summary["rerank_on"], summary["rerank_off"]
    for k, label in [("correct", "Correct answers"), ("supported", "Answers passing the support check"),
                     ("expected_page_cited", "Expected page cited (answerable qs)"),
                     ("expected_in_top_k", "Expected page in retrieved top-k"),
                     ("mean_expected_rank", "Mean rank of expected page"),
                     ("mrr", "Mean reciprocal rank (MRR)"),
                     ("recall_at_1", "Recall@1 (expected page ranked first)"),
                     ("recall_at_3", "Recall@3 (expected page in top 3)"),
                     ("mean_retrieval_s", "Mean retrieval latency (s)"), ("mean_answer_s", "Mean end-to-end latency (s)")]:
        out.append(f"| {label} | {on[k]}{'/' + str(on['n']) if k in ('correct', 'supported') else ''} | "
                   f"{off[k]}{'/' + str(off['n']) if k in ('correct', 'supported') else ''} |")
    out += ["", "## Per question", "", "| Id | Type | Rerank | Correct | Keywords | Expected page cited | Supported | Image | Rank | Retrieval s | Answer s |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['id']} | {r['type']} | {'on' if r['rerank'] else 'off'} | {'✅' if r['correct'] else '❌'} | "
                   f"{r['keyword_hits']} | {r['expected_page_cited']} | {r['supported']} | {r['image_shown']} | "
                   f"{r['expected_rank_in_retrieval']} | {r['retrieval_s']} | {r['answer_s']} |")
    out += ["", "## Answers", ""]
    for r in rows:
        out += [f"### {r['id']} (rerank {'on' if r['rerank'] else 'off'})", "", f"**Q:** {r['question']}", "",
                f"**A:** {r['answer']}", "", f"Sources: {', '.join(r['sources']) or 'none'}", ""]
        if r["warnings"]:
            out += [f"Warnings: {'; '.join(r['warnings'])}", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--materials", default="course_materials", help="folder with the course files")
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES, help="file names inside --materials")
    ap.add_argument("--questions", default="data/results/questions.json")
    ap.add_argument("--out", default="data/results")
    ap.add_argument("--modes", default="on,off", help="comma list of rerank modes to run")
    ap.add_argument("--rescore", metavar="RESULTS_JSON", help="re-judge a saved run instead of running the services")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    out = Path(args.out)
    if args.rescore:
        meta, summary, rows = rescore(Path(args.rescore), questions)
        (out / "eval_latest.md").write_text(to_markdown(meta, summary, rows), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    from course_assistant.app_core import Assistant
    from course_assistant.config import load_settings

    settings = load_settings()
    settings.ensure_dirs()
    assistant = Assistant(settings)
    materials = Path(args.materials)
    used = []
    for name in args.files:
        path = materials / name
        if not path.exists():
            print(f"skip (missing): {path}")
            continue
        t0 = time.time()
        rec, status = assistant.add_document(path)
        print(f"{status:9s} {name} ({rec.n_pages} pages, {rec.n_chunks} chunks, {time.time() - t0:.1f}s)")
        used.append(name)

    modes = [m.strip() == "on" for m in args.modes.split(",")]
    rows: list[dict] = []
    for use_reranker in modes:
        for q in questions:
            r = run_question(assistant, q, use_reranker)
            rows.append(r)
            print(f"[rerank {'on ' if use_reranker else 'off'}] {q['id']:18s} correct={r['correct']!s:5s} "
                  f"cited={r['expected_page_cited']!s:5s} supported={r['supported']!s:5s} "
                  f"rank={r['expected_rank_in_retrieval']} {r['answer_s']}s")

    meta = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": used,
            "n_questions": len(questions), "top_k": settings.top_k, "mock": settings.mock_services,
            "models": {"llm": settings.llm.model, "text_embed": settings.text_embed.model,
                       "visual_embed": settings.visual_embed.model, "rerank": settings.rerank.model}}
    summary = summarise(rows)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / f"eval_{stamp}.json").write_text(json.dumps({"meta": meta, "summary": summary, "results": rows},
                                                       indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "eval_latest.md").write_text(to_markdown(meta, summary, rows), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out / f'eval_{stamp}.json'} and {out / 'eval_latest.md'}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    sys.exit(main())
