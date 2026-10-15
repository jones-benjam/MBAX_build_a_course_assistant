"""Gradio interface. Handler functions are plain Python (testable without a browser);
`build_app()` wires them into Blocks. Run with `python app.py`."""
from __future__ import annotations

import html
from pathlib import Path

import gradio as gr

from .app_core import Assistant
from .ingest import UnsupportedFormat, describe_supported_formats
from .qa import Answer, Source
from .quiz import Quiz, QuizError
from .services import ServiceUnavailable

MAX_QUIZ_QUESTIONS = 10


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def doc_choices(assistant: Assistant) -> list[tuple[str, str]]:
    return [(f"{d.name}  ({d.n_pages} pages)", d.doc_id) for d in assistant.library.list_documents()]


def library_table(assistant: Assistant) -> list[list]:
    return [[d.name, d.ext.lstrip("."), d.n_pages, d.n_chunks, d.added_at.replace("T", " ").replace("+00:00", " UTC")]
            for d in assistant.library.list_documents()]


def _warn_md(warnings: list[str]) -> str:
    return "".join(f"\n> ⚠️ {html.escape(w)}" for w in warnings)


def source_gallery(sources: list[Source]) -> list[tuple[str, str]]:
    seen, out = set(), []
    for s in sources:
        if s.image_path and s.image_path.exists() and str(s.image_path) not in seen:
            seen.add(str(s.image_path))
            out.append((str(s.image_path), f"[S{s.id}] {s.label}"))
    return out


def sources_markdown(sources: list[Source]) -> str:
    if not sources:
        return "_No sources cited._"
    parts = []
    for s in sources:
        excerpt = html.escape(s.excerpt).replace("\n", " ")
        if len(excerpt) > 600:
            excerpt = excerpt[:600] + "…"
        parts.append(f"**[S{s.id}] {html.escape(s.label)}**\n\n> {excerpt}")
    return "\n\n".join(parts)


def answer_markdown(ans: Answer) -> str:
    if ans.error:
        head = f"### ⚠️ {html.escape(ans.answer)}"
    elif ans.missing_information:
        head = f"### ℹ️ Not found in the course materials\n\n{html.escape(ans.answer)}"
    else:
        head = ans.answer
    meta = []
    if ans.sources and not ans.missing_information:
        meta.append("✅ Sources support the answer" if ans.supported else "⚠️ Could not verify the answer against its sources")
    if ans.latency_s:
        meta.append(f"{ans.latency_s:.1f}s" + (" · reranked" if ans.reranked else ""))
    tail = ("\n\n<sub>" + " · ".join(meta) + "</sub>") if meta else ""
    return head + _warn_md(ans.warnings) + tail


# ----------------------------------------------------------------------------
# Handlers
# ----------------------------------------------------------------------------

def handle_add(assistant: Assistant, files) -> tuple[str, list[list], dict, dict, dict]:
    """Add uploaded files. Returns status, table and choice updates for every doc selector."""
    msgs = []
    for f in files or []:
        path = Path(f if isinstance(f, str) else getattr(f, "name", str(f)))
        try:
            rec, status = assistant.add_document(path)
            if status == "duplicate":
                msgs.append(f"↩️ **{rec.name}** is already in the library (same content); not added again.")
            else:
                msgs.append(f"✅ Added **{rec.name}**: {rec.n_pages} pages, {rec.n_chunks} chunks.")
        except UnsupportedFormat as e:
            msgs.append(f"❌ **{path.name}**: {e}")
        except ServiceUnavailable as e:
            msgs.append(f"❌ **{path.name}** was not added because a model service is unavailable ({e}). Try again later.")
        except Exception as e:  # conversion errors etc.
            msgs.append(f"❌ **{path.name}** could not be processed: {type(e).__name__}: {e}")
    if not msgs:
        msgs.append("No files selected.")
    choices = doc_choices(assistant)
    upd = gr.update(choices=choices)
    return "\n\n".join(msgs), library_table(assistant), upd, gr.update(choices=choices, value=[]), gr.update(choices=choices, value=[])


def handle_remove(assistant: Assistant, doc_id: str | None) -> tuple[str, list[list], dict, dict, dict]:
    if not doc_id:
        msg = "Select a document to remove."
    else:
        rec = assistant.library.get(doc_id)
        name = rec.name if rec else doc_id
        msg = f"🗑️ Removed **{name}** and all of its indexed content." if assistant.remove_document(doc_id) else f"**{name}** was not found."
    choices = doc_choices(assistant)
    return msg, library_table(assistant), gr.update(choices=choices, value=None), gr.update(choices=choices, value=[]), gr.update(choices=choices, value=[])


def handle_ask(assistant: Assistant, question: str, selected_docs: list[str] | None, use_reranker: bool):
    doc_ids = set(selected_docs) if selected_docs else None
    if not assistant.library.list_documents():
        return "### ℹ️ No course materials yet\n\nAdd documents in the **Materials** tab first.", [], ""
    ans = assistant.ask(question, doc_ids=doc_ids, use_reranker=use_reranker)
    return answer_markdown(ans), source_gallery(ans.sources), sources_markdown(ans.sources)


def handle_generate_quiz(assistant: Assistant, selected_docs: list[str] | None, topic: str, n_questions: int, use_reranker: bool):
    """Returns: quiz state (dict), status markdown, then MAX_QUIZ_QUESTIONS radio updates, then feedback md + gallery."""
    doc_ids = set(selected_docs) if selected_docs else None
    hidden = [gr.update(visible=False, choices=[], value=None, label="")] * MAX_QUIZ_QUESTIONS
    try:
        quiz = assistant.generate_quiz(doc_ids=doc_ids, topic=topic, n_questions=int(n_questions), use_reranker=use_reranker)
    except QuizError as e:
        return None, f"### ⚠️ {html.escape(str(e))}", *hidden, "", []
    radios = []
    for i in range(MAX_QUIZ_QUESTIONS):
        if i < len(quiz.questions):
            q = quiz.questions[i]
            labels = [f"{'ABCDEF'[j]}. {opt}" for j, opt in enumerate(q.options)]
            radios.append(gr.update(visible=True, label=f"Q{q.number}. {q.question}", choices=labels, value=None))
        else:
            radios.append(gr.update(visible=False, choices=[], value=None, label=""))
    scope = ", ".join(quiz.doc_names)
    status = (f"**Quiz {quiz.quiz_id}** — {len(quiz.questions)} questions from {html.escape(scope)}"
              + (f", topic *{html.escape(quiz.topic)}*" if quiz.topic else "")
              + ". Answers stay hidden until you submit or reveal." + _warn_md(quiz.warnings))
    return quiz.to_dict(), status, *radios, "", []


def _selected_index(value: str | None) -> int | None:
    if not value:
        return None
    letter = value.split(".", 1)[0].strip()
    return "ABCDEF".index(letter) if letter in "ABCDEF" else None


def handle_submit_quiz(assistant: Assistant, quiz_state: dict | None, *answers):
    if not quiz_state:
        return "Generate a quiz first.", []
    quiz = Quiz.from_dict(quiz_state)
    responses = {q.number: _selected_index(answers[i] if i < len(answers) else None) for i, q in enumerate(quiz.questions)}
    result = assistant.grade_quiz(quiz, responses)
    lines = [f"## Score: {result.score} / {result.total}  ({result.percent:.0f}%)"]
    gallery: list[tuple[str, str]] = []
    for fb in result.feedback:
        icon = "✅" if fb.correct else ("⬜" if fb.selected_index is None else "❌")
        picked = f"you chose **{fb.selected_letter}**" if fb.selected_letter else "unanswered"
        lines.append(f"### {icon} Q{fb.number}. {html.escape(fb.question)}\n"
                     f"Correct answer: **{fb.correct_letter}. {html.escape(fb.options[fb.correct_index])}** ({picked})\n\n"
                     f"{html.escape(fb.explanation)}\n\n" + sources_markdown(fb.sources))
        for img, cap in source_gallery(fb.sources):
            gallery.append((img, f"Q{fb.number} · {cap}"))
    return "\n\n".join(lines), gallery


def handle_reveal_quiz(assistant: Assistant, quiz_state: dict | None):
    if not quiz_state:
        return "Generate a quiz first.", []
    quiz = Quiz.from_dict(quiz_state)
    lines = ["## Answer key"]
    gallery = []
    for q in quiz.questions:
        lines.append(f"**Q{q.number}. {html.escape(q.question)}** → **{q.answer_letter}. {html.escape(q.options[q.answer_index])}**\n\n"
                     f"{html.escape(q.explanation)}\n\n" + sources_markdown(q.sources))
        for img, cap in source_gallery(q.sources):
            gallery.append((img, f"Q{q.number} · {cap}"))
    return "\n\n".join(lines), gallery


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------

CSS = """
.source-gallery img { background: #fff; }
footer { display: none !important; }
"""


def build_app(assistant: Assistant) -> gr.Blocks:
    mode = "offline mock services" if assistant.services.is_mock else "class model endpoints"
    with gr.Blocks(title="Course Assistant") as demo:
        gr.Markdown(f"# 📚 Course Assistant\nAsk questions and generate practice quizzes from your course materials. "
                    f"Every answer cites the document and page/slide and shows the original slide. <sub>Running on {mode}.</sub>")

        with gr.Tabs():
            # ---------------- Materials ----------------
            with gr.Tab("📁 Materials"):
                gr.Markdown(describe_supported_formats() + " Uploading the same file twice does not duplicate it.")
                with gr.Row():
                    with gr.Column(scale=1):
                        upload = gr.File(label="Add course files", file_count="multiple",
                                         file_types=[".pdf", ".pptx", ".docx", ".txt", ".md"])
                        add_btn = gr.Button("Add to library", variant="primary")
                        remove_dd = gr.Dropdown(label="Remove a document", choices=doc_choices(assistant), value=None)
                        remove_btn = gr.Button("Remove selected", variant="stop")
                    with gr.Column(scale=2):
                        status = gr.Markdown("")
                        table = gr.Dataframe(headers=["Document", "Type", "Pages", "Chunks", "Added"],
                                             value=library_table(assistant), interactive=False, wrap=True, label="Library")

            # ---------------- Ask ----------------
            with gr.Tab("💬 Ask"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ask_docs = gr.CheckboxGroup(label="Limit to documents (optional)", choices=doc_choices(assistant), value=[])
                        ask_rerank = gr.Checkbox(label="Use multimodal reranker", value=assistant.settings.use_reranker)
                    with gr.Column(scale=3):
                        question = gr.Textbox(label="Your question", lines=2,
                                              placeholder="e.g. Find the meme about vibe coding on prod and explain it")
                        ask_btn = gr.Button("Ask", variant="primary")
                        answer_md = gr.Markdown(label="Answer")
                        answer_gallery = gr.Gallery(label="Supporting slides / pages", columns=2, height=420,
                                                    object_fit="contain", elem_classes=["source-gallery"])
                        with gr.Accordion("Source excerpts", open=True):
                            sources_md = gr.Markdown()

            # ---------------- Quiz ----------------
            with gr.Tab("📝 Quiz"):
                with gr.Row():
                    with gr.Column(scale=1):
                        quiz_docs = gr.CheckboxGroup(label="Course material (none selected = all)", choices=doc_choices(assistant), value=[])
                        topic = gr.Textbox(label="Topic filter (optional)", placeholder="e.g. prompt engineering")
                        n_q = gr.Slider(1, MAX_QUIZ_QUESTIONS, value=5, step=1, label="Number of questions")
                        quiz_rerank = gr.Checkbox(label="Use multimodal reranker", value=assistant.settings.use_reranker)
                        gen_btn = gr.Button("Generate quiz", variant="primary")
                    with gr.Column(scale=3):
                        quiz_status = gr.Markdown()
                        quiz_state = gr.State(None)
                        radios = [gr.Radio(choices=[], label="", visible=False) for _ in range(MAX_QUIZ_QUESTIONS)]
                        with gr.Row():
                            submit_btn = gr.Button("Submit answers", variant="primary")
                            reveal_btn = gr.Button("Reveal answers")
                        feedback_md = gr.Markdown()
                        feedback_gallery = gr.Gallery(label="Sources for the answers", columns=2, height=360,
                                                      object_fit="contain", elem_classes=["source-gallery"])

        # ---------------- Wiring ----------------
        lib_outputs = [status, table, remove_dd, ask_docs, quiz_docs]
        add_btn.click(lambda files: handle_add(assistant, files), inputs=[upload], outputs=lib_outputs)
        remove_btn.click(lambda d: handle_remove(assistant, d), inputs=[remove_dd], outputs=lib_outputs)
        ask_inputs = [question, ask_docs, ask_rerank]
        ask_btn.click(lambda q, d, r: handle_ask(assistant, q, d, r), inputs=ask_inputs, outputs=[answer_md, answer_gallery, sources_md])
        question.submit(lambda q, d, r: handle_ask(assistant, q, d, r), inputs=ask_inputs, outputs=[answer_md, answer_gallery, sources_md])
        gen_btn.click(lambda d, t, n, r: handle_generate_quiz(assistant, d, t, n, r), inputs=[quiz_docs, topic, n_q, quiz_rerank],
                      outputs=[quiz_state, quiz_status, *radios, feedback_md, feedback_gallery])
        submit_btn.click(lambda s, *a: handle_submit_quiz(assistant, s, *a), inputs=[quiz_state, *radios],
                         outputs=[feedback_md, feedback_gallery])
        reveal_btn.click(lambda s: handle_reveal_quiz(assistant, s), inputs=[quiz_state], outputs=[feedback_md, feedback_gallery])
    return demo
