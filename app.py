"""Launch the Course Assistant UI.  Usage: python app.py [--host 0.0.0.0] [--port 7860]"""
from __future__ import annotations

import argparse
import logging

from course_assistant.app_core import Assistant
from course_assistant.ui import CSS, build_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Course Assistant (hybrid RAG over course materials)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("bm25s").setLevel(logging.WARNING)

    assistant = Assistant()
    demo = build_app(assistant)
    # allowed_paths lets the browser load rendered page images from the data directory.
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, css=CSS,
                allowed_paths=[str(assistant.settings.data_dir.resolve())], show_error=True)


if __name__ == "__main__":
    main()
