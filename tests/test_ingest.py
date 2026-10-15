import shutil
from pathlib import Path

import pytest

from course_assistant import ingest
from course_assistant.ingest import Chunk, UnsupportedFormat, chunk_pages, parse_document
from course_assistant.library import Library
from tests.fixtures import make_docx, make_pdf, make_pptx

needs_libreoffice = pytest.mark.skipif(not ingest.libreoffice_available(), reason="LibreOffice not installed")


def test_parse_pdf_keeps_text_and_renders_each_page(tmp_path):
    pdf = make_pdf(tmp_path / "syllabus.pdf", ["Grading: 40% exams, 60% projects", "Office hours: Tuesday 2pm"])
    pages = parse_document(pdf, "doc1", "syllabus.pdf", tmp_path / "pages")
    assert [p.page_number for p in pages] == [1, 2]
    assert "40% exams" in pages[0].text and "Tuesday" in pages[1].text
    for p in pages:
        assert p.image_path.exists() and p.image_path.suffix == ".png"
        assert p.image_path.with_suffix(".txt").read_text() == p.text


@needs_libreoffice
def test_parse_pptx_converts_slides_and_keeps_notes(tmp_path):
    pptx = make_pptx(
        tmp_path / "week2.pptx",
        [("Vibe Coding", "Never vibe code on prod"), ("Agents", "Tools, memory, planning")],
        notes={1: "Show the meme here"},
    )
    pages = parse_document(pptx, "doc2", "week2.pptx", tmp_path / "pages")
    assert len(pages) == 3  # two text slides + one image-only slide
    assert "Never vibe code on prod" in pages[0].text
    assert "Speaker notes: Show the meme here" in pages[0].text
    assert all(p.image_path.exists() for p in pages)
    chunks = chunk_pages(pages)
    image_only = [c for c in chunks if c.page_number == 3]
    assert len(image_only) == 1 and "no extractable text" in image_only[0].text


@needs_libreoffice
def test_parse_docx_renders_pages(tmp_path):
    d = make_docx(tmp_path / "notes.docx", ["Course policy on late work.", "Submit via Canvas."])
    pages = parse_document(d, "doc3", "notes.docx", tmp_path / "pages")
    assert pages and "late work" in pages[0].text and pages[0].image_path.exists()


def test_parse_docx_text_fallback_without_libreoffice(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "libreoffice_available", lambda: False)
    d = make_docx(tmp_path / "notes.docx", ["Course policy on late work."])
    pages = parse_document(d, "doc3", "notes.docx", tmp_path / "pages")
    assert "late work" in pages[0].text and pages[0].image_path is None


def test_parse_text_splits_into_pseudo_pages(tmp_path):
    txt = tmp_path / "readme.md"
    txt.write_text("\n".join(f"line {i} about topic {i % 5}" for i in range(400)))
    pages = parse_document(txt, "doc4", "readme.md", tmp_path / "pages")
    assert len(pages) > 1 and all(len(p.text) <= ingest.TEXT_PAGE_CHARS for p in pages)
    assert "".join(p.text for p in pages).count("line") == 400


def test_unsupported_extension_explains_accepted_formats(tmp_path):
    f = tmp_path / "data.xlsx"
    f.write_bytes(b"x")
    with pytest.raises(UnsupportedFormat, match="Accepted: PDF"):
        parse_document(f, "x", "data.xlsx", tmp_path / "pages")


def test_chunks_are_page_bounded_with_overlap(tmp_path):
    pdf = make_pdf(tmp_path / "long.pdf", ["word " * 600, "other " * 50])
    pages = parse_document(pdf, "d", "long.pdf", tmp_path / "pages")
    chunks = chunk_pages(pages, chunk_size=400, chunk_overlap=50)
    assert len([c for c in chunks if c.page_number == 1]) > 1
    assert all(len(c.text) <= 400 for c in chunks)
    assert all(c.image_path == pages[c.page_number - 1].image_path for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    roundtrip = Chunk.from_dict(chunks[0].to_dict())
    assert roundtrip == chunks[0]


class RecordingListener:
    def __init__(self):
        self.added: list[str] = []
        self.removed: list[str] = []

    def add_chunks(self, chunks):
        self.added.extend(c.chunk_id for c in chunks)

    def remove_document(self, doc_id):
        self.removed.append(doc_id)


def test_library_add_dedup_remove(settings, tmp_path):
    lib = Library(settings)
    listener = RecordingListener()
    lib.add_listener(listener)
    pdf = make_pdf(tmp_path / "syllabus.pdf", ["Grading policy", "Schedule"])

    rec, status = lib.add_document(pdf)
    assert status == "added" and rec.n_pages == 2 and rec.n_chunks >= 2
    assert Path(rec.stored_path).exists() and (settings.pages_dir / rec.doc_id / "page_0001.png").exists()
    assert len(listener.added) == rec.n_chunks

    # Same content, different filename -> duplicate, nothing re-indexed.
    copy = tmp_path / "syllabus (1).pdf"
    shutil.copyfile(pdf, copy)
    rec2, status2 = lib.add_document(copy)
    assert status2 == "duplicate" and rec2.doc_id == rec.doc_id
    assert len(lib.list_documents()) == 1 and len(listener.added) == rec.n_chunks

    # Manifest persists across instances.
    assert Library(settings).get(rec.doc_id) is not None
    assert len(Library(settings).all_chunks()) == rec.n_chunks

    # Remove purges copy, page images, chunk store and notifies indexes.
    assert lib.remove_document(rec.doc_id) is True
    assert listener.removed == [rec.doc_id]
    assert not Path(rec.stored_path).exists()
    assert not (settings.pages_dir / rec.doc_id).exists()
    assert lib.all_chunks() == [] and lib.list_documents() == []
    assert lib.remove_document(rec.doc_id) is False


def test_library_rejects_unsupported_and_leaves_no_trace(settings, tmp_path):
    lib = Library(settings)
    bad = tmp_path / "x.xlsx"
    bad.write_bytes(b"not a doc")
    with pytest.raises(UnsupportedFormat):
        lib.add_document(bad)
    assert list(settings.uploads_dir.iterdir()) == [] and lib.list_documents() == []


def test_library_rolls_back_when_parse_fails(settings, tmp_path):
    lib = Library(settings)
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4 garbage")
    with pytest.raises(Exception):
        lib.add_document(corrupt)
    assert list(settings.uploads_dir.iterdir()) == [] and list(settings.index_dir.iterdir()) == []
    assert lib.list_documents() == []
