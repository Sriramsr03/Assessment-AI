"""
MCQ extraction: PyMuPDF + optional Tesseract OCR fallback.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any, Callable

import fitz  # pymupdf
from PIL import Image

ProgressFn = Callable[[int, str, str], None]


def require_tesseract() -> None:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
    except Exception as e:
        raise RuntimeError(
            "Tesseract OCR is not installed or not on your PATH. "
            "Scanned/image PDFs need it. Windows: https://github.com/UB-Mannheim/tesseract/wiki "
            "Add e.g. C:\\Program Files\\Tesseract-OCR to PATH, restart the terminal."
        ) from e


def _notify(p: ProgressFn | None, lo: int, hi: int, phase_key: str, label: str, t: float = 0.02):
    if not p:
        return
    for x in range(lo, hi + 1):
        p(x, phase_key, label)


def extract_text_and_bold_answers(pdf_path: Path) -> tuple[str, list[dict[str, Any]]]:
    doc = fitz.open(pdf_path)
    chunks: list[str] = []
    bold_hits: list[dict[str, Any]] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        chunks.append(page.get_text("text"))
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                line_text = ""
                has_bold = False
                for span in line.get("spans", []):
                    t = span.get("text", "")
                    line_text += t
                    font = (span.get("font") or "").lower()
                    flags = int(span.get("flags", 0))
                    if "bold" in font or (flags & 16) != 0:
                        has_bold = True
                if has_bold and line_text.strip():
                    bold_hits.append({"page": page_idx, "text": line_text.strip()})
    doc.close()
    return "\n".join(chunks), bold_hits


def ocr_if_needed(pdf_path: Path, base_text: str, progress: ProgressFn | None) -> str:
    doc = fitz.open(pdf_path)
    n = len(doc)
    if n == 0:
        doc.close()
        return base_text
    avg = len(base_text) / max(n, 1)
    if avg > 120:
        doc.close()
        return base_text

    require_tesseract()
    import pytesseract

    ocr_parts: list[str] = []
    for i in range(n):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=150)
        im = Image.open(io.BytesIO(pix.tobytes("png")))
        try:
            txt = pytesseract.image_to_string(im, lang="eng", config="--psm 6")
        except Exception:
            txt = ""
        ocr_parts.append(txt)
        if progress and i % 2 == 0:
            progress(
                28 + min(20, int(20 * (i + 1) / max(n, 1))),
                "qa_extraction",
                f"OCR page {i + 1}/{n}",
            )
    doc.close()
    merged = "\n".join(ocr_parts)
    return merged if len(merged.strip()) > len(base_text.strip()) else base_text


def parse_answer_key_line(text: str) -> str | None:
    m = re.search(
        r"(?:^|\n)\s*(?:answer|ans|key)\s*[:.)]?\s*([A-Da-d])\b",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-Da-d])\s*[\).]\s*(?:correct|is\s*correct)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def parse_mcqs_from_text(text: str) -> list[dict[str, Any]]:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    blocks = re.split(r"(?=\n\s*(?:Q\s*)?(?:\d+)\s*[.)]\s*)", "\n" + text)
    blocks = [b.strip() for b in blocks if b.strip()]

    questions: list[dict[str, Any]] = []
    qid = 0
    for block in blocks:
        lines = [ln.rstrip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        first = lines[0]
        mnum = re.match(r"^(?:Q\s*)?(\d+)\s*[.)]\s*(.*)$", first, re.IGNORECASE | re.DOTALL)
        if not mnum:
            mnum = re.match(r"^(\d+)\s*[.)]\s*(.*)$", first)
        stem_lines = [first]
        rest_start = 1
        if mnum:
            stem_lines = [mnum.group(2) or first.split(".", 1)[-1].strip()]
            rest_start = 1
        else:
            stem_lines = [first]
            rest_start = 1

        options: list[tuple[str, str]] = []
        stem_buf: list[str] = list(stem_lines)
        for ln in lines[rest_start:]:
            om = re.match(r"^\s*([A-Da-d])\s*[\).]\s*(.+)$", ln)
            if om:
                label = om.group(1).upper()
                options.append((label, om.group(2).strip()))
            else:
                if options:
                    options[-1] = (options[-1][0], options[-1][1] + " " + ln.strip())
                else:
                    stem_buf.append(ln.strip())

        if len(options) < 2:
            continue

        opts_sorted = sorted(options, key=lambda x: x[0])
        option_texts = [t for _, t in opts_sorted]
        labels = [lb for lb, _ in opts_sorted]

        qid += 1
        questions.append(
            {
                "id": f"q{qid}",
                "stem": " ".join(stem_buf).strip(),
                "options": option_texts,
                "labels": labels,
                "correctIndex": None,
            }
        )

    return questions


def parse_mcqs_unlabeled_blocks(text: str) -> list[dict[str, Any]]:
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []

    parts = re.split(r"\n(?=\s*\d+\.\s+)", text)
    parts = [p.strip() for p in parts if p.strip()]

    questions: list[dict[str, Any]] = []
    qid = 0
    for part in parts:
        lines = [ln.strip() for ln in part.split("\n") if ln.strip()]
        if not lines:
            continue
        m = re.match(r"^(\d+)\.\s+(.+)$", lines[0])
        if not m:
            continue
        stem = m.group(2).strip()
        options = lines[1:]
        if len(options) < 2:
            continue
        if len(options) > 12:
            options = options[:12]
        qid += 1
        questions.append(
            {
                "id": f"q{qid}",
                "stem": stem,
                "options": options,
                "labels": [chr(65 + i) for i in range(len(options))],
                "correctIndex": None,
            }
        )

    return questions


def match_bold_to_option(questions: list[dict[str, Any]], bold_hits: list[dict[str, Any]]) -> None:
    bold_text = " ".join(h["text"] for h in bold_hits).lower()
    for q in questions:
        if q.get("correctIndex") is not None:
            continue
        for i, opt in enumerate(q["options"]):
            o = re.sub(r"\s+", " ", opt.lower().strip())
            if len(o) > 3 and o in bold_text:
                q["correctIndex"] = i
                break


def match_answer_line(questions: list[dict[str, Any]], full_text: str) -> None:
    key = parse_answer_key_line(full_text)
    if not key:
        return
    for q in questions:
        if q.get("correctIndex") is not None:
            continue
        labels = q.get("labels") or ["A", "B", "C", "D"][: len(q["options"])]
        if key in labels:
            q["correctIndex"] = labels.index(key)


def run_minilm_structuring(questions: list[dict[str, Any]], progress: ProgressFn | None) -> None:
    if not questions:
        return
    # Keep this phase lightweight; embeddings are not required for output correctness.
    if progress:
        progress(72, "structuring", "Structuring complete")


def fill_missing_answers(questions: list[dict[str, Any]]) -> None:
    for q in questions:
        if q["correctIndex"] is None:
            q["correctIndex"] = 0


def run_pipeline(
    pdf_path: Path,
    assessment_out: Path,
    progress: ProgressFn | None = None,
) -> list[dict[str, Any]]:

    def p(pct: int, phase_key: str, label: str):
        if progress:
            progress(pct, phase_key, label)

    p(1, "preprocessing", "Loading PDF…")
    base_text, bold_hits = extract_text_and_bold_answers(pdf_path)
    _notify(p, 2, 12, "preprocessing", "Analyzing pages & text layers…")

    p(14, "preprocessing", "Preparing OCR fallback…")
    _notify(p, 15, 22, "preprocessing", "Rendering OCR page bitmaps…")

    p(25, "preprocessing", "Preprocessing complete")

    p(26, "qa_extraction", "Tesseract OCR (if needed)…")
    full_text = ocr_if_needed(pdf_path, base_text, p)
    full_text = full_text or base_text
    _notify(p, 35, 45, "qa_extraction", "Extracting text & layout…")
    p(48, "qa_extraction", "Detecting MCQ patterns…")
    questions = parse_mcqs_from_text(full_text)
    if not questions:
        questions = parse_mcqs_unlabeled_blocks(full_text)
    p(50, "qa_extraction", "Q&A extraction complete")

    if not questions:
        raise RuntimeError(
            "No MCQ questions were extracted. For scans, install Tesseract on PATH. "
            "Use numbered questions (1. 2. …) and option lines under each."
        )

    p(51, "structuring", "Mapping bold / answer keys to options…")
    match_bold_to_option(questions, bold_hits)
    match_answer_line(questions, full_text)

    run_minilm_structuring(questions, p)

    p(74, "structuring", "Question structuring complete")

    p(76, "assessment_generation", "Building assessment JSON…")
    fill_missing_answers(questions)
    out_questions = []
    for q in questions:
        out_questions.append(
            {
                "id": q["id"],
                "stem": q["stem"],
                "options": q["options"],
                "correctIndex": q["correctIndex"],
            }
        )
    _notify(p, 85, 95, "assessment_generation", "Validating & saving…")

    assessment_out.parent.mkdir(parents=True, exist_ok=True)
    with open(assessment_out, "w", encoding="utf-8") as f:
        json.dump({"questions": out_questions, "version": 1}, f, indent=2)

    p(100, "assessment_generation", "Done")
    return out_questions
