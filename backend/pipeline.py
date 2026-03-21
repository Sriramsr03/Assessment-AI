"""
MCQ extraction: PyMuPDF, Tesseract OCR, OpenCV photo path, MiniLM structuring.
"""
from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import fitz  # pymupdf
import numpy as np
from PIL import Image

ProgressFn = Callable[[int, str, str], None]
_embedder = None


def require_tesseract() -> None:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
    except Exception as e:
        raise RuntimeError(
            "Tesseract OCR is not installed or not on your PATH. "
            "Written/photo PDFs need it. Windows: https://github.com/UB-Mannheim/tesseract/wiki "
            "Add e.g. C:\\Program Files\\Tesseract-OCR to PATH, restart the terminal."
        ) from e


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _notify(p: ProgressFn | None, lo: int, hi: int, phase_key: str, label: str, t: float = 0.02):
    if not p:
        time.sleep(t)
        return
    for x in range(lo, hi + 1):
        p(x, phase_key, label)
        time.sleep(t * 0.15)


def pixmap_to_bgr(pix: fitz.Pixmap) -> np.ndarray:
    samples = np.frombuffer(pix.samples, dtype=np.uint8)
    h, w = pix.h, pix.w
    if pix.n == 4:
        arr = samples.reshape(h, w, 4)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if pix.n == 3:
        return samples.reshape(h, w, 3)
    if pix.n == 1:
        g = samples.reshape(h, w)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported pixmap channels: {pix.n}")


def preprocess_photo_page_cv(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def left_margin_tick_score(gray: np.ndarray, y0: int, y1: int, page_w: int) -> float:
    h, w = gray.shape[:2]
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))
    x1 = max(8, int(0.24 * page_w))
    roi = gray[y0:y1, 0:x1]
    if roi.size == 0:
        return 0.0
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return float(np.mean(binary > 0))


def group_tesseract_words_into_lines(data: dict) -> list[dict[str, Any]]:
    n = len(data["text"])
    groups: dict[tuple[int, int, int], list[tuple[int, int, int, int, str]]] = {}
    for i in range(n):
        conf = int(data["conf"][i])
        if conf < 0:
            continue
        t = (data["text"][i] or "").strip()
        if not t:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        left = int(data["left"][i])
        top = int(data["top"][i])
        wi = int(data["width"][i])
        hi = int(data["height"][i])
        groups.setdefault(key, []).append((left, top, wi, hi, t))

    lines_out: list[dict[str, Any]] = []
    for key in sorted(groups.keys()):
        parts = sorted(groups[key], key=lambda x: x[0])
        left = min(p[0] for p in parts)
        top = min(p[1] for p in parts)
        right = max(p[0] + p[2] for p in parts)
        bottom = max(p[1] + p[3] for p in parts)
        text = " ".join(p[4] for p in parts)
        lines_out.append(
            {
                "left": left,
                "top": top,
                "width": right - left,
                "height": bottom - top,
                "text": text,
            }
        )
    return lines_out


def parse_lines_photo_mcq_with_ticks(
    lines: list[dict[str, Any]],
    gray: np.ndarray,
) -> list[dict[str, Any]]:
    page_h, page_w = gray.shape[:2]
    questions: list[dict[str, Any]] = []
    i = 0
    qid = 0
    while i < len(lines):
        raw = lines[i]["text"].strip()
        raw = re.sub(r"^[\s|•]+", "", raw)
        m = re.match(r"^(\d{1,2})\s*[\.)]\s*(.+)$", raw)
        if not m:
            i += 1
            continue
        stem = m.group(2).strip()
        opts: list[str] = []
        scores: list[float] = []
        i += 1
        while i < len(lines):
            nxt = lines[i]["text"].strip()
            nxt = re.sub(r"^[\s|•]+", "", nxt)
            if re.match(r"^\d{1,2}\s*[\.)]\s+", nxt):
                break
            if len(nxt) < 2:
                i += 1
                continue
            opts.append(nxt)
            y0 = int(lines[i]["top"])
            y1 = int(lines[i]["top"] + lines[i]["height"])
            scores.append(left_margin_tick_score(gray, y0, y1, page_w))
            i += 1
            if len(opts) >= 8:
                break

        if len(opts) < 2:
            continue

        qid += 1
        spread = max(scores) - min(scores) if scores else 0.0
        if spread >= 0.012:
            correct_idx = int(np.argmax(scores))
        else:
            correct_idx = None

        questions.append(
            {
                "id": f"q{qid}",
                "stem": stem,
                "options": opts,
                "labels": [chr(65 + j) for j in range(len(opts))],
                "correctIndex": correct_idx,
            }
        )
    return questions


def extract_photo_pdf_cv(
    pdf_path: Path,
    progress: ProgressFn | None,
) -> list[dict[str, Any]]:
    require_tesseract()
    import pytesseract

    doc = fitz.open(pdf_path)
    n = len(doc)
    all_q: list[dict[str, Any]] = []
    base_id = 0

    for pi in range(n):
        page = doc.load_page(pi)
        pix = page.get_pixmap(dpi=300)
        img_bgr = pixmap_to_bgr(pix)
        gray = preprocess_photo_page_cv(img_bgr)

        pil = Image.fromarray(gray)
        data = pytesseract.image_to_data(
            pil,
            output_type=pytesseract.Output.DICT,
            lang="eng",
            config="--psm 6",
        )

        lines = group_tesseract_words_into_lines(data)
        page_qs = parse_lines_photo_mcq_with_ticks(lines, gray)
        if not page_qs:
            blob = pytesseract.image_to_string(pil, lang="eng", config="--psm 6")
            page_qs = parse_mcqs_unlabeled_blocks(blob)
        for q in page_qs:
            base_id += 1
            q["id"] = f"q{base_id}"
            all_q.append(q)

        if progress:
            progress(
                22 + int(22 * (pi + 1) / max(n, 1)),
                "qa_extraction",
                f"Photo CV + OCR page {pi + 1}/{n}",
            )

    doc.close()
    return all_q


def should_use_photo_cv(doc: fitz.Document, base_text: str) -> bool:
    n = len(doc)
    if n == 0:
        return False
    avg = len(base_text.strip()) / max(n, 1)
    return avg < 120


def preprocess_pdf_page_image(pix_bytes: bytes) -> bytes:
    arr = np.frombuffer(pix_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return pix_bytes
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 14
    )
    ok, buf = cv2.imencode(".png", th)
    return buf.tobytes() if ok else pix_bytes


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
        pix = page.get_pixmap(dpi=200)
        raw = pix.tobytes("png")
        pre = preprocess_pdf_page_image(raw)
        im = Image.open(io.BytesIO(pre))
        try:
            txt = pytesseract.image_to_string(im, lang="eng")
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
    model = _get_embedder()
    stems = [q["stem"] for q in questions]
    if progress:
        progress(62, "structuring", "MiniLM: encoding question stems…")
    model.encode(stems, show_progress_bar=False, batch_size=16)
    if progress:
        progress(72, "structuring", "MiniLM: structuring complete")


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

    p(14, "preprocessing", "OpenCV preprocessing (scan/photo pages)…")
    _notify(p, 15, 22, "preprocessing", "Rendering page bitmaps…")

    p(25, "preprocessing", "Preprocessing complete")

    doc_probe = fitz.open(pdf_path)
    use_photo_cv = should_use_photo_cv(doc_probe, base_text)
    doc_probe.close()

    questions: list[dict[str, Any]] = []
    full_text = ""

    if use_photo_cv:
        p(26, "qa_extraction", "Photo PDF: OpenCV preprocess + Tesseract (layout)…")
        photo_qs = extract_photo_pdf_cv(pdf_path, p)
        if photo_qs:
            questions = photo_qs
            full_text = "\n".join(f"{q['stem']} {' '.join(q['options'])}" for q in questions)
            p(50, "qa_extraction", "Photo CV: extracted questions + tick marks")

    if not questions:
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
        p(28, "qa_extraction", "Fallback: photo CV (OCR layout + tick marks)…")
        photo_fallback = extract_photo_pdf_cv(pdf_path, p)
        if photo_fallback:
            questions = photo_fallback
            full_text = "\n".join(f"{q['stem']} {' '.join(q['options'])}" for q in questions)
            p(50, "qa_extraction", "Photo CV: extracted questions + tick marks")

    if not questions:
        raise RuntimeError(
            "No MCQ questions were extracted. For scans, install Tesseract on PATH. "
            "Use numbered questions (1. 2. …) and option lines under each."
        )

    p(51, "structuring", "Mapping bold / answer keys / ticks to options…")
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
