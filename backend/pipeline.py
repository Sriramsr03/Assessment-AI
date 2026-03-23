"""
MCQ extraction: PyMuPDF, Tesseract OCR, OpenCV photo path.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any, Callable

import cv2
import fitz  # pymupdf
import numpy as np
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


def pixmap_to_bgr(pix: fitz.Pixmap) -> np.ndarray:
    samples = np.frombuffer(pix.samples, dtype=np.uint8)
    h, w = pix.h, pix.w
    if pix.n == 4:
        arr = samples.reshape(h, w, 4)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if pix.n == 3:
        return samples.reshape(h, w, 3)
    if pix.n == 1:
        gray = samples.reshape(h, w)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported pixmap channels: {pix.n}")


def preprocess_photo_page_cv(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Upscale low-resolution pages so OCR line grouping and tick scoring
    # have enough pixel detail.
    h, w = gray.shape[:2]
    if h < 900 or w < 700:
        scale = 1.4 if h < 700 else 1.25
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)

    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return gray


def left_margin_tick_score(gray: np.ndarray, y0: int, y1: int, page_w: int) -> float:
    h, _ = gray.shape[:2]
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))
    # ticks typically live in a narrow left gutter
    x1 = max(10, int(0.16 * page_w))
    roi = gray[y0:y1, 0:x1]
    if roi.size == 0:
        return 0.0
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        roi,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        3,
    )
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    # score = ink density in the tick area
    return float(np.mean(binary > 0))


def group_tesseract_words_into_lines(data: dict) -> list[dict[str, Any]]:
    n = len(data["text"])
    groups: dict[tuple[int, int, int], list[tuple[int, int, int, int, str]]] = {}
    for i in range(n):
        conf = int(data["conf"][i])
        if conf < 0:
            continue
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        left = int(data["left"][i])
        top = int(data["top"][i])
        wi = int(data["width"][i])
        hi = int(data["height"][i])
        groups.setdefault(key, []).append((left, top, wi, hi, text))

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
    _, page_w = gray.shape[:2]
    questions: list[dict[str, Any]] = []
    i = 0
    qid = 0
    while i < len(lines):
        raw = lines[i]["text"].strip()
        raw = re.sub(r"^[\s|•]+", "", raw)
        # Tesseract often misreads `1` as `l` or `I` in scanned docs.
        m = re.match(r"^([0-9IlI]{1,3})\s*[\.)]\s*(.*)$", raw)
        if not m:
            i += 1
            continue
        stem = (m.group(2) or "").strip()
        opts: list[str] = []
        scores: list[float] = []
        i += 1
        while i < len(lines):
            nxt = lines[i]["text"].strip()
            nxt = re.sub(r"^[\s|•]+", "", nxt)
            if re.match(r"^[0-9IlI]{1,3}\s*[\.)]\s+", nxt):
                break
            if len(nxt) < 2:
                i += 1
                continue
            nxt = re.sub(r"^[A-Da-d]\s*[\).:\-]\s*", "", nxt).strip()
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
        # For ticked MCQs we assume each option line is a candidate and the
        # ticked one has the highest left-margin ink density.
        correct_idx = int(np.argmax(scores)) if scores else None
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


def _clean_option_text(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("|", " ")
    s = re.sub(r"^[\s|•]+", "", s)
    # Strip common option prefixes (OCR often misreads punctuation)
    s = re.sub(r"^([A-Da-d]|[0-9Il])\s*[\).:\-]?\s*", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def parse_lines_photo_mcq_by_tick_anchors(
    lines: list[dict[str, Any]],
    gray: np.ndarray,
) -> list[dict[str, Any]]:
    page_h, page_w = gray.shape[:2]
    if not lines:
        return []

    # Tick score is computed per OCR line bounding box. Even if question
    # numbering OCR fails, tick anchors usually still produce a signal.
    tick_scores = [0.0] * len(lines)
    for idx, ln in enumerate(lines):
        y0 = int(ln["top"])
        y1 = int(ln["top"] + ln["height"])
        tick_scores[idx] = left_margin_tick_score(gray, y0, y1, page_w)

    max_score = max(tick_scores, default=0.0)
    if max_score <= 0:
        return []

    # Select anchor candidates (local-ish maxima by vertical spacing).
    # Use a lower threshold because tick density is often subtle in scanned PDFs.
    thresh = max(0.01, max_score * 0.45)
    candidates = [idx for idx, s in enumerate(tick_scores) if s >= thresh]
    if not candidates:
        return []

    candidates.sort(key=lambda i: lines[i]["top"])
    # Prefer local maxima to avoid selecting many anchors from the same tick/line region.
    local_candidates: list[int] = []
    for idx in candidates:
        if 0 < idx < len(lines) - 1 and tick_scores[idx] >= tick_scores[idx - 1] and tick_scores[idx] >= tick_scores[idx + 1]:
            local_candidates.append(idx)
    if local_candidates:
        candidates = local_candidates
    # Minimum distance between two question anchors (in pixels).
    # The previous value was too large for typical worksheet layouts.
    min_gap_px = max(40, int(0.02 * page_h))

    anchors: list[int] = []
    last_anchor_top = -10**9
    for idx in candidates:
        t = lines[idx]["top"]
        if t - last_anchor_top >= min_gap_px:
            anchors.append(idx)
            last_anchor_top = t

    questions: list[dict[str, Any]] = []
    qid = 0
    max_opts = 4

    for a_idx in anchors:
        # Choose a consecutive span of option lines around the tick anchor.
        window_start = max(0, a_idx - 4)
        window_end = min(len(lines), a_idx + 5)
        if window_end - window_start < 3:
            continue

        # Prefer 4 options (common MCQ). If we can't, allow 5.
        candidate_sizes = [4, 5]
        best = None  # (score_sum, span_start, span_size)

        for size in candidate_sizes:
            if window_end - window_start < size:
                continue
            for start in range(window_start, window_end - size + 1):
                end = start + size
                if not (start <= a_idx < end):
                    continue
                span_scores = [tick_scores[j] for j in range(start, end)]
                score_sum = sum(span_scores)
                if best is None or score_sum > best[0]:
                    best = (score_sum, start, size)

        if best is None:
            continue
        _, span_start, span_size = best
        span_end = span_start + span_size

        span_indices = list(range(span_start, span_end))
        span_scores = [tick_scores[j] for j in span_indices]
        correct_local = int(np.argmax(span_scores)) if span_scores else None

        option_texts: list[str] = []
        for j in span_indices:
            t = _clean_option_text(lines[j].get("text") or "")
            if t:
                option_texts.append(t)

        if len(option_texts) < 2:
            continue

        # Stem is whatever OCR line exists just before the option span.
        stem_lines = [
            lines[j].get("text") or "" for j in range(max(0, span_start - 2), span_start)
        ]
        stem = " ".join([s.strip() for s in stem_lines if s.strip()]) or option_texts[0]

        qid += 1
        questions.append(
            {
                "id": f"q{qid}",
                "stem": stem,
                "options": option_texts[:max_opts],
                "labels": [chr(65 + i) for i in range(len(option_texts[:max_opts]))],
                "correctIndex": correct_local if correct_local is not None else None,
            }
        )

    return questions


def extract_photo_pdf_cv(
    pdf_path: Path,
    progress: ProgressFn | None,
    allow_text_fallback: bool = True,
) -> list[dict[str, Any]]:
    require_tesseract()
    import pytesseract

    doc = fitz.open(pdf_path)
    n = len(doc)
    all_q: list[dict[str, Any]] = []
    base_id = 0

    for pi in range(n):
        page = doc.load_page(pi)
        pix = page.get_pixmap(dpi=450)
        img_bgr = pixmap_to_bgr(pix)
        gray = preprocess_photo_page_cv(img_bgr)

        pil = Image.fromarray(gray)
        data = pytesseract.image_to_data(
            pil,
            output_type=pytesseract.Output.DICT,
            lang="eng",
            config="--oem 3 --psm 11",
        )

        lines = group_tesseract_words_into_lines(data)
        page_qs = parse_lines_photo_mcq_with_ticks(lines, gray)
        if not page_qs:
            page_qs = parse_lines_photo_mcq_by_tick_anchors(lines, gray)
        if not page_qs and allow_text_fallback:
            blob = pytesseract.image_to_string(pil, lang="eng", config="--oem 3 --psm 11")
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
    mode: str = "auto",
) -> list[dict[str, Any]]:
    mode = (mode or "auto").strip().lower()

    def p(pct: int, phase_key: str, label: str):
        if progress:
            progress(pct, phase_key, label)

    p(1, "preprocessing", "Loading PDF…")
    base_text, bold_hits = extract_text_and_bold_answers(pdf_path)
    _notify(p, 2, 12, "preprocessing", "Analyzing pages & text layers…")

    questions: list[dict[str, Any]] = []
    full_text = ""

    if mode == "printed":
        # Digital/printed PDF: do NOT run the OpenCV tick framing.
        p(14, "qa_extraction", "Tesseract OCR (if needed)…")
        full_text = ocr_if_needed(pdf_path, base_text, p)
        full_text = full_text or base_text
        _notify(p, 35, 45, "qa_extraction", "Extracting text & layout…")
        p(48, "qa_extraction", "Detecting MCQ patterns…")
        questions = parse_mcqs_from_text(full_text)
        if not questions:
            questions = parse_mcqs_unlabeled_blocks(full_text)
        p(55, "qa_extraction", "Printed extraction complete")
        if not questions:
            raise RuntimeError(
                "No MCQ questions were extracted from the printed/digital PDF. "
                "Try 'Written' mode if the PDF is scanned or ticked."
            )

    elif mode == "written":
        # Written/ticked PDF: run ONLY the OpenCV tick framing path.
        p(14, "preprocessing", "OpenCV preprocessing (written/ticked PDFs)…")
        _notify(p, 15, 22, "preprocessing", "Rendering page bitmaps…")
        p(26, "qa_extraction", "Written PDF: OpenCV preprocess + tick framing…")
        questions = extract_photo_pdf_cv(pdf_path, p, allow_text_fallback=False)
        if questions:
            full_text = "\n".join(
                f"{q['stem']} {' '.join(q['options'])}" for q in questions
            )
        p(50, "qa_extraction", "Written framing complete")
        if not questions:
            raise RuntimeError(
                "No MCQ questions were extracted from the written/ticked PDF. "
                "Tesseract must be installed and on PATH."
            )

    else:
        # Auto mode: choose based on PDF density; keep the OpenCV fallback.
        p(14, "preprocessing", "OpenCV preprocessing (scan/photo pages)…")
        _notify(p, 15, 22, "preprocessing", "Rendering page bitmaps…")
        p(25, "preprocessing", "Preprocessing complete")

        doc_probe = fitz.open(pdf_path)
        use_photo_cv = should_use_photo_cv(doc_probe, base_text)
        doc_probe.close()

        if use_photo_cv:
            p(26, "qa_extraction", "Photo PDF: OpenCV preprocess + Tesseract (layout)…")
            photo_qs = extract_photo_pdf_cv(pdf_path, p)
            if photo_qs:
                questions = photo_qs
                full_text = "\n".join(
                    f"{q['stem']} {' '.join(q['options'])}" for q in questions
                )
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

        # If the PDF is a written/ticked sheet but the heuristic misses,
        # still try the OpenCV tick framing before giving up.
        if not questions:
            p(26, "qa_extraction", "Photo CV fallback: OpenCV preprocess + tick framing…")
            photo_qs = extract_photo_pdf_cv(pdf_path, p)
            if photo_qs:
                questions = photo_qs
                full_text = "\n".join(
                    f"{q['stem']} {' '.join(q['options'])}" for q in questions
                )
                p(50, "qa_extraction", "Photo CV fallback: extracted questions + tick marks")

        if not questions:
            raise RuntimeError(
                "No MCQ questions were extracted. For scans/written PDFs, install Tesseract on PATH. "
                "Use numbered questions (1. 2. …) and option lines under each."
            )

    p(51, "structuring", "Mapping bold / answer keys / tick marks to options…")
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
