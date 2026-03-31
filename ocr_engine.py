from functools import lru_cache

import cv2
import easyocr
import numpy as np


@lru_cache(maxsize=1)
def _get_easyocr_reader():
    # Load once per process to avoid re-initialization overhead on Streamlit reruns.
    return easyocr.Reader(["en"], gpu=False)


def segment_lines(image: np.ndarray):
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)

    contours, _ = cv2.findContours(
        detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    line_images = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 15:  # filter noise
            line = image[y : y + h, x : x + w]
            line_images.append((y, line))

    line_images = sorted(line_images, key=lambda x: x[0])  # top to bottom
    return [img for _, img in line_images]


def extract_text_lines(image: np.ndarray):
    reader = _get_easyocr_reader()
    lines = segment_lines(image)

    if not lines:
        lines = [image]

    full_text = []
    line_confidences = []

    for line in lines:
        result = reader.readtext(line)
        if not result:
            continue

        line_text = " ".join([res[1] for res in result if len(res) >= 2]).strip()
        confs = [float(res[2]) for res in result if len(res) >= 3]
        if line_text:
            full_text.append(line_text)
        if confs:
            line_confidences.append(float(np.mean(confs)))

    text = "\n".join(full_text).strip()
    confidence = float(np.mean(line_confidences)) if line_confidences else 0.0
    return text, confidence


def easyocr_extract(image: np.ndarray):
    return extract_text_lines(image)


def hybrid_ocr(image: np.ndarray):
    # Backward-compatible function name; now EasyOCR-only.
    try:
        text, confidence = easyocr_extract(image)
        return {"text": text, "confidence": confidence, "source": "easyocr"}
    except Exception:
        return {"text": "", "confidence": 0.0, "source": "easyocr"}