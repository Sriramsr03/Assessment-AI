import logging
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pdf2image import convert_from_path

try:
    import easyocr
except ImportError:  # pragma: no cover - runtime dependency
    easyocr = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


QUESTION_RE = re.compile(r"^\s*(\d+)\s*[\.\)]\s*(.+)$")
OPTION_RE = re.compile(r"^\s*([A-Da-d])\s*[\.\)\:\-]?\s*(.+)$")


@dataclass
class LineRegion:
    x: int
    y: int
    w: int
    h: int
    image: np.ndarray


class HandwrittenMCQExtractor:
    """
    Extract handwritten MCQ questions/options and selected answers from PDF files.
    """

    def __init__(
        self,
        dpi: int = 300,
        density_threshold: float = 0.25,
        use_trocr: bool = False,
        debug: bool = False,
        debug_dir: Optional[str] = None,
        poppler_path: Optional[str] = None,
    ) -> None:
        self.dpi = dpi
        self.density_threshold = density_threshold
        self.use_trocr = use_trocr
        self.debug = debug
        self.debug_dir = debug_dir
        self.poppler_path = poppler_path or os.getenv("POPPLER_PATH")
        self.reader = None
        self._load_ocr_engine()

    def _load_ocr_engine(self) -> None:
        if self.use_trocr:
            logger.info("TrOCR mode enabled. Falling back to EasyOCR if unavailable.")
        if easyocr is None:
            raise RuntimeError(
                "easyocr is not installed. Install dependencies before running extraction."
            )
        self.reader = easyocr.Reader(["en"], gpu=False)
        logger.info("EasyOCR reader initialized.")

    def pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if not pdf_path.lower().endswith(".pdf"):
            raise ValueError("Input file must be a PDF.")

        logger.info("Converting PDF to images: %s", pdf_path)
        kwargs = {"dpi": self.dpi}
        if self.poppler_path:
            kwargs["poppler_path"] = self.poppler_path
        try:
            pil_images = convert_from_path(pdf_path, **kwargs)
        except Exception as exc:
            message = str(exc)
            if "Unable to get page count" in message:
                raise RuntimeError(
                    "Poppler is not configured. Install Poppler and either add "
                    "'pdftoppm' to PATH or provide a valid Poppler bin folder path."
                ) from exc
            raise
        if not pil_images:
            raise ValueError("No pages were extracted from the PDF.")

        images: List[np.ndarray] = []
        for index, pil_img in enumerate(pil_images):
            rgb = np.array(pil_img)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            images.append(bgr)
            logger.debug("Converted page %d to image.", index + 1)
        return images

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided for preprocessing.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )
        rotated = self._correct_skew(binary)
        return rotated

    def _correct_skew(self, binary_image: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(binary_image > 0))
        if len(coords) == 0:
            return binary_image

        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        h, w = binary_image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        corrected = cv2.warpAffine(
            binary_image,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        logger.debug("Skew corrected by angle: %.2f", angle)
        return corrected

    def detect_lines(
        self,
        binary_image: np.ndarray,
        min_height: int = 18,
        max_height_ratio: float = 0.2,
    ) -> List[LineRegion]:
        if binary_image is None or binary_image.size == 0:
            return []

        h, w = binary_image.shape[:2]
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions: List[LineRegion] = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            if ch < min_height:
                continue
            if ch > int(h * max_height_ratio):
                continue
            if cw < int(w * 0.2):
                continue

            pad = 4
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w, x + cw + pad)
            y1 = min(h, y + ch + pad)
            crop = binary_image[y0:y1, x0:x1]
            regions.append(LineRegion(x=x0, y=y0, w=x1 - x0, h=y1 - y0, image=crop))

        regions.sort(key=lambda r: (r.y, r.x))
        logger.info("Detected %d candidate text lines.", len(regions))
        return self._merge_nearby_regions(regions)

    def _merge_nearby_regions(
        self,
        regions: List[LineRegion],
        vertical_gap: int = 8,
    ) -> List[LineRegion]:
        if not regions:
            return []

        merged: List[LineRegion] = []
        current = regions[0]

        for reg in regions[1:]:
            current_bottom = current.y + current.h
            if abs(reg.y - current_bottom) <= vertical_gap:
                x0 = min(current.x, reg.x)
                y0 = min(current.y, reg.y)
                x1 = max(current.x + current.w, reg.x + reg.w)
                y1 = max(current.y + current.h, reg.y + reg.h)
                combined = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                combined[current.y - y0 : current.y - y0 + current.h, current.x - x0 : current.x - x0 + current.w] = current.image
                combined[reg.y - y0 : reg.y - y0 + reg.h, reg.x - x0 : reg.x - x0 + reg.w] = np.maximum(
                    combined[
                        reg.y - y0 : reg.y - y0 + reg.h,
                        reg.x - x0 : reg.x - x0 + reg.w,
                    ],
                    reg.image,
                )
                current = LineRegion(x=x0, y=y0, w=x1 - x0, h=y1 - y0, image=combined)
            else:
                merged.append(current)
                current = reg
        merged.append(current)
        return merged

    def extract_text(self, line_image: np.ndarray) -> str:
        if line_image is None or line_image.size == 0:
            return ""

        if self.use_trocr:
            text = self._extract_with_trocr(line_image)
            if text:
                return self._clean_text(text)

        # EasyOCR expects more standard foreground/background for OCR quality.
        ocr_img = cv2.bitwise_not(line_image)
        results = self.reader.readtext(ocr_img, detail=0, paragraph=True)
        text = " ".join(results).strip() if results else ""
        return self._clean_text(text)

    def _extract_with_trocr(self, line_image: np.ndarray) -> str:
        """
        Optional TrOCR path. Returns empty string if dependencies are unavailable.
        """
        try:
            from PIL import Image
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except Exception:
            logger.warning("TrOCR dependencies unavailable. Using EasyOCR fallback.")
            return ""

        try:
            if not hasattr(self, "_trocr_processor"):
                self._trocr_processor = TrOCRProcessor.from_pretrained(
                    "microsoft/trocr-base-handwritten"
                )
                self._trocr_model = VisionEncoderDecoderModel.from_pretrained(
                    "microsoft/trocr-base-handwritten"
                )
            pil_img = Image.fromarray(cv2.bitwise_not(line_image))
            pixel_values = self._trocr_processor(images=pil_img, return_tensors="pt").pixel_values
            generated_ids = self._trocr_model.generate(pixel_values)
            text = self._trocr_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            return text.strip()
        except Exception as exc:
            logger.warning("TrOCR extraction failed: %s", exc)
            return ""

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        text = text.replace(" ,", ",").replace(" .", ".")
        return text

    def detect_tick(self, option_region: np.ndarray) -> bool:
        if option_region is None or option_region.size == 0:
            return False

        total_pixels = option_region.size
        active_pixels = np.count_nonzero(option_region)
        density = active_pixels / float(total_pixels)
        logger.debug("Option mark density: %.4f", density)
        return density > self.density_threshold

    def build_qna(
        self,
        line_entries: List[Tuple[LineRegion, str]],
    ) -> List[Dict[str, object]]:
        questions: List[Dict[str, object]] = []
        current_question: Optional[Dict[str, object]] = None
        selected_candidates: Dict[str, float] = {}

        for region, text in line_entries:
            if not text:
                continue

            qmatch = QUESTION_RE.match(text)
            if qmatch:
                if current_question is not None:
                    current_question["answer"] = (
                        max(selected_candidates, key=selected_candidates.get)
                        if selected_candidates
                        else None
                    )
                    questions.append(current_question)
                current_question = {
                    "question": qmatch.group(2).strip(),
                    "options": {"A": "", "B": "", "C": "", "D": ""},
                    "answer": None,
                }
                selected_candidates = {}
                continue

            if current_question is None:
                continue

            omatch = OPTION_RE.match(text)
            if omatch:
                label = omatch.group(1).upper()
                value = omatch.group(2).strip()
                if label in current_question["options"]:
                    current_question["options"][label] = value

                    # Detect marks from a narrow left-side region near option labels.
                    mark_window = self._extract_mark_window(region.image)
                    score = self._tick_score(mark_window)
                    if self.detect_tick(mark_window):
                        selected_candidates[label] = score
                continue

            # Multi-line continuation for previous option or question.
            self._append_to_last_slot(current_question, text)

        if current_question is not None:
            current_question["answer"] = (
                max(selected_candidates, key=selected_candidates.get)
                if selected_candidates
                else None
            )
            questions.append(current_question)

        return questions

    @staticmethod
    def _append_to_last_slot(question_obj: Dict[str, object], text: str) -> None:
        options: Dict[str, str] = question_obj.get("options", {})  # type: ignore
        last_key = None
        for key in ["D", "C", "B", "A"]:
            if options.get(key):
                last_key = key
                break
        if last_key:
            options[last_key] = f"{options[last_key]} {text}".strip()
        else:
            question_obj["question"] = f"{question_obj['question']} {text}".strip()

    @staticmethod
    def _extract_mark_window(option_image: np.ndarray) -> np.ndarray:
        h, w = option_image.shape[:2]
        left = option_image[:, : max(1, int(w * 0.2))]
        # Dilate to emphasize pen strokes (tick/scribble/filled bubble).
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.dilate(left, kernel, iterations=1)

    @staticmethod
    def _tick_score(mark_region: np.ndarray) -> float:
        active = np.count_nonzero(mark_region)
        total = max(mark_region.size, 1)
        return active / total

    def extract_from_pdf(self, pdf_path: str) -> List[Dict[str, object]]:
        pages = self.pdf_to_images(pdf_path)
        all_entries: List[Tuple[LineRegion, str]] = []

        for page_index, image in enumerate(pages):
            processed = self.preprocess_image(image)
            lines = self.detect_lines(processed)

            if self.debug:
                self._save_debug_visual(page_index, image, lines)

            for line in lines:
                text = self.extract_text(line.image)
                all_entries.append((line, text))

        qna = self.build_qna(all_entries)
        logger.info("Extraction completed. Questions parsed: %d", len(qna))
        return qna

    def _save_debug_visual(
        self,
        page_index: int,
        original_image: np.ndarray,
        lines: List[LineRegion],
    ) -> None:
        if not self.debug:
            return
        output_dir = self.debug_dir or tempfile.gettempdir()
        os.makedirs(output_dir, exist_ok=True)

        canvas = original_image.copy()
        for reg in lines:
            cv2.rectangle(canvas, (reg.x, reg.y), (reg.x + reg.w, reg.y + reg.h), (0, 255, 0), 2)

        path = os.path.join(output_dir, f"debug_page_{page_index + 1}.jpg")
        cv2.imwrite(path, canvas)
        logger.info("Saved debug page visualization: %s", path)


def create_app(
    density_threshold: float = 0.25,
    use_trocr: bool = False,
    debug: bool = False,
    debug_dir: Optional[str] = None,
    poppler_path: Optional[str] = None,
) -> FastAPI:
    app = FastAPI(title="Handwritten MCQ Extractor API")
    extractor = HandwrittenMCQExtractor(
        density_threshold=density_threshold,
        use_trocr=use_trocr,
        debug=debug,
        debug_dir=debug_dir,
        poppler_path=poppler_path,
    )

    @app.post("/extract-handwritten")
    async def extract_handwritten(file: UploadFile = File(...)) -> List[Dict[str, object]]:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        temp_path = ""
        try:
            contents = await file.read()
            if not contents:
                raise HTTPException(status_code=400, detail="Empty PDF file.")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(contents)
                temp_path = temp_pdf.name

            result = extractor.extract_from_pdf(temp_path)
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Failed extracting handwritten MCQ data.")
            raise HTTPException(
                status_code=500,
                detail=f"Extraction failed: {str(exc)}",
            ) from exc
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    return app


app = create_app()

