import json
import os
import re
from typing import Any, Dict, List

import google.generativeai as genai
import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image


# Load environment variables from .env in project root (if present).
load_dotenv()


def _extract_json_array(text: str):
    text = text.strip()
    if not text:
        return []

    # Strip markdown fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Try direct parse first.
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except Exception:
        pass

    # Fallback: parse first JSON array-like block.
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _normalize_mcqs(mcqs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for item in mcqs:
        question = str(item.get("question", "")).strip()
        options = item.get("options", {})
        if not isinstance(options, dict):
            continue
        clean_options = {}
        for key, value in options.items():
            k = str(key).lower().strip()
            if k in {"a", "b", "c", "d"}:
                clean_options[k] = str(value).strip()
        answer = str(
            item.get("answer", item.get("correct_answer", item.get("correct_option", "")))
        ).lower().strip()
        if answer not in {"a", "b", "c", "d"}:
            answer = ""

        if question and len(clean_options) >= 2:
            normalized.append(
                {"question": question, "options": clean_options, "answer": answer}
            )
    return normalized


def _pick_working_model(preferred: str = "gemini-1.5-flash"):
    """
    Select a usable Gemini model for generate_content.
    """
    candidates = [
        preferred,
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro-latest",
        "gemini-1.5-pro",
    ]

    def _can_generate(model_name: str) -> bool:
        try:
            model = genai.GenerativeModel(model_name)
            # Probe with minimal token usage to verify this model truly supports generate_content.
            _ = model.generate_content(
                "ping",
                generation_config={"max_output_tokens": 1, "temperature": 0},
            )
            return True
        except Exception:
            return False

    # Try known candidates first (must pass real generate_content probe).
    for name in candidates:
        if _can_generate(name):
            return genai.GenerativeModel(name)

    # Last resort: discover models available to this API key.
    try:
        for model_info in genai.list_models():
            methods = getattr(model_info, "supported_generation_methods", []) or []
            model_name = getattr(model_info, "name", "")
            if "generateContent" in methods and model_name:
                # list_models often returns names like "models/gemini-1.5-flash"
                if _can_generate(model_name):
                    return genai.GenerativeModel(model_name)
    except Exception:
        pass

    raise RuntimeError(
        "No compatible Gemini model found for this API key/project. "
        "Check key permissions and enabled Gemini API."
    )


def _to_pil_images(images: List[np.ndarray]) -> List[Image.Image]:
    pil_images: List[Image.Image] = []
    for img in images[:3]:
        if img is None or getattr(img, "size", 0) == 0:
            continue
        if len(img.shape) == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(rgb))
    return pil_images


def extract_mcqs_with_gemini(
    raw_ocr_text: str, images: List[np.ndarray] | None = None, api_key: str = ""
) -> List[Dict[str, Any]]:
    key = api_key.strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError("Gemini API key missing. Set GEMINI_API_KEY in .env or environment.")

    if not raw_ocr_text or not raw_ocr_text.strip():
        return []

    genai.configure(api_key=key)
    model = _pick_working_model("gemini-1.5-flash")

    prompt = f"""
You are an OCR + document understanding assistant for MCQ sheets.
Use BOTH the noisy OCR text and image(s) to reconstruct MCQs accurately.
Important: if a tick/check mark indicates a selected/correct option, capture it.

Rules:
1) Return ONLY valid JSON array.
2) Each item must be:
   {{
     "question": "string",
     "options": {{
       "a": "string",
       "b": "string",
       "c": "string",
       "d": "string"
     }},
     "answer": "a|b|c|d|"
   }}
3) Include all 4 options whenever visible in image.
4) Set "answer" to the option letter if clearly ticked/marked; else empty string.
5) Fix obvious OCR noise where reasonable, but do not invent unrelated content.
6) If no MCQ can be inferred, return [].

OCR INPUT:
{raw_ocr_text}
""".strip()

    pil_images = _to_pil_images(images or [])
    content_parts = [prompt] + pil_images if pil_images else [prompt]
    response = model.generate_content(content_parts)
    text = getattr(response, "text", "") or ""
    parsed = _extract_json_array(text)
    return _normalize_mcqs(parsed)

