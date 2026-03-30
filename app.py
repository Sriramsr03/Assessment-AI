import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

from mcq import HandwrittenMCQExtractor, LineRegion


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploaded_pdfs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_pdf(uploaded_file) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = uploaded_file.name.replace(" ", "_")
    output_path = UPLOAD_DIR / f"{timestamp}_{safe_name}"
    with open(output_path, "wb") as file_obj:
        file_obj.write(uploaded_file.getbuffer())
    return output_path


def extract_with_progress(
    extractor: HandwrittenMCQExtractor,
    pdf_path: Path,
    progress_bar,
    progress_text,
) -> List[Dict[str, object]]:
    pages = extractor.pdf_to_images(str(pdf_path))
    total_steps = max(len(pages) * 3 + 2, 1)
    current_step = 0

    def update_progress(message: str) -> None:
        nonlocal current_step
        current_step += 1
        progress = min(current_step / total_steps, 1.0)
        progress_bar.progress(progress)
        progress_text.text(message)

    update_progress("PDF loaded. Starting extraction...")

    all_entries: List[Tuple[LineRegion, str]] = []
    for page_no, image in enumerate(pages, start=1):
        processed = extractor.preprocess_image(image)
        update_progress(f"Page {page_no}: preprocessing complete")

        lines = extractor.detect_lines(processed)
        update_progress(f"Page {page_no}: detected {len(lines)} lines")

        for line in lines:
            text = extractor.extract_text(line.image)
            all_entries.append((line, text))
        update_progress(f"Page {page_no}: OCR complete")

    result = extractor.build_qna(all_entries)
    update_progress("Building question/answer mapping...")
    update_progress("Extraction complete")
    return result


def render_results(results: List[Dict[str, object]]) -> None:
    if not results:
        st.warning("No questions were detected from the uploaded PDF.")
        return

    st.success(f"Extraction finished. Total questions: {len(results)}")
    for idx, item in enumerate(results, start=1):
        st.subheader(f"Question {idx}")
        st.write(item.get("question", ""))
        options = item.get("options", {})
        for label in ["A", "B", "C", "D"]:
            st.write(f"{label}. {options.get(label, '')}")
        st.write(f"Detected Answer: **{item.get('answer')}**")
        st.divider()

    st.json(results)


def main() -> None:
    st.set_page_config(page_title="Handwritten MCQ Extractor", layout="wide")
    st.title("Handwritten MCQ PDF Extractor")
    st.caption("Upload a handwritten MCQ PDF, extract questions/options/answers, and preview results.")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    threshold = st.slider("Tick detection threshold", min_value=0.05, max_value=0.90, value=0.25, step=0.01)
    debug_mode = st.checkbox("Debug mode (save line-box visualizations)", value=False)
    poppler_path = st.text_input(
        "Poppler bin path (Windows, optional)",
        value=os.getenv("POPPLER_PATH", ""),
        placeholder=r"C:\path\to\poppler\Library\bin",
        help="If PATH is not configured, provide folder containing pdftoppm.exe.",
    ).strip()

    if uploaded_file is not None:
        saved_path = save_uploaded_pdf(uploaded_file)
        st.info(f"File saved locally: `{saved_path}`")

        if st.button("Extract", type="primary"):
            progress_bar = st.progress(0.0)
            progress_text = st.empty()
            try:
                extractor = HandwrittenMCQExtractor(
                    density_threshold=threshold,
                    debug=debug_mode,
                    poppler_path=poppler_path or None,
                )
                results = extract_with_progress(extractor, saved_path, progress_bar, progress_text)
                render_results(results)
            except Exception as exc:
                st.error(f"Extraction failed: {exc}")
                if "Poppler" in str(exc) or "Unable to get page count" in str(exc):
                    st.info(
                        "Set Poppler bin path above (folder containing pdftoppm.exe), "
                        "or add Poppler to your system PATH."
                    )
    else:
        st.write("Please upload a PDF file to continue.")


if __name__ == "__main__":
    main()

