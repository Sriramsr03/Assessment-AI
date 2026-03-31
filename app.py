import streamlit as st
import numpy as np
import cv2
from ocr_preprocess import preprocess_array
from ocr_engine import hybrid_ocr
from extractor import extract_mcqs
from gemini_extractor import extract_mcqs_with_gemini
from pdfutils import pdf_to_images
import json

st.set_page_config(page_title="AI MCQ Extractor", layout="wide")

st.title("📄 AI-Based MCQ Extraction System (EasyOCR)")
st.caption("OCR by EasyOCR + optional Gemini 1.5 Flash refinement for MCQ extraction.")

with st.sidebar:
    st.subheader("Gemini Settings")
    use_gemini = st.checkbox("Use Gemini 1.5 Flash for extraction", value=True)

uploaded_file = st.file_uploader(
    "Upload PDF or Image",
    type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file:
    file_type = uploaded_file.type
    images = []

    if file_type == "application/pdf":
        images = pdf_to_images(uploaded_file)
        st.success(f"{len(images)} pages loaded")
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Unable to decode image. Please upload a valid PNG/JPG/JPEG file.")
            st.stop()
        images = [img]
        st.success("1 image loaded")

    if st.button("Run OCR"):
        full_text = ""

        with st.spinner("Processing..."):
            ocr_progress = st.progress(0, text="Starting OCR...")
            total_pages = max(1, len(images))
            for i, img in enumerate(images):
                ocr_progress.progress(
                    int((i / total_pages) * 100),
                    text=f"OCR in progress: page {i+1}/{total_pages}"
                )

                processed = preprocess_array(img)

                result = hybrid_ocr(processed)

                full_text += result["text"] + "\n"

            ocr_progress.progress(100, text="OCR completed")

        st.session_state["ocr_text"] = full_text

    if "ocr_text" in st.session_state:
        st.subheader("📃 Raw OCR Text")
        with st.expander("View OCR text", expanded=True):
            st.text_area(
                "OCR Result",
                value=st.session_state["ocr_text"],
                height=220,
                key="raw_ocr_text_area",
            )

    if "ocr_text" in st.session_state:
        if st.button("Extract MCQs"):
            extraction_progress = st.progress(0, text="Preparing extraction...")
            extraction_progress.progress(35, text="Cleaning OCR text...")
            ocr_text = st.session_state["ocr_text"]

            if use_gemini:
                extraction_progress.progress(60, text="Analyzing with Gemini 1.5 Flash...")
                try:
                    mcqs = extract_mcqs_with_gemini(ocr_text, images=images)
                    if not mcqs:
                        mcqs = extract_mcqs(ocr_text)
                        st.info("Gemini returned empty result. Used regex fallback extraction.")
                except Exception as exc:
                    mcqs = extract_mcqs(ocr_text)
                    st.warning(f"Gemini failed, used regex fallback. Details: {exc}")
            else:
                extraction_progress.progress(60, text="Applying regex extraction...")
                mcqs = extract_mcqs(ocr_text)

            extraction_progress.progress(80, text="Structuring MCQs...")
            extraction_progress.progress(100, text="Extraction completed")

            st.subheader("✅ Extracted MCQs")
            if not mcqs:
                st.warning("No questions detected. Try clearer scan/crop or higher resolution input.")
            for i, q in enumerate(mcqs):
                st.markdown(f"### Q{i+1}: {q['question']}")
                for k, v in q["options"].items():
                    st.write(f"{k}) {v}")
                if q.get("answer"):
                    st.success(f"Detected Answer: {str(q['answer']).upper()}")

            st.subheader("📦 JSON Output")
            st.json(mcqs)

            st.download_button(
                "Download JSON",
                data=json.dumps(mcqs, indent=2),
                file_name="mcqs.json",
                mime="application/json"
            )