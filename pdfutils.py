import fitz  # PyMuPDF
import numpy as np
import cv2

def pdf_to_images(uploaded_file):
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    images = []

    for page in doc:
        pix = page.get_pixmap()

        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)

        # Convert RGBA → BGR (for OpenCV)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        images.append(img)

    return images