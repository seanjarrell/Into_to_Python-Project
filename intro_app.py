import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import io
import re

# Streamlit Set up
st.set_page_config(page_title="Receipt Scanner")
st.title("Receipt Scanner")
st.write("Take a photo or upload an image containing one or more receipts.")

# Load in the YOLOv26 Engine. 
#This Engine was created and trained using Google Antigravity for a seperate project that we are both working on for with Dr. Rayfield 
#Therefore AI was used for this training, but it was not used for defining the engine and bringing into the pipeline
@st.cache_resource
def load_engine():
    from yolo_engine import ReceiptEngine
    return ReceiptEngine(conf=0.30, buffer=20)

# Load EasyOCR
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False, verbose=False)

# Load both models when the app starts
engine = load_engine()
reader = load_ocr()

# Helper functions 

def get_store_name(text_lines):
    # The store name is usually the first line on the receipt
    if text_lines:
        return text_lines[0].title()
    return "UnknownStore"

def get_date(text_lines):
    # Search every line for a date pattern
    for line in text_lines:
        match = re.search(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}', line)
        if match:
            return match.group(0)
    return "NoDate"

def make_filename(store, date):
    # Remove special characters so the filename is clean
    store = re.sub(r'[^a-zA-Z0-9]', '', store)
    date  = re.sub(r'[^0-9\-]',    '', date).replace('/', '-')
    return f"{store}_{date}.jpg"

def scan_receipt(image_file):
    # Read the uploaded image
    image_file.seek(0)
    image = Image.open(image_file).convert("RGB")

    # Use YOLO26 to find all receipts in the image
    crops = engine.detect_and_crop_all(image, return_pil=True)

    # If YOLO finds nothing, treat the whole image as one receipt
    if not crops:
        crops = [image]

    st.success(f"Found {len(crops)} receipt(s)!")

    # Loop through each detected receipt
    for i, crop in enumerate(crops):
        st.subheader(f"Receipt {i + 1}")

        # Run OCR to read the text on this receipt
        text_results = reader.readtext(np.array(crop))

        # Pull out just the text from each OCR result
        text_lines = []
        for (box, text, confidence) in text_results:
            if confidence > 0.35:
                text_lines.append(text.strip())

        # Extract store name and date
        store    = get_store_name(text_lines)
        date     = get_date(text_lines)
        filename = make_filename(store, date)

        # Show the cropped receipt image
        st.image(crop, caption=filename, use_column_width=True)

        # Show what was found
        st.write(f"**Store:** {store}")
        st.write(f"**Date:** {date}")
        st.write(f"**Filename:** {filename}")

        # Save the image to memory and offer a download button
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=93)

        st.download_button(
            label=f" Download {filename}",
            data=buf.getvalue(),
            file_name=filename,
            mime="image/jpeg",
            key=f"download_{i}"
        )

        st.divider()

# Camera and upload input
tab_camera, tab_upload = st.tabs([" Camera", " Upload"])

with tab_camera:
    photo = st.camera_input("Take a picture of your receipt(s)")
    if photo:
        scan_receipt(photo)

with tab_upload:
    upload = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"])
    if upload:
        scan_receipt(upload)
