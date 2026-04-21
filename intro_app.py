# ============================================================
# Receipt Scanner - Intro to Python
# University of North Florida
# Built with Streamlit, YOLO26, and EasyOCR
# ============================================================

import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import io
import re

# Set up the web page title and layout
st.set_page_config(page_title="UNF Receipt Scanner", page_icon="🧾", layout="centered")

# ---- Page Header ----
# Show the UNF logo next to the app title
logo_col, title_col = st.columns([1, 4])

with logo_col:
    st.image("UNF_Logo.png", width=80)

with title_col:
    st.markdown(
        '<div style="background-color:navy; padding:10px 16px; border-radius:8px;">'
        '<span style="color:white; font-size:1.4rem; font-weight:bold;">Receipt Scanner</span><br>'
        '<span style="color:#ccd9f0; font-size:0.85rem;">Intro to Python &middot; University of North Florida</span>'
        '</div>',
        unsafe_allow_html=True
    )

st.divider()

# ---- Load AI Models ----
# @st.cache_resource means the models only load once, not every time the page refreshes
@st.cache_resource
def load_engine():
    from yolo_engine import ReceiptEngine
    return ReceiptEngine(conf=0.30, buffer=20)

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False, verbose=False)

engine = load_engine()
reader = load_ocr()

# ---- Helper Functions ----

# Get the store name from the first line of OCR text
def get_store_name(text_lines):
    if text_lines:
        return text_lines[0].title()
    return "UnknownStore"

# Search through each line looking for a date like 02/14/2026
def get_date(text_lines):
    for line in text_lines:
        match = re.search(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}', line)
        if match:
            return match.group(0)
    return "NoDate"

# Build a clean filename from the store name and date
def make_filename(store, date):
    store = re.sub(r'[^a-zA-Z0-9]', '', store)
    date  = re.sub(r'[^0-9\-]', '', date)
    return f"{store}_{date}.jpg"

# Shrink large photos so they don't slow things down
def resize_image(image):
    max_size = 2000
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image

# ---- Main Function ----

def scan_receipt(image_source):
    # Step 1: Open and resize the image
    image_source.seek(0)
    image = Image.open(image_source).convert("RGB")
    image = resize_image(image)

    # Step 2: Use YOLO26 to find all receipts in the photo
    with st.spinner("Detecting receipts..."):
        crops = engine.detect_and_crop_all(image, return_pil=True)

    # If nothing was detected, just use the whole image
    if not crops:
        crops = [image]

    st.success(f"Found {len(crops)} receipt(s)!")

    # Step 3: Loop through each receipt that was found
    for i, crop in enumerate(crops):

        # Step 4: Use EasyOCR to read the text on the receipt
        with st.spinner(f"Reading receipt {i + 1}..."):
            ocr_results = reader.readtext(np.array(crop))

        # Only keep text that OCR is confident about (above 35%)
        text_lines = []
        for (box, text, confidence) in ocr_results:
            if confidence > 0.35:
                text_lines.append(text.strip())

        # Step 5: Show the receipt and let the user correct the details
        st.subheader(f"Receipt {i + 1}")

        image_col, info_col = st.columns(2)

        with image_col:
            st.image(crop, use_container_width=True)

        with info_col:
            # Pre-fill the fields with what OCR found, user can edit if needed
            store = st.text_input("Store name", get_store_name(text_lines), key=f"store_{i}")
            date  = st.text_input("Date",       get_date(text_lines),       key=f"date_{i}")

            # Show what the filename will be
            filename = make_filename(store, date)
            st.caption(f"Will save as: {filename}")

            # Step 6: Save button — only downloads when clicked
            image_bytes = io.BytesIO()
            crop.save(image_bytes, format="JPEG", quality=93)

            st.download_button(
                label="Save Receipt",
                data=image_bytes.getvalue(),
                file_name=filename,
                mime="image/jpeg",
                key=f"save_{i}"
            )

        st.divider()

# ---- Camera and Upload Tabs ----
tab_camera, tab_upload = st.tabs(["  Camera  ", "Upload"])

with tab_camera:
    photo = st.camera_input("Take a photo", label_visibility="collapsed")
    if photo:
        scan_receipt(photo)

with tab_upload:
    uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file:
        scan_receipt(uploaded_file)

st.caption("Built with YOLO26 + EasyOCR · Intro to Python · UNF")
