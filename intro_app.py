#Installing Libraries
import streamlit as st
import easyocr
import numpy as np
from PIL import Image, ImageOps
import io
import re

#Streamlit Page set up
st.set_page_config(page_title="UNF Receipt Scanner", page_icon="🧾", layout="centered")
logo_col, title_col = st.columns([1, 4])
with logo_col:
    st.image("UNF_Logo.png", width=80)
with title_col:
    st.title("Receipt Scanner")
    st.caption("Intro to Python · University of North Florida")
st.divider()

# Loading our trained YOLOv26 model
@st.cache_resource
def load_engine():
    from yolo_engine import ReceiptEngine
    return ReceiptEngine(conf=0.15, buffer=20)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False, verbose=False)
engine = load_engine()
reader = load_ocr()

#Looking for Store Name on the receipts
def get_store_name(text_lines):
    if text_lines:
        return text_lines[0].title()
    return "UnknownStore"

#Scan for dates on the receipts
#This portion is going to look for the symbols that are in the MM/DD?YYYY
def get_date(text_lines):
    for line in text_lines:
        match = re.search(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}', line)
        if match:
            return match.group(0)
    return "NoDate"

#Filename
def make_filename(store, date):
    store = re.sub(r'[^a-zA-Z0-9]', '', store)
    date  = re.sub(r'[^0-9\-]', '', date)
    return f"{store}_{date}.jpg"

# Resizing and readjusting the image to fet a more accurate scan
def resize_image(image):
    image = ImageOps.exif_transpose(image)
    max_size = 1500
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image

#Scanning function
def scan_receipt(image_source):
    image_source.seek(0)
    image = Image.open(image_source).convert("RGB")
    image = resize_image(image)
    with st.spinner("Detecting Receipts"):
        crops = engine.detect_and_crop_all(image, return_pil=True)
    if not crops:
        crops = [image]
    st.success(f"Found {len(crops)} receipt(s)!")
    for i, crop in enumerate(crops):
        with st.spinner(f"Reading receipt {i + 1}..."):
            ocr_results = reader.readtext(np.array(crop))
        text_lines = [text.strip() for (box, text, confidence) in ocr_results if confidence > 0.35]
        st.subheader(f"Receipt {i + 1}")
        image_col, info_col = st.columns(2)
        with image_col:
            st.image(crop, width="stretch")
        with info_col:
            store = st.text_input("Store name", get_store_name(text_lines), key=f"store_{i}")
            date  = st.text_input("Date",       get_date(text_lines),       key=f"date_{i}")
            filename = make_filename(store, date)
            st.caption(f"Will save as: {filename}")
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

#Camera and Upload tab
tab_camera, tab_upload = st.tabs(["  Camera  ", "Upload"])
with tab_camera:
    photo = st.camera_input("Take a photo", label_visibility="collapsed")
    if photo:
        scan_receipt(photo)
with tab_upload:
    uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file:
        scan_receipt(uploaded_file)
