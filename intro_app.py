import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import io
import re
 
st.set_page_config(page_title="Receipt Scanner", page_icon="🧾")
 
st.title("🧾 Receipt Scanner")
st.write("Upload or take a photo of a receipt to save it with the right name.")
 
# Load models once and cache them
@st.cache_resource
def load_engine():
    from yolo_engine import ReceiptEngine
    return ReceiptEngine(conf=0.30, buffer=20)
 
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False, verbose=False)
 
engine = load_engine()
reader = load_ocr()
 
# Pull the store name from the first line of the receipt
def get_store_name(lines):
    return lines[0].title() if lines else "UnknownStore"
 
# Search for a date pattern like 02/14/2026
def get_date(lines):
    for line in lines:
        match = re.search(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}', line)
        if match:
            return match.group(0)
    return "NoDate"
 
# Build a clean filename from store name and date
def make_filename(store, date):
    store = re.sub(r'[^a-zA-Z0-9]', '', store)
    date  = re.sub(r'[^0-9\-]', '', date)
    return f"{store}_{date}.jpg"
 
# Shrink large images so OCR doesn't run out of memory
def resize(image, max_px=2000):
    w, h = image.size
    if max(w, h) <= max_px:
        return image
    scale = max_px / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
 
# Main function that runs when the user submits a photo
def scan(source):
    source.seek(0)
    image = Image.open(source).convert("RGB")
    image = resize(image)
 
    with st.spinner("Detecting receipts..."):
        crops = engine.detect_and_crop_all(image, return_pil=True) or [image]
 
    st.success(f"Found {len(crops)} receipt(s)!")
 
    for i, crop in enumerate(crops):
        st.subheader(f"Receipt {i + 1}")
 
        with st.spinner("Reading text..."):
            results = reader.readtext(np.array(crop))
 
        lines = [t.strip() for (_, t, c) in results if c > 0.35]
 
        col1, col2 = st.columns(2)
 
        with col1:
            st.image(crop, use_container_width=True)
 
        with col2:
            store = st.text_input("Store name", get_store_name(lines), key=f"s{i}")
            date  = st.text_input("Date",       get_date(lines),       key=f"d{i}")
            fname = make_filename(store, date)
            st.caption(f"📄 {fname}")
 
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=93)
            st.download_button("💾 Save", buf.getvalue(), fname, "image/jpeg", key=f"dl{i}")
 
        st.divider()
 
# Camera and upload tabs
tab1, tab2 = st.tabs(["📷 Camera", "📁 Upload"])
 
with tab1:
    photo = st.camera_input("Take a photo", label_visibility="collapsed")
    if photo:
        scan(photo)
 
with tab2:
    upload = st.file_uploader("Upload image", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if upload:
        scan(upload)
