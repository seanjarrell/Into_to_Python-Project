import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import io
import re
 
st.set_page_config(page_title="UNF Receipt Scanner", page_icon="🧾", layout="centered")
 
# Simple UNF-themed styling — navy and white
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f4f6f9; }
[data-testid="stHeader"] { background: transparent; }
.unf-header {
    background: #002f6c;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.unf-header-text h2 { color: white; margin: 0; font-size: 1.4rem; }
.unf-header-text p  { color: #a8c4e8; margin: 0; font-size: 0.85rem; }
.card {
    background: white;
    border-radius: 10px;
    padding: 1.25rem;
    margin: 1rem 0;
    border: 1px solid #dde3ed;
}
.stButton > button {
    background: #002f6c !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
    width: 100%;
    height: auto !important;
    padding: 0.6rem 1rem !important;
}
.stButton > button:hover { background: #00408a !important; }
[data-testid="stDownloadButton"] > button {
    background: white !important;
    color: #002f6c !important;
    border: 2px solid #002f6c !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100%;
    height: auto !important;
}
[data-baseweb="tab-list"] { background: white !important; border-radius: 8px !important; border: 1px solid #dde3ed !important; }
[aria-selected="true"][data-baseweb="tab"] { background: #002f6c !important; color: white !important; }
[data-baseweb="tab"] { color: #555 !important; font-weight: 600 !important; }
hr { border-color: #dde3ed !important; }
</style>
""", unsafe_allow_html=True)
 
# Header with UNF logo
col_logo, col_text = st.columns([1, 4])
with col_logo:
    st.image("UNF_Logo.png", width=80)
with col_text:
    st.markdown("""
    <div style="padding-top:0.4rem">
        <h2 style="color:#002f6c; margin:0; font-size:1.4rem;">Receipt Scanner</h2>
        <p style="color:#555; margin:0; font-size:0.85rem;">Intro to Python · University of North Florida</p>
    </div>
    """, unsafe_allow_html=True)
 
st.divider()
 
# Load models once
@st.cache_resource
def load_engine():
    from yolo_engine import ReceiptEngine
    return ReceiptEngine(conf=0.30, buffer=20)
 
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False, verbose=False)
 
engine = load_engine()
reader = load_ocr()
 
# Helper functions
def get_store_name(lines):
    return lines[0].title() if lines else "UnknownStore"
 
def get_date(lines):
    for line in lines:
        m = re.search(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}', line)
        if m:
            return m.group(0)
    return "NoDate"
 
def make_filename(store, date):
    store = re.sub(r'[^a-zA-Z0-9]', '', store)
    date  = re.sub(r'[^0-9\-]', '', date)
    return f"{store}_{date}.jpg"
 
def resize(image, max_px=2000):
    w, h = image.size
    if max(w, h) <= max_px:
        return image
    scale = max_px / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
 
# Scan function
def scan(source):
    source.seek(0)
    image = resize(Image.open(source).convert("RGB"))
 
    with st.spinner("Detecting receipts..."):
        crops = engine.detect_and_crop_all(image, return_pil=True) or [image]
 
    st.success(f"Found {len(crops)} receipt(s)!")
 
    for i, crop in enumerate(crops):
        with st.spinner(f"Reading receipt {i + 1}..."):
            results = reader.readtext(np.array(crop))
 
        lines = [t.strip() for (_, t, c) in results if c > 0.35]
 
        st.subheader(f"Receipt {i + 1}")
        col1, col2 = st.columns(2)
 
        with col1:
            st.image(crop, use_container_width=True)
 
        with col2:
            store = st.text_input("Store name", get_store_name(lines), key=f"s{i}")
            date  = st.text_input("Date",       get_date(lines),       key=f"d{i}")
            fname = make_filename(store, date)
            st.caption(f"📄 Will save as: {fname}")
 
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=93)
            st.download_button("💾 Save Receipt", buf.getvalue(), fname, "image/jpeg", key=f"dl{i}")
 
        st.divider()
 
# Input tabs
tab1, tab2 = st.tabs(["📷  Camera", "📁  Upload"])
 
with tab1:
    photo = st.camera_input("Take a photo", label_visibility="collapsed")
    if photo:
        scan(photo)
 
with tab2:
    upload = st.file_uploader("Upload image", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if upload:
        scan(upload)
 
st.caption("Built with YOLO26 + EasyOCR · Intro to Python · UNF")
