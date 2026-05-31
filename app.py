import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import io

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceID · MobileNetV2",
    page_icon="🧠",
    layout="centered",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e0e0ff;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}

.title-block {
    text-align: center;
    padding: 2rem 0 1rem;
}

.title-block h1 {
    font-size: 2.2rem;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #7c6aff 0%, #4fc3f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}

.title-block p {
    color: #6b7280;
    font-size: 0.9rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.result-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2d2d5e;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
    text-align: center;
}

.result-name {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #7c6aff;
    letter-spacing: 1px;
}

.result-conf {
    font-size: 1rem;
    color: #9ca3af;
    margin-top: 0.4rem;
}

.conf-high { color: #4ade80; }
.conf-mid  { color: #fbbf24; }
.conf-low  { color: #f87171; }

.info-pill {
    display: inline-block;
    background: #1e1e3a;
    border: 1px solid #3d3d7a;
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    font-size: 0.75rem;
    color: #9ca3af;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin: 0.25rem;
}

.top5-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 0.5rem;
}

.top5-table th {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6b7280;
    padding: 0.5rem;
    border-bottom: 1px solid #2d2d5e;
    text-align: left;
}

.top5-table td {
    padding: 0.5rem;
    font-size: 0.9rem;
    border-bottom: 1px solid #1e1e3a;
}

.bar-fill {
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg, #7c6aff, #4fc3f7);
}

.divider {
    border: none;
    border-top: 1px solid #1e1e3a;
    margin: 1.5rem 0;
}

.stButton > button {
    background: linear-gradient(135deg, #7c6aff, #4fc3f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

section[data-testid="stSidebar"] {
    background: #0d0d1a !important;
    border-right: 1px solid #1e1e3a;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE    = 128
NUM_CLASSES = 22

# Default class labels — replace with your actual names in sidebar
DEFAULT_LABELS = [f"Person_{i+1:02d}" for i in range(NUM_CLASSES)]
# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang tải model AI…")
def load_model():
    return tf.keras.models.load_model("FaceID_MobileNetV2_Transfer.keras")

model = load_model()

# ─── Sidebar — Label Manager ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏷️ Tên nhãn (22 classes)")
    st.caption("Nhập tên thực của từng người để hiển thị đúng kết quả.")

    if "labels" not in st.session_state:
        st.session_state.labels = DEFAULT_LABELS.copy()

    edited_labels = []
    for i in range(NUM_CLASSES):
        val = st.text_input(f"Class {i}", value=st.session_state.labels[i], key=f"lbl_{i}", label_visibility="collapsed")
        edited_labels.append(val.strip() or DEFAULT_LABELS[i])

    if st.button("💾 Lưu nhãn"):
        st.session_state.labels = edited_labels
        st.success("Đã lưu!")

    labels = st.session_state.labels

    st.markdown("---")
    st.markdown("### ⚙️ Cài đặt")
    conf_threshold = st.slider("Ngưỡng tin cậy", 0.0, 1.0, 0.5, 0.05)
    top_k = st.selectbox("Hiển thị Top-K", [3, 5, 10], index=1)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
  <h1>⬡ FACE ID</h1>
  <p>MobileNetV2 Transfer Learning · 22 Classes</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<span class="info-pill">128×128 input</span><span class="info-pill">TF 2.x</span><span class="info-pill">Softmax output</span>', unsafe_allow_html=True)

# ─── Inference Helper ─────────────────────────────────────────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = arr[np.newaxis, ...]          # (1, 128, 128, 3)
    return arr

def predict(arr: np.ndarray):
    preds = model.predict(arr, verbose=0)[0]   # (22,)
    top_idx = np.argsort(preds)[::-1]
    return preds, top_idx

def conf_class(c):
    if c >= 0.75: return "conf-high"
    if c >= 0.5:  return "conf-mid"
    return "conf-low"

# ─── Input Mode ───────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)

tab_cam, tab_upload = st.tabs(["📷  Camera Live", "🖼️  Upload Ảnh"])

def show_result(preds, top_idx):
    best_i    = top_idx[0]
    best_name = labels[best_i]
    best_conf = float(preds[best_i])
    cc        = conf_class(best_conf)

    if best_conf < conf_threshold:
        st.warning(f"⚠️ Độ tin cậy thấp ({best_conf:.1%}) — không vượt ngưỡng {conf_threshold:.0%}")
    else:
        st.markdown(f"""
        <div class="result-card">
          <div class="result-name">{best_name}</div>
          <div class="result-conf">Độ tin cậy: <span class="{cc}"><b>{best_conf:.1%}</b></span></div>
        </div>
        """, unsafe_allow_html=True)

    # Top-K table
    st.markdown(f"**Top {top_k} dự đoán**")
    rows_html = ""
    for rank, idx in enumerate(top_idx[:top_k]):
        name = labels[idx]
        conf = float(preds[idx])
        bar_w = int(conf * 180)
        rows_html += f"""
        <tr>
          <td style="color:#6b7280;font-size:0.8rem">{rank+1}</td>
          <td>{name}</td>
          <td style="text-align:right;font-family:Space Mono,monospace">{conf:.1%}</td>
          <td style="width:120px;padding-left:1rem">
            <div style="background:#1e1e3a;border-radius:3px;height:6px;">
              <div class="bar-fill" style="width:{bar_w}px;max-width:100%;"></div>
            </div>
          </td>
        </tr>"""

    st.markdown(f"""
    <table class="top5-table">
      <thead><tr><th>#</th><th>Tên</th><th>Conf.</th><th>Bar</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

# ─── Tab: Camera ──────────────────────────────────────────────────────────────
with tab_cam:
    st.caption("Cho phép trình duyệt truy cập camera. Ảnh không được lưu hay gửi đi ngoài phiên này.")
    cam_img = st.camera_input("📸 Chụp khuôn mặt")

    if cam_img:
        img = Image.open(cam_img)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, caption="Ảnh chụp", use_container_width=True)
        with col2:
            with st.spinner("Đang phân tích…"):
                arr           = preprocess(img)
                preds, top_idx = predict(arr)
            show_result(preds, top_idx)

# ─── Tab: Upload ──────────────────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader("Chọn ảnh (JPG / PNG)", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, caption=uploaded.name, use_container_width=True)
        with col2:
            with st.spinner("Đang phân tích…"):
                arr           = preprocess(img)
                preds, top_idx = predict(arr)
            show_result(preds, top_idx)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.caption("🔒 Xử lý hoàn toàn trên server · Không lưu dữ liệu · MobileNetV2 Transfer Learning")
