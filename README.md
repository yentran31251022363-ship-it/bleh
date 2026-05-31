# FaceID · MobileNetV2 Transfer Learning

Streamlit app nhận diện khuôn mặt dùng model MobileNetV2 Transfer Learning, deploy trên Streamlit Cloud.

## Tính năng
- 📷 **Camera live** — chụp ảnh trực tiếp từ trình duyệt
- 🖼️ **Upload ảnh** — hỗ trợ JPG/PNG
- 🏷️ **Quản lý nhãn** — đặt tên thực cho 22 class trong sidebar
- 📊 **Top-K kết quả** — xem phân phối xác suất
- ⚙️ **Ngưỡng tin cậy** — lọc dự đoán thấp

## Cấu trúc

```
faceid_app/
├── app.py                          # Main Streamlit app
├── FaceID_MobileNetV2_Transfer.keras  # Model (upload lên GitHub LFS hoặc HuggingFace)
├── requirements.txt
└── .streamlit/
    └── config.toml
```

## Deploy lên Streamlit Cloud

1. Push repo lên GitHub (bao gồm file `.keras`)
2. Vào [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Chọn repo, branch `main`, file `app.py`
4. Click **Deploy**

> **Lưu ý**: File model ~24MB — nếu vượt giới hạn GitHub (100MB thì ok, 25MB LFS warning),
> dùng [Git LFS](https://git-lfs.com/) hoặc host model trên HuggingFace Hub rồi download trong `app.py`.

## Input / Output Model

| Item | Giá trị |
|------|---------|
| Input shape | `(None, 128, 128, 3)` |
| Preprocessing | Resize → float32 (model tự normalize) |
| Output | Softmax · 22 classes |
| Backbone | MobileNetV2 1.00 @ 128px |

## Đặt tên nhãn

Mở sidebar trong app → nhập tên thực của từng người vào ô `Class 0` → `Class 21` → **Lưu nhãn**.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```
