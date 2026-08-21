import streamlit as st

st.set_page_config(page_title="AIC2026 Retrieval UI", layout="wide")
st.title("AIC2026 — Hệ thống truy xuất video vòng sơ tuyển")
st.markdown(
    """
Dùng menu bên trái để vào từng module. Ở cả ba dạng, người dùng chỉ cần dán nguyên
văn truy vấn do BTC cung cấp vào một ô; backend tự hiểu cấu trúc và trả top 100:
- **Textual KIS** — tìm 100 khung hình theo mô tả văn bản.
- **Q&A** — tìm 100 khung hình và sinh answer cho từng kết quả.
- **TRAKE** — tự tách và căn chỉnh chuỗi khoảnh khắc theo thời gian.
- **Export** — xem/kiểm tra và xuất file `.csv` / `submission.zip` để nộp bài.

Backend API mặc định: `http://127.0.0.1:8000` (chạy `uvicorn app.main:app --port 8000`
trong thư mục `backend/`).
"""
)
