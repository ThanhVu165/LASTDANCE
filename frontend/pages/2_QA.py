import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.title("2. Q&A")
st.caption("Nhập nguyên văn truy vấn Q&A của BTC trong một ô. Hệ thống trả top 100 và answer cho từng khung hình.")

text = st.text_area(
    "Truy vấn",
    height=160,
    placeholder="Ví dụ: Trong video quay cảnh bữa tiệc, người phụ nữ mặc váy đỏ đang cầm ly màu gì?",
)
with st.expander("Tuỳ chọn submission"):
    query_id = st.text_input("Query ID", "query-2-qa")

if st.button("🔍 Tìm kiếm", type="primary") and text.strip():
    try:
        health = requests.get(f"{API}/health", timeout=10).json()
    except requests.RequestException as exc:
        st.error(f"Không kết nối được backend: {exc}")
        st.stop()
    if not health.get("vqa_ready", False):
        st.error(
            "Backend hiện không truy cập được CUDA. Hãy dừng backend và chạy lại "
            "bằng `C:\\LASTDANCE\\backend\\.venv\\Scripts\\python.exe -m uvicorn "
            "app.main:app --host 127.0.0.1 --port 8000 --reload`. "
            f"Runtime hiện tại: {health.get('python_executable', 'không rõ')}"
        )
        st.stop()
    with st.spinner("Đang tìm kiếm..."):
        resp = requests.post(
            f"{API}/search/qa",
            json={"text": text},
            timeout=1800,
        )
    if resp.status_code == 200:
        st.session_state["qa_results"] = resp.json()["results"]
    else:
        st.error(resp.text)

results = st.session_state.get("qa_results", [])
if results:
    st.write(f"**Top {len(results)} kết quả:**")
    for i, r in enumerate(results, start=1):
        c1, c2 = st.columns([1, 3])
        with c1:
            img_url = f"{API}/video/{r['video_id']}/keyframe/{r['local_idx']}"
            st.image(img_url, use_container_width=True)
        with c2:
            st.write(f"**#{i}** · `{r['video_id']}` · frame `{r['frame_id']}` · score {r['score']:.3f}")
            new_answer = st.text_input("Answer", value=r["answer"], key=f"answer_{i}")
            r["answer"] = new_answer
        st.divider()

    if st.button("➕ Thêm tất cả vào submission"):
        rows = [
            {
                "query_id": query_id,
                "query_type": "qa",
                "rank": i,
                "video_id": r["video_id"],
                "frame_ids": [r["frame_id"]],
                "answer": r["answer"],
            }
            for i, r in enumerate(results, start=1)
        ]
        resp = requests.post(
            f"{API}/submission/add",
            json={"query_id": query_id, "rows": rows, "replace": True},
            timeout=60,
        )
        if resp.status_code == 200:
            st.success(f"Đã thêm {len(rows)} dòng vào submission '{query_id}'.")
        else:
            st.error(resp.text)
