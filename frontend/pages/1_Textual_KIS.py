import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.title("1. Textual KIS")
st.caption("Nhập nguyên văn truy vấn của BTC. Hệ thống luôn trả về top 100.")

text = st.text_area(
    "Truy vấn",
    height=140,
    placeholder="Ví dụ: Tìm video về một diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời...",
)
with st.expander("Tuỳ chọn submission"):
    query_id = st.text_input("Query ID", "query-1-kis")

if st.button("🔍 Tìm kiếm", type="primary") and text.strip():
    with st.spinner("Đang tìm kiếm..."):
        resp = requests.post(f"{API}/search/kis", json={"text": text}, timeout=300)
    if resp.status_code == 200:
        st.session_state["kis_results"] = resp.json()["results"]
    else:
        st.error(resp.text)

results = st.session_state.get("kis_results", [])
if results:
    st.write(f"**Top {len(results)} kết quả** (đã xếp hạng theo điểm re-rank):")
    n_cols = 4
    for row_start in range(0, len(results), n_cols):
        cols = st.columns(n_cols)
        for col, (i, r) in zip(cols, enumerate(results[row_start : row_start + n_cols], start=row_start + 1)):
            with col:
                img_url = (
                    f"{API}/video/{r['video_id']}/frame/{r['frame_id']}"
                    if r.get("is_source_frame")
                    else f"{API}/video/{r['video_id']}/keyframe/{r['local_idx']}"
                )
                st.image(img_url, use_container_width=True)
                source_label = " · source-frame" if r.get("is_source_frame") else ""
                st.caption(
                    f"#{i} · {r['video_id']} · frame {r['frame_id']} · "
                    f"{r['score']:.3f}{source_label}"
                )

    if st.button("➕ Thêm tất cả vào submission"):
        rows = [
            {
                "query_id": query_id,
                "query_type": "kis",
                "rank": i,
                "video_id": r["video_id"],
                "frame_ids": [r["frame_id"]],
                "answer": None,
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
