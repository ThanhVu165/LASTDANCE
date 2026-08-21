import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.title("3. TRAKE")
st.caption(
    "Nhập nguyên văn truy vấn TRAKE của BTC trong một ô. Hệ thống tự tách chuỗi "
    "khoảnh khắc, căn chỉnh theo thời gian và trả top 100 tổ hợp."
)

text = st.text_area(
    "Truy vấn",
    height=190,
    placeholder=(
        "Ví dụ: Tìm 4 khoảnh khắc chính khi vận động viên thực hiện cú nhảy: "
        "(1) giậm nhảy, (2) bay qua xà, (3) tiếp đất, (4) đứng dậy."
    ),
)
with st.expander("Tuỳ chọn submission"):
    query_id = st.text_input("Query ID", "query-3-trake")

if st.button("🔍 Tìm kiếm", type="primary") and text.strip():
    with st.spinner("Đang hiểu truy vấn, truy xuất video và căn chỉnh chuỗi..."):
        resp = requests.post(
            f"{API}/search/trake",
            json={"text": text},
            timeout=600,
        )
    if resp.status_code == 200:
        payload = resp.json()
        st.session_state["trake_results"] = payload["results"]
        st.session_state["trake_moments"] = payload["moments"]
    else:
        st.error(resp.text)

moments = st.session_state.get("trake_moments", [])
if moments:
    st.write("**Các khoảnh khắc hệ thống đã tách:**")
    for index, moment in enumerate(moments, start=1):
        st.write(f"{index}. {moment}")

results = st.session_state.get("trake_results", [])
if results:
    st.write(f"**Top {len(results)} tổ hợp kết quả:**")
    for index, result in enumerate(results, start=1):
        st.write(
            f"**#{index}** · video `{result['video_id']}` · "
            f"frames {result['frame_ids']} · score {result['score']:.3f}"
        )
        columns = st.columns(len(result["local_idxs"]))
        for column, local_idx, frame_id in zip(
            columns,
            result["local_idxs"],
            result["frame_ids"],
        ):
            with column:
                image_url = (
                    f"{API}/video/{result['video_id']}/frame/{frame_id}"
                    if result.get("is_source_frames")
                    else f"{API}/video/{result['video_id']}/keyframe/{local_idx}"
                )
                st.image(image_url, use_container_width=True)
        st.divider()

    if st.button("➕ Thêm toàn bộ top 100 vào submission"):
        rows = [
            {
                "query_id": query_id,
                "query_type": "trake",
                "rank": index,
                "video_id": result["video_id"],
                "frame_ids": result["frame_ids"],
                "answer": None,
            }
            for index, result in enumerate(results, start=1)
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
