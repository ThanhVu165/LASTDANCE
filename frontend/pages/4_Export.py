import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.title("4. Export & Nộp bài")
st.caption(
    "Kiểm tra định dạng, xuất từng file CSV, hoặc đóng gói submission.zip theo đúng "
    "yêu cầu của hệ thống thi (https://sotuyenaic.oj.io.vn/)."
)

query_id = st.text_input("Query ID cần xem/xuất", "query-1-kis")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("👀 Xem submission"):
        resp = requests.get(f"{API}/submission/{query_id}", timeout=60)
        if resp.status_code == 200:
            st.write(resp.json().get("rows", []))
        else:
            st.error(resp.text)

with col2:
    if st.button("✅ Kiểm tra định dạng"):
        resp = requests.get(f"{API}/submission/{query_id}/validate", timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            if data["ok"]:
                st.success("Định dạng hợp lệ, sẵn sàng xuất!")
            else:
                for err in data["errors"]:
                    st.error(err)
        else:
            st.error(resp.text)

with col3:
    if st.button("🗑️ Xoá submission"):
        requests.delete(f"{API}/submission/{query_id}", timeout=60)
        st.info(f"Đã xoá dữ liệu của '{query_id}'.")

st.subheader("Xuất 1 file CSV")
csv_filename = st.text_input("Tên file CSV", f"{query_id}.csv")
if st.button("⬇️ Tải CSV"):
    resp = requests.get(f"{API}/submission/{query_id}/export", timeout=60)
    if resp.status_code == 200:
        st.download_button("Download CSV", data=resp.text, file_name=csv_filename, mime="text/csv")
    else:
        st.error(resp.text)

st.divider()
st.subheader("Đóng gói submission.zip (nhiều query cùng lúc)")
st.caption("Mỗi dòng: `query_id, tên file .csv` — ví dụ `query-1-kis, query-1-kis.csv`")
mapping_text = st.text_area("Danh sách query_id -> filename", height=120, value=f"{query_id}, {query_id}.csv")

if st.button("📦 Tạo submission.zip", type="primary"):
    files = {}
    for line in mapping_text.splitlines():
        if "," not in line:
            continue
        qid, fname = [p.strip() for p in line.split(",", 1)]
        if qid and fname:
            files[qid] = fname

    resp = requests.post(f"{API}/submission/zip", json={"files": files}, timeout=120)
    if resp.status_code == 200:
        st.download_button("Download submission.zip", data=resp.content, file_name="submission.zip", mime="application/zip")
    else:
        st.error(resp.text)
