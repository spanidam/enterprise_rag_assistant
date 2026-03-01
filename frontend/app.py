import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Enterprise RAG Assistant", layout="wide")
st.title("Enterprise RAG Assistant")

# ---- Sidebar: upload PDF ----
st.sidebar.header("Upload a document (PDF)")
uploaded = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])  # [2](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
if uploaded is not None:
    if st.sidebar.button("Upload to RAG"):
        files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
        r = requests.post(f"{API_BASE}/ingest", files=files)
        if r.ok:
            st.sidebar.success(f"Uploaded: {uploaded.name}")
        else:
            st.sidebar.error(f"Upload failed: {r.status_code} {r.text}")

st.sidebar.markdown("---")
if st.sidebar.button("Health Check"):
    r = requests.get(f"{API_BASE}/health")
    st.sidebar.write(r.status_code, r.text)

# ---- Chat state ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):  # [3](https://docs.streamlit.io/develop/api-reference/chat)
        st.markdown(msg["content"])
        # Show citations if present
        if msg.get("sources"):
            st.markdown("**Citations:**")
            for s in msg["sources"]:
                chunk_file = s["chunk_file"]
                with st.expander(f"{s['id']} — {chunk_file}"):
                    # Fetch chunk text from backend for preview
                    resp = requests.get(f"{API_BASE}/chunk/{chunk_file}")
                    if resp.ok:
                        st.write(resp.json()["text"][:2000])
                    else:
                        st.write("Could not load chunk preview.")

# Chat input
user_prompt = st.chat_input("Ask a question…")  # [4](https://docs.streamlit.io/develop/api-reference/chat/st.chat_input)
if user_prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Call backend /ask
    payload = {"question": user_prompt}
    r = requests.post(f"{API_BASE}/ask", json=payload)

    if not r.ok:
        assistant_text = f"Error: {r.status_code} {r.text}"
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.error(assistant_text)
    else:
        data = r.json()
        assistant_text = data.get("answer", "")
        sources = data.get("sources", [])

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_text, "sources": sources}
        )

        with st.chat_message("assistant"):
            st.markdown(assistant_text)
            if sources:
                st.markdown("**Citations:**")
                for s in sources:
                    chunk_file = s["chunk_file"]
                    with st.expander(f"{s['id']} — {chunk_file}"):
                        resp = requests.get(f"{API_BASE}/chunk/{chunk_file}")
                        if resp.ok:
                            st.write(resp.json()["text"][:2000])
                        else:
                            st.write("Could not load chunk preview.")