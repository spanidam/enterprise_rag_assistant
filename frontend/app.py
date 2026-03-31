import streamlit as st
import requests

API_BASE = "https://enterprise-rag-assistant-1034.onrender.com"

st.set_page_config(page_title="Enterprise RAG Assistant", layout="wide")
st.title("Enterprise RAG Assistant")

# -------------------------------
# Sidebar: PDF upload (Future Work)
# -------------------------------
st.sidebar.header("Upload a document (PDF)")
uploaded = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

if uploaded is not None:
    st.sidebar.info(
        "📄 PDF upload is a planned feature. "
        "The current demo uses a pre‑ingested document corpus."
    )

st.sidebar.markdown("---")
if st.sidebar.button("Health Check"):
    try:
        r = requests.get(f"{API_BASE}/health", timeout=10)
        st.sidebar.success(f"Backend status: {r.status_code}")
    except requests.exceptions.RequestException:
        st.sidebar.error("Backend not reachable")

# -------------------------------
# Chat state
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# Chat input
# -------------------------------
user_prompt = st.chat_input("Ask a question (min 5 characters)…")

if user_prompt:
    # User message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Backend call with error handling
    try:
        response = requests.post(
            f"{API_BASE}/ask",
            json={"question": user_prompt},
            timeout=30
        )

        # Validation error
        if response.status_code == 422:
            st.warning("Please enter a question with at least 5 characters.")
            st.stop()

        # Backend error
        if response.status_code != 200:
            st.error("Backend error. Please try again later.")
            st.stop()

        data = response.json()

    except requests.exceptions.RequestException:
        st.error("Cannot connect to backend.")
        st.stop()

    # Assistant response
    assistant_text = data.get("answer", "")
    confidence = data.get("confidence_score")
    coverage = data.get("citation_coverage")
    abstained = data.get("abstained")
    reason = data.get("abstain_reason", "")

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_text}
    )

    with st.chat_message("assistant"):
        st.markdown(assistant_text)

        if confidence is not None:
            st.caption(
                f"Confidence: {confidence} | "
                f"Citation coverage: {coverage}"
            )

        if abstained:
            st.warning(reason)