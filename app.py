import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation


load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources: list[dict], retrieval_source: str) -> None:
    if not sources:
        return
    st.caption(f"Phương pháp truy xuất: {retrieval_source}")
    with st.expander(f"Nguồn tham khảo ({len(sources)})"):
        for index, source in enumerate(sources, 1):
            metadata = source["metadata"]
            title = metadata.get("title") or metadata.get("source") or "Không rõ nguồn"
            url = metadata.get("url")
            score = float(source.get("score", 0.0))
            st.markdown(f"**[Document {index}] {title}** — score: `{score:.4f}`")
            if url:
                st.markdown(f"[Mở nguồn]({url})")
            st.caption(source.get("content", "")[:500])

with st.sidebar:
    st.title("RAG Chatbot")
    st.caption("Trợ lý hỏi đáp về luật, chiến thuật và kiến thức cờ vua")
    top_k = st.slider("Số chunks", 3, 10, 5)

st.title("RAG Chatbot")
st.caption("Câu trả lời được tạo từ bộ tài liệu đã thu thập và kèm nguồn đối chiếu.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(
                message.get("sources", []),
                message.get("retrieval_source", "none"),
            )

query = st.chat_input("Nhập câu hỏi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm tài liệu và tạo câu trả lời..."):
            result = generate_with_citation(query, top_k=top_k)
        answer = result["answer"]
        st.markdown(answer)
        render_sources(result["sources"], result["retrieval_source"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
        }
    )
