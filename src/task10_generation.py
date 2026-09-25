"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import os
import re

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_MAX_ATTEMPTS = max(1, int(os.getenv("LLM_MAX_ATTEMPTS", "2")))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Trả lời chỉ từ context được cung cấp.
Trả lời cùng ngôn ngữ với người dùng.
Nếu input chỉ là một cụm từ/chủ đề, hãy hiểu là yêu cầu giải thích tổng quan.
Mỗi khẳng định phải có citation chính xác dạng [Document N], ví dụ [Document 1].
Không dùng dạng [1], [Source 1] hoặc (Document 1).
Không sử dụng kiến thức bên ngoài context.
Nếu thiếu evidence, hãy từ chối xác minh."""

SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."


def _has_valid_citations(answer: str, source_count: int) -> bool:
    """Đảm bảo câu trả lời có citation và mọi số đều trỏ tới sources."""
    citations = [int(value) for value in re.findall(r"\[Document (\d+)\]", answer)]
    return bool(citations) and all(1 <= value <= source_count for value in citations)


def _normalize_citations(answer: str, source_count: int) -> str:
    """Chuẩn hóa các citation phổ biến của LLM về contract [Document N]."""

    def replace(match: re.Match) -> str:
        value = int(match.group(1))
        return f"[Document {value}]" if 1 <= value <= source_count else match.group(0)

    patterns = (
        r"\[(?:Document|Doc|Source)\s*#?\s*(\d+)\]",
        r"\((?:Document|Doc|Source)\s*#?\s*(\d+)\)",
        r"\[(\d+)\]",
    )
    normalized = answer
    for pattern in patterns:
        normalized = re.sub(pattern, replace, normalized, flags=re.IGNORECASE)
    return normalized


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        source_url = metadata.get("url") or metadata["source"]
        citation_index = chunk.get("_citation_index", index)
        parts.append(
            f"[Document {citation_index} | Title: {metadata['title']} | "
            f"Source: {metadata['source']} | URL: {source_url}]\n"
            f"{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    provider = LLM_PROVIDER.strip().lower()
    if not LLM_MODEL:
        raise ValueError("Missing LLM_MODEL")

    if provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        client = OpenAI(
            api_key=api_key,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_ATTEMPTS - 1,
        )
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return (response.choices[0].message.content or "").strip()

    if provider == "gemini":
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY")
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(LLM_TIMEOUT_SECONDS * 1000),
                retry_options=types.HttpRetryOptions(attempts=LLM_MAX_ATTEMPTS),
            ),
        )
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            ),
        )
        return (response.text or "").strip()

    if provider == "anthropic":
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY")
        client = Anthropic(
            api_key=api_key,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_ATTEMPTS - 1,
        )
        response = client.messages.create(
            model=LLM_MODEL,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=1000,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return "".join(
            block.text
            for block in response.content
            if getattr(block, "type", "") == "text"
        ).strip()

    raise ValueError(f"Unsupported LLM provider: {provider}")


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    query = query.strip()
    if not query or top_k <= 0:
        return {
            "answer": SAFE_REFUSAL,
            "sources": [],
            "retrieval_source": "none",
        }

    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": SAFE_REFUSAL,
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    citation_by_id = {
        chunk["id"]: index
        for index, chunk in enumerate(chunks, 1)
    }
    context_chunks = [
        {**chunk, "_citation_index": citation_by_id[chunk["id"]]}
        for chunk in reordered
    ]
    context = format_context(context_chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
    except Exception as error:
        return {
            "answer": SAFE_REFUSAL,
            "sources": [],
            "retrieval_source": "none",
            "failure_reason": "provider_error",
            "error_type": type(error).__name__,
        }

    answer = _normalize_citations(answer, len(reordered))
    if not answer or not _has_valid_citations(answer, len(reordered)):
        retrieval_method = chunks[0]["retrieval_method"]
        retrieval_source = (
            retrieval_method
            if retrieval_method in {"hybrid", "pageindex"}
            else "hybrid"
        )
        return {
            "answer": SAFE_REFUSAL,
            "sources": chunks,
            "retrieval_source": retrieval_source,
            "failure_reason": "invalid_citation",
        }

    retrieval_method = chunks[0]["retrieval_method"]
    retrieval_source = (
        retrieval_method
        if retrieval_method in {"hybrid", "pageindex"}
        else "hybrid"
    )
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    print(generate_with_citation("test query"))
