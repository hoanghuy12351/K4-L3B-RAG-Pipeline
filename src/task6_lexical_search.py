"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""

CORPUS: list[dict] = []
import re
from functools import lru_cache

from .task4_chunking_indexing import get_collection


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Okapi

    if not corpus:
        raise ValueError("Corpus không được rỗng")

    tokenized_corpus = [
        re.findall(
            r"\w+",
            item["content"].lower(),
            flags=re.UNICODE,
        )
        for item in corpus
    ]

    return BM25Okapi(tokenized_corpus)


@lru_cache(maxsize=1)
def _load_corpus_and_index():
    """Đọc Chroma và dựng BM25 một lần trong mỗi tiến trình."""
    collection_data = get_collection().get(
        include=["documents", "metadatas"],
    )
    corpus = [
        {"id": item_id, "content": content, "metadata": metadata}
        for item_id, content, metadata in zip(
            collection_data["ids"],
            collection_data["documents"],
            collection_data["metadatas"],
        )
    ]
    return corpus, build_bm25_index(corpus) if corpus else None


def clear_bm25_cache() -> None:
    """Gọi sau khi index thay đổi trong cùng tiến trình."""
    _load_corpus_and_index.cache_clear()


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    query = query.strip()

    if not query or top_k <= 0:
        return []

    corpus = CORPUS
    bm25 = None
    if not corpus:
        corpus, bm25 = _load_corpus_and_index()

    if not corpus:
        return []

    query_tokens = re.findall(
        r"\w+",
        query.lower(),
        flags=re.UNICODE,
    )

    if not query_tokens:
        return []

    tokenized_corpus = [
        re.findall(
            r"\w+",
            item["content"].lower(),
            flags=re.UNICODE,
        )
        for item in corpus
    ]

    if bm25 is None:
        bm25 = build_bm25_index(corpus)
    scores = bm25.get_scores(query_tokens)

    indices = sorted(
        range(len(corpus)),
        key=lambda index: (
            float(scores[index]),
            len(set(query_tokens) & set(tokenized_corpus[index])),
        ),
        reverse=True,
    )

    results = []

    for index in indices:
        overlap = set(query_tokens) & set(tokenized_corpus[index])

        if not overlap:
            continue

        item = corpus[index]

        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                "score": float(scores[index]),
                "metadata": item["metadata"],
                "retrieval_method": "bm25",
            }
        )

        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
