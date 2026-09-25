"""
Task 7 — Reciprocal Rank Fusion.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score. Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.

Lưu ý: RRF score chỉ phản ánh thứ hạng, không dùng để quyết định fallback.
Module này dùng RRF bằng Python thuần; Jina hoặc model self-host chỉ là lựa chọn
reranker nâng cao, không bắt buộc cho contract cơ bản.
"""


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists và trả hybrid SearchResult."""
    if top_k <= 0:
        return []

    if k < 0:
        raise ValueError("k phải lớn hơn hoặc bằng 0")

    scores = {}
    items = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            item_id = item["id"]

            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)

            items[item_id] = item

    ranked_ids = sorted(
        scores,
        key=lambda item_id: scores[item_id],
        reverse=True,
    )

    results = []

    for item_id in ranked_ids[:top_k]:
        result = items[item_id].copy()
        result["score"] = scores[item_id]
        result["retrieval_method"] = "hybrid"
        results.append(result)

    return results


if __name__ == "__main__":
    dense_results = [
        {"id": "chunk-1", "content": "Dense result 1", "score": 0.92},
        {"id": "chunk-2", "content": "Dense result 2", "score": 0.81},
    ]
    bm25_results = [
        {"id": "chunk-2", "content": "BM25 result 1", "score": 8.4},
        {"id": "chunk-3", "content": "BM25 result 2", "score": 6.1},
    ]

    for result in rerank_rrf([dense_results, bm25_results], top_k=3):
        print(
            result["id"],
            f"rrf_score={result['score']:.6f}",
            result["retrieval_method"],
        )
