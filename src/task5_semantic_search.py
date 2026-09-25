"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from .task4_chunking_indexing import embed_texts, get_collection


# tìm đoạn văn liên quan đến query, trả về nhưng chunk có độ tương đồng cao nhất với query, dựa trên embedding vector của query và các chunk trong cơ sở dữ liệu vector.
def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    # dense result: kết quả trả về từ dense, là các kết quả tìm được bằng pp tìm kiếm(semantic search) dựa trên embedding vector của query và các chunk trong cơ sở dữ liệu vector.
    query = query.strip()
    if not query or top_k <= 0:
        return []
    query_vector = embed_texts([query])[
        0
    ]  # embed_text: chuyển query thành vector embedding, trả về 1 list chứa 1 vector embedding của query
    respone = get_collection().query(  # hàm tìm vecto trong db gần với vecto truy vấn này
        query_embeddings=[query_vector],  # vecto câu hỏi người dùng
        n_results=top_k,  # lấy ra     top_k kết quả gần nhất
        include=[
            "documents",
            "metadatas",
            "distances",
        ],  # yêu cầu trả về các trường dữ liệu: nội dung, metadata và khoảng cách ngoài id mặc định
    )
    result = []
    for item_id, content, metadata, distance in zip(
        respone["ids"][0],
        respone["documents"][0],
        respone["metadatas"][0],
        respone["distances"][0],
    ):
        result.append(
            {
                "id": item_id,
                "content": content,
                "score": max(
                    0.0, 1.0 - distance
                ),  # tính điểm tương đồng từ khoảng cách cosine
                "metadata": metadata,
                "retrieval_method": "dense",  # phương pháp truy xuất là dense
            }
        )
    result.sort(
        key=lambda item: item["score"], reverse=True
    )  # sắp xếp kết quả theo điểm giảm dần, lấy score làm tiêu chí
    return result[:top_k]  # trả về top_k kết quả có điểm cao nhất


if __name__ == "__main__":
    for result in semantic_search("test query", top_k=3):
        print(result)
