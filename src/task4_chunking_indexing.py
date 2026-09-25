"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os  # thư viện thao tác với hệ điều hành
import re  # thư viện để tìm kiếm và thao tác với chuỗi

# trong Rag, re dùng làm sạch, chuẩn hóa văn bản trước chunking, vd: search, findall, sub(tìm, thay thế), split(chia)
from functools import (
    lru_cache,
)  # thư viện để lưu trữ kết quả của hàm, tránh tính toán lại
from dotenv import load_dotenv  # thư viện để load biến môi trường từ file .env
from pathlib import Path
from src.contracts import (
    Document,
    validate_document,
)  # validate_document thường dùng để kiểm tra document(tài liệu) có đúng format(định dạng) trước khi chunking(chia nhỏ văn bản) và indexing(đưa vào vector database) không.

# cổng kiểm tra dữ liệu đầu vào, đảm bảo dữ liệu đầu ra đúng định dạng, tránh lỗi khi chạy pipeline
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"  # phương pháp chunking: recursive, token, sentence, paragraph, character
# chia văn bản bằng cách sử dụng danh sách các ký tự phân tách, ví dụ: "\n\n", "\n", ". ", " ", "".

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024  # chiều của vector embedding

COLLECTION_NAME = "rag_documents"  # tên collection trong ChromaDB(csdl lưu vecto), dùng để lưu trữ các vector embedding của các chunks.
# rag_documents        -> tài liệu chính của chatbot
# rag_news             -> chỉ tin tức
# rag_legal            -> chỉ quy định/chính sách
# test_documents       -> dữ liệu thử nghiệm
load_dotenv()


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []

    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        # lấy nội dung của file markdown, loại bỏ khoảng trắng đầu và cuối, trả về 1 dòng chuỗi, nếu file rỗng thì bỏ qua
        if not content:
            continue

        title = path.stem
        url = None

        for line in content.splitlines():
            if line.startswith("# "):
                title = line.removeprefix("# ").strip()

            if line.startswith("**Source:**"):
                url = line.removeprefix("**Source:**").strip()

        doc_type = "legal" if "legal" in path.parts else "news"

        document = {
            "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
            "content": content,
            "metadata": {
                "source": path.name,
                "title": title,
                "doc_type": doc_type,
                "url": url,
            },
        }

        validate_document(
            document
        )  # kiem tra dung contract khong, neeus dung thi them vao documents, neu khong dung thi bao loi
        documents.append(document)

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for document in documents:
        texts = splitter.split_text(document["content"])
        # dùng split_text để chia nội dung của document thành các chunk nhỏ hơn dựa trên các tham số đã định nghĩa ở trên.
        for index, text in enumerate(texts):
            text = text.strip()

            if not text:
                continue

            chunk = {
                "id": f"{document['id']}::chunk-{index}",  # tạo id riêng từng chunk
                "content": text,
                "metadata": {
                    **document["metadata"],
                    "chunk_index": index,
                },
            }

            validate_document(chunk, require_chunk=True)
            chunks.append(chunk)

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Chuyển danh sách văn bản thành embedding vectors."""
    if not texts:
        return []

    provider = os.getenv(
        "EMBEDDING_PROVIDER",
        "sentence_transformers",
    )
    # os.getenv("TÊN_BIẾN", "giá_trị_mặc_định")
    model_name = os.getenv(
        "EMBEDDING_MODEL",
        EMBEDDING_MODEL,
    )

    if provider != "sentence_transformers":
        raise ValueError(f"Unsupported embedding provider: {provider}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    # tạo obj model embedding
    vectors = model.encode(
        texts,
        batch_size=16,  # xử lí tối đa 16 text 1 lượt
        show_progress_bar=len(texts) > 32,  # nếu trên 32 text, hiện progress bar
        normalize_embeddings=True,
    )

    return (
        vectors.tolist()
    )  # cần chuyển sang list để lưu trữ trong ChromaDB, vì numpy array không thể lưu trực tiếp vào csdl vector


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""

    texts = [chunk["content"] for chunk in chunks]
    vectors = embed_texts(texts)
    if len(vectors) != len(chunks):
        raise ValueError("Số embedding không khớp với số chunk")

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    return chunks


def get_collection():  # mở hoặc tạo 1 chroma collection để lưu các vecto embedding
    """Mở Chroma collection dùng cosine distance(khoảng cách cosine-tìm kiếm độ tương đồng)."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    # nếu thư mục chưa tồn tại thì tạo thư mục CHROMA_DIR để lưu trữ dữ liệu của ChromaDB, nếu đã tồn tại thì bỏ qua
    client = (
        chromadb.PersistentClient(  # lưu dữ liệu xuống ổ đĩa, không chỉ giữ trong RAm
            path=str(CHROMA_DIR)
        )
    )  # để kết nối và thao tác với ChromaDB tại thư mục đó.

    return client.get_or_create_collection(  # lấy hoặc tạo collection trong ChromaDB, nếu collection đã tồn tại thì lấy ra, nếu chưa tồn tại thì tạo mới
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # quy định cách chroma so sánh vecto
    )


def index_to_vectorstore(
    chunks: list[dict],
) -> None:  # lưu chunk, vecto, metadata vào csdl vector
    """Upsert chunks vào ChromaDB."""  # update+ insert
    # TODO: Upsert ids, documents, embeddings và metadatas.
    #
    collection = get_collection()
    metadatas = []
    for chunk in chunks:
        metadata = chunk["metadata"]
        metadatas.append(metadata)
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=metadatas,
    )
    # collection.upsert(
    # #     ids=[chunk["id"] for chunk in chunks],
    # #     documents=[chunk["content"] for chunk in chunks],
    # #     embeddings=[chunk["embedding"] for chunk in chunks],
    # #     metadatas=[chunk["metadata"] for chunk in chunks],
    # # )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


if __name__ == "__main__":
    run_pipeline()
