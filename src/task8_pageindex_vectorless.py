"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pageindex import PageIndexClient


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
ROOT_DIR = Path(__file__).parent.parent
LEGAL_DIR = ROOT_DIR / "data" / "landing" / "legal"
CACHE_PATH = ROOT_DIR / "pageindex_doc_ids.json"
SOURCES_PATH = LEGAL_DIR / "sources.json"

# Một PDF tiếng Việt là đủ để minh họa nhánh fallback mà không upload
# toàn bộ corpus lớn lên dịch vụ ngoài.
PAGEINDEX_FILES = ("fide_laws_of_chess_2018_vi.pdf",)


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _source_manifest() -> dict[str, dict]:
    items = _load_json(SOURCES_PATH, [])
    return {
        item["filename"]: item
        for item in items
        if isinstance(item, dict) and item.get("filename")
    }


def _extract_nodes(payload: dict) -> list[dict]:
    """Lấy danh sách node từ các dạng response phổ biến của PageIndex."""
    value = payload.get("result")
    if value is None:
        value = payload.get("results")
    if value is None:
        value = payload.get("nodes")
    if value is None:
        value = payload.get("retrieved_nodes")

    if isinstance(value, dict):
        for key in ("nodes", "results", "retrieved_nodes", "items"):
            if isinstance(value.get(key), list):
                value = value[key]
                break

    return value if isinstance(value, list) else []


def _node_content_items(node: dict) -> list[tuple[str, str]]:
    """Lấy text trực tiếp hoặc các relevant_content lồng nhau của một node."""
    direct_content = (
        node.get("text")
        or node.get("content")
        or node.get("summary")
        or node.get("node_summary")
    )
    if isinstance(direct_content, str) and direct_content.strip():
        return [("text", direct_content.strip())]

    items = []

    def walk(value) -> None:
        if isinstance(value, list):
            for child in value:
                walk(child)
            return
        if not isinstance(value, dict):
            return

        content = value.get("relevant_content")
        if isinstance(content, str) and content.strip():
            physical_index = str(value.get("physical_index") or len(items) + 1)
            items.append((physical_index, content.strip()))
        for child in value.values():
            if isinstance(child, (list, dict)):
                walk(child)

    walk(node.get("relevant_contents", []))
    return items


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    if not PAGEINDEX_API_KEY:
        raise ValueError("Missing PAGEINDEX_API_KEY")

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    cache = _load_json(CACHE_PATH, {})
    manifest = _source_manifest()

    for filename in PAGEINDEX_FILES:
        if filename in cache and cache[filename].get("doc_id"):
            print(f"Cached: {filename}")
            continue

        path = LEGAL_DIR / filename
        if not path.exists():
            print(f"Skipped missing file: {path}")
            continue

        response = client.submit_document(str(path))
        doc_id = response.get("doc_id")
        if not doc_id:
            raise ValueError(f"PageIndex không trả doc_id cho {filename}")

        source = manifest.get(filename, {})
        cache[filename] = {
            "doc_id": doc_id,
            "title": source.get("title", path.stem.replace("_", " ").title()),
            "url": source.get("url"),
        }
        CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Uploaded: {filename} -> {doc_id}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    query = query.strip()
    if not query or top_k <= 0 or not PAGEINDEX_API_KEY:
        return []

    cache = _load_json(CACHE_PATH, {})
    if not cache:
        return []

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    results = []

    for filename, source in cache.items():
        doc_id = source.get("doc_id")
        if not doc_id or not client.is_retrieval_ready(doc_id):
            continue

        submitted = client.submit_query(doc_id=doc_id, query=query, thinking=False)
        retrieval_id = submitted.get("retrieval_id")
        if not retrieval_id:
            continue

        deadline = time.monotonic() + 120
        response = {}
        while time.monotonic() < deadline:
            response = client.get_retrieval(retrieval_id)
            status = response.get("status")
            if status == "completed":
                break
            if status == "failed":
                response = {}
                break
            time.sleep(2)

        nodes = _extract_nodes(response)
        for rank, node in enumerate(nodes, start=1):
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("node_id") or node.get("id") or rank)
            raw_score = node.get("score")
            score = float(raw_score) if isinstance(raw_score, (int, float)) else 1.0 / rank
            for content_index, (content_id, content) in enumerate(
                _node_content_items(node),
                start=1,
            ):
                results.append(
                    {
                        "id": f"pageindex::{doc_id}::{node_id}::{content_id}",
                        "content": content,
                        "score": score - (content_index - 1) * 1e-6,
                        "metadata": {
                            "source": filename,
                            "title": source.get("title", filename),
                            "doc_type": "legal",
                            "url": source.get("url"),
                            "chunk_index": len(results),
                        },
                        "retrieval_method": "pageindex",
                    }
                )

    unique_results = {}
    for result in results:
        current = unique_results.get(result["id"])
        if current is None or result["score"] > current["score"]:
            unique_results[result["id"]] = result

    return sorted(
        unique_results.values(),
        key=lambda item: item["score"],
        reverse=True,
    )[:top_k]


if __name__ == "__main__":
    upload_documents()
