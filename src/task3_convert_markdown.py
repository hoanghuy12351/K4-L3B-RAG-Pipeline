"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng MarkItDown để convert PDF/DOCX.
    2. Đọc JSON và giữ metadata ở đầu file Markdown.
    3. Giữ cấu trúc thư mục legal/ và news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.

Cài đặt:
    Dependency MarkItDown đã được khai báo trong pyproject.toml.
    
-> Hoặc dùng công cụ nào bạn quen khác Markitdown
"""

import json
from pathlib import Path

from markitdown import MarkItDown


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs() -> None:
    """Convert PDF/DOCX và thêm metadata nguồn ở đầu Markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = MarkItDown()

    manifest_path = legal_dir / "sources.json"
    manifest = {}
    if manifest_path.exists():
        manifest = {
            item["filename"]: item
            for item in json.loads(manifest_path.read_text(encoding="utf-8"))
        }

    for path in sorted(legal_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".doc", ".docx"}:
            continue
        source = manifest.get(path.name, {})
        title = source.get("title", path.stem.replace("_", " ").title())
        url = source.get("url")
        result = converter.convert(str(path))
        content = result.text_content.strip()
        if not content:
            raise ValueError(f"Không trích xuất được nội dung: {path}")
        header = (
            f"# {title}\n\n"
            f"**Source:** {url or path.name}\n\n"
            "**Publisher:** International Chess Federation (FIDE)\n\n"
            "**Document type:** legal\n\n---\n\n"
        )
        output = output_dir / f"{path.stem}.md"
        output.write_text(header + content, encoding="utf-8")
        print(f"Saved: {output}")


def convert_news_articles() -> None:
    """Convert article JSON và giữ metadata kiểm chứng nguồn."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {"url", "title", "date_crawled", "content_markdown"}

    for path in sorted(news_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = required - data.keys()
        if missing:
            raise ValueError(f"{path.name} thiếu metadata: {sorted(missing)}")
        content = data["content_markdown"].strip()
        if not content:
            raise ValueError(f"Nội dung rỗng: {path}")
        header = (
            f"# {data['title']}\n\n"
            f"**Source:** {data['url']}\n\n"
            f"**Crawled:** {data['date_crawled']}\n\n"
            f"**Publisher:** {data.get('publisher', 'Unknown')}\n\n"
            f"**License:** {data.get('license', 'Unknown')}\n\n"
            "**Document type:** news\n\n---\n\n"
        )
        output = output_dir / f"{path.stem}.md"
        output.write_text(header + content, encoding="utf-8")
        print(f"Saved: {output}")


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
