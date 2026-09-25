"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Ví dụ tài liệu: học phí, học bổng, ký túc xá, quy trình đăng ký.
Nếu website chặn crawler, hãy chọn nguồn công khai khác; không vượt WAF.
"""

import json
from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

DOCUMENT_SOURCES = {
    "fide_general_competition_rules.pdf": {
        "title": "FIDE General Regulations for Competitions",
        "url": "https://handbook.fide.com/files/handbook/Competition_Rules.pdf",
    },
    "fide_online_chess_regulations.pdf": {
        "title": "FIDE Online Chess Regulations",
        "url": "https://handbook.fide.com/files/handbook/OnlineChessRegulations.pdf",
    },
    "fide_anti_cheating_protection_measures.pdf": {
        "title": "FIDE Anti-Cheating Protection Measures",
        "url": "https://handbook.fide.com/files/handbook/ACCProtectionMeasures.pdf",
    },
    "fide_arbiters_manual_2025.pdf": {
        "title": "FIDE Arbiters' Manual 2025",
        "url": (
            "https://arbiters.fide.com/wp-content/uploads/Publications/Manual/"
            "Arbiters_Manual_2025.pdf"
        ),
    },
    "fide_chess_olympiad_2026_regulations.pdf": {
        "title": "FIDE Chess Olympiad 2026 Regulations",
        "url": (
            "https://handbook.fide.com/files/handbook/"
            "Olympiad2026MainCompetition.pdf"
        ),
    },
}


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Tải ít nhất 3 PDF/DOCX từ nguồn công khai."""
    setup_directory()
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "ChessMentor-RAG/1.0 (educational dataset collector)"}
    )

    manifest: list[dict[str, str]] = []
    for filename, source in DOCUMENT_SOURCES.items():
        url = source["url"]
        output = DATA_DIR / filename
        if output.exists() and output.read_bytes()[:4] == b"%PDF":
            print(f"Exists: {output}")
        else:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ValueError(f"Nguồn không trả về PDF hợp lệ: {url}")
            output.write_bytes(response.content)
            print(f"Saved: {output}")

        manifest.append(
            {
                "filename": filename,
                "title": source["title"],
                "url": url,
                "publisher": "International Chess Federation (FIDE)",
            }
        )

    (DATA_DIR / "sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    setup_directory()
    download_documents()
