"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://handbook.fide.com/chapter/E012023",
    "https://en.wikibooks.org/wiki/Chess/Playing_The_Game",
    "https://en.wikibooks.org/wiki/Chess/Basic_Openings",
    "https://en.wikibooks.org/wiki/Chess/Strategy",
    "https://en.wikibooks.org/wiki/Chess/Tactics",
    "https://en.wikibooks.org/wiki/Chess/The_Endgame",
    "https://en.wikibooks.org/wiki/Chess/Arranging_The_Board",
    "https://en.wikibooks.org/wiki/Chess/Notating_The_Game",
    "https://en.wikibooks.org/wiki/Chess/Checkmates",
    "https://en.wikibooks.org/wiki/Chess/Tournaments",
]


async def crawl_article(url: str) -> dict:
    """Thu thập nội dung công khai và trả về schema của Task 2."""
    return await asyncio.to_thread(_crawl_article_sync, url)


def _crawl_article_sync(url: str) -> dict:
    headers = {"User-Agent": "ChessMentor-RAG/1.0 (educational dataset collector)"}

    if urlparse(url).netloc == "en.wikibooks.org":
        page_title = unquote(urlparse(url).path.removeprefix("/wiki/"))
        api_url = "https://en.wikibooks.org/w/api.php"
        response = requests.get(
            api_url,
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "redirects": 1,
                "titles": page_title,
                "format": "json",
                "formatversion": 2,
            },
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        page = response.json()["query"]["pages"][0]
        if page.get("missing"):
            raise ValueError(f"Không tìm thấy trang Wikibooks: {page_title}")
        title = page["title"]
        content = page.get("extract", "").strip()
        license_name = "CC BY-SA 4.0"
    else:
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for node in soup.select("script, style, nav, footer, header, aside"):
            node.decompose()
        root = soup.select_one("main, article, #content") or soup.body
        if root is None:
            raise ValueError(f"Không tìm thấy nội dung HTML: {url}")
        title = (soup.title.get_text(" ", strip=True) if soup.title else url)
        content = root.get_text("\n", strip=True)
        content = re.sub(r"\n{3,}", "\n\n", content)
        license_name = "Official public regulations"

    if len(content) < 500:
        raise ValueError(f"Nội dung quá ngắn hoặc crawl lỗi: {url}")

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": f"# {title}\n\n{content}",
        "publisher": urlparse(url).netloc,
        "license": license_name,
    }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
