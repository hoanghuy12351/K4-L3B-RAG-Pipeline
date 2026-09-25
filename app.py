import html as html_lib
import json
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.task4_chunking_indexing import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    chunk_documents,
    load_documents,
)
from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_rrf
from src.task9_retrieval_pipeline import SCORE_THRESHOLD
from src.task10_generation import generate_with_citation

load_dotenv()

ROOT_DIR = Path(__file__).parent
LANDING_DIR = ROOT_DIR / "data" / "landing"
GOLDEN_DATASET_PATH = ROOT_DIR / "group_project" / "evaluation" / "golden_dataset.json"

STAGES = [
    ("ask", "💬", "Ask"),
    ("retrieve", "🔎", "Retrieve"),
    ("rerank", "🧠", "Rerank"),
    ("generate", "🤖", "Generate"),
    ("done", "⚖️", "Answer"),
]

SUGGESTIONS = [
    {
        "icon": "♜",
        "label": "Nhập thành",
        "question": "Khi nào tôi được phép nhập thành (castling)?",
    },
    {
        "icon": "⚠️",
        "label": "Nước đi không hợp lệ",
        "question": "Điều gì xảy ra sau khi một đấu thủ đi một nước không hợp lệ?",
    },
    {
        "icon": "⏱️",
        "label": "Kiểm soát thời gian",
        "question": "Điều gì xảy ra khi đồng hồ của một đấu thủ hết thời gian?",
    },
    {
        "icon": "🛡️",
        "label": "Chống gian lận",
        "question": "FIDE quy định thế nào về gian lận trong thi đấu cờ vua online?",
    },
]

OUT_OF_SCOPE_QUESTION = "Công thức nấu phở bò truyền thống gồm những gì?"

BOARD_ROWS = [
    ["♜", "♞", "♝", "♛", "♚", "♝", "♞", "♜"],
    ["♟"] * 8,
    [""] * 8,
    [""] * 8,
    [""] * 8,
    [""] * 8,
    ["♙"] * 8,
    ["♖", "♘", "♗", "♕", "♔", "♗", "♘", "♖"],
]

CSS = """
<style>
:root {
  --bg: #FAF7F0;
  --panel: #FFFFFF;
  --card: #FFFFFF;
  --border: #E3DCC9;
  --text: #241E14;
  --text-secondary: #5B5346;
  --accent: #A97B15;
  --success: #1E7A34;
  --warning: #B4670B;
  --danger: #C62828;
  --track: #ECE6D6;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stApp { background: var(--bg); color: var(--text); }
section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
.app-header { display: flex; flex-direction: column; align-items: center; gap: 0.3rem; padding: 1.2rem 0 0.6rem; text-align: center; }
.app-title { font-size: 2rem; font-weight: 800; letter-spacing: 0.04em; color: var(--text); }
.app-subtitle { color: var(--text-secondary); font-style: italic; }
.status-badge { margin-top: 0.4rem; padding: 0.25rem 0.9rem; border-radius: 999px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.05em; border: 1px solid var(--border); }
.status-badge.online { color: var(--success); border-color: var(--success); animation: pulseDot 2s infinite; }
.status-badge.warn { color: var(--warning); border-color: var(--warning); }
@keyframes pulseDot { 0%,100% { box-shadow: 0 0 0 0 rgba(30,122,52,0.35); } 50% { box-shadow: 0 0 0 6px rgba(30,122,52,0); } }
.hero { display: flex; align-items: center; gap: 2.2rem; background: var(--panel); border: 1px solid var(--border); border-radius: 18px; padding: 1.6rem; margin: 0.6rem 0 1.4rem; flex-wrap: wrap; }
.chess-board { display: flex; flex-direction: column; border: 2px solid var(--accent); border-radius: 6px; overflow: hidden; box-shadow: 0 8px 30px rgba(0,0,0,0.45); }
.board-row { display: flex; }
.square { width: 34px; height: 34px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; }
.square.light { background: #2A2118; }
.square.dark { background: #14100C; }
.piece { animation: floatPiece 3.4s ease-in-out infinite; display: inline-block; }
.square:nth-child(odd) .piece { animation-delay: 0.4s; }
@keyframes floatPiece { 0%,100% { transform: translateY(0px); } 50% { transform: translateY(-3px); } }
.hero-text { flex: 1; min-width: 260px; }
.hero-kicker { color: var(--accent); font-size: 0.75rem; font-weight: 700; letter-spacing: 0.12em; margin-bottom: 0.4rem; }
.hero-title { font-size: 1.4rem; font-weight: 700; line-height: 1.4; margin-bottom: 0.5rem; }
.hero-desc { color: var(--text-secondary); font-size: 0.92rem; line-height: 1.5; }
.section-label { color: var(--text-secondary); font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin: 0.9rem 0 0.5rem; }
.suggestion-card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 0.9rem; height: 128px; display: flex; flex-direction: column; gap: 0.35rem; animation: fadeInUp 0.5s ease both; box-shadow: 0 1px 3px rgba(36,30,20,0.06); }
.suggestion-icon { font-size: 1.3rem; }
.suggestion-label { color: var(--accent); font-weight: 700; font-size: 0.85rem; }
.suggestion-question { color: var(--text-secondary); font-size: 0.78rem; line-height: 1.3; }
@keyframes fadeInUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.stepper { display: flex; align-items: center; justify-content: center; gap: 0; padding: 0.8rem 0 0.4rem; }
.step { display: flex; flex-direction: column; align-items: center; gap: 0.3rem; min-width: 64px; }
.step-dot { width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1rem; border: 2px solid var(--border); background: var(--card); color: var(--text-secondary); }
.step.active .step-dot { border-color: var(--accent); color: var(--accent); box-shadow: 0 0 0 4px rgba(169,123,21,0.18); animation: stepPulse 1s infinite; }
.step.done .step-dot { border-color: var(--success); color: var(--success); }
.step-label { font-size: 0.7rem; color: var(--text-secondary); }
.step.active .step-label { color: var(--accent); }
.step.done .step-label { color: var(--success); }
.step-connector { flex: 1; height: 2px; background: var(--border); min-width: 24px; margin-bottom: 1.1rem; }
.step-connector.done { background: var(--success); }
@keyframes stepPulse { 0%,100% { box-shadow: 0 0 0 4px rgba(169,123,21,0.18); } 50% { box-shadow: 0 0 0 8px rgba(169,123,21,0.05); } }
.confidence-block { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 0.8rem 1rem; margin: 0.6rem 0; box-shadow: 0 1px 3px rgba(36,30,20,0.06); }
.confidence-header { display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-secondary); letter-spacing: 0.05em; text-transform: uppercase; }
.confidence-pct { font-weight: 800; font-size: 0.9rem; }
.confidence-track { position: relative; height: 8px; border-radius: 4px; background: var(--track); margin: 0.5rem 0; overflow: visible; }
.confidence-fill { height: 100%; border-radius: 4px; transition: width 0.4s ease; }
.confidence-threshold { position: absolute; top: -3px; width: 2px; height: 14px; background: var(--text-secondary); }
.confidence-status { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.03em; }
.evidence-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 0.7rem 0.9rem; margin-bottom: 0.4rem; animation: fadeInUp 0.4s ease both; box-shadow: 0 1px 3px rgba(36,30,20,0.06); }
.evidence-top { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.3rem; }
.evidence-rank { color: var(--accent); font-weight: 800; font-size: 0.8rem; }
.evidence-badge { font-size: 0.65rem; font-weight: 700; padding: 0.1rem 0.5rem; border-radius: 999px; letter-spacing: 0.05em; }
.evidence-badge.legal { background: rgba(169,123,21,0.15); color: var(--accent); }
.evidence-badge.news { background: rgba(30,122,52,0.15); color: var(--success); }
.evidence-method { margin-left: auto; color: var(--text-secondary); font-size: 0.68rem; text-transform: uppercase; }
.evidence-title { font-weight: 600; font-size: 0.88rem; margin-bottom: 0.4rem; }
.evidence-bar-track { height: 5px; border-radius: 3px; background: var(--track); }
.evidence-bar-fill { height: 100%; border-radius: 3px; background: var(--accent); }
.evidence-meta { color: var(--text-secondary); font-size: 0.68rem; margin-top: 0.3rem; }
.evidence-snippet { margin-top: 0.5rem; padding: 0.5rem 0.6rem; background: var(--track); border-radius: 8px; font-size: 0.78rem; line-height: 1.4; color: var(--text); }
.oos-banner { background: rgba(198,40,40,0.08); border: 1px solid var(--danger); color: var(--danger); border-radius: 10px; padding: 0.6rem 0.9rem; font-size: 0.82rem; font-weight: 600; margin-bottom: 0.5rem; }
.oos-trigger { border: 1px dashed var(--border); border-radius: 12px; padding: 0.6rem 0.9rem; margin-top: 0.5rem; color: var(--text-secondary); font-size: 0.8rem; }
.compare-card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem; box-shadow: 0 1px 3px rgba(36,30,20,0.06); }
.compare-rank { color: var(--accent); font-weight: 800; font-size: 0.78rem; margin-right: 0.4rem; }
.compare-title { font-size: 0.82rem; font-weight: 600; color: var(--text); }
.compare-tag { font-size: 0.62rem; font-weight: 700; padding: 0.05rem 0.4rem; border-radius: 999px; background: rgba(30,122,52,0.15); color: var(--success); margin-left: 0.4rem; }
.compare-meta { color: var(--text-secondary); font-size: 0.68rem; margin-top: 0.15rem; }
.chunk-strip { display: flex; gap: 3px; height: 30px; border-radius: 8px; overflow: hidden; margin: 0.6rem 0; }
.chunk-block { display: flex; align-items: center; justify-content: center; font-size: 0.62rem; font-weight: 700; color: #0B0D10; min-width: 3px; }
.golden-summary { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.5rem 0 0.8rem; }
.golden-chip { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 0.5rem 0.9rem; box-shadow: 0 1px 3px rgba(36,30,20,0.06); }
.golden-chip .value { font-size: 1.2rem; font-weight: 800; color: var(--accent); }
.golden-chip .label { font-size: 0.68rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }
</style>
"""

CHUNK_PALETTE = ["#D4AF37", "#B08D2B", "#8C9E5E", "#5E9E8C", "#5E7A9E", "#7A5E9E", "#9E5E7A"]


@st.cache_data(show_spinner=False)
def load_legal_sources() -> list[dict]:
    path = LANDING_DIR / "legal" / "sources.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_news_sources() -> list[dict]:
    items = []
    news_dir = LANDING_DIR / "news"
    for path in sorted(news_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        items.append(
            {
                "filename": path.name,
                "title": data.get("title", path.stem),
                "url": data.get("url"),
            }
        )
    return items


@st.cache_data(show_spinner=False, ttl=60)
def chunk_counts_by_stem() -> dict[str, int]:
    try:
        from src.task4_chunking_indexing import get_collection

        data = get_collection().get(include=["metadatas"])
    except Exception:
        return {}
    counts: dict[str, int] = {}
    for metadata in data.get("metadatas", []):
        stem = Path(metadata.get("source", "")).stem
        counts[stem] = counts.get(stem, 0) + 1
    return counts


@st.cache_data(show_spinner=False, ttl=30)
def backend_health() -> dict:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    key_name = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }.get(provider)
    has_key = bool(os.getenv(key_name, "")) if key_name else False
    try:
        from src.task4_chunking_indexing import get_collection

        indexed_chunks = get_collection().count()
    except Exception:
        indexed_chunks = 0
    if indexed_chunks == 0:
        return {"ok": False, "reason": "Chưa index dữ liệu (chạy task4)"}
    if not has_key:
        return {"ok": False, "reason": f"Thiếu {key_name}"}
    return {"ok": True, "reason": "RAG ONLINE"}


def render_header() -> None:
    health = backend_health()
    state = "online" if health["ok"] else "warn"
    dot = "🟢" if health["ok"] else "🟡"
    badge_text = "RAG ONLINE" if health["ok"] else health["reason"].upper()
    st.markdown(
        '<div class="app-header">'
        '<div class="app-title">♟️ FIDE CHESS LAW ASSISTANT</div>'
        '<div class="app-subtitle">Ask the rules. See the evidence.</div>'
        f'<div class="status-badge {state}">{dot} {html_lib.escape(badge_text)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_hero_board() -> None:
    rows_html = []
    for r, row in enumerate(BOARD_ROWS):
        cells = []
        for c, piece in enumerate(row):
            shade = "light" if (r + c) % 2 == 0 else "dark"
            piece_html = f'<span class="piece">{piece}</span>' if piece else ""
            cells.append(f'<div class="square {shade}">{piece_html}</div>')
        rows_html.append(f'<div class="board-row">{"".join(cells)}</div>')

    legal_count = len(load_legal_sources())
    news_count = len(load_news_sources())
    st.markdown(
        '<div class="hero">'
        f'<div class="chess-board">{"".join(rows_html)}</div>'
        '<div class="hero-text">'
        '<div class="hero-kicker">AI-POWERED LEGAL KNOWLEDGE RETRIEVAL</div>'
        "<div class=\"hero-title\">Hiểu luật cờ vua FIDE.<br>Theo dõi từng bằng chứng.</div>"
        '<div class="hero-desc">Mọi câu trả lời được truy hồi và trích dẫn trực tiếp từ '
        f"{legal_count + news_count} tài liệu gốc — {legal_count} văn bản pháp lý FIDE và "
        f"{news_count} bài viết chuyên môn.</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_suggestions() -> str | None:
    st.markdown('<div class="section-label">💡 Câu hỏi gợi ý</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    picked = None
    for col, item in zip(cols, SUGGESTIONS):
        with col:
            st.markdown(
                '<div class="suggestion-card">'
                f'<div class="suggestion-icon">{item["icon"]}</div>'
                f'<div class="suggestion-label">{html_lib.escape(item["label"])}</div>'
                f'<div class="suggestion-question">{html_lib.escape(item["question"])}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
            if st.button("Hỏi câu này →", key=f"suggest-{item['label']}", use_container_width=True):
                picked = item["question"]

    st.markdown(
        '<div class="oos-trigger">🌐 <b>Demo an toàn</b> — thử một câu hỏi hoàn toàn '
        "ngoài phạm vi cờ vua để xem hệ thống từ chối đúng cách thay vì bịa thông tin.</div>",
        unsafe_allow_html=True,
    )
    if st.button("Hỏi câu hỏi NGOÀI phạm vi →", key="suggest-oos"):
        picked = OUT_OF_SCOPE_QUESTION
    return picked


def render_stepper(slot, active_stage: str) -> None:
    order = [key for key, _, _ in STAGES]
    active_index = order.index(active_stage)
    parts = []
    for index, (_, icon, label) in enumerate(STAGES):
        if index < active_index:
            state = "done"
        elif index == active_index:
            state = "active"
        else:
            state = "pending"
        dot = "✓" if state == "done" else icon
        parts.append(
            f'<div class="step {state}"><div class="step-dot">{dot}</div>'
            f'<div class="step-label">{label}</div></div>'
        )
        if index < len(STAGES) - 1:
            connector_state = "done" if index < active_index else ""
            parts.append(f'<div class="step-connector {connector_state}"></div>')
    slot.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_confidence_meter(score: float, threshold: float, retrieval_source: str) -> None:
    pct = max(0.0, min(1.0, score)) * 100
    threshold_pct = max(0.0, min(1.0, threshold)) * 100
    if retrieval_source == "none":
        color, status = "var(--danger)", "KHÔNG ĐỦ CĂN CỨ — TỪ CHỐI TRẢ LỜI"
    elif score >= threshold:
        color, status = "var(--success)", "ĐỘ TIN CẬY CAO"
    elif score >= threshold - 0.15:
        color, status = "var(--warning)", "ĐỘ TIN CẬY TRUNG BÌNH"
    else:
        color, status = "var(--warning)", "ĐIỂM DENSE THẤP — ĐÃ THỬ FALLBACK"

    st.markdown(
        '<div class="confidence-block">'
        '<div class="confidence-header">'
        "<span>RAG CONFIDENCE · dense cosine score</span>"
        f'<span class="confidence-pct" style="color:{color}">{pct:.0f}%</span>'
        "</div>"
        '<div class="confidence-track">'
        f'<div class="confidence-fill" style="width:{pct:.1f}%; background:{color};"></div>'
        f'<div class="confidence-threshold" style="left:{threshold_pct:.1f}%;"></div>'
        "</div>"
        f'<div class="confidence-status" style="color:{color}">{status}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_evidence_card(
    source: dict, rank: int, max_score: float, show_snippet: bool = False
) -> None:
    metadata = source.get("metadata", {})
    relative = (source.get("score", 0.0) / max_score * 100) if max_score else 0.0
    doc_type = metadata.get("doc_type", "doc")
    badge_class = "legal" if doc_type == "legal" else "news"
    title = metadata.get("title") or metadata.get("source", "Untitled")
    method = source.get("retrieval_method", "")
    chunk_index = metadata.get("chunk_index", "?")
    content = source.get("content", "")

    snippet_html = ""
    if show_snippet:
        snippet = content[:240].strip()
        if len(content) > 240:
            snippet += "…"
        snippet_html = f'<div class="evidence-snippet">{html_lib.escape(snippet)}</div>'

    st.markdown(
        '<div class="evidence-card">'
        '<div class="evidence-top">'
        f'<span class="evidence-rank">#{rank}</span>'
        f'<span class="evidence-badge {badge_class}">{html_lib.escape(doc_type.upper())}</span>'
        f'<span class="evidence-method">{html_lib.escape(method)}</span>'
        "</div>"
        f'<div class="evidence-title">{html_lib.escape(title)}</div>'
        '<div class="evidence-bar-track">'
        f'<div class="evidence-bar-fill" style="width:{relative:.0f}%"></div>'
        "</div>"
        f'<div class="evidence-meta">Relative rank score · chunk #{chunk_index}</div>'
        f"{snippet_html}"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander(f"📄 Xem toàn bộ ngữ cảnh — {title}"):
        st.markdown(content)
        url = metadata.get("url")
        if url:
            st.markdown(f"[Mở tài liệu gốc]({url})")


def render_developer_view(message: dict) -> None:
    with st.expander("⚙ Developer View — raw retrieval trace"):
        dense_hits = message.get("dense_hits", [])
        if dense_hits:
            st.markdown("**Dense search (semantic)**")
            for hit in dense_hits[:5]:
                st.text(f"{hit['score']:.3f}  {hit['metadata']['title'][:42]}")
        else:
            st.caption(
                "Bấm **🔬 Xem cosine score chính xác** phía trên để tải dense search "
                "(không tự chạy để tránh chậm mỗi lượt chat)."
            )
        st.caption(
            "So sánh với BM25 và hybrid (RRF): xem tab 🆚 Semantic vs Hybrid."
        )
        st.markdown("**Fused sources gửi vào LLM**")
        st.json(
            [
                {
                    "rank": i + 1,
                    "title": s["metadata"]["title"],
                    "score": round(s["score"], 4),
                    "method": s["retrieval_method"],
                }
                for i, s in enumerate(message.get("sources", []))
            ]
        )
        st.caption(
            f"retrieval_source: `{message.get('retrieval_source')}` · "
            f"score_threshold: `{SCORE_THRESHOLD}`"
        )
        if message.get("response_time_seconds") is not None:
            st.caption(
                f"response_time: `{message['response_time_seconds']:.2f}s`"
            )
        if message.get("error_type"):
            st.caption(f"provider_error: `{message['error_type']}`")


def render_confidence_section(message: dict, msg_key: str, top_k: int) -> None:
    retrieval_source = message.get("retrieval_source", "none")

    if "best_dense_score" in message:
        render_confidence_meter(message["best_dense_score"], SCORE_THRESHOLD, retrieval_source)
        return

    hint = (
        "⚠️ Dùng fallback PageIndex — điểm dense ban đầu dưới threshold."
        if retrieval_source == "pageindex"
        else "Trả lời dựa trên hybrid retrieval (dense + BM25)."
    )
    st.caption(hint)
    if st.button("🔬 Xem cosine score chính xác", key=f"confcheck-{msg_key}"):
        query = message.get("query", "")
        try:
            dense_hits = semantic_search(query, top_k=top_k) if query else []
        except Exception:
            dense_hits = []
        message["best_dense_score"] = dense_hits[0]["score"] if dense_hits else 0.0
        message["dense_hits"] = dense_hits
        render_confidence_meter(message["best_dense_score"], SCORE_THRESHOLD, retrieval_source)


def render_answer_extras(message: dict, dev_view: bool, msg_key: str, top_k: int) -> None:
    sources = message.get("sources") or []
    retrieval_source = message.get("retrieval_source", "none")
    failure_reason = message.get("failure_reason")

    if retrieval_source == "none":
        if failure_reason == "provider_error":
            st.warning(
                "Dịch vụ sinh câu trả lời đang bận, hết quota hoặc mất kết nối. "
                "Vui lòng thử lại sau."
            )
        else:
            st.markdown(
                '<div class="oos-banner">🚫 Ngoài phạm vi kiến thức — hệ thống từ chối suy đoán '
                "để tránh bịa thông tin, đúng theo nguyên tắc grounded generation.</div>",
                unsafe_allow_html=True,
            )
    else:
        render_confidence_section(message, msg_key, top_k)

    if failure_reason == "invalid_citation":
        st.warning(
            "Đã tìm thấy evidence nhưng LLM không trả citation đúng định dạng. "
            "Các nguồn truy xuất được giữ lại bên dưới để kiểm tra."
        )

    if sources:
        st.markdown('<div class="section-label">🔎 Evidence</div>', unsafe_allow_html=True)
        max_score = max(s["score"] for s in sources) or 1.0
        for rank, source in enumerate(sources, start=1):
            render_evidence_card(source, rank, max_score, show_snippet=rank <= 3)
    else:
        st.info("Không tìm thấy evidence đủ tin cậy trong knowledge base.")

    if dev_view:
        render_developer_view(message)


def render_compare_card(rank: int, result: dict, tag: str = "") -> None:
    metadata = result.get("metadata", {})
    title = metadata.get("title", metadata.get("source", "Untitled"))
    tag_html = f'<span class="compare-tag">{html_lib.escape(tag)}</span>' if tag else ""
    st.markdown(
        '<div class="compare-card">'
        f'<span class="compare-rank">#{rank}</span>'
        f'<span class="compare-title">{html_lib.escape(title)}</span>{tag_html}'
        f'<div class="compare-meta">score {result.get("score", 0.0):.4f} · '
        f'chunk #{metadata.get("chunk_index", "?")}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_compare_tab(default_top_k: int) -> None:
    st.markdown('<div class="section-label">🆚 Semantic (dense) vs Hybrid (RRF: dense+BM25)</div>', unsafe_allow_html=True)
    st.caption(
        "So sánh trực tiếp thứ hạng khi chỉ dùng dense embedding so với sau khi fuse với "
        "BM25 bằng Reciprocal Rank Fusion — minh hoạ tại sao hybrid retrieval hữu ích hơn."
    )
    query_ab = st.text_input(
        "Câu hỏi để so sánh",
        value=SUGGESTIONS[0]["question"],
        key="compare_query",
    )
    if st.button("🔍 So sánh", key="compare_run") and query_ab.strip():
        dense = semantic_search(query_ab, top_k=default_top_k)
        sparse = lexical_search(query_ab, top_k=default_top_k)
        hybrid = rerank_rrf([dense, sparse], top_k=default_top_k)
        st.session_state["compare_result"] = {
            "query": query_ab,
            "dense": dense,
            "hybrid": hybrid,
        }

    cached = st.session_state.get("compare_result")
    if not cached:
        st.info("Nhập câu hỏi rồi bấm **So sánh** để xem kết quả.")
        return

    dense = cached["dense"]
    hybrid = cached["hybrid"]
    st.caption(f"Kết quả cho: _{cached['query']}_")
    dense_ranks = {item["id"]: i + 1 for i, item in enumerate(dense)}

    col_dense, col_hybrid = st.columns(2)
    with col_dense:
        st.markdown("**🔵 Semantic only (dense cosine)**")
        for i, item in enumerate(dense, start=1):
            render_compare_card(i, item)
    with col_hybrid:
        st.markdown("**🟣 Hybrid (sau RRF)**")
        for i, item in enumerate(hybrid, start=1):
            previous_rank = dense_ranks.get(item["id"])
            if previous_rank is None:
                tag = "🆕 nhờ BM25"
            elif previous_rank != i:
                tag = f"dense #{previous_rank}"
            else:
                tag = ""
            render_compare_card(i, item, tag)

    st.caption(
        "Lưu ý: RRF score chỉ phản ánh thứ hạng gộp, không cùng thang đo với cosine "
        "similarity — không nên so sánh trực tiếp hai cột số điểm."
    )


@st.cache_data(show_spinner=False)
def cached_documents() -> list[dict]:
    return load_documents()


def render_chunking_tab() -> None:
    st.markdown('<div class="section-label">📐 Chunking thật từ Task 4</div>', unsafe_allow_html=True)
    st.caption(
        f"Recursive character splitter · chunk_size={CHUNK_SIZE} · chunk_overlap={CHUNK_OVERLAP} "
        "— đúng tham số đang dùng khi index vào ChromaDB."
    )
    documents = cached_documents()
    if not documents:
        st.info("Chưa có tài liệu chuẩn hoá trong data/standardized/.")
        return

    titles = [doc["metadata"]["title"] for doc in documents]
    index = st.selectbox("Chọn tài liệu", range(len(documents)), format_func=lambda i: titles[i])
    document = documents[index]
    chunks = chunk_documents([document])

    st.caption(f"{len(document['content']):,} ký tự gốc → **{len(chunks)} chunks**")

    blocks = "".join(
        f'<div class="chunk-block" style="flex:{max(len(c["content"]), 1)}; '
        f'background:{CHUNK_PALETTE[i % len(CHUNK_PALETTE)]}" title="Chunk {i} · '
        f'{len(c["content"])} ký tự">{i}</div>'
        for i, c in enumerate(chunks)
    )
    st.markdown(f'<div class="chunk-strip">{blocks}</div>', unsafe_allow_html=True)

    for i, chunk in enumerate(chunks):
        with st.expander(f"Chunk #{i} · {len(chunk['content'])} ký tự"):
            st.code(chunk["content"])


@st.cache_data(show_spinner=False)
def load_golden_dataset() -> list[dict]:
    if not GOLDEN_DATASET_PATH.exists():
        return []
    return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))


def evaluate_golden_item(item: dict, top_k: int) -> dict:
    result = generate_with_citation(item["question"], top_k=top_k)
    actual_sources = sorted({s["metadata"]["source"] for s in result["sources"]})
    expected_sources = set(item.get("expected_sources", []))
    answerable = item.get("answerable", True)
    if answerable:
        passed = result["retrieval_source"] != "none" and bool(
            expected_sources & set(actual_sources)
        )
    else:
        passed = result["retrieval_source"] == "none"
    return {
        **item,
        "actual_answer": result["answer"],
        "actual_sources": actual_sources,
        "retrieval_source": result["retrieval_source"],
        "passed": passed,
    }


def render_golden_tab(default_top_k: int) -> None:
    st.markdown(
        '<div class="section-label">✅ Golden Set — 15 câu đánh giá</div>',
        unsafe_allow_html=True,
    )
    golden_items = load_golden_dataset()
    if not golden_items:
        st.info("Chưa có group_project/evaluation/golden_dataset.json.")
        return

    st.caption(
        f"{len(golden_items)} câu hỏi golden — chạy qua đúng "
        "`generate_with_citation` thật, không phải dữ liệu giả lập. "
        "Đạt/Trượt xét bằng: câu trả lời không bị từ chối và có ít nhất một "
        "nguồn trùng với `expected_sources` (với câu ngoài phạm vi thì Đạt "
        "nghĩa là hệ thống từ chối đúng)."
    )

    if st.button("▶️ Chạy toàn bộ qua pipeline", key="golden_run"):
        progress = st.progress(0.0, text="Đang chạy...")
        results = []
        for i, item in enumerate(golden_items, start=1):
            results.append(evaluate_golden_item(item, default_top_k))
            progress.progress(
                i / len(golden_items), text=f"{i}/{len(golden_items)} · {item['id']}"
            )
        progress.empty()
        st.session_state["golden_results"] = results

    results = st.session_state.get("golden_results")
    if not results:
        st.info("Bấm nút phía trên để chạy 15 câu golden qua pipeline thật.")
        return

    passed_count = sum(1 for r in results if r["passed"])
    refusal_items = [r for r in results if not r.get("answerable", True)]
    refusal_correct = sum(1 for r in refusal_items if r["passed"])
    refusal_display = f"{refusal_correct}/{len(refusal_items)}" if refusal_items else "—"
    st.markdown(
        '<div class="golden-summary">'
        f'<div class="golden-chip"><div class="value">{passed_count}/{len(results)}</div>'
        '<div class="label">Đạt / Tổng</div></div>'
        f'<div class="golden-chip"><div class="value">{refusal_display}</div>'
        '<div class="label">Từ chối đúng (out-of-domain)</div></div>'
        f'<div class="golden-chip"><div class="value">{default_top_k}</div>'
        '<div class="label">Top-k đang dùng</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    for r in results:
        status_icon = "✅" if r["passed"] else "❌"
        with st.expander(f"{status_icon} {r['id']} · {r['category']} · {r['question'][:70]}"):
            st.markdown(f"**Câu hỏi:** {r['question']}")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Expected answer**")
                st.caption(r["expected_answer"])
                st.caption(
                    "Expected sources: "
                    + (", ".join(r.get("expected_sources", [])) or "—")
                )
            with col_b:
                st.markdown("**Actual answer (pipeline thật)**")
                st.caption(r["actual_answer"])
                st.caption("Actual sources: " + (", ".join(r["actual_sources"]) or "—"))
            st.caption(
                f"retrieval_source: `{r['retrieval_source']}` · "
                f"difficulty: `{r['difficulty']}` · answerable: `{r['answerable']}`"
            )


st.set_page_config(page_title="FIDE Chess Law Assistant", page_icon="♟️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### 📚 Knowledge Base")
    legal_sources = load_legal_sources()
    news_sources = load_news_sources()
    counts = chunk_counts_by_stem()

    c1, c2, c3 = st.columns(3)
    c1.metric("Docs", len(legal_sources) + len(news_sources))
    c2.metric("Legal", len(legal_sources))
    c3.metric("News", len(news_sources))

    with st.expander(f"⚖ Legal documents ({len(legal_sources)})"):
        for item in legal_sources:
            stem = Path(item["filename"]).stem
            n_chunks = counts.get(stem, 0)
            flag = " 🇻🇳" if item.get("language") == "vi" else ""
            st.markdown(f"**{item['title']}**{flag}")
            st.caption(f"{n_chunks} chunks indexed" if n_chunks else "chưa index")

    with st.expander(f"📰 Knowledge articles ({len(news_sources)})"):
        for item in news_sources:
            stem = Path(item["filename"]).stem
            n_chunks = counts.get(stem, 0)
            st.markdown(f"**{item['title']}**")
            st.caption(f"{n_chunks} chunks indexed" if n_chunks else "chưa index")

    st.divider()
    top_k = st.slider("Số chunks truy hồi", 3, 10, 5)
    st.caption(f"Score threshold hiện tại: **{SCORE_THRESHOLD:.2f}**")
    dev_view = st.toggle("⚙ Developer View", value=False)
    if st.button("🗑 Xoá hội thoại", use_container_width=True):
        st.session_state.messages = []

render_header()

chat_query = st.chat_input("Hỏi về luật cờ vua FIDE...")

tab_chat, tab_compare, tab_chunking, tab_golden = st.tabs(
    ["💬 Chat", "🆚 Semantic vs Hybrid", "📐 Chunking Demo", "✅ Golden Set"]
)

with tab_chat:
    if not st.session_state.messages:
        render_hero_board()
        picked_question = render_suggestions()
    else:
        picked_question = None

    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                render_answer_extras(message, dev_view, msg_key=str(index), top_k=top_k)

    query = picked_question or chat_query

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            stepper_slot = st.empty()
            render_stepper(stepper_slot, "ask")
            time.sleep(0.1)
            render_stepper(stepper_slot, "retrieve")
            time.sleep(0.15)
            render_stepper(stepper_slot, "rerank")
            time.sleep(0.1)
            render_stepper(stepper_slot, "generate")

            response_started = time.perf_counter()
            result = generate_with_citation(query, top_k=top_k)
            response_time_seconds = time.perf_counter() - response_started

            render_stepper(stepper_slot, "done")
            time.sleep(0.15)
            stepper_slot.empty()

            st.markdown(result["answer"])

            assistant_message = {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "retrieval_source": result["retrieval_source"],
                "query": query,
                "response_time_seconds": response_time_seconds,
                "failure_reason": result.get("failure_reason"),
                "error_type": result.get("error_type"),
            }
            render_answer_extras(
                assistant_message,
                dev_view,
                msg_key=str(len(st.session_state.messages)),
                top_k=top_k,
            )
            st.session_state.messages.append(assistant_message)

with tab_compare:
    render_compare_tab(top_k)

with tab_chunking:
    render_chunking_tab()

with tab_golden:
    render_golden_tab(top_k)
