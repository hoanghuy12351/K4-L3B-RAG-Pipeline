"""Đánh giá A/B dense-only và hybrid + RRF trên golden dataset.

Chương trình lưu tiến độ sau từng mẫu để có thể chạy lại mà không gọi API lại
cho những phần đã hoàn thành.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .task10_generation import (
    SAFE_REFUSAL,
    SYSTEM_PROMPT,
    _has_valid_citations,
    call_llm,
    format_context,
    reorder_for_llm,
)
from .task9_retrieval_pipeline import retrieve


ROOT_DIR = Path(__file__).parent.parent
EVALUATION_DIR = ROOT_DIR / "group_project" / "evaluation"
GOLDEN_PATH = EVALUATION_DIR / "golden_dataset.json"
RAW_RESULTS_PATH = EVALUATION_DIR / "evaluation_results.json"
REPORT_PATH = EVALUATION_DIR / "RESULT.md"
TOP_K = 5
CONFIGS = {
    "dense_only": False,
    "hybrid_rrf": True,
}
METRIC_NAMES = (
    "faithfulness",
    "answer_relevance",
    "context_recall",
    "context_precision",
)

load_dotenv()


def _is_quota_error(error: Exception) -> bool:
    message = str(error).upper()
    return "RESOURCE_EXHAUSTED" in message or "QUOTA" in message


def load_golden_dataset(path: Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    """Đọc và kiểm tra tối thiểu schema của golden dataset."""
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or len(dataset) < 15:
        raise ValueError("Golden dataset phải là list có ít nhất 15 mẫu")

    required = {"question", "expected_answer", "expected_context"}
    seen_questions = set()
    for index, item in enumerate(dataset):
        if not isinstance(item, dict):
            raise ValueError(f"Golden case {index} phải là object")
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Golden case {index} thiếu {sorted(missing)}")
        if any(not str(item[key]).strip() for key in required):
            raise ValueError(f"Golden case {index} có trường bắt buộc rỗng")
        question = str(item["question"]).strip()
        if question in seen_questions:
            raise ValueError(f"Câu hỏi trùng: {question}")
        seen_questions.add(question)
    return dataset


def _generate_from_chunks(query: str, chunks: list[dict]) -> dict[str, Any]:
    """Sinh câu trả lời từ chunks có sẵn để hai cấu hình dùng cùng generator."""
    if not chunks:
        return {"answer": SAFE_REFUSAL, "sources": []}

    citation_by_id = {
        chunk["id"]: index
        for index, chunk in enumerate(chunks, 1)
    }
    reordered = reorder_for_llm(chunks)
    context_chunks = [
        {**chunk, "_citation_index": citation_by_id[chunk["id"]]}
        for chunk in reordered
    ]
    context = format_context(context_chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    last_error: Exception | None = None
    answer = ""
    for attempt in range(3):
        try:
            answer = call_llm(SYSTEM_PROMPT, user_message)
            last_error = None
            break
        except Exception as error:
            last_error = error
            if _is_quota_error(error):
                break
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
    if last_error is not None:
        return {
            "answer": SAFE_REFUSAL,
            "sources": [],
            "generation_error": f"{type(last_error).__name__}: {last_error}",
        }

    if not answer or not _has_valid_citations(answer, len(chunks)):
        return {
            "answer": SAFE_REFUSAL,
            "sources": [],
            "generation_error": "LLM response had missing or invalid citations",
        }
    return {"answer": answer, "sources": chunks}


def run_configuration(
    sample: dict[str, Any],
    config_name: str,
    use_reranking: bool,
) -> dict[str, Any]:
    """Chạy retrieval và generation cho một mẫu."""
    started = time.perf_counter()
    chunks = retrieve(
        sample["question"],
        top_k=TOP_K,
        score_threshold=0.0,  # Tắt fallback để A/B chỉ khác retrieval strategy.
        use_reranking=use_reranking,
    )
    generated = _generate_from_chunks(sample["question"], chunks)
    elapsed = time.perf_counter() - started

    sources = generated["sources"]
    return {
        "id": sample.get("id") or sample["question"],
        "config": config_name,
        "question": sample["question"],
        "reference_answer": sample["expected_answer"],
        "reference_context": sample["expected_context"],
        "expected_sources": sample.get("expected_sources", []),
        "answerable": bool(sample.get("answerable", True)),
        "category": sample.get("category", "unknown"),
        "difficulty": sample.get("difficulty", "unknown"),
        "response": generated["answer"],
        "contexts": [source["content"] for source in sources],
        "source_ids": [source["id"] for source in sources],
        "source_files": [source["metadata"]["source"] for source in sources],
        "retrieval_methods": [source["retrieval_method"] for source in sources],
        "best_dense_score": (
            float(chunks[0]["score"])
            if config_name == "dense_only" and chunks
            else None
        ),
        "latency_seconds": round(elapsed, 3),
        "generation_error": generated.get("generation_error"),
        "metrics": {},
        "metric_errors": {},
    }


def _build_ragas_components():
    """Tạo evaluator LLM và embedding local cho RAGAS 0.4.x."""
    from ragas.cache import DiskCacheBackend
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    model = (
        os.getenv("EVALUATOR_MODEL", "").strip()
        or os.getenv("LLM_MODEL", "").strip()
    )
    cache = DiskCacheBackend(cache_dir=str(ROOT_DIR / ".cache" / "ragas"))

    if provider == "gemini":
        import instructor
        from google import genai
        from ragas.llms.base import InstructorLLM

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY")
        # RAGAS 0.4.3 không truyền use_async khi patch google-genai client.
        # Patch trực tiếp để các metric.ascore() dùng được async Gemini client.
        google_client = genai.Client(api_key=api_key)
        patched_client = instructor.from_genai(
            google_client,
            use_async=True,
            model=model,
        )
        evaluator_llm = InstructorLLM(
            client=patched_client,
            model=model,
            provider="google",
            cache=cache,
            temperature=0,
        )
    elif provider == "openai":
        from openai import AsyncOpenAI

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        evaluator_llm = llm_factory(
            model,
            provider="openai",
            client=AsyncOpenAI(api_key=api_key),
            cache=cache,
            temperature=0,
        )
    elif provider == "anthropic":
        from anthropic import AsyncAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY")
        evaluator_llm = llm_factory(
            model,
            provider="anthropic",
            client=AsyncAnthropic(api_key=api_key),
            cache=cache,
            temperature=0,
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER for evaluation: {provider}")

    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip()
    evaluator_embeddings = embedding_factory(
        "huggingface",
        model=embedding_model,
        interface="modern",
        cache=cache,
    )
    return {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevance": AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            strictness=1,
        ),
        "context_recall": ContextRecall(llm=evaluator_llm),
        "context_precision": ContextPrecision(llm=evaluator_llm),
    }


async def score_record(record: dict[str, Any], metrics: dict[str, Any]) -> None:
    """Tính bốn metric, giữ lỗi từng metric để có thể resume."""
    calls = {
        "faithfulness": {
            "user_input": record["question"],
            "response": record["response"],
            "retrieved_contexts": record["contexts"],
        },
        "answer_relevance": {
            "user_input": record["question"],
            "response": record["response"],
        },
        "context_recall": {
            "user_input": record["question"],
            "retrieved_contexts": record["contexts"],
            "reference": record["reference_answer"],
        },
        "context_precision": {
            "user_input": record["question"],
            "reference": record["reference_answer"],
            "retrieved_contexts": record["contexts"],
        },
    }

    for name, metric in metrics.items():
        if record["metrics"].get(name) is not None:
            continue
        if not record["contexts"] and name != "answer_relevance":
            record["metrics"][name] = 1.0 if not record["answerable"] else 0.0
            continue
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                result = await metric.ascore(**calls[name])
                value = float(result.value)
                record["metrics"][name] = (
                    None if math.isnan(value) else round(value, 6)
                )
                record["metric_errors"].pop(name, None)
                last_error = None
                break
            except Exception as error:
                last_error = error
                if _is_quota_error(error):
                    break
                if attempt < 2:
                    await asyncio.sleep(2 ** (attempt + 1))
        if last_error is not None:
            record["metrics"][name] = None
            record["metric_errors"][name] = (
                f"{type(last_error).__name__}: {last_error}"
            )


def _save_results(records: list[dict[str, Any]]) -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    RAW_RESULTS_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _averages(records: list[dict[str, Any]], config: str) -> dict[str, float | None]:
    selected = [record for record in records if record["config"] == config]
    output: dict[str, float | None] = {}
    for metric in METRIC_NAMES:
        values = [
            record["metrics"].get(metric)
            for record in selected
            if record["metrics"].get(metric) is not None
        ]
        output[metric] = round(statistics.fmean(values), 4) if values else None
    return output


def calibrate_threshold(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Chọn threshold tối đa balanced accuracy trên nhãn answerable."""
    dense = [
        record
        for record in records
        if record["config"] == "dense_only" and record["best_dense_score"] is not None
    ]
    if not dense:
        return {"recommended": None, "balanced_accuracy": None}

    best = (0.0, -1.0)
    for step in range(101):
        threshold = step / 100
        positives = [record for record in dense if record["answerable"]]
        negatives = [record for record in dense if not record["answerable"]]
        tpr = (
            sum(record["best_dense_score"] >= threshold for record in positives)
            / len(positives)
            if positives else 1.0
        )
        tnr = (
            sum(record["best_dense_score"] < threshold for record in negatives)
            / len(negatives)
            if negatives else 1.0
        )
        balanced_accuracy = (tpr + tnr) / 2
        if balanced_accuracy > best[1]:
            best = (threshold, balanced_accuracy)
    return {
        "recommended": best[0],
        "balanced_accuracy": round(best[1], 4),
        "in_domain_scores": [
            round(record["best_dense_score"], 4)
            for record in dense if record["answerable"]
        ],
        "out_of_domain_scores": [
            round(record["best_dense_score"], 4)
            for record in dense if not record["answerable"]
        ],
    }


def write_report(records: list[dict[str, Any]]) -> None:
    averages = {config: _averages(records, config) for config in CONFIGS}
    calibration = calibrate_threshold(records)
    metric_label = {
        "faithfulness": "Faithfulness",
        "answer_relevance": "Answer relevance",
        "context_recall": "Context recall",
        "context_precision": "Context precision",
    }
    expected_record_count = len(load_golden_dataset()) * len(CONFIGS)
    complete = (
        len(records) == expected_record_count
        and all(
            all(record["metrics"].get(name) is not None for name in METRIC_NAMES)
            for record in records
        )
    )
    evaluator_model = (
        os.getenv("EVALUATOR_MODEL", "").strip()
        or os.getenv("LLM_MODEL", "").strip()
    )
    lines = [
        "# RAG evaluation results",
        "",
        "## Run information",
        "",
        f"- Evaluation status: {'COMPLETE' if complete else 'INCOMPLETE'}",
        f"- Framework: RAGAS 0.4.3",
        f"- Generator/evaluator provider: {os.getenv('LLM_PROVIDER', '')}",
        f"- Generator model: {os.getenv('LLM_MODEL', '')}",
        f"- Evaluator model: {evaluator_model}",
        f"- Embedding model: {os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')}",
        f"- Evaluated golden samples: {len({record['id'] for record in records})}/"
        f"{len(load_golden_dataset())}",
        f"- `top_k`: {TOP_K}",
        "",
        "## Configurations",
        "",
        "- **Config A — dense-only:** semantic retrieval, không RRF, không fallback.",
        "- **Config B — hybrid + RRF:** semantic + BM25, hợp nhất bằng RRF, không fallback.",
        "- Hai cấu hình dùng cùng corpus, generator, evaluator, prompt và `top_k`.",
        "",
        "## Overall scores",
        "",
        "| Metric | Dense-only | Hybrid + RRF | Delta B−A |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in METRIC_NAMES:
        left = averages["dense_only"][metric]
        right = averages["hybrid_rrf"][metric]
        delta = round(right - left, 4) if left is not None and right is not None else None
        lines.append(
            f"| {metric_label[metric]} | {left if left is not None else 'N/A'} "
            f"| {right if right is not None else 'N/A'} "
            f"| {delta if delta is not None else 'N/A'} |"
        )

    config_means = {}
    for config, values in averages.items():
        valid = [value for value in values.values() if value is not None]
        config_means[config] = statistics.fmean(valid) if valid else float("nan")
    comparable = {
        key: value for key, value in config_means.items()
        if not math.isnan(value)
    }
    winner = max(comparable, key=comparable.get) if comparable else "N/A"
    lines.extend([
        "",
        "## A/B comparison",
        "",
        f"- Cấu hình có điểm trung bình cao hơn: **{winner}**.",
        f"- Dense-only mean: {config_means['dense_only']:.4f}.",
        f"- Hybrid + RRF mean: {config_means['hybrid_rrf']:.4f}.",
        "- Trade-off: hybrid có thêm chi phí dựng BM25/RRF nhưng tận dụng được từ khóa chính xác.",
        "",
        "## Threshold calibration",
        "",
        f"- Threshold đề xuất: {calibration.get('recommended')}.",
        f"- Balanced accuracy: {calibration.get('balanced_accuracy')}.",
        f"- In-domain dense scores: {calibration.get('in_domain_scores', [])}.",
        f"- Out-of-domain dense scores: {calibration.get('out_of_domain_scores', [])}.",
        "",
        "## Worst performers",
        "",
        "| Config | ID | Mean score | Question | Error type |",
        "| --- | --- | ---: | --- | --- |",
    ])
    ranked = []
    for record in records:
        values = [
            value for value in record["metrics"].values()
            if value is not None
        ]
        mean_score = statistics.fmean(values) if values else 0.0
        errors = []
        if record.get("generation_error"):
            errors.append("generation")
        if record.get("metric_errors"):
            errors.append("evaluation")
        if not errors:
            errors.append("retrieval/data")
        ranked.append((mean_score, record, ", ".join(errors)))
    for mean_score, record, error_type in sorted(ranked, key=lambda item: item[0])[:5]:
        question = record["question"].replace("|", "\\|")
        lines.append(
            f"| {record['config']} | {record['id']} | {mean_score:.4f} "
            f"| {question} | {error_type} |"
        )

    lines.extend([
        "",
        "## Recommendations",
        "",
        "1. Dùng threshold đã hiệu chỉnh thay cho giá trị mặc định nếu balanced accuracy tốt hơn.",
        "2. Kiểm tra thủ công các mẫu có context recall hoặc faithfulness thấp.",
        "3. Giữ hybrid + RRF khi nó cải thiện điểm tổng; nếu không, rà tokenizer BM25 cho corpus đa ngôn ngữ.",
        "4. Đánh giá PageIndex fallback riêng vì A/B trên đây cố ý cô lập retrieval strategy.",
        "",
        f"Raw results: `{RAW_RESULTS_PATH.relative_to(ROOT_DIR).as_posix()}`.",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run_evaluation(limit: int | None = None, force: bool = False) -> None:
    dataset = load_golden_dataset()
    if limit is not None:
        dataset = dataset[:limit]

    records: list[dict[str, Any]] = []
    if RAW_RESULTS_PATH.exists() and not force:
        records = json.loads(RAW_RESULTS_PATH.read_text(encoding="utf-8"))
    existing = {(record["id"], record["config"]): record for record in records}

    for sample in dataset:
        sample_id = sample.get("id") or sample["question"]
        for config_name, use_reranking in CONFIGS.items():
            key = (sample_id, config_name)
            if key in existing:
                continue
            print(f"Generating {sample_id} / {config_name}...")
            record = run_configuration(sample, config_name, use_reranking)
            records.append(record)
            existing[key] = record
            _save_results(records)

    metrics = _build_ragas_components()
    selected_keys = {
        (sample.get("id") or sample["question"], config)
        for sample in dataset
        for config in CONFIGS
    }
    for record in records:
        if (record["id"], record["config"]) not in selected_keys:
            continue
        if all(record["metrics"].get(name) is not None for name in METRIC_NAMES):
            continue
        print(f"Scoring {record['id']} / {record['config']}...")
        await score_record(record, metrics)
        _save_results(records)

    evaluated = [
        record for record in records
        if (record["id"], record["config"]) in selected_keys
    ]
    write_report(evaluated)
    print(f"Saved: {RAW_RESULTS_PATH}")
    print(f"Saved: {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    dataset = load_golden_dataset()
    print(f"Golden dataset valid: {len(dataset)} samples")
    if args.validate_only:
        return
    asyncio.run(run_evaluation(limit=args.limit, force=args.force))


if __name__ == "__main__":
    main()
