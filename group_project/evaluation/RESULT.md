# RAG evaluation results

## Run information

- Evaluation status: COMPLETE
- Framework: RAGAS 0.4.3
- Generator/evaluator provider: openai
- Generator model: gpt-4o-mini
- Evaluator model: gpt-4o-mini
- Embedding model: BAAI/bge-m3
- Evaluated golden samples: 15/15
- `top_k`: 5

## Configurations

- **Config A — dense-only:** semantic retrieval, không RRF, không fallback.
- **Config B — hybrid + RRF:** semantic + BM25, hợp nhất bằng RRF, không fallback.
- Hai cấu hình dùng cùng corpus, generator, evaluator, prompt và `top_k`.

## Overall scores

| Metric | Dense-only | Hybrid + RRF | Delta B−A |
| --- | ---: | ---: | ---: |
| Faithfulness | 0.8611 | 0.8933 | 0.0322 |
| Answer relevance | 0.7707 | 0.7898 | 0.0191 |
| Context recall | 0.7756 | 0.9256 | 0.15 |
| Context precision | 0.7011 | 0.732 | 0.0309 |

## A/B comparison

- Cấu hình có điểm trung bình cao hơn: **hybrid_rrf**.
- Dense-only mean: 0.7771.
- Hybrid + RRF mean: 0.8352.
- Chênh lệch trung bình Hybrid − Dense: 0.0581.
- Trade-off: hybrid có thêm chi phí dựng BM25/RRF nhưng tận dụng được từ khóa chính xác.

## Threshold calibration

- Threshold đề xuất: 0.45.
- Balanced accuracy: 1.0.
- In-domain dense scores: [0.6435, 0.6806, 0.7107, 0.6125, 0.6405, 0.7318, 0.7105, 0.7349, 0.6826, 0.7287, 0.5782, 0.6609, 0.6668].
- Out-of-domain dense scores: [0.3752, 0.4491].

## Worst performers

| Config | ID | Mean score | Question | Error type |
| --- | --- | ---: | --- | --- |
| dense_only | golden_011 | 0.2045 | Which conditions permanently remove the right to castle, and which conditions temporarily prevent castling under the FIDE Laws effective from 2023? | retrieval/data |
| dense_only | golden_003 | 0.3379 | When is a chess game automatically drawn because of fivefold repetition or the 75-move rule under the FIDE Laws effective from 2023? | retrieval/data |
| hybrid_rrf | golden_003 | 0.4233 | When is a chess game automatically drawn because of fivefold repetition or the 75-move rule under the FIDE Laws effective from 2023? | retrieval/data |
| dense_only | golden_001 | 0.4774 | Under the FIDE Laws of Chess effective from 2023, what penalties apply to a player's first and second completed illegal moves? | retrieval/data |
| hybrid_rrf | golden_011 | 0.7387 | Which conditions permanently remove the right to castle, and which conditions temporarily prevent castling under the FIDE Laws effective from 2023? | retrieval/data |

## Recommendations

1. Kiểm thử và áp dụng threshold 0.45 thay cho 0.55; trên golden set giá trị này đạt balanced accuracy 1.0.
2. Ưu tiên phân tích `golden_011`, `golden_003` và `golden_001`, là các mẫu có điểm thấp nhất.
3. Dùng **hybrid_rrf** làm cấu hình retrieval mặc định; nó cải thiện điểm trung bình 0.0581.
4. Đánh giá PageIndex fallback riêng vì A/B trên đây cố ý cô lập retrieval strategy.

Raw results: `group_project/evaluation/evaluation_results.json`.
