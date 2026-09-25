# RAG evaluation results

## Run information

- Evaluation status: INCOMPLETE
- Framework: RAGAS 0.4.3
- Generator/evaluator provider: gemini
- Generator model: gemini-3.6-flash
- Evaluator model: gemini-3.6-flash
- Embedding model: BAAI/bge-m3
- Evaluated golden samples: 1/15
- `top_k`: 5

## Configurations

- **Config A — dense-only:** semantic retrieval, không RRF, không fallback.
- **Config B — hybrid + RRF:** semantic + BM25, hợp nhất bằng RRF, không fallback.
- Hai cấu hình dùng cùng corpus, generator, evaluator, prompt và `top_k`.

## Overall scores

| Metric | Dense-only | Hybrid + RRF | Delta B−A |
| --- | ---: | ---: | ---: |
| Faithfulness | 1.0 | N/A | N/A |
| Answer relevance | 0.6875 | N/A | N/A |
| Context recall | 0.0 | N/A | N/A |
| Context precision | N/A | N/A | N/A |

## A/B comparison

- Cấu hình có điểm trung bình cao hơn: **dense_only**.
- Dense-only mean: 0.5625.
- Hybrid + RRF mean: nan.
- Trade-off: hybrid có thêm chi phí dựng BM25/RRF nhưng tận dụng được từ khóa chính xác.

## Threshold calibration

- Threshold đề xuất: 0.0.
- Balanced accuracy: 1.0.
- In-domain dense scores: [0.6435].
- Out-of-domain dense scores: [].

## Worst performers

| Config | ID | Mean score | Question | Error type |
| --- | --- | ---: | --- | --- |
| hybrid_rrf | golden_001 | 0.0000 | Under the FIDE Laws of Chess effective from 2023, what penalties apply to a player's first and second completed illegal moves? | evaluation |
| dense_only | golden_001 | 0.5625 | Under the FIDE Laws of Chess effective from 2023, what penalties apply to a player's first and second completed illegal moves? | evaluation |

## Recommendations

1. Dùng threshold đã hiệu chỉnh thay cho giá trị mặc định nếu balanced accuracy tốt hơn.
2. Kiểm tra thủ công các mẫu có context recall hoặc faithfulness thấp.
3. Giữ hybrid + RRF khi nó cải thiện điểm tổng; nếu không, rà tokenizer BM25 cho corpus đa ngôn ngữ.
4. Đánh giá PageIndex fallback riêng vì A/B trên đây cố ý cô lập retrieval strategy.

Raw results: `group_project/evaluation/evaluation_results.json`.
