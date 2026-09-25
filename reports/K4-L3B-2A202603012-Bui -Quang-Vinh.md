# Individual contribution report

## Thông tin

* Họ và tên: Bùi Quang Vinh
* Mã học viên: 2A202603012
* Nhóm: K4-L3B
* Repository/branch: https://github.com/hoanghuy12351/K4-L3B-RAG-Pipeline @ main

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm                                                         | File/commit/PR                                          | Trạng thái |
| ------------------ | ------------------------------------------------------------------------------ | ------------------------------------------------------- | ---------- |
| Audit Task 1–10    | Kiểm tra yêu cầu lab, rà soát các task đã hoàn thành và phần còn thiếu         | `README.md`, `src/`, `tests/`                           | Done       |
| Retrieval pipeline | Kiểm tra luồng Dense + BM25 → RRF → fallback và các tham số `top_k`, threshold | `task7_reranking.py`, `task9_retrieval_pipeline.py`     | Done       |
| Task 8–10          | Kiểm tra PageIndex fallback, generation và citation                            | `task8_pageindex_vectorless.py`, `task10_generation.py` | Done       |
| Testing            | Kiểm tra contract, acceptance test và luồng end-to-end                         | `tests/`, `group_project/evaluation/RESULT.md`          | Done       |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Dùng dense cosine score để kiểm tra fallback, không dùng RRF score.
   **Lý do/evidence:** Hai loại score có thang đo khác nhau.
   **Trade-off:** Cần giữ dense result riêng nhưng threshold chính xác hơn.

2. **Quyết định:** Threshold cần được calibration theo dữ liệu.
   **Lý do/evidence:** Evaluation đề xuất `0.45`, trong khi code mặc định là `0.55`.
   **Trade-off:** Phải điều chỉnh lại khi corpus thay đổi.

## Kiểm thử và kết quả

* Test hoặc query tôi đã dùng: `pytest tests/test_contracts.py -q`, `pytest tests/test_acceptance.py -q`, `pytest -q`.
* Kết quả trước/sau nếu có: xác minh pipeline Task 1–10 đã được tích hợp; `top_k=5`, RRF chạy một lần và fallback dùng dense score.
* Lỗi đã phát hiện và cách xử lý: phát hiện threshold giữa code và evaluation chưa đồng nhất; cần cấu hình lại theo kết quả calibration.

## Điều còn hạn chế

* Một hạn chế cụ thể của phần tôi làm: chủ yếu tập trung vào audit, kiểm thử và kiểm tra tích hợp pipeline.
* Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: bổ sung smoke test end-to-end cho PageIndex và generation.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

* Ngày: 25/09/2026
* Tên thành viên: Bùi Quang Vinh
