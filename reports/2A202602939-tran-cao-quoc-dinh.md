# Individual contribution report

## Thông tin

- Họ và tên: Trần Cao Quốc Dinh
- Mã học viên: 2A202602939
- Nhóm: 
- Repository/branch: https://github.com/hoanghuy12351/K4-L3B-RAG-Pipeline @ main

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| UI/UX chatbot (Streamlit) | Redesign toàn bộ `app.py`: theme "Luxury Chess + AI Lab", layout, chat flow | `app.py` | Done |
| Animated retrieval pipeline | Stepper trực quan Ask → Retrieve → Rerank → Generate → Answer, dữ liệu là kết quả thật của `generate_with_citation`, chỉ animate thứ tự hiển thị | `app.py::render_stepper` | Done |
| Evidence/citation panel + RAG Confidence meter | Card nguồn có badge Legal/News, relative rank score, expander xem ngữ cảnh + link gốc; đồng hồ confidence dùng đúng cosine score gốc so với `SCORE_THRESHOLD` | `app.py::render_evidence_card`, `render_confidence_meter` | Done |
| Document Explorer (sidebar) | Liệt kê 6 tài liệu legal + 13 bài news thật từ `data/landing/`, số chunks đã index đọc trực tiếp từ ChromaDB | `app.py::chunk_counts_by_stem`, sidebar section | Done |
| Developer View | Hiện breakdown dense/BM25/fused score thật để "show your work" | `app.py::render_developer_view` | Done |
| So sánh Semantic vs Hybrid (A/B) | Tab so sánh trực tiếp thứ hạng dense-only và sau khi fuse RRF+BM25 cho cùng một câu hỏi | `app.py::render_compare_tab` | Done |
| Demo chunking thật | Tab chọn tài liệu, gọi đúng `chunk_documents` của Task 4 (chunk_size=500, overlap=50) và trực quan hoá chunk theo tỉ lệ độ dài | `app.py::render_chunking_tab` | Done |
| Demo an toàn ngoài phạm vi | Câu hỏi ngoài domain cờ vua + banner cảnh báo rõ khi hệ thống từ chối trả lời (`retrieval_source == "none"`) | `app.py::OUT_OF_SCOPE_QUESTION`, `.oos-banner` | Done |
| Setup môi trường dev | Tạo `.venv`, `.env`, đồng bộ `SCORE_THRESHOLD=0.55` đã hiệu chỉnh và cấu hình provider với nhóm | `.env`, `.venv/` | Done |

Ghi chú: các thay đổi trên hiện ở trạng thái uncommitted trên nhánh `main` (`git diff --stat app.py`: 631 dòng thêm / 26 dòng xoá) — sẽ commit trước khi nộp.

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Không hiển thị trực tiếp `sources[0]["score"]` (RRF hoặc PageIndex) làm "RAG Confidence", mà gọi riêng `semantic_search()` để lấy cosine score gốc dùng cho đồng hồ confidence.
   **Lý do/evidence:** `task7_reranking.py` (dòng 7) và `task9_retrieval_pipeline.py` (dòng 11) đều ghi rõ RRF score chỉ phản ánh thứ hạng, không cùng thang đo với cosine similarity dùng cho `SCORE_THRESHOLD`. Nếu hiển thị thẳng RRF score (~0.01–0.03) cạnh threshold 0.55, đồng hồ sẽ luôn báo "gần 0% tin cậy" dù kết quả truy hồi đúng — sai lệch kỹ thuật nghiêm trọng khi demo trước hội đồng.
   **Trade-off:** Tốn thêm một lệnh gọi `semantic_search()` mỗi câu hỏi (embedding model chạy local, chi phí thấp) để đổi lấy con số hiển thị đúng bản chất, thay vì đẹp nhưng sai.

2. **Quyết định:** Bàn cờ minh hoạ trong giao diện chỉ tĩnh/trang trí (có animation nhẹ), không dựng animation nước đi tự động theo nội dung từng câu hỏi.
   **Lý do/evidence:** `generate_with_citation` (task10) chỉ trả `answer`/`sources`/`retrieval_source`, không có structured move data. Muốn "diễn" đúng nước đi cho câu hỏi bất kỳ bắt buộc phải hardcode bảng ánh xạ câu hỏi mẫu → animation, rủi ro lộ dàn dựng nếu giám khảo hỏi câu ngoài kịch bản đã chuẩn bị.
   **Trade-off:** Giảm hiệu ứng "wow" so với concept ban đầu của nhóm, đổi lại demo trung thực với năng lực thật của hệ thống RAG, không có phần giả lập.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: `python -m py_compile app.py` (kiểm tra cú pháp, pass) và review thủ công đối chiếu với `docs/MODULE_CONTRACTS.md` (`SearchResult`, `GenerationResult`) để đảm bảo UI đọc đúng field thật của pipeline.
- Kết quả trước/sau nếu có: chưa chạy được `streamlit run app.py` trực tiếp trên trình duyệt tính đến thời điểm viết báo cáo — máy hết dung lượng ổ đĩa (còn ~7GB) và RAM gần cạn khiến `pip install` bị kill giữa chừng nhiều lần; đã chuyển sang cài tối giản đúng các thư viện `app.py` cần (bỏ `crawl4ai`/`playwright`/`patchright`/`ragas`/`datasets` không liên quan tới UI) để hoàn tất cài đặt.
- Lỗi đã phát hiện và cách xử lý:
  1. `picked_question or st.chat_input(...)` bị short-circuit khiến ô chat input không render khi bấm suggestion-card → tách `chat_query = st.chat_input(...)` gọi độc lập trước khi kết hợp.
  2. Thang đo RRF/PageIndex score không khớp với `SCORE_THRESHOLD` (nêu ở mục quyết định #1) → sửa bằng cách gọi `semantic_search()` riêng cho hiển thị.
  3. Streamlit chạy lại toàn bộ nội dung mọi tab ở mỗi lần rerun; do `task6_lexical_search.CORPUS` không được cache nên tab "Semantic vs Hybrid" sẽ âm thầm rebuild toàn bộ BM25 index mỗi khi người dùng tương tác ở bất kỳ đâu trong app, kể cả tab Chat → chuyển sang chỉ tính toán khi bấm nút "So sánh" tường minh.

## Điều còn hạn chế

- Chưa live-test trên trình duyệt thực tế do giới hạn tài nguyên máy (disk/RAM) khi cài dependencies — cần chạy `streamlit run app.py` và kiểm tra thủ công (golden path + câu hỏi ngoài phạm vi) trước buổi demo.
- Relative rank score trên evidence card là điểm chuẩn hoá tương đối trong tập kết quả trả về, không phải chỉ số similarity tuyệt đối — cần giải thích rõ điều này nếu hội đồng hỏi.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: hoàn tất live-test trên trình duyệt, chụp/quay demo thật để đối chiếu với animation đã thiết kế.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-25
- Tên thành viên: Trần Cao Quốc Dinh
