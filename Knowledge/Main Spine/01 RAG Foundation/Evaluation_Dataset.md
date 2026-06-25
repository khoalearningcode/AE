# Evaluation Dataset — RAG Foundation Deep Dive

## Status

Stage: 01 RAG Foundation  
Current level: deep dive note  
Last updated: 2026-06-25

## Vị trí học

- Parent note: RAG Foundation
- Related note: RAG Evaluation, Retriever, Context Construction
- Vai trò: xây bộ câu hỏi/evidence chuẩn để đo RAG bằng số và chạy regression sau mỗi thay đổi.

## Must understand

- Không có eval set thì không biết RAG tốt thật hay chỉ “demo thấy ổn”.
- Eval set cần query, reference answer, ground-truth evidence và negative/hard-negative cases.
- Synthetic QA hữu ích nhưng cần review, vì LLM có thể tạo câu hỏi quá dễ hoặc leak wording.
- Eval phải có unanswerable, ambiguous, permission-sensitive và time-sensitive queries.
- Regression eval quan trọng hơn một lần benchmark.

## Must practice

- Tạo golden QA set 50-100 câu cho một corpus nhỏ.
- Gắn ground-truth evidence ở cấp document/chunk/page/section.
- Tạo hard negatives: cùng topic nhưng không support answer.
- Tạo negative queries không có answer trong corpus.
- Tách eval set theo query type và difficulty.
- Chạy lại eval sau khi đổi chunking, embedding, retriever, reranker hoặc prompt.

## Can explain when ready

- Golden QA khác synthetic QA ở đâu?
- Hard negative giúp phát hiện lỗi gì?
- Vì sao eval set quá dễ làm RAG có vẻ tốt giả?
- Regression eval nên chạy khi thay đổi tầng nào?

---

## 1. Eval dataset cần có gì?

Một record tối thiểu:

```json
{
  "query": "Khách enterprise có được refund sau 30 ngày không?",
  "reference_answer": "Phụ thuộc SLA riêng và approval rule, không theo standard refund 30 ngày.",
  "positive_evidence": [
    {
      "document_id": "enterprise_sla_2026",
      "section": "Refund Exception",
      "chunk_id": "enterprise_sla_2026_chunk_12"
    }
  ],
  "hard_negatives": [
    "standard_refund_2026_chunk_03",
    "payment_terms_2026_chunk_07"
  ],
  "metadata_constraints": {
    "version": "latest",
    "permission_group": "employee"
  },
  "query_type": "policy_condition",
  "difficulty": "medium"
}
```

---

## 2. Các loại query cần có

```text
factual lookup
definition
policy condition
comparison
multi-hop
summarization
table/chart question
code/API question
unanswerable question
ambiguous question
time-sensitive question
permission-sensitive question
```

Nếu chỉ có factual lookup dễ, eval sẽ không bắt được lỗi production.

---

## 3. Golden QA set

Golden QA là bộ câu hỏi được review kỹ.

Cần có:

```text
query rõ ràng
answer chuẩn
evidence chuẩn
metadata/filter cần áp dụng
labels relevance nếu cần ranking eval
```

Ưu điểm:

```text
đáng tin
phù hợp domain thật
dùng được cho regression
```

Nhược điểm:

```text
tốn công
cần người hiểu tài liệu
khó scale nhanh
```

---

## 4. Synthetic QA generation

Synthetic QA dùng LLM tạo câu hỏi từ corpus.

Flow:

```text
sample document sections
-> generate questions
-> generate reference answer
-> attach source evidence
-> human review / spot check
-> add hard negatives
```

Rủi ro:

```text
câu hỏi quá giống wording trong document
câu hỏi quá dễ
answer bị hallucinate từ LLM generator
ground-truth evidence sai
không đại diện user query thật
```

Synthetic QA nên dùng để scale ban đầu, không thay thế hoàn toàn human-labeled eval.

---

## 5. Hard negatives

Hard negative là chunk nhìn có vẻ liên quan nhưng không support answer.

Ví dụ:

```text
Query:
"enterprise refund after 30 days"

Positive:
enterprise refund SLA exception

Hard negatives:
standard refund within 30 days
enterprise payment terms
warranty policy
```

Hard negatives giúp test:

```text
dense semantic-near-but-wrong
reranker có phân biệt điều kiện không
LLM có bị context noise dẫn sai không
```

---

## 6. Negative và unanswerable queries

RAG production phải biết khi nào không trả lời.

Unanswerable cases:

```text
query hỏi thông tin không có trong corpus
query thiếu điều kiện quan trọng
query nằm ngoài quyền truy cập
query hỏi future/current info không có source mới
```

Metric cần đo:

```text
abstention accuracy
unsupported answer rate
false answer rate
```

---

## 7. Split eval set

Nên chia:

```text
dev set: tune chunking/retrieval/prompt
test set: báo cáo chất lượng
regression set: chạy lại sau mỗi thay đổi
```

Không tune prompt hoặc threshold trực tiếp trên test set rồi báo số đẹp.

---

## 8. Regression eval

Chạy lại eval khi đổi:

```text
parser
chunking strategy
embedding model
vector DB metric/index
top-k
metadata filter
hybrid fusion
reranker model
context construction
prompt
LLM model
```

Mỗi thay đổi có thể làm một metric tốt hơn nhưng metric khác xấu đi.

---

## 9. Kết luận

Eval dataset là tài sản của RAG system. Nó biến việc “thử thấy ổn” thành quy trình đo được, debug được và regression được. Không có eval set, bạn không biết thay đổi nào thật sự cải thiện RAG.
