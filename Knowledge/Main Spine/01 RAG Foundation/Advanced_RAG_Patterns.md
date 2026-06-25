# Advanced RAG Patterns — RAG Foundation Deep Dive

## Status

Stage: 01 RAG Foundation  
Current level: deep dive note  
Last updated: 2026-06-25

## Vị trí học

- Parent note: RAG Foundation
- Related note: Retriever, Context Construction, RAG Evaluation
- Vai trò: định vị các pattern RAG nâng cao và khi nào nên học/dùng sau khi baseline đã đo được.

## Must understand

- Advanced RAG không thay thế foundation; nó chỉ giải quyết lỗi cụ thể sau khi baseline đã đo được.
- Không nên nhảy vào GraphRAG/Agentic RAG nếu chưa có recall@k, MRR, faithfulness, latency.
- Query rewriting, multi-query, self-query và multi-hop đã thuộc nhóm retrieval nâng cao cơ bản.
- HyDE, corrective RAG, agentic RAG, GraphRAG và multimodal RAG tăng sức mạnh nhưng cũng tăng complexity/eval cost.

## Must practice

- Bắt đầu từ basic evaluated RAG.
- Thêm từng pattern một và chạy ablation.
- Chỉ giữ pattern nếu metric hoặc failure case cải thiện rõ.
- Log query type để route pattern phù hợp.

## Can explain when ready

- HyDE giải quyết lỗi gì?
- Corrective RAG khác retry retrieval thường ở đâu?
- Agentic RAG nên dùng khi nào?
- GraphRAG phù hợp domain nào?
- Multimodal RAG cần thêm eval gì?

---

## 1. Thứ tự học đúng

```text
Basic RAG
-> evaluated RAG
-> reranked RAG
-> observable RAG
-> permission-aware RAG
-> agentic RAG
```

Không nên học advanced pattern trước khi biết baseline đang fail ở đâu.

---

## 2. Query rewriting

Viết lại query để rõ hơn, nhất là trong chat history.

```text
"cái đó áp dụng cho intern không?"
-> "Chính sách nghỉ phép năm 2026 có áp dụng cho intern không?"
```

Dùng khi query mơ hồ, có đại từ, thiếu context.

---

## 3. Multi-query retrieval

Tạo nhiều query variants để tăng recall.

```text
"enterprise refund policy"
"refund SLA for enterprise customers"
"ngoại lệ hoàn tiền khách doanh nghiệp"
```

Dùng khi docs và user dùng từ khác nhau, đa ngôn ngữ hoặc nhiều synonym. Cần dedup/rerank để giảm noise.

---

## 4. HyDE

HyDE generate hypothetical document/answer rồi embed cái hypothetical text để retrieve.

Flow:

```text
user query
-> LLM tạo hypothetical answer/document
-> embed hypothetical text
-> retrieve documents gần hypothetical text
```

Mạnh khi query quá ngắn hoặc thiếu keyword. Rủi ro là hypothetical text có thể bias retrieval sai.

---

## 5. Self-query retriever

LLM biến natural language query thành:

```text
semantic query + metadata filter
```

Phù hợp khi metadata schema tốt: product catalog, policy, legal, ticket, CRM.

Permission filter không để LLM quyết định; backend enforce từ auth context.

---

## 6. Corrective RAG

Corrective RAG kiểm tra retrieval có đủ tốt không. Nếu yếu thì sửa:

```text
retrieve
-> evaluate retrieved evidence
-> nếu yếu: rewrite query / retrieve lại / dùng web/tool khác / abstain
```

Dùng khi retrieval hay empty/noisy và query phức tạp.

---

## 7. Multi-hop RAG

Retrieve nhiều bước khi answer cần evidence nối tiếp.

```text
Hop 1: policy X thuộc team nào?
Hop 2: manager của team đó là ai?
Hop 3: manager có quyền approve không?
```

Cần giới hạn hop, log evidence từng hop, dừng khi thiếu evidence.

---

## 8. Agentic RAG

Agent quyết định:

```text
có cần retrieve không
retrieve nguồn nào
query nào
có cần retrieve tiếp không
khi nào dừng
```

Dùng khi có nhiều nguồn/tool và workflow phức tạp. Không dùng cho mọi chatbot vì latency/cost/debug khó hơn.

---

## 9. GraphRAG

GraphRAG dùng entity/relationship/community summaries để retrieve theo graph.

Phù hợp khi:

```text
nhiều entity liên quan nhau
câu hỏi cần relationship
tài liệu nhiều nguồn
knowledge network quan trọng
```

Không phù hợp nếu chỉ cần FAQ/policy lookup đơn giản.

---

## 10. Multimodal RAG

Retrieve text + image + table + chart.

Cần thêm:

```text
OCR
table parser
image caption/embedding
layout-aware chunks
multimodal eval
source/citation cho non-text evidence
```

---

## 11. Decision rule

Chỉ thêm advanced pattern nếu nó giải một failure mode rõ:

| Failure | Pattern có thể thử |
| --- | --- |
| query mơ hồ | query rewriting |
| synonym/ngôn ngữ khác | multi-query / HyDE |
| query có metadata condition | self-query |
| retrieval yếu/empty | corrective RAG |
| evidence nhiều bước | multi-hop |
| nhiều tools/sources | agentic RAG |
| relationship-heavy | GraphRAG |
| table/image/chart | multimodal RAG |

---

## 12. Kết luận

Advanced RAG chỉ có ý nghĩa khi baseline đã có eval và observability. Nếu chưa đo được retrieval recall, faithfulness, citation accuracy, latency và failure modes, thêm advanced pattern chỉ làm hệ thống phức tạp hơn mà chưa chắc tốt hơn.
