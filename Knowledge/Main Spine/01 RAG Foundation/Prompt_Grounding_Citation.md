# Prompt Grounding / Citation — RAG Foundation Deep Dive

## Status

Stage: 01 RAG Foundation  
Current level: deep dive note  
Last updated: 2026-06-25

## Vị trí học

- Parent note: RAG Foundation
- Related note: Context Construction, Rerank, RAG Evaluation
- Vai trò: ép LLM trả lời dựa trên evidence, biết từ chối khi thiếu context, cite đúng source và không tự bịa.

## Must understand

- Grounded prompt không đảm bảo tuyệt đối hết hallucination; nó tạo ràng buộc và format để kiểm tra.
- Citation không phải trang trí; citation là cơ chế audit claim về source/chunk.
- “Use only context” chưa đủ nếu context nhiễu, mâu thuẫn, thiếu điều kiện hoặc citation map sai.
- Abstention là hành vi production cần có: context không đủ thì nói không đủ.
- Claim-level citation tốt hơn cite dồn cuối đoạn.
- Citation đúng ID chưa chắc claim được support; cần kiểm tra citation faithfulness.

## Must practice

- Viết prompt yêu cầu dùng only evidence blocks và cite `[E1]`, `[E2]`.
- Tạo case context thiếu và kiểm tra model có abstain không.
- Tạo case source mâu thuẫn và yêu cầu model báo conflict.
- Tạo claim-level citation: mỗi claim factual có citation riêng.
- Post-check citation: claim có thật sự nằm trong evidence được cite không.

## Can explain when ready

- Groundedness khác correctness ở đâu?
- Fake citation và weak citation khác nhau thế nào?
- Khi nào nên trả lời một phần thay vì từ chối toàn bộ?
- Vì sao quote/evidence extraction trước answer có thể giảm hallucination?

---

## 1. Grounded prompt là gì?

Grounded prompt yêu cầu model trả lời dựa trên context được cung cấp.

```text
Question:
{user_question}

Context:
[E1] ...
[E2] ...
[E3] ...

Instructions:
- Use only the provided context.
- Cite evidence IDs for every factual claim.
- If the context is insufficient, say "insufficient context".
- Do not invent facts.
```

Nhưng phải hiểu đúng:

```text
Grounded prompt giảm xác suất hallucination.
Grounded prompt không thay thế retrieval, citation mapping, verifier hoặc eval.
```

RAGAS paper nhấn mạnh RAG evaluation phải xét cả retrieval, faithfulness và generation quality, nên grounding không thể chỉ dựa vào prompt nghe có vẻ chặt. ([RAGAS][1])

---

## 2. Vì sao “use only context” vẫn fail?

Prompt có thể nói:

```text
Use only the provided context.
If you don't know, say you don't know.
```

Nhưng vẫn fail khi:

```text
context có evidence đúng nhưng bị nhiễu
context không đủ nhưng model vẫn cố trả lời
model cite source liên quan nhưng không support claim
source mâu thuẫn nhưng model tự hòa trộn
query yêu cầu ngoài phạm vi context
```

Ví dụ:

```text
Context:
[E1] Standard customers can request refund within 30 days.
[E2] Enterprise customers follow a separate SLA.

Question:
Enterprise customers có được refund sau 45 ngày không?

Bad answer:
Có, khách enterprise được refund sau 45 ngày [E2].
```

Vấn đề: `[E2]` chỉ nói theo SLA riêng, không nói chắc chắn refund sau 45 ngày.

---

## 3. Abstention

Abstention là khả năng từ chối trả lời khi context không đủ.

Answer tốt:

```text
Context cho biết khách enterprise theo SLA riêng [E2],
nhưng không nêu rule cụ thể sau 45 ngày.
Vì vậy chưa đủ context để kết luận.
```

Khi nên abstain:

```text
không có retrieved context
context chỉ cùng topic nhưng không trả lời trực tiếp
context thiếu điều kiện quan trọng
sources mâu thuẫn nhưng không có metadata để resolve
query hỏi ngoài phạm vi tài liệu
```

Nếu context trả lời một phần, không nên từ chối toàn bộ. Nên nói phần biết và phần thiếu.

---

## 4. Citation

Citation tốt phải map được về:

```text
chunk_id
document_id
source title/file
page/section
version/status
retrieved score/rerank score nếu cần debug
```

Claim-level citation:

```text
Standard customers can request refunds within 30 days [E1].
Enterprise customers follow a separate SLA [E2].
Refund exceptions after the standard window require manager approval [E3].
```

Không tốt:

```text
Standard refund là 30 ngày, enterprise theo SLA riêng, và exception cần manager approval. [E1][E2][E3]
```

Vì citation bị dồn cuối, khó biết claim nào được source nào support.

---

## 5. Fake citation và weak citation

Fake citation:

```text
Answer:
Enterprise refund sau 45 ngày luôn được approve [E2].

[E2]:
Enterprise customers follow a separate SLA.
```

Weak citation:

```text
Answer:
Refund được xử lý trong 3 ngày [E1].

[E1]:
Refund requests should be submitted within 30 days.
```

Cùng topic refund nhưng không support claim “xử lý trong 3 ngày”.

Cách giảm:

```text
cite từng factual claim
không cite nếu source không trực tiếp support claim
extract quote/evidence trước khi answer
post-check claim support bằng verifier nếu cần
không cho phép citation ngoài context IDs
```

---

## 6. Prompt pattern thực dụng

```text
You answer using only the evidence blocks.

Rules:
- Every factual claim must cite one or more evidence IDs.
- Do not cite an evidence block unless it directly supports the claim.
- If evidence is insufficient, say what is known and what is missing.
- If sources conflict, explain the conflict and prefer official/latest/specific sources.
- Do not use outside knowledge unless explicitly allowed.

Evidence:
[E1] ...
[E2] ...

Question:
...
```

---

## 7. Eval cần có

| Metric | Ý nghĩa |
| --- | --- |
| Citation accuracy | Citation có trỏ đúng evidence không |
| Citation faithfulness | Claim có phản ánh đúng source được cite không |
| Abstention accuracy | Thiếu context thì có từ chối đúng không |
| Groundedness | Claim có support bởi context không |
| Unsupported claim rate | Tỷ lệ claim không có evidence |

---

## 8. Kết luận

Prompt grounding là lớp cuối để buộc LLM dùng evidence đúng cách. Nó không thay thế retrieval, reranking và context construction, nhưng nó quyết định answer cuối có cite được, audit được và biết từ chối khi thiếu thông tin hay không.

[1]: https://arxiv.org/abs/2309.15217 "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
