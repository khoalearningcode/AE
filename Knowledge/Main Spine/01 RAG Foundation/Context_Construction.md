# Context Construction — RAG Foundation Deep Dive

## Status

Stage: 01 RAG Foundation  
Current level: deep dive note  
Last updated: 2026-06-25

## Vị trí học

- Parent note: RAG Foundation
- Related note: Retriever, Rerank, HybridSearch
- Vai trò: mở rộng phần biến retrieved/reranked chunks thành context có thể đưa vào prompt, có citation, ít nhiễu và kiểm soát được conflict/token budget.

## Must understand

- Retrieval/reranking tốt chưa đủ; context construction quyết định evidence được chọn, sắp xếp, nén, cite và đưa vào prompt như thế nào.
- Context dài hơn không tự động tốt hơn; context nhiễu có thể làm LLM trả lời sai.
- Dedup, grouping, ordering, compression và citation mapping là các bước production RAG cần có.
- Evidence quan trọng nằm giữa context dài dễ bị bỏ qua hoặc bị nhiễu bởi các chunk xung quanh.
- Conflict/version/recency phải được xử lý bằng metadata và rule rõ ràng, không để LLM tự đoán.
- Context construction cần eval riêng bằng context precision, context recall, citation accuracy, groundedness và token efficiency.

## Must practice

- Từ reranked candidates, build final context có evidence IDs ổn định.
- Dedup theo `chunk_id`, text hash, `parent_id` và document/source.
- Group evidence theo `document_id`, section, page hoặc parent.
- Thử ordering theo relevance, document order, source grouping và strongest-first.
- Implement extractive compression cho chunk dài.
- Thêm citation mapping từ `[E1]` về `doc_id`, source, page, section, chunk_id`.
- Test conflict case: policy cũ/mới, draft/official, general rule/specific exception.
- Đo token usage trước/sau compression và ảnh hưởng tới answer groundedness.

## Can explain when ready

- Vì sao context construction kém vẫn làm RAG sai dù retrieval đúng?
- Dedup theo chunk, text, parent và source khác nhau thế nào?
- Khi nào nên dùng extractive compression thay vì abstractive summarization?
- Vì sao citation ID phải stable và map được về source?
- Conflict giữa source cũ/mới nên xử lý bằng rule nào?
- Token budget nên phân bổ cho retrieved context như thế nào?

---

## 1. Context construction là gì?

Sau retrieval/reranking, hệ thống chưa có prompt tốt. Nó mới có danh sách candidate chunks.

```text
Retriever/reranker trả về chunks.
Context construction quyết định chunks đó được chọn, sắp xếp, nén, gộp, cite và xử lý mâu thuẫn như thế nào.
```

Câu quan trọng:

```text
Retrieval tốt nhưng context construction kém vẫn làm RAG trả lời sai.
```

Ví dụ:

```text
Top-5 có evidence đúng.
Nhưng hệ thống nhét thêm 15 chunk nhiễu.
Evidence đúng nằm giữa context dài.
LLM bỏ qua hoặc bị nhiễu.
-> Answer sai.
```

Paper “Lost in the Middle” cho thấy model long-context thường dùng thông tin tốt hơn khi relevant information nằm ở đầu/cuối context, và kém hơn khi nằm ở giữa. ([Lost in the Middle][1])

---

## 2. Context construction nằm ở đâu?

Pipeline đầy đủ hơn:

```text
User query
-> query rewrite / decomposition
-> retrieval
-> reranking
-> context construction
-> prompt assembly
-> LLM answer
```

Context construction gồm:

```text
1. Chọn chunk nào giữ lại.
2. Loại chunk trùng.
3. Gom chunk theo source/section.
4. Mở rộng parent context nếu cần.
5. Nén context nếu quá dài.
6. Sắp xếp thứ tự evidence.
7. Gắn citation ID.
8. Xử lý conflict/version/recency.
9. Cắt theo token budget.
10. Đưa vào prompt theo format rõ ràng.
```

Nếu retrieval là tìm nguyên liệu, context construction là bước biến nguyên liệu thành phần context có thể dùng được.

---

## 3. Từ top-k chunks đến context engineering

| Giai đoạn | Cách làm | Điểm yếu |
| --- | --- | --- |
| Naive RAG | top-k chunks -> prompt | nhiễu, trùng, không xử lý conflict |
| Reranked RAG | retrieve nhiều -> rerank top-k | vẫn chưa tối ưu ordering/token budget |
| Long-context RAG | nhét nhiều context hơn | lost-in-the-middle, cost cao, nhiễu |
| Production RAG | dedup + group + compress + cite | pipeline phức tạp hơn |
| Advanced RAG | compression + reorder + conflict-aware | cần eval kỹ |

LangChain có `ContextualCompressionRetriever`, một pattern bọc quanh base retriever để compress retrieved docs trước khi đưa vào LLM. ([LangChain Contextual Compression][2])

---

## 4. Context ordering

Không chỉ chọn chunk nào, mà còn phải quyết định chunk nào đặt ở đâu trong prompt.

Naive ordering:

```text
sort by reranker_score descending
-> đưa vào prompt theo thứ tự đó
```

Cách này đơn giản nhưng chưa chắc tối ưu vì:

```text
evidence quan trọng có thể nằm giữa context dài
chunks cùng source/section bị tách rời làm mất mạch
```

Các chiến lược ordering:

| Chiến lược | Khi phù hợp |
| --- | --- |
| Theo relevance score | câu hỏi đơn giản, chunk độc lập |
| Theo document order | answer cần logic theo trình tự |
| Group by source rồi order trong group | cần giữ source/section boundary |
| Strongest evidence first | cần giảm nguy cơ model bỏ qua evidence chính |
| Long-context reorder | context dài, muốn đẩy evidence quan trọng về đầu/cuối |

Ví dụ policy:

```text
Section 1: Standard refund
Section 2: Exceptions
Section 3: Enterprise SLA
```

Nếu đảo thứ tự lung tung, LLM dễ hiểu sai điều kiện. Context ordering ảnh hưởng trực tiếp đến việc LLM có dùng đúng evidence hay không.

LlamaIndex có `LongContextReorder`, dùng reordering để đưa nodes có relevance cao về đầu/cuối context. ([LlamaIndex LongContextReorder][3])

---

## 5. Deduplication

Retrieval thường trả về nhiều chunk gần giống nhau.

Ví dụ:

```text
Dense search lấy chunk A.
Sparse search cũng lấy chunk A.
Multi-query lấy lại chunk A dưới query khác.
Parent-child lấy nhiều child cùng parent.
```

Nếu không dedup:

```text
lặp thông tin
tốn token
giảm diversity evidence
LLM tưởng một fact được xác nhận bởi nhiều nguồn dù chỉ lặp từ một nguồn
```

Các cấp dedup:

| Cấp | Ý nghĩa |
| --- | --- |
| `chunk_id` | cùng chunk thì giữ một bản |
| normalized text/hash | khác ID nhưng text gần y hệt |
| `parent_id` | nhiều child cùng section thì gom hoặc chọn child tốt nhất |
| source/document | tránh top-k toàn từ một file khi cần nhiều nguồn |

Ví dụ lỗi:

```text
[1] Refund allowed within 30 days.
[2] Refund allowed within 30 days.
[3] Refund allowed within 30 days.
[4] Enterprise SLA exception...
```

LLM có thể bị bias bởi thông tin lặp “30 days” và bỏ qua exception.

Rule:

```text
Dedup trước context assembly.
Group gần trùng theo parent/source.
Giữ diversity vừa đủ.
```

---

## 6. Context compression

Context compression là giảm độ dài context nhưng vẫn giữ evidence relevant.

Ba kiểu chính:

```text
extractive compression
abstractive compression
structured compression
```

### Extractive compression

Chỉ giữ câu/đoạn liên quan nhất từ chunk.

```text
Original:
Section Refund Policy:
- Standard refund within 30 days.
- Payment methods.
- Enterprise customers follow separate SLA.
- Warranty is handled by support.

Query:
"enterprise refund sau 30 ngày"

Compressed:
Enterprise customers follow a separate SLA for refund exceptions after the standard refund period.
```

Điểm mạnh:

```text
ít hallucination hơn abstractive
giữ wording/source
dễ cite
```

Điểm yếu:

```text
có thể bỏ mất context cần thiết
cần sentence-level selection tốt
```

### Abstractive compression

Dùng model/LLM tóm tắt chunk theo query.

```text
Theo tài liệu, khách enterprise không theo rule refund 30 ngày tiêu chuẩn mà theo SLA riêng.
```

Điểm mạnh là gọn và gom được nhiều chunk. Điểm yếu là có nguy cơ hallucinate, khó citation chính xác và có thể làm mất wording pháp lý/technical.

Với legal/finance/compliance:

```text
extractive > abstractive
```

Với customer support/general docs, structured summary có thể chấp nhận nếu vẫn trace được source.

### Structured compression

Nén thành format có cấu trúc:

```text
Source: refund_policy_2026.pdf, page 7
Relevant facts:
- Standard customers: refund within 30 days.
- Enterprise customers: follow separate SLA.
- After 30 days: manager approval required.
```

Điểm mạnh:

```text
dễ đọc
dễ cite nếu map source rõ
giảm noise
tốt cho LLM answer
```

Điểm yếu là cần schema tốt và vẫn có rủi ro nếu extraction sai.

---

## 7. Evidence grouping

Thay vì đưa chunks rời rạc:

```text
Chunk 1
Chunk 2
Chunk 3
Chunk 4
```

Gom theo source/section:

```text
Source A: HR Policy 2026
- Section: Annual Leave
- Relevant chunks: ...

Source B: Employee Handbook
- Section: Probation
- Relevant chunks: ...
```

Grouping giúp LLM hiểu:

```text
chunk nào thuộc cùng tài liệu
source nào đáng tin hơn
section nào nói cùng một rule
có conflict giữa source A và B không
```

Format tốt:

```text
[Source 1]
Title: HR Policy 2026
Version: latest
Effective date: 2026-01-01
Section: Refund Policy
Chunks:
- [S1-C1] ...
- [S1-C2] ...

[Source 2]
Title: Enterprise SLA
Version: 2025
Effective date: 2025-09-01
Section: Refund Exception
Chunks:
- [S2-C1] ...
```

Format này giúp citation mapping và conflict handling dễ hơn.

---

## 8. Citation mapping

Mỗi chunk đưa vào prompt cần có ID rõ ràng.

Không nên:

```text
Refund is allowed within 30 days...
Enterprise customers follow SLA...
```

Nên:

```text
[E1]
Source: refund_policy_2026.pdf
Page: 7
Section: Refund
Text: Refund is allowed within 30 days...

[E2]
Source: enterprise_sla_2026.pdf
Page: 3
Section: Refund Exception
Text: Enterprise customers follow a separate SLA...
```

Citation mapping giúp:

```text
answer có source
debug được answer đến từ chunk nào
audit được hệ thống
user kiểm chứng được
detect hallucination dễ hơn
```

Rule:

```text
Mỗi context block phải có stable citation ID.
Citation ID phải map được về doc_id/page/section/chunk_id.
Không cite source không có trong context.
```

---

## 9. Token budget

Context window của LLM không chỉ dành cho retrieved chunks. Nó còn chứa:

```text
system prompt
developer instruction
user query
chat history
retrieved context
tool results
output budget
```

Ví dụ model có 32k tokens không có nghĩa là nên nhét 32k tokens retrieved context.

Ví dụ allocation:

```text
System/developer instructions: 1k
Chat history: 2k
User query: 0.5k
Retrieved context: 20k
Answer budget: 4k
Safety margin: 4.5k
```

Rule thực dụng:

```text
Retrieve rộng.
Rerank/nén/chọn hẹp.
Không nhồi context chỉ vì còn token.
```

---

## 10. Lost-in-the-middle

Lost-in-the-middle là hiện tượng LLM khó dùng thông tin nằm giữa context dài.

Ví dụ:

```text
Chunk 1: irrelevant
Chunk 2: irrelevant
Chunk 3: irrelevant
Chunk 4: evidence đúng
Chunk 5: irrelevant
Chunk 6: irrelevant
Chunk 7: irrelevant
```

Evidence đúng nằm giữa và bị bao quanh bởi noise. Model có thể bỏ qua.

Cách giảm:

```text
không nhồi quá nhiều chunk
rerank trước khi assemble
đặt evidence quan trọng ở đầu hoặc cuối
dùng long-context reorder
nén context
group evidence liên quan gần nhau
lặp concise evidence summary ở đầu context nếu cần
```

---

## 11. Conflict handling

Retrieved chunks không phải lúc nào cũng đồng ý với nhau.

Ví dụ:

```text
Source A, HR Policy 2024:
Annual leave = 12 days

Source B, HR Policy 2026:
Annual leave = 14 days
```

Nếu LLM thấy cả hai mà không có rule xử lý conflict, nó có thể trả lời sai:

```text
"Nhân viên có 12-14 ngày nghỉ phép."
```

Conflict xảy ra vì:

```text
tài liệu cũ và mới cùng tồn tại
draft và official version cùng index
policy nội bộ và FAQ public khác nhau
regional rules khác nhau
exception document mâu thuẫn rule chung
retrieved chunk thiếu metadata
```

Rule xử lý:

```text
official > draft
latest > archived
policy > FAQ
specific exception > general rule
effective date phù hợp > outdated
```

Prompt/context nên hiển thị source status:

```text
[Source A]
Status: archived
Date: 2024

[Source B]
Status: official/latest
Date: 2026
```

Nếu resolve được bằng metadata, resolve. Nếu không resolve được, nói rõ sources mâu thuẫn.

---

## 12. Recency handling

Không phải lúc nào mới hơn cũng đúng hơn. Nhưng với policy, giá, luật, release note, product spec, bug status hoặc quy trình nội bộ, recency rất quan trọng.

Ví dụ:

```text
Query:
"Chính sách nghỉ phép hiện tại là gì?"

Context có:
HR Policy 2023
HR Policy 2024
HR Policy 2026
```

Phải ưu tiên latest/effective source.

Query khác:

```text
"Chính sách nghỉ phép năm 2024 là gì?"
```

Tài liệu 2024 mới đúng.

Metadata cần có:

```text
created_at
updated_at
effective_date
version
status: draft/official/archived
is_active
valid_from
valid_to
```

Không nên chỉ dùng upload time. Một file cũ upload hôm nay không phải policy mới.

---

## 13. Parent-child context

Parent-child retrieval tạo thêm một quyết định:

```text
Rerank child chunk hay parent chunk?
Đưa child hay parent vào prompt?
```

Pattern tốt:

```text
1. Retrieve child chunks nhỏ.
2. Rerank child chunks.
3. Map child -> parent section.
4. Chỉ lấy parent của child relevant.
5. Compress parent theo query nếu parent quá dài.
6. Cite cả child evidence và parent source.
```

Ví dụ:

```text
Child retrieved:
"Enterprise refund follows separate SLA."

Parent section:
Full Refund Policy section gồm standard rule + enterprise exception.
```

Nếu chỉ đưa child, có thể thiếu context. Nếu đưa nguyên parent quá dài, dễ noise.

Cách nhớ:

```text
Child để find, parent để understand.
```

---

## 14. Table và code context

Table context không nên là row trơ trọi:

```text
A | 20M | 40%
```

Nên giữ:

```text
Table: Loan Approval Rules
Columns: Segment | Min income | Max DTI
Row: Segment A | Min income 20M | Max DTI 40%
Footnote: applies to salaried customers only
```

Code context không nên là một dòng code trơ trọi:

```python
rho_target = rho_target_after_step * r_keep_per_sample
```

Nên giữ:

```text
File: finetune_hierarchical_action_aware.py
Class: ActionAwarePRouter
Function: forward
Purpose: layer skipping router
Relevant code:
...
```

Với code RAG, context construction phải giữ file path, class/function name, imports nếu cần, caller/callee nếu query hỏi behavior, comments/docstring và stack trace liên quan.

---

## 15. Context format tốt

Format context rõ ràng:

```text
You are given evidence blocks. Use only these blocks.

[E1]
Source: HR Policy 2026
Status: official
Effective date: 2026-01-01
Section: Refund Policy
Page: 7
Text:
...

[E2]
Source: Enterprise SLA 2026
Status: official
Effective date: 2026-02-01
Section: Refund Exceptions
Page: 3
Text:
...

Instructions:
- Cite evidence IDs like [E1], [E2].
- If evidence conflicts, prioritize official/latest/specific source.
- If evidence is insufficient, say so.
```

Điểm mạnh:

```text
dễ cite
dễ debug
giảm hallucination
LLM hiểu source boundaries
conflict handling rõ hơn
```

---

## 16. Algorithm baseline

```text
Input:
- reranked_chunks
- user_query
- max_context_tokens
- user_permissions

Steps:
1. Enforce permission/security filter.
2. Remove inactive/outdated docs nếu query hỏi hiện tại.
3. Deduplicate by chunk_id/text_hash.
4. Group by document_id/section/parent_id.
5. Within each group, sort by page/section order.
6. Select top groups by best reranker score.
7. Expand parent context nếu child thiếu context.
8. Compress long chunks/parents theo query.
9. Handle conflicts using metadata priority.
10. Allocate token budget per source/group.
11. Reorder important evidence to beginning/end.
12. Assign citation IDs.
13. Build final prompt context.
```

Pseudo-logic:

```text
candidate_groups = group_by_source_and_section(reranked_chunks)

for group in candidate_groups:
    group.score = max(chunk.rerank_score)
    group.status_priority = official/latest/specificity

sort groups by:
    permission valid
    status_priority
    relevance score
    recency if query requires current info

for group in selected_groups:
    compress if too long
    assign citation IDs

final_context = reorder_for_long_context(selected_evidence)
```

---

## 17. Token budget theo nhóm evidence

Không nên để một source ăn hết context.

Ví dụ:

```text
Top source A có 10 chunks rất giống nhau.
Source B có exception quan trọng nhưng chỉ 1 chunk.
```

Nếu chỉ sort theo score, A chiếm hết budget và B bị loại.

Cần budget theo group:

```text
max_chunks_per_source = 2-3
max_tokens_per_source = 1000-2000
reserve_tokens_for_conflict_source = yes
```

Pattern:

```text
1. Chọn source/group đa dạng.
2. Trong mỗi group, chọn chunk tốt nhất.
3. Nếu còn budget, thêm surrounding chunks.
```

---

## 18. Context construction và hallucination

RAG hallucination không chỉ do LLM. Nó có thể do context:

```text
thiếu evidence
evidence đúng bị lẫn noise
chunk conflict không xử lý
citation ID sai
context quá dài
source outdated
table/code mất cấu trúc
```

Fix không phải chỉ đổi LLM. Fix là:

```text
rerank tốt hơn
dedup
group
compress
reorder
conflict/recency handling
```

---

## 19. Eval context construction

Cần eval riêng context construction, không chỉ retrieval.

| Metric | Ý nghĩa |
| --- | --- |
| Context precision | Bao nhiêu context thật sự relevant |
| Context recall | Evidence cần thiết có nằm trong final context không |
| Citation accuracy | Citation có trỏ đúng evidence không |
| Answer groundedness | Answer có dựa đúng context không |
| Conflict resolution accuracy | Có xử lý đúng source mâu thuẫn không |
| Token efficiency | Bao nhiêu token context tạo ra answer đúng |
| Faithfulness | Answer có thêm fact ngoài context không |

Ablation nên test:

```text
A. Top-k raw chunks
B. Reranked chunks
C. Rerank + dedup
D. Rerank + dedup + grouping
E. Rerank + dedup + grouping + compression
F. E + long-context reorder
G. F + conflict/recency rules
```

Kỳ vọng:

```text
Context recall giữ nguyên hoặc tăng.
Context precision tăng.
Answer groundedness tăng.
Token usage giảm.
Citation accuracy tăng.
```

---

## 20. Lỗi phổ biến

| Lỗi | Triệu chứng | Cách sửa |
| --- | --- | --- |
| Nhồi quá nhiều chunk | context dài, noise cao, answer sai | rerank + final_context_k nhỏ + compression |
| Không dedup | nhiều chunk cùng ý chiếm hết prompt | dedup by chunk_id/text_hash/parent_id |
| Mất source boundary | LLM không biết câu nào thuộc source nào | evidence block có `[E1]`, source, page, section |
| Không xử lý conflict | policy 2024 và 2026 bị trộn | filter latest/active hoặc conflict priority rule |
| Citation map sai | answer cite `[E2]` nhưng fact nằm ở `[E4]` | stable evidence IDs, citation post-check |
| Evidence quan trọng nằm giữa context dài | model bỏ qua evidence chính | long-context reorder, giảm context length |
| Dùng upload time làm recency | file cũ upload lại bị coi là latest | dùng effective_date/version/status |
| Table/code mất cấu trúc | LLM hiểu sai row hoặc dòng code | giữ header, schema, file/class/function context |

---

## 21. Baseline cho RAG doanh nghiệp

```text
Input:
reranked top-20 hoặc top-50 candidates

Process:
1. Permission + active/latest filter
2. Dedup chunk/text/parent
3. Group by document_id + section
4. Select top 3-5 evidence groups
5. Limit 1-3 chunks per group
6. Expand parent context nếu cần
7. Compress long chunks extractively
8. Reorder: strongest evidence first, supporting evidence later
9. Assign citation IDs
10. Build prompt with source metadata
```

Context format:

```text
[E1]
Source: ...
Version/status: ...
Page/section: ...
Text: ...

[E2]
Source: ...
Version/status: ...
Page/section: ...
Text: ...
```

Instruction:

```text
Use only the evidence blocks.
Cite evidence IDs.
If evidence is insufficient, say so.
If evidence conflicts, explain conflict and prioritize latest official source.
```

---

## 22. Thứ tự học thực dụng

```text
1. Token budget
2. Citation mapping
3. Deduplication
4. Evidence grouping
5. Context ordering
6. Lost-in-the-middle
7. Context compression
8. Conflict handling
9. Recency handling
10. Domain-specific context: table/code/legal/finance
```

---

## 23. Kết luận

Context construction là phần nhiều RAG tutorial bỏ qua, nhưng production RAG không thể thiếu.

Naive:

```text
retrieval top-5
-> concatenate chunks
-> prompt
```

Production hơn:

```text
reranked candidates
-> permission/version filter
-> dedup
-> group by source/section
-> compress
-> reorder
-> assign citations
-> handle conflict/recency
-> final context
-> LLM
```

Mindset:

```text
Retriever tìm evidence.
Reranker chọn evidence.
Context construction biến evidence thành prompt dùng được.
```

Nếu context construction kém, LLM có thể sai dù retriever đã tìm đúng chunk. Trong RAG doanh nghiệp, đây là nơi kiểm soát grounding, citation, token cost, conflict, recency và reliability của toàn bộ hệ thống.

[1]: https://arxiv.org/abs/2307.03172 "Lost in the Middle: How Language Models Use Long Contexts"
[2]: https://reference.langchain.com/python/langchain-classic/retrievers/contextual_compression/ContextualCompressionRetriever "ContextualCompressionRetriever - LangChain Reference"
[3]: https://developers.llamaindex.ai/python/examples/node_postprocessor/longcontextreorder/ "LongContextReorder - LlamaIndex"
