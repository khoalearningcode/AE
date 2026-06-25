# RAG Evaluation — RAG Foundation Deep Dive

## Status

Stage: 01 RAG Foundation  
Current level: deep dive note  
Last updated: 2026-06-25

## Vị trí học

- Parent note: RAG Foundation
- Related note: Retriever, Rerank, Context Construction, Prompt Grounding Citation
- Vai trò: đo RAG theo từng tầng thay vì chỉ nhìn answer cuối.

## Must understand

- RAG evaluation phải tách retrieval-level, generation-level và system-level.
- Answer sai có thể do parsing, chunking, embedding, filtering, retrieval, reranking, context construction, prompt hoặc generation.
- Recall@k là metric đầu tiên cho retriever; MRR/nDCG quan trọng cho ranking; faithfulness/groundedness quan trọng cho generation.
- Correctness và faithfulness khác nhau.
- Metric offline phải nối với failure cases thật, không chỉ một con số đẹp.

## Must practice

- Tạo eval set có query, reference answer, ground-truth evidence.
- Đo Recall@k, Precision@k, MRR, nDCG cho retriever/reranker.
- Đo answer correctness, faithfulness, citation accuracy cho generation.
- Log latency, token usage, cost/query và failure rate.
- Chạy ablation: dense-only, hybrid, hybrid+rerank, hybrid+rerank+context construction.

## Can explain when ready

- Vì sao “answer đúng” vẫn có thể fail trong strict RAG?
- Context precision và context recall khác nhau thế nào?
- Khi nào dùng MRR, khi nào dùng nDCG?
- System metric nào quyết định production có deploy được không?

---

## 1. Vì sao RAG evaluation khó?

Không nên hỏi chung:

```text
RAG có tốt không?
```

Phải hỏi:

```text
Retriever có lấy đúng evidence không?
Context có đủ và ít nhiễu không?
LLM có trả lời đúng và bám context không?
Hệ thống có nhanh, rẻ, ổn định, an toàn không?
```

RAG là pipeline nhiều tầng:

```text
document parsing
-> chunking
-> embedding
-> indexing
-> retrieval
-> reranking
-> context construction
-> prompt grounding
-> generation
-> citation
```

Answer cuối sai không đủ để kết luận “LLM yếu”.

RAGAS docs đi theo hướng component-wise evaluation với metrics như faithfulness, answer relevancy, context recall và context precision. ([Ragas Metrics][1])

---

## 2. Retrieval-level metrics

| Metric | Ý nghĩa | Khi quan trọng |
| --- | --- | --- |
| Recall@k | Evidence đúng có nằm trong top-k không | metric đầu tiên cho retriever |
| Precision@k | Top-k có bao nhiêu chunk relevant | đo noise |
| MRR | Evidence đúng đầu tiên nằm rank mấy | đo chunk đúng có lên cao không |
| nDCG | Relevant chunk có xếp đúng thứ tự không | khi relevance có nhiều mức |
| Hit rate | Có ít nhất một evidence đúng không | QA đơn giản |
| Context recall | Context có đủ evidence cần thiết không | multi-hop/multi-evidence |
| Context precision | Context có ít rác không | tránh LLM bị nhiễu |

LlamaIndex retrieval evaluation dùng các ranking metrics như hit-rate, MRR, precision, recall, AP và nDCG để so retrieved results với ground-truth context. ([LlamaIndex Evaluation][2])

---

## 3. Recall@k

Recall@k trả lời:

```text
Trong top-k retrieved chunks, có evidence đúng không?
```

Ví dụ:

```text
top_k = 5
retrieved:
1. payment policy
2. refund policy standard
3. warranty
4. enterprise SLA exception
5. shipping policy
```

Nếu evidence đúng là chunk 4:

```text
Recall@5 = hit
Recall@3 = miss
```

Với RAG:

```text
Nếu evidence đúng không vào candidate set,
LLM gần như không có cơ hội trả lời grounded.
```

---

## 4. Precision@k

Precision@k đo trong top-k có bao nhiêu chunk relevant.

Recall cao nhưng precision thấp:

```text
evidence đúng có trong context
nhưng bị bao quanh bởi nhiều noise
LLM vẫn có thể trả lời sai
```

Precision thường được cải thiện bằng reranker, dedup và context construction.

---

## 5. MRR và nDCG

MRR quan tâm evidence đúng đầu tiên đứng rank mấy.

```text
rank 1 -> 1.0
rank 2 -> 0.5
rank 5 -> 0.2
```

MRR tốt khi chỉ cần chunk đúng đầu tiên lên cao.

nDCG hợp khi relevance có nhiều mức:

```text
3 = trực tiếp answer query
2 = relevant một phần
1 = cùng topic nhưng không đủ answer
0 = irrelevant
```

nDCG đo ranking có đưa chunk relevance cao lên trước không.

---

## 6. Generation-level metrics

| Metric | Ý nghĩa |
| --- | --- |
| Answer correctness | Answer có đúng với reference không |
| Answer relevancy | Answer có trả lời đúng câu hỏi không |
| Faithfulness | Answer có bám context không |
| Groundedness | Claim có evidence support không |
| Citation accuracy | Citation có trỏ đúng source support claim không |
| Hallucination rate | Tỷ lệ claim bịa/không có trong context |

Faithfulness khác correctness:

| Case | Correct? | Faithful? | Ý nghĩa |
| --- | --- | --- | --- |
| A | Có | Có | tốt |
| B | Có | Không | đúng nhưng dùng ngoài context |
| C | Không | Có | context sai/cũ nên answer sai theo context |
| D | Không | Không | hallucination hoặc reasoning sai |

Trong strict RAG, case B vẫn fail vì không grounded.

---

## 7. System-level metrics

| Metric | Ý nghĩa |
| --- | --- |
| Latency | tổng thời gian response |
| Retrieval latency | vector/BM25 search |
| Rerank latency | thời gian reranker |
| Generation latency | thời gian LLM |
| Token usage | input/output tokens |
| Cost per query | chi phí mỗi request |
| Failure rate | tỷ lệ lỗi |
| Timeout rate | tỷ lệ timeout |
| Cache hit rate | bao nhiêu request được cache |
| Permission violation rate | retrieve sai quyền không |
| Abstention accuracy | thiếu context thì từ chối đúng không |

Production RAG không chỉ cần đúng. Nó còn phải đủ nhanh, đủ rẻ, ổn định và an toàn.

---

## 8. Ablation nên chạy

```text
A. Dense retrieval only
B. Dense + metadata filter
C. Hybrid retrieval
D. Hybrid + reranker
E. Hybrid + reranker + context construction
F. E + grounded prompt/citation
```

Mục tiêu:

```text
biết thay đổi nào tăng recall
biết thay đổi nào tăng precision/groundedness
biết trade-off latency/cost
biết lỗi nằm ở tầng nào
```

---

## 9. Kết luận

RAG evaluation phải trả lời được ba câu:

```text
Evidence đúng có được lấy ra không?
Answer có bám evidence không?
Hệ thống có chạy được production không?
```

Nếu không tách metric theo tầng, rất dễ sửa sai chỗ: đổi LLM trong khi lỗi nằm ở chunking, đổi embedding trong khi lỗi nằm ở metadata filter, hoặc tăng top-k trong khi lỗi nằm ở context construction.

[1]: https://docs.ragas.io/en/v0.1.21/concepts/metrics/ "Ragas Metrics"
[2]: https://developers.llamaindex.ai/python/framework/module_guides/evaluating/ "LlamaIndex Evaluating"
