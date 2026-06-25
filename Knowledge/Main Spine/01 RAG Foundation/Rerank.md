# Rerank — RAG Foundation Deep Dive

## Status

Stage: 01 RAG Foundation  
Current level: deep dive note  
Last updated: 2026-06-25

## Vị trí học

- Parent note: RAG Foundation
- Related note: Retriever, HybridSearch, VectorDB
- Vai trò: mở rộng phần reranking, cross-encoder, LLM reranker, late interaction, ranking eval và context selection trong RAG Foundation.

## Must understand

- Reranking là bước xếp hạng lại candidates sau retrieval, không thay thế retrieval.
- Retriever ưu tiên recall; reranker ưu tiên precision.
- Bi-encoder nhanh vì encode query/document riêng, nhưng dễ mất chi tiết query-document interaction.
- Cross-encoder chính xác hơn vì đọc query và document cùng lúc, nhưng chậm hơn.
- LLM reranker linh hoạt cho query phức tạp, nhưng latency/cost cao và khó deterministic.
- ColBERT/late interaction nằm giữa dense retriever và cross-encoder: giữ token-level matching nhưng vẫn precompute document representation được.
- Reranker không cứu được nếu evidence đúng không nằm trong candidate list.

## Must practice

- So sánh retrieval top-k trước và sau rerank bằng cùng eval set.
- Chạy baseline: hybrid retrieve top-50, cross-encoder rerank, final top-5.
- Đo MRR, nDCG@k, Precision@k, Recall@k, latency và cost/query.
- Tạo hard negatives: semantic gần nhưng sai entity, sai điều kiện, sai version, sai clause.
- Test reranker với query tiếng Việt, tiếng Anh và mixed Vietnamese-English.
- Kiểm tra lỗi truncation khi chunk quá dài.
- Thử dedup/diversity sau rerank để tránh top-k toàn chunk trùng ý.

## Can explain when ready

- Vì sao cross-encoder thường chính xác hơn bi-encoder nhưng không dùng để search toàn corpus?
- Reranker sửa được lỗi gì và không sửa được lỗi gì?
- Khi nào dùng cross-encoder, khi nào dùng LLM reranker?
- Pointwise, pairwise và listwise reranking khác nhau thế nào?
- ColBERT giải quyết trade-off nào giữa dense retriever và cross-encoder?
- Rerank top bao nhiêu là hợp lý theo latency/cost?

---

## 1. Reranking là gì?

Reranking là bước xếp hạng lại các candidate chunks sau khi retriever đã lấy ra một danh sách rộng.

```text
User query
-> retriever lấy top-50 / top-100
-> reranker chấm lại candidates
-> lấy top-5 / top-10
-> đưa vào LLM
```

Tư duy đúng:

```text
Retriever = lấy rộng để không miss evidence.
Reranker = đọc kỹ để chọn evidence tốt nhất.
```

Vector search nhanh vì thường chỉ so vector. Nhưng nó không đọc sâu quan hệ giữa query và chunk. Reranker sinh ra để sửa phần ranking sau retrieval.

Sentence Transformers docs giải thích cross-encoder thường cho performance tốt hơn bi-encoder trong ranking, nhưng không thực tế để search corpus lớn vì không tạo embedding indexable như bi-encoder. ([Sentence Transformers Cross-Encoder][1])

---

## 2. Vì sao cần reranking?

Naive RAG:

```text
query
-> vector search top-5
-> LLM answer
```

Vấn đề là top-5 từ vector search có thể:

```text
semantic gần nhưng không đúng evidence
thiếu điều kiện quan trọng
nhầm entity/số/mã
chứa chunk quá rộng
chứa duplicate chunks
rank chunk đúng ở vị trí thấp
```

Ví dụ:

```text
Query:
"enterprise refund sau 30 ngày có được không?"

Retriever top results:
1. Standard refund within 30 days
2. Enterprise payment terms
3. Refund method via bank transfer
4. Enterprise refund SLA exception
5. Warranty policy
```

Chunk đúng là số 4. Nếu chỉ đưa top-3 vào LLM, model không có evidence đúng. Nếu retrieve top-50 rồi rerank, reranker có cơ hội đưa chunk 4 lên đầu.

Kết luận:

```text
Reranking không thay thế retrieval.
Reranking sửa ranking sau retrieval.
```

---

## 3. Bi-encoder retriever

Bi-encoder encode query và document riêng biệt.

```text
query -> query vector
chunk -> chunk vector
similarity(query_vector, chunk_vector)
```

Document vectors có thể precompute và lưu vào vector DB.

Điểm mạnh:

```text
rất nhanh
scale tốt với hàng triệu chunks
document embeddings precompute được
phù hợp first-stage retrieval
```

Điểm yếu:

```text
nén cả chunk thành một vector
dễ loãng khi chunk có nhiều ý
không đọc tương tác chi tiết giữa query và document
dễ semantic-near-but-wrong
```

Ví dụ:

```text
Query:
"refund after 30 days for enterprise"

Chunk:
"standard refund within 30 days"
```

Vector có thể thấy gần vì có `refund` và `30 days`, nhưng chunk không trả lời đúng intent `enterprise after 30 days`.

Vai trò:

```text
Bi-encoder = candidate generator.
Cross-encoder / LLM reranker = relevance judge.
```

---

## 4. Cross-encoder reranker

Cross-encoder đưa query và chunk vào cùng một model.

```text
[QUERY] enterprise refund after 30 days
[DOC] enterprise customers follow a separate refund SLA...
-> relevance score
```

Model nhìn cả query và document cùng lúc, nên attention giữa query tokens và document tokens xảy ra trực tiếp.

Điểm mạnh:

```text
relevance tốt hơn dense vector search
giảm semantic-near-but-wrong
tốt cho legal/policy/finance/code docs
có thể hiểu điều kiện, phủ định, entity, relation tốt hơn
```

Ví dụ:

```text
Query:
"delete user API"

Candidate A:
"create user API"

Candidate B:
"delete user API requires admin permission"
```

Dense có thể thấy A gần. Cross-encoder thường đẩy B lên.

Điểm yếu:

```text
chậm hơn bi-encoder
không thể chạy trên toàn bộ corpus lớn
tốn compute theo số candidates
chunk quá dài có thể bị truncate
```

Pattern chuẩn:

```text
bi-encoder / hybrid retriever lấy top-50
cross-encoder rerank top-50
lấy top-5
```

Nên dùng khi retrieval có nhiều noise, chunk đúng có vào candidate list nhưng rank thấp, domain cần citation chính xác hoặc câu hỏi có nhiều điều kiện.

---

## 5. LLM reranker

LLM reranker dùng LLM để đánh giá hoặc xếp hạng candidates.

Pointwise:

```text
Query: ...
Document: ...
Score relevance từ 0 đến 5.
```

Pairwise:

```text
Query: ...
Document A: ...
Document B: ...
Cái nào relevant hơn?
```

Listwise:

```text
Query: ...
Documents 1-10: ...
Xếp hạng documents theo relevance.
```

Điểm mạnh:

```text
linh hoạt
zero-shot/few-shot dùng được
xử lý query phức tạp tốt
có thể listwise compare
có thể giải thích vì sao chunk relevant
```

Điểm yếu:

```text
latency cao
cost cao
khó deterministic
context length giới hạn
nhạy với prompt và thứ tự input
không nên rerank quá nhiều candidates
```

Dùng LLM reranker khi:

```text
candidates ít, khoảng 5-20
query phức tạp
cần reasoning/listwise comparison
offline quality quan trọng hơn latency
domain không có reranker chuyên biệt
```

Không nên dùng LLM reranker trực tiếp cho top-100 trong app realtime nếu chưa có batching, caching hoặc routing tốt.

RankGPT là ví dụ nghiên cứu dùng generative LLMs cho relevance ranking trong information retrieval. ([RankGPT][2])

---

## 6. ColBERT và late interaction

ColBERT-style late interaction nằm giữa bi-encoder và cross-encoder.

Bi-encoder:

```text
query -> 1 vector
doc -> 1 vector
so similarity
```

Cross-encoder:

```text
query + doc -> cùng model -> score
```

ColBERT-style:

```text
query -> token vectors
doc -> token vectors
late interaction giữa query tokens và doc tokens
```

Nó encode query và doc độc lập, nên document representations có thể precompute. Nhưng nó không ép document thành đúng một vector; nó giữ nhiều token vectors để so fine-grained hơn.

Trực giác:

```text
Query:
"enterprise refund after 30 days"

Token-level matching:
enterprise <-> enterprise customers
refund <-> reimbursement/refund
30 days <-> standard period/30 days
```

Điểm mạnh:

```text
chính xác hơn dense single-vector trong nhiều case
giữ token-level matching
nhanh hơn cross-encoder ở scale lớn
document representation precompute được
có thể giải thích token nào match token nào
```

Điểm yếu:

```text
index lớn hơn dense vector đơn
storage/memory cao hơn
serving/indexing phức tạp hơn
tooling không đơn giản bằng dense retrieval
```

ColBERT paper mô tả late interaction architecture encode query/document độc lập bằng BERT, rồi dùng bước interaction rẻ hơn để giữ fine-grained similarity và cho phép precompute document representations. ([ColBERT][3])

---

## 7. Pointwise, pairwise, listwise

Pointwise chấm từng candidate độc lập.

```text
(query, chunk A) -> score A
(query, chunk B) -> score B
(query, chunk C) -> score C
```

Ưu điểm:

```text
đơn giản
dễ batch
dễ scale hơn pairwise/listwise
score từng chunk rõ
```

Nhược điểm:

```text
không nhìn toàn list
không biết chunk A và B có duplicate không
không tối ưu diversity
```

Pairwise so sánh từng cặp candidates.

```text
Given query:
A relevant hơn B không?
A relevant hơn C không?
B relevant hơn C không?
```

Ưu điểm là so sánh trực tiếp hơn pointwise. Nhược điểm là số cặp tăng nhanh: `n` candidates tạo `O(n^2)` comparisons.

Listwise xếp hạng cả list cùng lúc.

```text
Query
Candidates 1-10
-> ranking tốt nhất: [4, 1, 7, 2, ...]
```

Ưu điểm:

```text
nhìn toàn cục list
tốt cho diversity/redundancy
phù hợp LLM khi candidate ít
```

Nhược điểm:

```text
giới hạn context
nhạy với thứ tự input
khó dùng với list dài
latency/cost cao
```

Cách nhớ:

```text
Pointwise = chấm từng chunk.
Pairwise = so chunk A với chunk B.
Listwise = xếp hạng cả danh sách.
```

---

## 8. Pattern reranking chuẩn trong RAG

Pattern production phổ biến:

```text
1. Dense retriever top-50
2. Sparse/BM25 retriever top-50
3. Fusion/union candidates
4. Reranker score candidates
5. Lấy top-5/top-10
6. Deduplicate/context assembly
7. LLM answer
```

Config khởi điểm:

```text
dense_top_k = 30
sparse_top_k = 30
candidate_union = 40-60 unique chunks
rerank_top_n = 10
final_context_k = 4-6
```

Nếu latency cho phép:

```text
retrieve top-100
rerank top-100
final top-5
```

Nếu latency chặt:

```text
retrieve top-20
rerank top-20
final top-4
```

Nguyên tắc:

```text
Không rerank quá ít nếu retriever chưa đủ tốt.
Không rerank quá nhiều nếu latency/cost không chịu nổi.
```

---

## 9. Reranker sửa được lỗi gì?

| Lỗi | Ví dụ | Reranker giúp gì |
| --- | --- | --- |
| Semantic gần nhưng sai evidence | `chính sách hoàn tiền` retrieve `chính sách thanh toán` | đẩy chunk đúng về refund lên |
| Điều kiện bị bỏ qua | `enterprise refund sau 30 ngày` retrieve standard refund | ưu tiên chunk có enterprise + after 30 days |
| Exact entity bị nhầm | `A-134 refund` retrieve `A-143 refund` | ưu tiên exact entity nếu candidate có |
| Query phức tạp nhiều tiêu chí | so sánh refund normal vs enterprise sau 30 ngày | ưu tiên chunk chứa đủ điều kiện |
| Context noise | nhiều chunk cùng topic nhưng không trả lời trực tiếp | chọn candidate sắc hơn |

---

## 10. Reranker không sửa được lỗi gì?

Không sửa được nếu retriever miss evidence.

```text
retriever top-50 không có chunk đúng
-> reranker không thể chọn chunk đúng
```

Cách sửa là tăng recall trước:

```text
tăng top-k
hybrid search
query rewrite
multi-query
chunking lại
embedding model tốt hơn
metadata filter đúng hơn
```

Không sửa được parser/chunking quá tệ.

```text
"Enterprise customers are excluded."
```

Nếu chunk không cho biết excluded khỏi chính sách nào, reranker có thể rank cao nhưng LLM vẫn thiếu nghĩa. Cần parent-child retrieval, heading injection, table/code-aware chunking hoặc context expansion.

Không sửa được metadata/security filter sai. Reranker không phải lớp security. Permission filter phải enforce trước retrieval/rerank.

---

## 11. Reranking với hybrid search

Hybrid + reranking là combo mạnh:

```text
Dense search -> semantic candidates
Sparse search -> exact keyword candidates
Fusion -> union/RRF
Reranker -> chọn evidence đúng nhất
```

Ví dụ:

```text
Query:
"BAD_FPD10_ASOF threshold"

Dense candidates:
BAD_FPD30_ASOF threshold
BAD score logic

Sparse candidates:
BAD_FPD10_ASOF definition
BAD_FPD10_ASOF code

Reranker:
đưa đúng BAD_FPD10_ASOF lên đầu
```

Mindset:

```text
Hybrid tăng recall.
Reranker tăng precision.
```

---

## 12. Reranking và context budget

Reranker giúp tiết kiệm context.

Không rerank:

```text
retrieve top-10
-> đưa cả 10 chunks vào prompt
-> noise cao, tốn token
```

Có rerank:

```text
retrieve top-50
-> rerank
-> đưa top-5
-> ít noise hơn, evidence sắc hơn
```

Với LLM context window lớn, vẫn không nên nhét mọi thứ. Context dài dễ gây:

```text
lost in the middle
duplicate evidence
LLM bị nhiễu
cost cao
latency cao
```

Nghiên cứu “Lost in the Middle” cho thấy model long-context có thể dùng thông tin kém hơn khi thông tin relevant nằm ở giữa input so với đầu/cuối context. ([Lost in the Middle][4])

---

## 13. Reranker score

Reranker score là relevance score, không phải xác suất tuyệt đối.

Ví dụ:

```text
chunk A: 0.92
chunk B: 0.87
chunk C: 0.31
```

Không nên hiểu:

```text
0.92 = 92% chắc chắn đúng
```

Score phụ thuộc model, calibration, domain, query length và candidate quality.

Cách dùng đúng:

```text
sort candidates trong cùng query/setup
threshold mềm nếu đã calibrate
debug retrieval quality
detect low-confidence answer
```

Cách dùng sai:

```text
so score giữa các model khác nhau
đặt threshold theo cảm giác
coi score là truth
```

---

## 14. Rerank top bao nhiêu?

Không có số cố định, nhưng có baseline thực dụng.

| Bối cảnh | Config khởi điểm |
| --- | --- |
| App nhỏ / latency thấp | retrieve top-20, rerank top-20, final top-4 |
| RAG doanh nghiệp thường | retrieve top-50, rerank top-50, final top-5 hoặc top-8 |
| Query phức tạp / legal / multi-hop | retrieve top-100, rerank theo batch, final top-8 đến top-12 |
| Có LLM reranker | retrieve top-30, cross-encoder top-30, LLM listwise top-10, final top-5 |

Không nên LLM listwise rerank top-100 trực tiếp nếu latency/cost không cho phép.

---

## 15. Chọn reranker model

Checklist:

```text
Ngôn ngữ: tiếng Việt, tiếng Anh, mixed?
Domain: policy, legal, finance, code, support, academic?
Latency: realtime chatbot hay offline search?
Candidate size: rerank 20, 50 hay 100?
Deployment: local GPU, CPU, API?
Max input length: chunk dài bao nhiêu?
Score quality: eval set có cải thiện MRR/nDCG không?
```

Với tiếng Việt/Anh mixed, cần test thật. Reranker English tốt chưa chắc tốt với query tiếng Việt.

---

## 16. Domain notes

### Multilingual RAG

Môi trường Việt Nam thường có:

```text
query tiếng Việt
document tiếng Anh
document tiếng Việt
query mixed Việt-Anh
technical terms tiếng Anh
```

Ví dụ:

```text
"refund policy cho enterprise customer sau 30 ngày"
```

Reranker cần hiểu cả `refund policy`, `enterprise customer` và `sau 30 ngày`.

Cách xử lý:

```text
chọn multilingual reranker
translate query trước rerank nếu cần
giữ exact technical terms
eval riêng case Việt->Anh, Anh->Việt, mixed
```

### Code RAG

Code RAG cần reranking vì dense retrieval dễ lấy code gần nghĩa nhưng sai symbol.

```text
Query:
"ActionAwarePRouter rho_target_after_step"

Candidate A:
"HierarchicalADPPruner r_keep_final"

Candidate B:
"ActionAwarePRouter computes rho_target_after_step"
```

Reranker hoặc LLM reranker nên ưu tiên exact symbol match, definition vs usage, file path, import context và stack trace.

### Legal / finance / policy RAG

Reranker quan trọng vì domain này có nhiều điều kiện:

```text
enterprise vs normal
after 30 days vs within 30 days
exception vs standard rule
approval condition
policy version / effective date
```

Trong finance/risk, các field như `BAD_FPD10_ASOF`, `BAD_FPD30_ASOF`, `FPD30_OBS_ASOF` rất gần tên nhưng khác nghĩa. Hybrid + reranker là baseline nên có.

---

## 17. Eval reranker

Không nên nói “reranker tốt hơn” nếu chưa đo.

Cần dataset:

```text
query
candidate chunks
relevance labels
hard negatives
```

Label ví dụ:

```text
3 = directly answers query
2 = partially relevant
1 = same topic but not evidence
0 = irrelevant
```

Metrics:

| Metric | Ý nghĩa |
| --- | --- |
| MRR | Chunk đúng đứng càng cao càng tốt |
| nDCG@k | Ranking có phản ánh mức relevance không |
| Precision@k | Top-k sau rerank có bao nhiêu chunk hữu ích |
| Recall@k | Evidence vẫn còn trong top-k sau rerank không |
| Answer accuracy | LLM trả lời đúng hơn không |
| Groundedness | Answer có dựa đúng source không |
| Latency | Rerank thêm bao lâu |
| Cost/query | Tốn thêm bao nhiêu |

Ablation nên chạy:

```text
A. Dense retrieval only
B. Hybrid retrieval only
C. Dense + cross-encoder reranker
D. Hybrid + cross-encoder reranker
E. Hybrid + cross-encoder + LLM listwise reranker
```

Kỳ vọng:

```text
Retriever cải thiện Recall@k.
Reranker cải thiện MRR/nDCG/Precision@k.
End-to-end cải thiện answer correctness/groundedness.
```

---

## 18. Lỗi reranking phổ biến

| Lỗi | Triệu chứng | Cách sửa |
| --- | --- | --- |
| Rerank quá ít candidates | retrieve top-5 rồi rerank top-5, evidence rank 20 bị miss | retrieve top-50 rồi rerank |
| Chunk quá dài | reranker truncate hoặc mất đoạn quan trọng | chunk nhỏ hơn, rerank child, answer bằng parent |
| Reranker không hợp ngôn ngữ/domain | query Việt/doc Anh bị chấm sai | multilingual reranker, translate query, eval thật |
| Mất diversity | top-5 toàn chunk trùng ý từ cùng document | dedup by document/parent, MMR/diversity selection |
| Dùng reranker như security layer | private docs vào candidates | permission filter trước retrieval/rerank |
| Tin score tuyệt đối | `score > 0.8` coi là đúng | calibrate threshold bằng eval set |
| LLM reranker quá nhiều candidates | latency/cost tăng mạnh | cross-encoder trước, LLM rerank top ít hơn |

---

## 19. Latency và cost

Reranker thêm latency.

Ví dụ:

```text
retrieve: 30 ms
cross-encoder rerank top-50: 100-500 ms tùy model/hardware
LLM rerank top-10: vài trăm ms đến vài giây
```

Cách tối ưu:

```text
chỉ rerank top-N, không rerank toàn corpus
batch candidates
dùng smaller reranker cho realtime
cache rerank result cho query phổ biến
route query: query đơn giản bỏ LLM reranker
dùng cross-encoder trước, LLM reranker chỉ cho case khó
```

Pattern tốt:

```text
fast path:
hybrid retrieve + cross-encoder rerank

slow path:
query phức tạp hoặc low confidence
-> thêm LLM reranker / decomposition
```

---

## 20. Kiến trúc production

```text
User query
-> query rewrite / metadata filter
-> dense retriever top-50
-> sparse retriever top-50
-> fusion / union / dedup candidates
-> cross-encoder reranker
-> top-10 candidates
-> diversity / parent expansion / context compression
-> final top-5 context
-> LLM answer with citations
```

Nếu dùng LLM reranker:

```text
hybrid retrieve top-50
-> cross-encoder top-15
-> LLM listwise rerank top-15
-> final top-5
```

Không nên đưa thẳng top-50 vào LLM reranker nếu latency/cost không cho phép.

---

## 21. Thứ tự học thực dụng

```text
1. Bi-encoder vs cross-encoder
2. Cross-encoder reranker
3. Retrieve top-N -> rerank -> final top-k
4. Reranker metrics: MRR, nDCG, Precision@k, latency
5. Hybrid + reranking
6. LLM reranker
7. Pointwise / pairwise / listwise
8. ColBERT late interaction
9. Domain/multilingual reranker
10. Optimization: batching, caching, truncation, routing
```

---

## 22. Kết luận

Reranking là một trong những phần tạo khác biệt lớn giữa RAG demo và RAG production.

Naive RAG:

```text
vector search top-5
-> LLM
```

Production RAG tốt hơn:

```text
hybrid retrieve top-50/top-100
-> cross-encoder rerank
-> optional LLM listwise rerank cho query khó
-> final top-5/top-10
-> LLM answer
```

Mindset:

```text
Retriever phải lấy evidence đúng vào candidate set.
Reranker phải đẩy evidence đúng lên đầu.
Context assembly phải đưa đủ evidence nhưng không nhiễu.
```

Nếu evidence đúng không có trong candidates, reranker không cứu được. Nếu evidence đúng có nhưng bị rank thấp, reranker chính là lớp giúp RAG giảm hallucination và giảm context noise.

[1]: https://www.sbert.net/examples/cross_encoder/applications/README.html "Cross-Encoder Applications - Sentence Transformers"
[2]: https://github.com/sunnweiwei/RankGPT "RankGPT"
[3]: https://arxiv.org/abs/2004.12832 "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"
[4]: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00638/119630/Lost-in-the-Middle-How-Language-Models-Use-Long "Lost in the Middle"
