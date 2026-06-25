# RAG Foundation — Knowledge cần học kỹ

## Status

Stage: 01 RAG Foundation  
Current level: drafted → coding practice next  
Last updated: 2026-06-25

## Must understand

- RAG pipeline end-to-end.
- Data ingestion, parsing, metadata và document versioning.
- Chunking trade-off: size, overlap, semantic boundary.
- Deep dive: Chunking.
- Embedding, vector DB, metadata filter và hybrid search.
- Deep dive: Embedding.
- Deep dive: VectorDB.
- Deep dive: Retriever.
- Deep dive: HybridSearch.
- Deep dive: Rerank.
- Deep dive: Context Construction.
- Deep dive: Prompt Grounding / Citation.
- Deep dive: RAG Evaluation.
- Deep dive: Evaluation Dataset.
- Deep dive: Observability / Logging.
- Deep dive: RAG Failure Modes.
- Deep dive: Security / Permissions / Privacy.
- Deep dive: Data Freshness / Update Pipeline.
- Deep dive: Performance / Latency / Cost.
- Deep dive: Caching.
- Deep dive: Advanced RAG Patterns.
- Retrieval vs reranking vs context construction.
- Citation, faithfulness, evaluation và observability.

## Must practice

- Build simple retriever trên một tập Markdown/PDF nhỏ.
- Compare chunk size và top_k.
- Measure Recall@5, MRR hoặc context precision.
- Add reranker và so sánh trước/sau.
- Log retrieved chunks, citation, latency và failure cases.

## Can explain when ready

- Vì sao vector search alone chưa đủ cho production RAG?
- Chunking ảnh hưởng answer quality như thế nào?
- Debug một bad RAG answer theo từng tầng ra sao?

---

Roadmap xác định main spine là **RAG → Agents → Serving → GPU Inference → AI Factory**, và Stage 1 ghi rõ key skills là **chunking, vector DB, reranking, citations, RAG evaluation, observability**; project RAG cũng cần có **citations, recall@k, MRR, latency report, failure cases**. ([Khoa Learning Code](https://khoalearningcode.github.io/AI_Engineer_Roadmap/ "AI Systems Engineer Roadmap"))

## 1. RAG là gì?

**RAG — Retrieval-Augmented Generation** là kỹ thuật cho LLM truy cập external knowledge trước khi trả lời. Thay vì chỉ dựa vào kiến thức nằm trong weights của model, hệ thống sẽ retrieve tài liệu liên quan, đưa vào prompt, rồi yêu cầu LLM trả lời dựa trên context đó. LangChain docs mô tả RAG là cách xây Q&A/chatbot có thể trả lời dựa trên source information cụ thể. ([LangChain Docs](https://docs.langchain.com/oss/python/langchain/rag "Build a RAG agent with LangChain"))

Flow cơ bản:

```text
User query
→ retrieve relevant documents/chunks
→ inject context into prompt
→ LLM generates grounded answer
→ return answer + citations
```

Cần hiểu RAG không phải chỉ là “vector DB + LLM”. RAG là một pipeline gồm nhiều tầng:

```text
data ingestion
→ preprocessing
→ chunking
→ embedding
→ indexing
→ retrieval
→ reranking
→ context construction
→ generation
→ citation
→ evaluation
→ observability
```

---

## 2. Data ingestion / document loading

Đây là phần nền tảng quan trọng.

RAG bắt đầu từ dữ liệu. Dữ liệu có thể là:

```text
PDF
Word
HTML
Markdown
CSV
Google Docs
Notion
web pages
database records
images/OCR documents
logs
tickets
internal wiki
```

Cần học:

|Mảng|Nội dung cần nắm|
|---|---|
|**Document loader**|Cách đọc PDF, HTML, Markdown, CSV, database|
|**Parsing**|Trích text từ file, giữ cấu trúc heading/table/list|
|**OCR**|Dùng khi tài liệu là scan/image|
|**Metadata extraction**|Lưu source, page, section, timestamp, author, permission|
|**Cleaning**|Xóa header/footer lặp, ký tự lỗi, text rác|
|**Deduplication**|Loại document/chunk trùng|
|**Document versioning**|Biết tài liệu đang ở version nào|

Cần hiểu:

```text
Garbage in → garbage out.
Nếu parsing sai, chunking sai.
Nếu chunking sai, retrieval sai.
Nếu retrieval sai, generation dễ hallucinate.
```

Ví dụ lỗi:

```text
PDF có table quan trọng nhưng parser đọc lộn thứ tự cột
→ chunk sai nghĩa
→ LLM trả lời sai dù vector search vẫn retrieve đúng file
```

---

## 3. Document structure / metadata

RAG production không chỉ lưu text. Mỗi chunk nên có metadata.

Ví dụ metadata:

```json
{
  "source": "employee_policy.pdf",
  "page": 12,
  "section": "Refund Policy",
  "created_at": "2026-06-01",
  "department": "HR",
  "permission_group": "internal",
  "document_version": "v3"
}
```

Metadata dùng để:

```text
filter theo phòng ban
filter theo ngày
filter theo quyền truy cập
hiển thị citation
debug retrieved chunks
xóa/update document cũ
```

Qdrant docs hỗ trợ payload filtering cùng search API, nghĩa là metadata/payload có thể dùng để lọc kết quả khi search vector. ([Qdrant](https://qdrant.tech/documentation/search/search/ "Search"))

Cần hiểu:

> Metadata tốt giúp RAG không chỉ retrieve “đoạn gần nghĩa”, mà retrieve đúng **nguồn, đúng quyền, đúng version, đúng thời điểm**.

---

## 4. Chunking

Phạm vi cần mở rộng.

Chunking là chia document dài thành các đoạn nhỏ để embedding và retrieve.

Note mở rộng: Chunking đi sâu vào chunk paradox, fixed-size, recursive, semantic, parent-child, hierarchical, late, table-aware và code-aware chunking.

Cần học:

|Loại chunking|Ý nghĩa|
|---|---|
|**Fixed-size chunking**|Chia theo số token/ký tự cố định|
|**Recursive chunking**|Ưu tiên chia theo paragraph, sentence, rồi mới tới token|
|**Semantic chunking**|Chia theo ý nghĩa/semantic boundary|
|**Parent-child chunking**|Retrieve chunk nhỏ, nhưng đưa parent context lớn hơn vào prompt|
|**Hierarchical chunking**|Tổ chức chunk theo nhiều tầng: section → paragraph → sentence|
|**Late chunking**|Embed tài liệu dài trước rồi tách representation theo chunk, tùy model/pipeline|
|**Table-aware chunking**|Giữ table/caption/header để không mất nghĩa|
|**Code-aware chunking**|Chia theo function/class/file thay vì cắt ngang code|

Các nghiên cứu RAG gần đây cũng tập trung vào hierarchical/semantic chunking vì chunking truyền thống có thể không giữ đủ semantic meaning hoặc cấu trúc văn bản. ([arXiv](https://arxiv.org/abs/2507.09935 "Enhancing Retrieval Augmented Generation with Hierarchical Text Segmentation Chunking"))

Các tham số cần hiểu:

|Tham số|Ý nghĩa|
|---|---|
|**chunk size**|Chunk dài bao nhiêu token/ký tự|
|**chunk overlap**|Phần lặp giữa hai chunk|
|**separator**|Dựa vào heading, paragraph, sentence hay token|
|**parent context size**|Khi retrieve child chunk thì lấy thêm bao nhiêu context cha|
|**max context budget**|Tổng token được phép đưa vào LLM|

Trade-off:

```text
Chunk quá nhỏ:
+ retrieve chính xác hơn
- dễ mất context

Chunk quá lớn:
+ giữ nhiều context
- embedding bị loãng, retrieval kém chính xác, tốn token
```

---

## 5. Embedding

Embedding là biến text/image/document thành vector số để search theo nghĩa.

Note mở rộng: Embedding đi sâu vào dense/sparse/hybrid retrieval, similarity metric, normalization, domain/multilingual/instruction embedding và retrieval eval.

Flow:

```text
document chunk → embedding model → vector
query → embedding model → vector
compare query vector with document vectors
```

Cần học:

|Mảng|Nội dung|
|---|---|
|**Dense embedding**|Vector dense dùng semantic similarity|
|**Sparse embedding**|Vector thưa, gần keyword/BM25-style|
|**Embedding dimension**|Số chiều vector|
|**Cosine similarity**|So hướng vector|
|**Dot product**|So tích vô hướng|
|**Normalization**|Chuẩn hóa vector trước khi search|
|**Domain embedding**|Embedding model hợp domain sẽ retrieve tốt hơn|
|**Multilingual embedding**|Quan trọng nếu tài liệu/query có tiếng Việt/Anh|
|**Instruction embedding**|Một số embedding model cần prompt kiểu query/document|

Sentence Transformers mô tả semantic search là embed query và corpus vào vector space rồi tìm corpus embeddings gần nhất với query embedding, thường bằng cosine similarity. ([Hugging Face](https://huggingface.co/learn/cookbook/en/advanced_rag "Advanced RAG on Hugging Face documentation using ..."))

Cần hiểu:

```text
Embedding gần nhau không đồng nghĩa chắc chắn đúng.
Nó chỉ nói hai đoạn gần nhau theo representation của model.
```

Ví dụ lỗi:

```text
Query: "chính sách hoàn tiền"
Retrieved chunk: "chính sách thanh toán"

Semantic gần, nhưng không phải evidence đúng.
```

---

## 6. Vector DB / Indexing

Vector DB lưu embeddings và cho phép search gần nhất.

Các khái niệm vector DB nên được học theo năng lực hệ thống, không chỉ theo tên tool.

Cần học:

|Khái niệm|Ý nghĩa|
|---|---|
|**Collection/index**|Nơi lưu vectors|
|**Point/vector record**|Một chunk + vector + metadata|
|**ANN search**|Approximate nearest neighbor search để search nhanh|
|**Payload/metadata filter**|Lọc theo source, date, permission, category|
|**Upsert**|Insert/update vector|
|**Delete/update**|Xóa hoặc cập nhật document cũ|
|**Index type**|HNSW, IVF, PQ tùy tool|
|**Persistence**|Lưu index bền vững|
|**Sharding/replication**|Scale khi dữ liệu lớn|
|**Multi-vector**|Một document có nhiều vector: dense/sparse/image/text|

Qdrant hỗ trợ hybrid queries với sparse và dense vectors, ví dụ dùng Reciprocal Rank Fusion để kết hợp nhiều nguồn retrieval. ([Qdrant](https://qdrant.tech/documentation/search/hybrid-queries/ "Hybrid Queries"))

So sánh ở mức kiến thức:

|Tool|Nên hiểu|
|---|---|
|**Chroma**|Dễ dùng local/prototype|
|**Qdrant**|Self-host mạnh, filtering tốt, hybrid/multi-vector tốt|
|**Pinecone**|Managed vector DB, tiện production không muốn tự vận hành|
|**FAISS**|Library vector search mạnh, nhưng không phải DB đầy đủ|
|**Milvus/Weaviate**|Vector DB production, scale lớn|

---

## 7. Retrieval

Retrieval là lấy ra các chunk có khả năng liên quan tới query.

Cần học:

|Mảng|Nội dung|
|---|---|
|**top-k**|Lấy k kết quả đầu|
|**similarity score**|Điểm gần nhau giữa query và chunk|
|**metadata filtering**|Lọc trước/sau retrieval|
|**pre-filter vs post-filter**|Lọc trước search hay sau search|
|**threshold**|Chỉ lấy chunk trên ngưỡng score|
|**multi-query retrieval**|Tạo nhiều query biến thể|
|**query rewriting**|Viết lại query rõ hơn|
|**query decomposition**|Tách câu hỏi phức tạp thành câu hỏi nhỏ|
|**self-query retrieval**|LLM tạo query + metadata filter|
|**multi-hop retrieval**|Retrieve nhiều bước|
|**hybrid search**|Kết hợp dense semantic search và sparse keyword search|

LangChain docs phân biệt retrieval theo kiểu RAG agent linh hoạt với tool search và 2-step RAG chain nhanh hơn, cho thấy retrieval có nhiều pattern chứ không chỉ một vector search đơn giản. ([LangChain Docs](https://docs.langchain.com/oss/python/langchain/retrieval "Retrieval - Docs by LangChain"))

Cần hiểu:

```text
Retriever phải ưu tiên recall.
Tức là top-k nên chứa evidence đúng.
Nếu evidence đúng không vào top-k, LLM gần như không có cơ hội trả lời grounded.
```

---

## 8. Hybrid search

Đây là phần nên thêm vào.

Dense retrieval mạnh về semantic meaning:

```text
"refund policy" gần "money back rules"
```

Sparse/BM25 mạnh về exact keyword:

```text
mã sản phẩm
tên riêng
số hợp đồng
API name
error code
legal clause
```

Hybrid search kết hợp cả hai:

```text
dense vector search + sparse keyword search → fusion/rerank
```

Qdrant có tutorial hybrid search kết hợp dense và sparse vectors để tận dụng semantic understanding và keyword precision. ([Qdrant](https://qdrant.tech/course/essentials/day-3/pitstop-project/ "Project: Building a Hybrid Search Engine"))

Cần học:

|Khái niệm|Ý nghĩa|
|---|---|
|**BM25**|Keyword retrieval cổ điển|
|**Sparse vector**|Vector thưa đại diện keyword/sparse signal|
|**Dense vector**|Semantic vector|
|**RRF**|Reciprocal Rank Fusion, gộp ranking từ nhiều retriever|
|**Weighted fusion**|Gán trọng số dense/sparse|
|**Hybrid reranking**|Retrieve nhiều nguồn rồi rerank|

Khi nào cần hybrid:

```text
tài liệu kỹ thuật
legal/finance documents
mã lỗi
tên người
tên sản phẩm
câu hỏi có keyword rất quan trọng
```

---

## 9. Reranking

Phạm vi cần học sâu hơn.

Vector search thường nhanh nhưng chưa đủ chính xác. Reranker đọc kỹ hơn cặp:

```text
query + candidate chunk
```

và cho relevance score.

Cần học:

|Loại reranker|Ý nghĩa|
|---|---|
|**Cross-encoder reranker**|Đưa query + chunk vào cùng model, chấm relevance kỹ hơn|
|**Bi-encoder retriever**|Embed query/chunk riêng, search nhanh|
|**LLM reranker**|Dùng LLM chấm lại candidate|
|**ColBERT-style late interaction**|Cân bằng giữa tốc độ và token-level matching|
|**Pairwise reranking**|So sánh chunk A với chunk B|
|**Listwise reranking**|Xếp hạng cả list cùng lúc|

Trade-off:

```text
Retriever:
+ nhanh
+ scale lớn
- hiểu relevance chưa sâu

Reranker:
+ relevance tốt hơn
- chậm hơn
- tốn compute hơn
```

Cần hiểu pattern chuẩn:

```text
retrieve top-50 hoặc top-100
→ rerank
→ lấy top-5/top-10 đưa vào LLM
```

---

## 10. Context construction

Đây là phần hay bị thiếu trong RAG docs.

Sau retrieval/reranking, không phải cứ nhét chunk vào prompt là xong. Cần xây context.

Cần học:

|Mảng|Nội dung|
|---|---|
|**Context ordering**|Chunk nào đặt trước/sau|
|**Deduplication**|Loại chunk trùng|
|**Context compression**|Nén context dài|
|**Evidence grouping**|Gom chunk cùng source/section|
|**Citation mapping**|Mỗi chunk có ID để cite|
|**Token budget**|Không vượt context length|
|**Lost-in-the-middle**|Evidence ở giữa context có thể bị model bỏ qua|
|**Conflict handling**|Khi các chunk mâu thuẫn|
|**Recency handling**|Ưu tiên tài liệu mới hơn khi cần|

Cần hiểu:

```text
Retrieval tốt nhưng context construction kém vẫn làm RAG trả lời sai.
```

Ví dụ:

```text
Top-5 có evidence đúng
nhưng evidence bị đặt giữa 15 chunk nhiễu
→ LLM bỏ qua evidence
→ answer sai
```

---

## 11. Prompt grounding / citation

Các khái niệm cần nắm:

|Mảng|Nội dung|
|---|---|
|**Grounded prompt**|Yêu cầu trả lời dựa trên context|
|**Abstention**|Nếu context không có thì nói không biết|
|**Citation**|Gắn claim với source/chunk|
|**Quote/evidence extraction**|Trích evidence ngắn trước khi answer|
|**Answer format**|JSON, bullet, table, citation style|
|**Source constraint**|Không dùng outside knowledge nếu task yêu cầu|
|**Conflict instruction**|Nếu source mâu thuẫn thì báo mâu thuẫn|

Pattern:

```text
Question
Retrieved context with source IDs
Instruction:
- Use only provided context.
- Cite source IDs.
- If context is insufficient, say insufficient context.
- Do not invent.
```

RAGAS paper nhấn mạnh RAG evaluation khó vì phải xét cả retrieval có lấy context phù hợp không, LLM có khai thác context faithful không, và generation có chất lượng không. ([arXiv](https://arxiv.org/abs/2309.15217 "RAGAS: Automated Evaluation of Retrieval Augmented Generation"))

---

## 12. RAG evaluation

Evaluation nên được chia thành 3 tầng.

RAGAS docs nói RAG nên đánh giá theo từng component trong pipeline, với các metric như faithfulness, answer relevancy, context recall, context precision, context utilization. ([Ragas](https://docs.ragas.io/en/v0.1.21/concepts/metrics/ "Metrics"))

### 12.1 Retrieval-level metrics

|Metric|Ý nghĩa|
|---|---|
|**Recall@k**|Top-k có chứa evidence đúng không|
|**Precision@k**|Top-k có nhiều chunk đúng không|
|**MRR**|Evidence đúng đầu tiên nằm rank mấy|
|**NDCG**|Relevant chunk có được xếp cao không|
|**Hit rate**|Có ít nhất một chunk đúng trong top-k không|
|**Context recall**|Retrieved context có bao phủ đủ thông tin không|
|**Context precision**|Retrieved context có ít nhiễu không|

RAGAS định nghĩa context precision là metric đánh giá retrieved contexts có hữu ích để trả lời question hay không bằng cách so với reference answer. ([Ragas](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/ "Context Precision"))

### 12.2 Generation-level metrics

|Metric|Ý nghĩa|
|---|---|
|**Answer correctness**|Câu trả lời đúng không|
|**Answer relevancy**|Có trả lời đúng câu hỏi không|
|**Faithfulness**|Câu trả lời có bám context không|
|**Groundedness**|Claim có được support bởi evidence không|
|**Citation accuracy**|Citation có hỗ trợ đúng claim không|
|**Hallucination rate**|Tỷ lệ answer bịa hoặc không có trong context|

### 12.3 System-level metrics

|Metric|Ý nghĩa|
|---|---|
|**Latency**|Tổng thời gian response|
|**Retrieval latency**|Thời gian vector search/rerank|
|**Generation latency**|Thời gian LLM sinh answer|
|**Token usage**|Input/output token|
|**Cost per query**|Chi phí mỗi câu|
|**Failure rate**|Tỷ lệ lỗi|
|**Timeout rate**|Tỷ lệ request timeout|

LlamaIndex docs cũng chia evaluation thành retrieval evaluation, tức đánh giá retriever với questions bằng ranking metrics, và dataset generation để tạo question-context pairs từ corpus. ([Developer Documentation](https://developers.llamaindex.ai/python/framework/module_guides/evaluating/ "Evaluating | Developer Documentation - LlamaParse"))

---

## 13. Evaluation dataset

Đây là phần cần có trong RAG production.

Không có eval set thì không biết RAG có tốt thật không.

Cần học:

|Mảng|Nội dung|
|---|---|
|**Golden QA set**|Bộ câu hỏi + answer chuẩn|
|**Ground-truth evidence**|Chunk/document nào support answer|
|**Synthetic QA generation**|Dùng LLM tạo câu hỏi từ corpus, sau đó review|
|**Human-labeled eval**|Người gán nhãn relevant chunk/answer|
|**Hard queries**|Câu hỏi paraphrase, multi-hop, mơ hồ|
|**Negative queries**|Câu hỏi không có answer trong corpus|
|**Regression eval**|Chạy lại eval sau mỗi thay đổi|

Cần có các loại câu hỏi:

```text
factual lookup
definition
policy condition
comparison
multi-hop
summarization
table/chart question
unanswerable question
ambiguous question
time-sensitive question
permission-sensitive question
```

---

## 14. Observability / logging

Observability cần được mở rộng thành trace đầy đủ.

Cần log:

```text
user query
rewritten query nếu có
retrieved chunk IDs
retrieved scores
reranker scores
final context sent to LLM
prompt version
model name/version
temperature/top_p
answer
citations
latency từng stage
token usage
cost
user feedback
error/failure type
```

Roadmap nhấn mạnh Stage 1 RAG cần logs/evaluation và portfolio proof có retrieval metrics/failure cases; các nhánh LLMOps cũng nhấn mạnh evaluation, logging, monitoring, prompt/version tracking, feedback loop, security, deployment. ([Khoa Learning Code](https://khoalearningcode.github.io/AI_Engineer_Roadmap/ "AI Systems Engineer Roadmap"))

Cần hiểu:

```text
Không log retrieved chunks thì không debug được RAG.
Không log prompt/model version thì không biết thay đổi nào làm hệ thống tốt/xấu.
Không log latency/token thì không tối ưu cost được.
```

---

## 15. RAG failure modes

Đây là phần rất nên thêm vào docs.

RAG sai không phải chỉ vì “LLM yếu”. Phải tách lỗi theo tầng.

|Tầng|Lỗi thường gặp|
|---|---|
|**Ingestion**|Parse PDF lỗi, mất table, OCR sai|
|**Chunking**|Chunk mất context, cắt ngang ý, overlap sai|
|**Embedding**|Model không hợp domain/ngôn ngữ|
|**Indexing**|Vector cũ, metadata sai, duplicate|
|**Retrieval**|Top-k không có evidence đúng|
|**Filtering**|Metadata filter loại mất tài liệu đúng|
|**Reranking**|Reranker xếp sai hoặc quá chậm|
|**Context construction**|Nhồi quá nhiều chunk nhiễu|
|**Generation**|LLM hallucinate dù có context|
|**Citation**|Citation không support claim|
|**Permission**|User thấy tài liệu không được phép xem|
|**Freshness**|Trả lời theo tài liệu cũ|
|**Evaluation**|Eval set quá dễ hoặc bị leak|

Pattern debug:

```text
Answer sai
→ kiểm tra top-k có evidence đúng không
→ nếu không: lỗi retrieval/chunking/embedding/filter
→ nếu có: kiểm tra context construction/prompt/generation
→ nếu answer đúng nhưng citation sai: lỗi citation mapping
→ nếu chỉ sai ngoài production: nghĩ tới distribution shift hoặc stale data
```

---

## 16. Security / permissions / privacy

Phần này rất quan trọng cho enterprise RAG và roadmap cũng có hướng enterprise/private data, permissions, privacy governance. ([Khoa Learning Code](https://khoalearningcode.github.io/AI_Engineer_Roadmap/ "AI Systems Engineer Roadmap"))

Cần học:

|Mảng|Nội dung|
|---|---|
|**Access control**|User chỉ retrieve tài liệu được phép xem|
|**Permission-aware retrieval**|Filter theo user/group/role|
|**PII handling**|Nhận diện và che thông tin nhạy cảm|
|**Data privacy**|Không gửi dữ liệu riêng tư ra model/API không được phép|
|**Prompt injection**|Document chứa lệnh độc hại đánh lừa LLM|
|**Data exfiltration**|User cố trích dữ liệu ngoài quyền|
|**Audit log**|Log ai hỏi gì, retrieved gì|
|**Tenant isolation**|Không lẫn dữ liệu giữa khách hàng/phòng ban|

Ví dụ lỗi nguy hiểm:

```text
User A hỏi chính sách lương
retriever trả về chunk của phòng HR chỉ dành cho manager
→ RAG leak dữ liệu
```

---

## 17. Data freshness / update pipeline

RAG khác fine-tuning ở chỗ knowledge có thể cập nhật bằng external data. Nhưng phải quản lý freshness.

Cần học:

|Mảng|Nội dung|
|---|---|
|**Incremental indexing**|Chỉ index phần thay đổi|
|**Re-indexing**|Rebuild index khi parser/chunker/embedding thay đổi|
|**Document deletion**|Xóa vector khi tài liệu bị xóa|
|**Versioning**|Biết chunk thuộc version nào|
|**Staleness detection**|Phát hiện tài liệu cũ|
|**Recency ranking**|Ưu tiên source mới hơn khi phù hợp|
|**Source conflict**|Hai tài liệu mâu thuẫn thì xử lý thế nào|

Cần hiểu:

```text
Nếu document update nhưng vector DB chưa update,
RAG sẽ trả lời theo knowledge cũ.
```

---

## 18. Performance / latency / cost

Ngay từ RAG Foundation cũng nên học performance, vì Stage sau của roadmap là LLM Serving/GPU Inference.

Cần học:

|Mảng|Nội dung|
|---|---|
|**Embedding latency**|Thời gian embed query/document|
|**Vector search latency**|Thời gian search DB|
|**Rerank latency**|Reranker có thể là bottleneck|
|**LLM latency**|Thời gian sinh answer|
|**Token cost**|Context dài làm tăng cost|
|**Caching**|Cache query embedding, retrieved results, final answer|
|**Batching**|Batch embedding/indexing|
|**Streaming**|Trả token dần cho user|
|**Timeout/fallback**|Nếu reranker/LLM quá chậm thì fallback|

Cần hiểu:

```text
RAG quality cao nhưng latency quá lâu thì production vẫn fail.
```

---

## 19. Caching

Nên thêm caching vào docs.

Các loại cache:

|Cache|Ý nghĩa|
|---|---|
|**Embedding cache**|Query giống nhau không cần embed lại|
|**Retrieval cache**|Query giống nhau dùng lại top-k|
|**LLM response cache**|Câu hỏi giống nhau dùng lại answer|
|**Document parsing cache**|Không parse lại file chưa đổi|
|**Reranker cache**|Cache score của query-chunk pair|

Trade-off:

```text
Cache giúp nhanh/rẻ hơn
nhưng phải xử lý stale cache khi document update
```

---

## 20. Advanced RAG patterns

Đây là phần mở rộng sau khi foundation đã ổn.

|Pattern|Ý nghĩa|
|---|---|
|**Query rewriting**|Viết lại query rõ hơn|
|**Multi-query retrieval**|Tạo nhiều query để tăng recall|
|**HyDE**|Generate hypothetical answer/document rồi retrieve|
|**Self-query retriever**|LLM tạo query + metadata filter|
|**Contextual compression**|Nén retrieved context|
|**Parent-child retrieval**|Retrieve chunk nhỏ, trả context lớn|
|**Multi-hop RAG**|Retrieve nhiều bước cho câu hỏi phức tạp|
|**Corrective RAG**|Nếu retrieval yếu thì sửa/truy vấn lại|
|**Agentic RAG**|Agent quyết định search/open/verify nhiều bước|
|**GraphRAG**|Dùng graph/entity relationship cho retrieval|
|**Multimodal RAG**|Retrieve text + image + table + chart|

Nhưng với roadmap, thứ tự nên là:

```text
Basic RAG
→ evaluated RAG
→ reranked RAG
→ observable RAG
→ permission-aware RAG
→ agentic RAG
```

Không nên nhảy vào GraphRAG/Agentic RAG khi chưa đo được recall@k, MRR, faithfulness, latency.

---

# Bảng tổng hợp kiến thức

|Mảng|Nội dung cần nắm|
|---|---|
|**RAG concept**|RAG là gì, vì sao cần external knowledge, khác fine-tuning/prompting thế nào|
|**Data ingestion**|load PDF/HTML/Markdown/DB, parsing, OCR, cleaning, dedup|
|**Metadata**|source, page, section, timestamp, permission, version|
|**Chunking**|chunk size, overlap, recursive, semantic, parent-child, hierarchical, table/code-aware chunking|
|**Embedding**|dense embedding, cosine similarity, dot product, multilingual/domain embedding|
|**Vector DB**|Chroma/Qdrant/Pinecone/FAISS, collection, index, payload, filter, upsert/delete|
|**Retrieval**|top-k, similarity score, threshold, metadata filtering, query rewriting, multi-query|
|**Hybrid search**|dense + sparse/BM25, RRF, keyword + semantic retrieval|
|**Reranking**|cross-encoder, bi-encoder vs cross-encoder, LLM reranker, retrieve top-N then rerank top-k|
|**Context construction**|ordering, dedup, compression, token budget, citation mapping, conflict handling|
|**Prompt grounding**|context injection, source constraint, abstention, citation, answer format|
|**RAG evaluation**|recall@k, precision@k, MRR, NDCG, context precision/recall, faithfulness, correctness|
|**Eval dataset**|golden QA, ground-truth evidence, synthetic QA, hard negatives, unanswerable queries|
|**Observability**|query logs, retrieved chunks, scores, prompt version, model version, latency, tokens, failure cases|
|**Security**|permission-aware retrieval, PII, prompt injection, audit logs, data leakage|
|**Freshness**|re-indexing, incremental update, versioning, stale document detection|
|**Performance**|latency, cost, caching, batch embedding, streaming, timeout/fallback|
|**Failure analysis**|phân loại lỗi ingestion/chunking/retrieval/reranking/generation/citation/security|

---

# Dòng roadmap chuẩn hóa

> **RAG Foundation:** hiểu RAG pipeline từ data ingestion → chunking → embedding → vector indexing → retrieval → reranking → context construction → grounded generation → citation → evaluation → observability. Nắm dense/sparse/hybrid retrieval, metadata filtering, query rewriting, reranker, RAG metrics, failure analysis, permission/security, freshness, latency/cost.  
> **Mức cần đạt:** đọc hoặc build một RAG system là biết lỗi nằm ở tầng nào: data parsing, chunking, embedding, retrieval, reranking, context, prompt, generation, citation, security hay latency; biết đo bằng recall@k, MRR, context precision/recall, faithfulness, answer correctness, latency và token cost.

---

# Cấu trúc ghi chú

```text
RAG Foundation

1. RAG overview
2. RAG vs Prompting vs Fine-tuning
3. Data ingestion
   - loaders
   - parsing
   - OCR
   - cleaning
   - deduplication
4. Metadata and document schema
5. Chunking
   - fixed-size
   - recursive
   - semantic
   - parent-child
   - hierarchical
   - table/code-aware chunking
6. Embedding
   - dense embedding
   - cosine similarity
   - dot product
   - domain/multilingual embedding
7. Vector database
   - collection/index
   - upsert/delete
   - metadata filtering
   - Chroma/Qdrant/Pinecone/FAISS
8. Retrieval
   - top-k
   - similarity score
   - threshold
   - metadata filtering
   - query rewriting
   - multi-query retrieval
9. Hybrid search
   - BM25/sparse
   - dense retrieval
   - RRF/fusion
10. Reranking
   - cross-encoder
   - LLM reranker
   - retrieve top-N then rerank top-k
11. Context construction
   - token budget
   - ordering
   - deduplication
   - compression
   - citation mapping
12. Prompt grounding
   - source-based answer
   - abstention
   - citations
   - conflict handling
13. RAG evaluation
   - recall@k
   - precision@k
   - MRR
   - NDCG
   - context precision/recall
   - faithfulness
   - answer correctness
14. Evaluation dataset
   - golden QA
   - ground-truth evidence
   - hard queries
   - unanswerable queries
15. Observability
   - query logs
   - retrieved chunks
   - reranker scores
   - prompt/model version
   - latency
   - token usage
   - failure cases
16. Security and permissions
   - access control
   - PII
   - prompt injection
   - audit logs
17. Freshness and indexing lifecycle
18. Performance and cost
   - caching
   - batching
   - streaming
   - timeout/fallback
19. RAG failure modes
20. Advanced RAG patterns
   - HyDE
   - corrective RAG
   - agentic RAG
   - GraphRAG
   - multimodal RAG
```

Để đạt mức **Production RAG Foundation**, phạm vi cần có **data ingestion, metadata, hybrid search, query transformation, context construction, eval dataset, security/permissions, freshness, performance/cost, caching, failure modes**. Đây là nền RAG cần học kỹ trước khi đi tiếp Agentic AI và LLM Serving.
