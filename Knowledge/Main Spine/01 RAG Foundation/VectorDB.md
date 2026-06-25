# Vector DB / Indexing — RAG Foundation Deep Dive

## Status

Stage: 01 RAG Foundation  
Current level: deep dive note  
Last updated: 2026-06-25

## Vị trí học

- Parent note: RAG Foundation
- Related note: Chunking, Embedding
- Vai trò: mở rộng phần vector DB, indexing, metadata filter và retrieval lifecycle trong RAG Foundation.

## Must understand

- Vector DB không chỉ là database lưu embedding; nó là retrieval index có storage, metadata, filter, update/delete và scale.
- Một record RAG tốt phải có vector, text/chunk, metadata, document version, permission và ID ổn định.
- ANN search đánh đổi latency và recall; index config ảnh hưởng trực tiếp tới chất lượng retrieval.
- Similarity metric phải khớp với embedding model: cosine, dot product hoặc L2.
- Metadata filter là bắt buộc trong RAG doanh nghiệp vì embedding similarity không hiểu security, version hay tenant boundary.
- Vector DB là retrieval index, không nên là source of truth duy nhất của tài liệu gốc.

## Must practice

- Ingest một tập Markdown/PDF nhỏ vào Chroma, Qdrant hoặc FAISS.
- Thiết kế metadata schema gồm `document_id`, `source`, `page`, `section`, `version`, `language`, `permission_group`.
- So sánh top-k retrieval khi đổi `chunk_size`, metric hoặc index/search config.
- Test metadata filter theo department, version, language và permission.
- Tạo case update document: upsert version mới, soft delete version cũ, tránh duplicate chunks.
- Log query, filters, retrieved IDs, scores, latency và empty results.

## Can explain when ready

- Vector DB khác FAISS ở điểm nào?
- Vì sao metadata filter quan trọng ngang với vector similarity?
- ANN search hy sinh gì để nhanh hơn exact search?
- Vì sao đổi embedding model thường cần rebuild hoặc tách collection/index?
- Khi nào dùng Chroma, Qdrant, Pinecone, Milvus, Weaviate hoặc FAISS?

---

## 1. Vector DB trong RAG là gì?

Vector DB là tầng lưu và tìm kiếm vector embedding để phục vụ retrieval. Trong RAG, nó thường đảm nhiệm bốn việc:

```text
1. Lưu embedding vector.
2. Lưu chunk text và metadata đi kèm.
3. Tìm nearest neighbors nhanh bằng index.
4. Hỗ trợ filter, update, delete, persistence và scale.
```

Flow cơ bản:

```text
document
-> chunk
-> embedding model
-> vector + text + metadata
-> vector DB

user query
-> query embedding
-> vector search + metadata filter
-> top-k candidates
-> rerank/context assembly
-> LLM answer
```

Câu cần nhớ:

```text
Embedding model quyết định vector có ý nghĩa không.
Vector DB quyết định vector đó được lưu, tìm, lọc, cập nhật và scale như thế nào.
```

Pinecone mô tả vector database là hệ thống index và lưu vector embeddings để similarity search nhanh, kèm các năng lực như metadata filtering, CRUD và scaling. ([Pinecone][1])

---

## 2. Vì sao không chỉ brute-force?

Với vài nghìn chunks, có thể so query vector với toàn bộ document vectors.

```text
query_vector
-> compare với 1,000 vectors
-> sort
-> lấy top-k
```

Nhưng khi dữ liệu lên đến hàng triệu hoặc hàng trăm triệu chunks, exact brute-force trở nên chậm và tốn tài nguyên.

Vector DB giải quyết bằng index:

```text
Không scan toàn bộ vector.
Dùng cấu trúc index để tìm gần đúng top-k nhanh hơn.
```

Đây là lý do phải hiểu ANN search.

---

## 3. Data model của một vector record

Một RAG vector record không nên chỉ có vector. Nó nên có đủ dữ liệu để retrieve, filter, cite, update và debug.

Ví dụ:

```json
{
  "id": "hr_policy_v3_chunk_0012",
  "vector": [0.12, -0.03, 0.44],
  "text": "Nhân viên toàn thời gian có 18 ngày nghỉ phép mỗi năm...",
  "metadata": {
    "document_id": "hr_policy_v3",
    "source": "HR Policy 2026.pdf",
    "page": 12,
    "section": "Annual Leave",
    "version": 3,
    "language": "vi",
    "department": "HR",
    "permission_group": "employee",
    "embedding_model": "text-embedding-model-x",
    "chunker": "recursive_v1",
    "is_active": true
  }
}
```

Metadata không phải phần phụ. Metadata quyết định:

```text
search đúng phạm vi
filter đúng quyền
citation đúng nguồn
update đúng version
debug được failure case
```

Qdrant gọi đơn vị dữ liệu là point: vector kèm payload. Qdrant cung cấp API để store, search và manage points. ([Qdrant GitHub][2])

---

## 4. Collection, index và schema

Tùy tool, tên gọi khác nhau:

| Tool | Khái niệm thường gặp |
| --- | --- |
| Qdrant | collection |
| Chroma | collection |
| Pinecone | index |
| Milvus | collection |
| Weaviate | collection/class |
| FAISS | index object |

Ý nghĩa chung:

```text
Collection/index = nơi chứa một tập vector cùng schema và search config.
```

Ví dụ collection cho policy nội bộ:

```text
name: company_policy_chunks
vector_size: 1024
distance_metric: cosine
metadata fields:
  source, page, section, version, department, permission_group, language
```

Không nên trộn bừa nhiều embedding model hoặc nhiều dimension vào cùng một collection. Nếu đổi embedding model, cần cân nhắc rebuild index hoặc tạo collection mới.

---

## 5. Similarity metric

Vector DB cần biết cách so sánh hai vector.

Metric phổ biến:

| Metric | Ý nghĩa | Ghi chú |
| --- | --- | --- |
| Cosine similarity | So hướng vector | Phổ biến cho text embedding |
| Dot product / inner product | Tích vô hướng | Thường dùng khi model được train/config cho dot |
| L2 / Euclidean distance | Khoảng cách hình học | Hay gặp trong ANN/indexing truyền thống |

Nguyên tắc:

```text
Metric phải khớp với embedding model và cách normalize vector.
```

Nếu model card khuyên cosine nhưng vector DB dùng dot product mà không normalize, ranking có thể lệch. Nếu vector đã normalize về length 1, dot product và cosine thường cho ranking tương đương, nhưng vẫn nên kiểm tra bằng eval.

Qdrant collection config hỗ trợ các distance metric phổ biến như cosine, dot và Euclidean. ([Qdrant Docs][3])

---

## 6. Exact search và ANN search

Exact search:

```text
So query với toàn bộ vectors
-> sort toàn bộ
-> lấy top-k chính xác nhất
```

ANN search:

```text
Dùng index để tìm gần đúng top-k
-> nhanh hơn nhiều
-> có thể miss một vài result tối ưu tuyệt đối
```

Trade-off cốt lõi:

| Loại | Điểm mạnh | Điểm yếu |
| --- | --- | --- |
| Exact search | Recall cao nhất | Chậm khi data lớn |
| ANN search | Nhanh, scale tốt hơn | Recall không tuyệt đối |
| ANN + rerank | Cân bằng tốt cho RAG | Pipeline phức tạp hơn |

Trong RAG, mục tiêu thường không phải exact top-1 tuyệt đối. Mục tiêu là evidence đúng phải lọt vào candidate set đủ lớn.

```text
ANN retrieve top 50
-> reranker chọn top 5
-> LLM answer từ context đã lọc
```

FAISS là library mạnh cho efficient similarity search và clustering của dense vectors, gồm nhiều thuật toán search trên tập vector lớn và có GPU support. ([FAISS][4])

---

## 7. Index type: HNSW, IVF, PQ

### HNSW

HNSW có thể hiểu là search bằng graph.

```text
Mỗi vector là một node.
Node nối với các vector gần nó.
Query đi qua graph để tiến dần tới vùng gần nhất.
```

Điểm mạnh:

```text
search nhanh
recall tốt
phổ biến trong vector DB
phù hợp dữ liệu vừa đến lớn
```

Điểm yếu:

```text
tốn RAM
build/update index có cost
cần tune tham số
```

Tham số hay gặp:

| Tham số | Ý nghĩa |
| --- | --- |
| `M` | số connection trong graph |
| `ef_construct` | độ kỹ khi build index |
| `ef_search` | độ kỹ khi search |

Trade-off:

```text
ef_search cao -> recall cao hơn, latency cao hơn
ef_search thấp -> nhanh hơn, dễ miss hơn
```

### IVF

IVF có thể hiểu là search bằng cluster/bucket.

```text
Chia vector space thành nhiều cụm.
Khi query đến, chỉ search trong vài cụm gần query nhất.
```

Điểm mạnh:

```text
tốt cho dataset lớn
giảm số vector cần scan
dễ kết hợp quantization
```

Điểm yếu:

```text
query vào sai bucket có thể miss result
cần train/build centroid
cần tune số cluster và số bucket search
```

Tham số hay gặp:

| Tham số | Ý nghĩa |
| --- | --- |
| `nlist` | số cluster/bucket |
| `nprobe` | số bucket được search |

Milvus mô tả IVF-series index là cách cluster vectors vào buckets bằng centroid-based partitioning. ([Milvus Index Explained][5])

### PQ

PQ, hay Product Quantization, là kỹ thuật nén vector.

```text
Vector float đầy đủ
-> mã nén nhỏ hơn
-> tiết kiệm RAM/storage
-> chấp nhận giảm precision
```

PQ phù hợp khi scale lớn và memory/storage là bottleneck. Đổi lại, recall có thể giảm và tuning phức tạp hơn.

Cách nhớ nhanh:

```text
HNSW = tìm bằng graph.
IVF = tìm bằng cluster/bucket.
PQ = nén vector để tiết kiệm memory.
```

---

## 8. Metadata filter

Trong RAG doanh nghiệp, search hiếm khi là “tìm top-k gần nhất trong toàn bộ database”. Thường phải là:

```text
Tìm top-k gần nhất
nhưng chỉ trong tài liệu user được phép xem
và chỉ trong version mới nhất
và chỉ trong phòng ban/ngôn ngữ/ngày hiệu lực phù hợp.
```

Ví dụ filter:

```json
{
  "department": "HR",
  "version_status": "latest",
  "permission_group": "employee",
  "language": "vi",
  "is_active": true
}
```

Vì embedding similarity không biết security. Nếu user hỏi “lương thưởng năm nay tính sao?”, vector search không filter có thể retrieve tài liệu private như `executive_compensation_private.pdf`.

RAG enterprise cần:

```text
metadata filter
permission-aware retrieval
tenant/user isolation
document versioning
source/date filter
```

Pinecone hỗ trợ metadata key-value trên record và search với metadata filter expression. ([Pinecone Metadata Filter][6])

---

## 9. Pre-filter và post-filter

Post-filter:

```text
1. ANN search top-k toàn DB.
2. Lọc metadata sau.
```

Rủi ro:

```text
ANN lấy top 10.
8 kết quả là private docs.
Filter xong còn 2 kết quả.
Relevant public doc nằm rank 30 nên bị miss.
```

Pre-filter:

```text
1. Lọc subset hợp lệ trước.
2. ANN search trong subset đó.
```

Pre-filter tốt hơn cho permission, tenant, version và date constraint, nhưng implementation khó hơn vì filter phải tích hợp vào retrieval path.

Kết luận thực dụng:

```text
Filter không chỉ là tiện ích.
Filter ảnh hưởng trực tiếp tới correctness, security và recall.
```

---

## 10. Upsert, update và delete

Upsert nghĩa là:

```text
id chưa tồn tại -> insert
id đã tồn tại -> update/replace
```

Trong RAG ingestion, upsert dùng khi:

```text
thêm document mới
re-embed document cũ
update metadata
ingest lại sau khi parser/chunking thay đổi
```

ID phải ổn định. Không nên dùng random UUID mỗi lần ingest.

Sai:

```text
id = random_uuid()
```

Hậu quả:

```text
mỗi lần ingest tạo duplicate chunks
search trả về nhiều bản trùng
document cũ không được thay thế
```

Tốt hơn:

```text
id = tenant_id/doc_id/version/chunk_id
```

hoặc:

```text
id = hash(document_id + chunk_index + chunk_text_hash)
```

Document lifecycle cần xử lý:

```text
policy đổi version
file bị xóa
user mất quyền truy cập
document bị archive
chunking strategy thay đổi
embedding model thay đổi
metadata sai cần sửa
```

Hard delete dùng khi document thật sự không được retrieve nữa. Soft delete dùng khi cần audit/versioning:

```json
{
  "is_active": false,
  "version_status": "archived"
}
```

Khi search thì filter `is_active = true`.

---

## 11. Persistence, backup và source of truth

Persistence nghĩa là index/data còn sau khi app restart.

Không có persistence:

```text
restart app
-> mất vector index
-> phải ingest lại
```

Có persistence:

```text
restart app
-> load lại index/vector data
-> tiếp tục search
```

Prototype local có thể dùng local folder, SQLite/DuckDB/Parquet hoặc disk-backed index. Production cần snapshot, backup, restore procedure, migration, monitoring và embedding model version tracking.

Điểm quan trọng:

```text
Vector DB không nên là source of truth duy nhất.
```

Nên tách:

```text
object storage / document store / database = source of truth
vector DB = retrieval index
```

Nếu vector DB hỏng hoặc cần rebuild, pipeline phải tái ingest được từ source documents.

---

## 12. Sharding và replication

Sharding là chia dữ liệu ra nhiều node.

```text
collection lớn
-> shard 1
-> shard 2
-> shard 3
```

Mục tiêu:

```text
chứa được nhiều dữ liệu hơn
tăng throughput
phân tán memory/storage
scale ngang
```

Điểm yếu:

```text
query phải fan-out nhiều shard
merge top-k từ nhiều shard
network latency tăng
balancing shard khó
update/delete phức tạp hơn
```

Replication là nhân bản dữ liệu/index ra nhiều bản.

```text
shard 1 replica A
shard 1 replica B
shard 1 replica C
```

Mục tiêu:

```text
high availability
fault tolerance
tăng read throughput
```

Trade-off:

```text
tốn storage hơn
write/update phải đồng bộ nhiều bản
consistency phức tạp hơn
```

Milvus giải thích sharding là split data across multiple nodes cho large datasets, còn replication tạo bản sao để tăng availability/fault tolerance. ([Milvus Sharding][7])

---

## 13. Multi-vector và hybrid search

Multi-vector nghĩa là một record/document có nhiều representation.

Ví dụ:

```json
{
  "id": "doc_1_chunk_3",
  "dense_vector": [0.12, 0.34],
  "sparse_vector": {"refund": 0.8, "policy": 0.4},
  "title_vector": [0.21, 0.18],
  "body_vector": [0.41, 0.72]
}
```

Dùng khi:

```text
dense + sparse retrieval
text + image retrieval
title và body cần embed riêng
document có nhiều field quan trọng
ColBERT-style late interaction
```

Hybrid search thường combine:

```text
dense vector search
+ sparse/BM25 search
+ optional full-text search
```

Dense tốt semantic:

```text
"trả lại tiền" gần "hoàn tiền" và "refund"
```

Sparse tốt exact match:

```text
"Điều 7.2"
"INV-2026-001"
"RTX 3090"
"BAD_FPD10_ASOF"
```

Vì dense score và BM25 score khác scale, không nên cộng thẳng một cách ngây thơ.

RRF, Reciprocal Rank Fusion, fusion dựa trên rank thay vì score gốc. Trực giác:

```text
Document xuất hiện rank cao ở nhiều nguồn
-> đáng ưu tiên hơn.
```

Qdrant hybrid query docs mô tả việc fuse dense, sparse và multivector results bằng RRF hoặc DBSF. ([Qdrant Hybrid Queries][8])

---

## 14. FAISS khác vector DB như thế nào?

FAISS là vector search library. Nó rất mạnh cho ANN, local search, benchmark, research, custom pipeline và GPU similarity search.

Vector DB là hệ thống rộng hơn:

```text
vector search
+ storage
+ metadata filtering
+ CRUD lifecycle
+ API service
+ persistence
+ sharding/replication
+ backup/restore
+ multi-tenant/security concerns
```

Nói gọn:

```text
FAISS = vector search library.
Vector DB = vector search + storage + metadata + lifecycle + ops + scale.
```

Dùng FAISS khi cần hiểu sâu indexing hoặc build prototype/custom retrieval. Dùng vector DB khi cần app vận hành lâu dài, có filter, update, persistence và production ops.

---

## 15. Chọn tool theo nhu cầu

| Nhu cầu | Tool nên nghĩ tới |
| --- | --- |
| Học RAG, demo local, notebook experiment | Chroma |
| Tự host production vừa đến lớn, filter tốt | Qdrant |
| Managed production, ít lo infra | Pinecone |
| Học ANN sâu, custom local search, GPU search | FAISS |
| Scale lớn, distributed vector DB | Milvus |
| Object + vector + structured filtering production | Weaviate |

Không chọn tool trước. Chọn yêu cầu hệ thống trước:

```text
data size
latency target
top-k/candidate size
metadata filter complexity
permission/tenant model
update/delete frequency
hybrid search needs
ops capability
cost constraint
```

---

## 16. Năng lực production cần có

### Ingestion

```text
batch insert
upsert
retry
dedup
document versioning
failed ingestion logs
```

### Search

```text
dense vector search
sparse/BM25 nếu cần
hybrid search
metadata filter
top-k tuning
score threshold
rerank candidate set
```

### Update/delete

```text
delete by ID
delete by document_id
update metadata
re-embed khi model đổi
soft delete/active flag
```

### Security

```text
tenant_id
user_id
permission_group
department
document visibility
```

Security phải nằm trước hoặc trong retrieval path. Không được để LLM nhìn thấy context private rồi mới lọc sau.

### Observability

```text
query
filters applied
top-k returned chunk IDs
scores
latency
empty results
reranker score
answer source citations
```

Không log được retrieval thì debug RAG rất khó.

---

## 17. Lỗi thường gặp

| Lỗi | Triệu chứng | Cách sửa |
| --- | --- | --- |
| Metadata thiếu | Search ra chunk nhưng không biết file/page/version | Lưu `document_id`, `source`, `page`, `section`, `version`, `permission_group` |
| Không filter permission | User retrieve được tài liệu private | Permission-aware retrieval, filter tenant/user/group |
| Duplicate chunks | Search trả về nhiều kết quả giống nhau | ID ổn định, upsert, delete/disable version cũ |
| Trộn embedding model | Score/ranking loạn | Tách collection hoặc rebuild khi đổi model |
| Top-k cố định mù | Query phức tạp thiếu evidence | Lấy candidate_top_k lớn hơn rồi rerank |
| Vector DB làm source of truth | Hỏng index là mất khả năng rebuild | Giữ tài liệu gốc ở document store/object storage |
| Metric sai | Retrieval kém dù data đúng | Match metric với embedding model, kiểm tra normalization |
| Post-filter quá muộn | Filter xong còn ít/không có result | Pre-filter hoặc tích hợp filter vào retrieval path |

---

## 18. Kiến trúc RAG với Vector DB

```text
Source Documents
-> Parser / Cleaner
-> Chunker
-> Embedding Model
-> Vector DB: vectors + text + metadata

User Query
-> Query Embedding
-> Vector Search + Metadata Filter
-> Candidate Chunks
-> Reranker
-> Context Assembly
-> LLM Answer + Citations
```

Vector DB nằm giữa embedding và retrieval, nhưng chất lượng của nó phụ thuộc toàn bộ pipeline: parsing, chunking, embedding, metadata schema, filter, lifecycle và eval.

---

## 19. Thứ tự học thực dụng

```text
1. Collection / point / payload
2. Similarity metric: cosine, dot, L2
3. ANN search
4. HNSW
5. Metadata filtering
6. Upsert / delete / versioning
7. Hybrid search: dense + sparse + RRF
8. Persistence / backup
9. Sharding / replication
10. Multi-vector
11. Tool comparison
```

---

## 20. Kết luận

Vector DB không phải chỉ là nơi gọi:

```python
similarity_search(query)
```

Một RAG engineer cần biết:

```text
vector được lưu theo schema nào
metadata nào bắt buộc phải có
search dùng metric gì
index ANN đánh đổi latency/recall ra sao
filter permission/date/source như thế nào
update/delete document cũ ra sao
có duplicate/version drift không
hybrid dense+sparse có cần không
production có backup/sharding/replication không
```

Mindset quan trọng:

```text
Nhiều lỗi RAG không nằm ở LLM.
Chúng nằm ở indexing/retrieval layer:
filter sai, version sai, duplicate chunk, top-k sai,
metric sai, hoặc document lifecycle không được quản lý.
```

[1]: https://www.pinecone.io/learn/vector-database/ "What is a Vector Database and How Does it Work?"
[2]: https://github.com/qdrant/qdrant "Qdrant GitHub"
[3]: https://qdrant.tech/documentation/concepts/collections/ "Qdrant Collections"
[4]: https://faiss.ai/index.html "Faiss Documentation"
[5]: https://milvus.io/docs/index-explained.md "Index Explained - Milvus Documentation"
[6]: https://docs.pinecone.io/guides/search/filter-by-metadata "Filter by metadata - Pinecone Docs"
[7]: https://milvus.io/ai-quick-reference/how-do-distributed-vector-databases-handle-sharding-and-replication "How distributed vector databases handle sharding and replication"
[8]: https://qdrant.tech/documentation/search/hybrid-queries/ "Qdrant Hybrid Queries"
