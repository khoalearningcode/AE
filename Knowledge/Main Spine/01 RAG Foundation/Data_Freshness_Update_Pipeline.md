# Data Freshness / Update Pipeline — RAG Foundation Deep Dive

## Status

Stage: 01 RAG Foundation  
Current level: deep dive note  
Last updated: 2026-06-25

## Vị trí học

- Parent note: RAG Foundation
- Related note: VectorDB, Context Construction, RAG Failure Modes
- Vai trò: quản lý vòng đời tài liệu, re-index, versioning, delete và tránh RAG trả lời bằng knowledge cũ.

## Must understand

- RAG khác fine-tuning vì knowledge nằm ở external data, nhưng external data phải được cập nhật đúng.
- Document update không đồng nghĩa vector DB đã update.
- Freshness đúng nên dựa vào version/effective date/status, không chỉ upload time.
- Khi parser/chunker/embedding model thay đổi, có thể cần re-index.
- Delete/permission change cũng là update quan trọng.

## Must practice

- Thiết kế metadata `version`, `effective_date`, `valid_from`, `valid_to`, `is_active`, `status`.
- Implement upsert với ID ổn định.
- Soft delete document cũ và filter `is_active=true`.
- Tạo update pipeline: detect changed docs, re-parse, re-chunk, re-embed, upsert.
- Test stale answer: tài liệu cũ và mới cùng tồn tại.

## Can explain when ready

- Incremental indexing khác full re-index ở đâu?
- Khi nào phải rebuild toàn bộ index?
- Soft delete và hard delete dùng khi nào?
- Vì sao upload date không đủ để ranking recency?

---

## 1. Lifecycle tài liệu

Tài liệu production không đứng yên:

```text
file mới được thêm
policy đổi version
file bị xóa
permission thay đổi
metadata sửa
parser/chunker đổi
embedding model đổi
document bị archived
```

RAG phải có update pipeline, không chỉ ingest một lần.

---

## 2. Metadata freshness

Cần có:

```text
document_id
version
status: draft/official/archived
is_active
created_at
updated_at
effective_date
valid_from
valid_to
source_hash
embedding_model
chunker_version
parser_version
```

Recency đúng không chỉ là upload time.

---

## 3. Incremental indexing

Flow:

```text
scan source docs
-> compare hash/version
-> only process changed docs
-> parse
-> chunk
-> embed
-> upsert
-> mark old chunks inactive if needed
```

Ưu điểm:

```text
nhanh hơn
rẻ hơn
ít downtime
```

Nhược điểm:

```text
logic phức tạp hơn
dễ sót delete/permission change
cần version tracking tốt
```

---

## 4. Re-indexing

Cần re-index khi:

```text
embedding model đổi
chunking strategy đổi
parser sửa lỗi lớn
metadata schema đổi
distance metric/index config đổi
data bị duplicate/stale nhiều
```

Re-index cần plan:

```text
build index mới song song
run eval
switch traffic
rollback nếu metric xấu
delete old index sau khi ổn định
```

---

## 5. Delete và archive

Hard delete:

```text
xóa vector khỏi DB
dùng khi tài liệu không được phép retrieve nữa
```

Soft delete:

```json
{
  "is_active": false,
  "status": "archived"
}
```

Dùng khi cần audit hoặc historical query.

Search hiện tại nên filter:

```text
is_active = true
status = official/latest
```

---

## 6. Conflict cũ/mới

Nếu query hỏi hiện tại:

```text
filter active/latest/effective
```

Nếu query hỏi thời điểm lịch sử:

```text
filter valid_from <= date <= valid_to
```

Nếu query không rõ:

```text
dùng latest official
ghi rõ version/date trong answer
```

---

## 7. Kết luận

Freshness là một dạng correctness. Nếu document update nhưng vector DB chưa update, RAG sẽ tự tin trả lời bằng knowledge cũ. Production RAG cần update pipeline, versioning, delete handling, eval sau re-index và rollback plan.
