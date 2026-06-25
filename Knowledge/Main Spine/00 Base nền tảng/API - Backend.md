# API / Backend — Knowledge cần học kỹ

## Status

Stage: 00 Base nền tảng  
Current level: concept → coding practice  
Last updated: 2026-06-25

## Must understand

- HTTP method, path, query, body, header và status code.
- REST endpoint design cho AI/RAG/LLM service.
- FastAPI route, Pydantic schema và validation.
- Request/response contract, error handling, logging.
- Streaming response cho chatbot/LLM.
- Auth/env/config ở mức cơ bản.

## Must practice

- Build API có `/health`, `/retrieve`, `/ask`, `/chat/stream`.
- Validate input/output bằng Pydantic.
- Thêm logging, config và error response rõ ràng.
- Chạy API local và bằng Docker.
- Test endpoint qua Swagger/curl.

## Can explain when ready

- Vì sao schema rõ giúp frontend hoặc service khác gọi API dễ hơn?
- Streaming khác response thường ở đâu?
- Khi API lỗi, cách phân biệt lỗi client và server như thế nào?

---

**API/backend** trong AI engineering không yêu cầu độ sâu như backend engineer, nhưng cần đủ năng lực để biến model/RAG/LLM pipeline thành một service dùng được.

Định nghĩa năng lực:

> **API/backend:** biết HTTP hoạt động thế nào, thiết kế REST API cơ bản, dùng FastAPI để expose model/RAG/LLM pipeline, hiểu request/response, validation, error handling, streaming token, auth/env/logging cơ bản.  
> **Mức đạt:** tự build được một API kiểu `/ask`, `/retrieve`, `/embed`, `/predict`, chạy local hoặc Docker, frontend/người khác gọi được.

---

## 1. HTTP cần hiểu gì?

HTTP là nền tảng của API. Không cần học quá sâu networking, nhưng phải hiểu:

|Mảng|Cần biết|Mức đạt|
|---|---|---|
|Method|`GET`, `POST`, `PUT`, `DELETE`|Biết khi nào dùng method nào|
|URL/path|`/ask`, `/users/{id}`, `/documents/search`|Thiết kế endpoint rõ nghĩa|
|Query params|`/search?q=rag&top_k=5`|Truyền tham số nhỏ|
|Body|JSON gửi trong request|Gửi prompt, text, config, list docs|
|Header|`Authorization`, `Content-Type`|Gửi token/API key|
|Status code|`200`, `400`, `401`, `404`, `500`|Biết API thành công hay lỗi gì|
|JSON|format trao đổi dữ liệu chính|Request/response rõ schema|

Ví dụ:

```http
POST /ask
Content-Type: application/json

{
  "question": "What is RAG?",
  "top_k": 5
}
```

Response:

```json
{
  "answer": "RAG is...",
  "sources": ["doc1.pdf", "doc2.pdf"]
}
```

Mức đạt: nhìn một API docs là hiểu request gửi gì, response nhận gì, lỗi do client hay server.

---

## 2. REST API cần hiểu gì?

REST API là cách thiết kế endpoint theo tài nguyên/action.

Với AI/RAG, thường sẽ có:

```text
POST /ask              hỏi chatbot RAG
POST /retrieve         lấy top-k documents
POST /embed            tạo embedding
POST /predict          model inference
GET  /health           kiểm tra server sống chưa
GET  /models           xem model đang dùng
```

Cần tránh thiết kế lộn xộn kiểu:

```text
/getAnswerNow
/do_everything
/test123
/api_final
```

Nên thiết kế rõ:

```text
POST /documents/upload
POST /documents/index
POST /retrieval/search
POST /chat/completions
GET  /health
```

Mức đạt: biết chia API theo chức năng, endpoint nhìn vào là hiểu.

---

## 3. FastAPI cần học gì?

FastAPI là phần nên ưu tiên nhất trong mảng backend vì hợp với Python AI.

Cần học:

|Mảng|Cần biết|Mức đạt|
|---|---|---|
|Route|`@app.get`, `@app.post`|Tạo endpoint|
|Pydantic schema|`BaseModel`|Validate input/output|
|Uvicorn|chạy server|Biết start API|
|Error handling|`HTTPException`|Trả lỗi đúng|
|Dependency|config/auth/db client|Dùng được ở mức cơ bản|
|Background task|xử lý việc nhẹ phía sau|Index file/chạy job đơn giản|
|StreamingResponse|stream token|Làm LLM chat streaming|
|Docs tự động|`/docs`|Test API bằng Swagger UI|

Ví dụ API đơn giản:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

class AskResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    answer = f"Answer for: {req.question}"
    return AskResponse(answer=answer, sources=["doc1"])
```

Chạy:

```bash
uvicorn app.main:app --reload --port 8000
```

Mức đạt: tự viết API bọc quanh RAG pipeline/model inference được.

---

## 4. Request / Response cần hiểu kỹ

Đây là phần cực quan trọng, vì AI API thường lỗi do input/output không rõ.

Ví dụ request tốt:

```json
{
  "question": "Explain attention mechanism",
  "top_k": 5,
  "stream": false
}
```

Response tốt:

```json
{
  "answer": "Attention allows...",
  "sources": [
    {
      "title": "transformer.pdf",
      "score": 0.82,
      "chunk_id": "chunk_12"
    }
  ],
  "latency_ms": 1240
}
```

Response lỗi tốt:

```json
{
  "detail": "question must not be empty"
}
```

Không nên response lung tung kiểu:

```json
{
  "result": "ok",
  "data": "...",
  "haha": true
}
```

Mức đạt: API có schema rõ, frontend/người khác gọi không phải đoán.

---

## 5. Streaming cần học gì?

Streaming rất quan trọng với LLM app.

Bình thường API chờ model trả lời xong rồi mới trả response:

```text
User gửi question → server generate xong toàn bộ → trả answer
```

Streaming thì model sinh token tới đâu gửi tới đó:

```text
User gửi question → server trả từng token/chunk dần dần
```

Tác dụng:

|Không streaming|Có streaming|
|---|---|
|Người dùng phải chờ lâu|Người dùng thấy token chạy ngay|
|Phù hợp predict/classification|Phù hợp chatbot|
|Code đơn giản hơn|Code phức tạp hơn một chút|

Trong FastAPI có thể dùng:

```python
from fastapi.responses import StreamingResponse

def token_generator():
    for token in ["Hello", " ", "world"]:
        yield token

@app.get("/stream")
def stream():
    return StreamingResponse(token_generator(), media_type="text/plain")
```

Với LLM/RAG, endpoint hay là:

```text
POST /chat/stream
```

Mức đạt: hiểu streaming khác response thường ở đâu, và tự làm được demo stream token đơn giản.

---

# Phần cần học kỹ nhất

## Ưu tiên 1: HTTP + request/response

Nếu không hiểu phần này thì dùng FastAPI chỉ là copy code.

Cần nắm:

```text
method
path
query parameter
body
header
status code
JSON schema
```

---

## Ưu tiên 2: FastAPI + Pydantic

Đây là core để expose AI model.

Cần nắm:

```text
@app.post
BaseModel
response_model
HTTPException
uvicorn
/docs
```

---

## Ưu tiên 3: API design cho AI service

Cần biết biến pipeline thành endpoint.

Ví dụ RAG pipeline:

```text
load docs
chunk
embed
store vector
retrieve
generate answer
```

Nên expose thành:

```text
POST /documents/index
POST /retrieve
POST /ask
POST /chat/stream
GET  /health
```

---

## Ưu tiên 4: Streaming

Phần này chưa cần học quá sâu ngay từ đầu, nhưng với LLM thì chắc chắn phải biết.

Cần đạt:

```text
hiểu vì sao chatbot cần streaming
biết StreamingResponse
biết stream từng chunk/token
biết frontend/client nhận stream khác response thường
```

---

# Những phần chưa cần học sâu ngay

|Chưa cần sâu|Vì sao|
|---|---|
|Django|Nặng, không cần cho AI API nhỏ|
|GraphQL|Chưa cần|
|WebSocket sâu|Sau này làm realtime phức tạp mới cần|
|Microservices sâu|Chưa cần ở giai đoạn base|
|Load balancing|Để sau khi deploy production|
|Backend architecture lớn|Không cần học lan man ban đầu|
|Database backend sâu|Biết CRUD cơ bản là đủ trước|

---

# Mini project nên làm

Làm một **RAG API mini** bằng FastAPI:

```text
rag-api/
  app/
    main.py
    schemas.py
    rag_pipeline.py
    retriever.py
    generator.py
  data/
    docs/
  requirements.txt
  README.md
```

Có các endpoint:

```text
GET  /health
POST /documents/index
POST /retrieve
POST /ask
POST /chat/stream
```

Yêu cầu chạy được:

```bash
uvicorn app.main:app --reload --port 8000
```

Test bằng:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "top_k": 5}'
```

Sau đó nâng cấp:

```text
thêm Dockerfile
thêm docker-compose với Qdrant
thêm .env
thêm logging
thêm streaming response
```

---

# Dòng roadmap chuẩn hóa thành

> **API/backend:** hiểu HTTP, REST API, method/path/query/body/header/status code, thiết kế request/response JSON rõ ràng, dùng FastAPI + Pydantic để expose model/RAG/LLM pipeline, có error handling/logging/config, biết streaming response cho chatbot.  
> **Mức đạt:** tự build được một API AI đơn giản gồm `/health`, `/retrieve`, `/ask`, `/chat/stream`, chạy local/Docker được, người khác hoặc frontend gọi được qua HTTP.

**API/backend** trong roadmap này không nhằm đào tạo backend engineer, mà nhằm biến AI model thành sản phẩm có thể gọi được. Đây là cầu nối giữa **model/research code** và **real application**.
