# AI Factory / GenAI Platform — Knowledge cần học kỹ

## Status

Stage: 05 AI Factory / GenAI Platform  
Current level: drafted → architecture mapping next  
Last updated: 2026-06-25

## Must understand

- Service boundary: model, retrieval, agent, tools, evaluation, monitoring.
- Tool registry, schema, permission và approval policy.
- Prompt/model/index/tool versioning.
- Request tracing xuyên service: traces, metrics, logs.
- Eval regression, feedback loop, rollback và runbook.
- Security, governance, config drift và cost tracking.

## Must practice

- Vẽ mini platform architecture.
- Trace một request qua retrieval/model/agent/tool.
- Định nghĩa eval regression cho một workflow.
- Viết runbook debug: retrieval failure, model failure, tool failure.
- Ghi config/version cần rollback khi lỗi.

## Can explain when ready

- AI Factory khác một app RAG/Agent riêng lẻ ở đâu?
- Vì sao service boundary quan trọng cho scale/debug/deploy?
- Khi production answer sai, trace/debug end-to-end như thế nào?

---

Stage 5 tương ứng với level **AI Systems Engineer / GenAI Platform Engineer**: không còn build từng app rời rạc nữa, mà biết gom **RAG + Agent + Model Serving + Evaluation + Monitoring + Deployment** thành một platform có service boundary rõ, reuse được, đo được, debug được, rollback được.

Roadmap định vị Stage 5 như sau: Stage 5 là **GenAI Platform / AI Factory Engineer**, mục tiêu là biến RAG, agents, serving, evaluation, monitoring thành production-style platform; key skills gồm **agent service, model service, retrieval service, tool registry, evaluation service, monitoring, deployment**. ([Khoalearningcode](https://khoalearningcode.github.io/AI_Engineer_Roadmap/ "AI Systems Engineer Roadmap"))

---

## 1. AI Factory / GenAI Platform là gì?

Hiểu đơn giản:

```text
RAG = một app biết trả lời theo tài liệu
Agent = một app biết dùng tool để hành động
LLM Serving = một service chạy model
GPU Inference = tối ưu model chạy trên GPU

AI Factory / GenAI Platform =
gom tất cả thành hệ thống platform reusable:
model service + retrieval service + agent service + tool registry + eval service + monitoring + deployment
```

Mục tiêu không còn là:

```text
build được một chatbot RAG.
```

Mà là:

```text
build được một platform để nhiều team/app có thể dùng chung retrieval, model serving, agent tools, evaluation, monitoring, auth, deployment.
```

---

## 2. Platform architecture tổng quát

Một Mini AI Factory nên có dạng:

```text
Client / App / UI
        ↓
API Gateway
        ↓
------------------------------------------------
| Agent Service     | Retrieval Service        |
| Model Service     | Evaluation Service       |
| Tool Registry     | Monitoring / Logging     |
------------------------------------------------
        ↓
Vector DB / Metadata DB / Object Storage / Model Runtime
```

Các service chính:

|Service|Vai trò|
|---|---|
|**API Gateway**|Cổng vào hệ thống, route request, auth, rate limit|
|**Model Service**|Serve LLM/embedding/reranker/model inference|
|**Retrieval Service**|Search documents, vector DB, metadata filter, citations|
|**Agent Service**|Planner, tool calling, state, verifier, trace|
|**Tool Registry**|Quản lý tool schema, permission, risk level|
|**Evaluation Service**|Chạy test RAG/agent/model tự động|
|**Monitoring Service**|Logs, metrics, traces, dashboard, alert|
|**Deployment Layer**|Docker, compose, CI/CD, rollback, config|

---

## 3. Service boundary

Đây là phần quan trọng nhất của platform.

Không nên gom tất cả vào một app:

```text
main.py chứa RAG + agent + model call + eval + logs + API + tool logic
```

Platform nên tách boundary:

```text
retrieval-service
model-service
agent-service
tool-registry-service
evaluation-service
monitoring/logging layer
```

### Vì sao cần service boundary?

|Lý do|Ý nghĩa|
|---|---|
|**Reuse**|Nhiều app dùng chung model/retrieval service|
|**Scale riêng**|Model service cần GPU, retrieval service cần DB/index|
|**Debug dễ**|Biết lỗi ở retrieval, agent, model hay gateway|
|**Deploy riêng**|Update reranker không cần deploy lại agent|
|**Security rõ**|Tool nào cần quyền, service nào được gọi|
|**Evaluation rõ**|Đo từng layer riêng biệt|

Ví dụ boundary tốt:

```text
Agent service không tự query vector DB trực tiếp.
Agent gọi retrieval-service/search.
Retrieval service chịu trách nhiệm search, filter, citation.
Agent service chỉ quyết định khi nào cần retrieve và dùng evidence ra sao.
```

---

## 4. API Gateway

API Gateway là cửa vào của platform.

Nó xử lý:

```text
routing
authentication
authorization
rate limiting
request validation
quota
logging
service discovery
versioning
```

Ví dụ route:

```text
POST /chat              → agent-service
POST /retrieve          → retrieval-service
POST /v1/chat/completions → model-service
POST /eval/run          → evaluation-service
GET  /health            → platform health
```

Kubernetes Ingress là một ví dụ cơ chế route HTTP/HTTPS traffic vào các backend service trong cluster, thường kèm load balancing, SSL termination và name-based routing. ([Kubernetes](https://kubernetes.io/docs/concepts/services-networking/ingress/ "Ingress"))

Cần học:

|Mảng|Nội dung|
|---|---|
|**Route**|Request nào đi service nào|
|**Auth**|User/API key/JWT|
|**Rate limit**|Chống spam/quá tải|
|**Quota**|Giới hạn token/request theo user/team|
|**Request ID**|Gắn ID để trace toàn hệ thống|
|**Timeout**|Không để request treo|
|**Retry policy**|Retry có kiểm soát|
|**Versioning**|`/v1`, `/v2`, model version, prompt version|

---

## 5. Model Service

Model service là service chuyên serve model.

Có thể gồm:

```text
LLM generation
embedding model
reranker model
classifier
vision model
```

API có thể là:

```text
POST /v1/chat/completions
POST /v1/embeddings
POST /rerank
GET  /models
GET  /health
GET  /metrics
```

Cần học:

|Mảng|Nội dung|
|---|---|
|**OpenAI-compatible API**|Dễ thay backend|
|**Streaming**|Trả token dần|
|**Model versioning**|Biết đang chạy model nào|
|**Health/readiness**|Model load xong chưa|
|**Benchmark**|TTFT, TPOT, tokens/sec, concurrency|
|**Fallback**|Model chính fail thì dùng model khác|
|**Config management**|max tokens, temperature default, context length|
|**Cost tracking**|token usage, cost/request|

Model service là chỗ nối Stage 3/4 vào platform.

---

## 6. Retrieval Service

Retrieval service là service chuyên xử lý knowledge.

Nó không chỉ search vector DB, mà quản lý toàn bộ retrieval pipeline:

```text
query
→ query rewrite nếu cần
→ dense/sparse/hybrid search
→ metadata/permission filter
→ rerank
→ return chunks + citations + scores
```

API có thể là:

```text
POST /documents/index
POST /documents/delete
POST /retrieve
POST /rerank
GET  /documents/{id}
GET  /collections
```

Cần học:

|Mảng|Nội dung|
|---|---|
|**Document ingestion**|Load/parse/clean documents|
|**Chunking pipeline**|Version chunk strategy|
|**Index lifecycle**|upsert/delete/re-index|
|**Metadata filtering**|source, page, section, time, permission|
|**Permission-aware retrieval**|User chỉ thấy docs được phép|
|**Hybrid retrieval**|Dense + sparse|
|**Reranking**|Cross-encoder/LLM reranker|
|**Citation mapping**|Chunk ID → source/page|
|**Freshness**|Tài liệu cũ/mới|
|**Retrieval metrics**|recall@k, MRR, NDCG, context precision|

---

## 7. Agent Service

Agent service là service điều phối task.

Nó quản lý:

```text
planner
tool router
tool execution
retrieval call
model call
state
memory
verifier
trace logs
guardrails
```

API có thể là:

```text
POST /agent/run
POST /agent/stream
GET  /agent/runs/{run_id}
GET  /agent/traces/{run_id}
POST /agent/approve/{action_id}
```

Cần học:

|Mảng|Nội dung|
|---|---|
|**Agent loop**|plan → tool → observe → verify → final|
|**State management**|Task state, session state|
|**Tool calls**|Function schema, arguments, output|
|**Trace logs**|Mọi bước agent đã làm|
|**Approval gate**|Action nhạy cảm cần duyệt|
|**Rollback**|Hủy/khôi phục action lỗi|
|**Failure recovery**|Tool fail thì retry/fallback|
|**Agent metrics**|task success, tool-call accuracy, groundedness|

Agent service là nơi Stage 2 đi vào platform.

---

## 8. Tool Registry

Tool registry là nơi quản lý các tool mà agent được phép dùng.

Tool không nên hard-code rải rác trong agent.

Mỗi tool nên có schema:

```json
{
  "name": "create_refund_ticket",
  "description": "Create a refund support ticket after eligibility verification.",
  "input_schema": {
    "order_id": "string",
    "reason": "string",
    "amount": "number"
  },
  "risk_level": "high",
  "requires_approval": true,
  "owner": "support-team",
  "timeout_ms": 5000
}
```

Cần học:

|Mảng|Nội dung|
|---|---|
|**Tool schema**|Input/output rõ ràng|
|**Tool description**|Model biết khi nào dùng|
|**Permission**|User/agent nào được gọi|
|**Risk level**|Low/medium/high|
|**Approval policy**|Tool nào cần người duyệt|
|**Timeout/retry**|Tool fail xử lý sao|
|**Audit log**|Ai gọi tool nào, lúc nào|
|**Versioning**|Tool schema thay đổi theo version|

Cần hiểu:

```text
Agent càng nhiều tool, tool registry càng quan trọng.
Không có registry thì khó kiểm soát quyền, trace, version và rủi ro.
```

---

## 9. Evaluation Service

Đây là phần cực quan trọng để platform không chỉ “chạy được” mà còn “đo được”.

Evaluation service chạy test tự động cho:

```text
RAG
agent
model serving
prompt
retrieval
reranker
tool calls
guardrails
```

MLflow hiện có hướng hỗ trợ tracking, evaluation, prompt management, tracing, model registry và agent/LLM workflows; Model Registry cung cấp model store tập trung với lineage, versioning, aliasing và metadata tagging cho lifecycle model. ([MLflow AI Platform](https://mlflow.org/docs/latest/ "MLflow Documentation | MLflow AI Platform"))

Cần học:

|Eval target|Metric|
|---|---|
|**Retrieval**|recall@k, MRR, NDCG, context precision/recall|
|**RAG answer**|faithfulness, answer correctness, citation accuracy|
|**Agent**|task success rate, tool-call accuracy, trajectory correctness|
|**Model serving**|TTFT, TPOT, tokens/sec, error rate|
|**Guardrails**|unsafe action blocked, approval required|
|**Regression**|version mới có làm tệ hơn version cũ không|
|**Cost**|cost/request, cost/successful task|

Evaluation service nên có:

```text
eval datasets
scheduled eval runs
before/after comparison
prompt/model/retriever version tracking
failure case report
benchmark table
```

---

## 10. Monitoring / Observability

Monitoring trả lời câu hỏi:

```text
Hệ thống đang chạy tốt không?
Chậm ở đâu?
Lỗi ở service nào?
Request này đi qua những bước nào?
Model nào tốn tiền nhất?
Agent fail vì tool hay retrieval?
```

OpenTelemetry là framework vendor-neutral để instrument, generate, collect và export telemetry data như **traces, metrics, logs**; đây là bộ ba cần hiểu khi làm observability. ([OpenTelemetry](https://opentelemetry.io/docs/ "Documentation"))

### Logs

Ghi event chi tiết:

```text
request_id
user_id
service_name
input size
retrieved chunks
tool calls
model name
error message
```

### Metrics

Số đo tổng hợp:

```text
latency p50/p95/p99
error rate
request count
tokens/sec
GPU memory
retrieval recall
agent success rate
cost/request
```

### Traces

Theo dõi request đi qua nhiều service:

```text
gateway
→ agent-service
→ retrieval-service
→ model-service
→ tool-service
→ final response
```

OpenTelemetry logs spec cũng nhấn mạnh logs có thể tương quan với traces và metrics thông qua resource/context, giúp truy vấn và phân tích hệ thống tốt hơn. ([OpenTelemetry](https://opentelemetry.io/docs/specs/otel/logs/ "OpenTelemetry Logging"))

---

## 11. Deployment

Stage này không cần thành DevOps chuyên sâu, nhưng phải biết deploy platform gọn.

Tối thiểu:

```text
Docker
Docker Compose
.env / config
health checks
logs
basic CI/CD
rollback
```

Nâng cao hơn:

```text
Kubernetes
Ingress
Secrets
ConfigMaps
Horizontal scaling
service discovery
```

Kubernetes là hệ thống open source để tự động hóa deployment, scaling và management của containerized applications. ([Kubernetes](https://kubernetes.io/ "Kubernetes"))

Cần học:

|Mảng|Nội dung|
|---|---|
|**Dockerfile**|Đóng gói từng service|
|**Compose**|Chạy multi-service local|
|**Health check**|Service sống chưa|
|**Readiness check**|Service sẵn sàng nhận traffic chưa|
|**Config**|Tách config khỏi code|
|**Secrets**|API key, DB password|
|**CI/CD basic**|Test → build image → deploy|
|**Rollback**|Version mới lỗi thì quay lại|
|**Blue/green/canary concept**|Deploy ít traffic trước|

Kubernetes Secrets dùng để lưu thông tin nhạy cảm như credentials/configs và có thể dùng với workloads hoặc Ingress TLS; đây là khái niệm cần biết khi platform bắt đầu có auth/API keys. ([Kubernetes](https://kubernetes.io/docs/concepts/configuration/secret/ "Secrets"))

---

## 12. Config, versioning, registry

Platform phải version hóa mọi thứ quan trọng.

Cần version:

```text
model version
prompt version
retriever version
chunking strategy version
embedding model version
reranker version
tool schema version
eval dataset version
deployment version
```

Nếu không version, sẽ không biết:

```text
Vì sao hôm qua answer đúng, hôm nay answer sai?
Do model đổi?
Do prompt đổi?
Do retriever đổi?
Do document index đổi?
Do tool schema đổi?
Do eval set đổi?
```

Cần học:

|Registry|Vai trò|
|---|---|
|**Model registry**|Quản lý model version/lifecycle|
|**Prompt registry**|Quản lý prompt template/version|
|**Tool registry**|Quản lý tool schema/permission|
|**Dataset registry**|Quản lý eval/data version|
|**Index registry**|Quản lý vector index version|
|**Experiment tracking**|Ghi lại config/metrics/artifacts|

---

## 13. Security / governance

AI Factory phải có governance, vì hệ thống có private data, tools, users, logs.

Cần học:

|Mảng|Nội dung|
|---|---|
|**Authentication**|Ai đang gọi hệ thống|
|**Authorization/RBAC**|User được dùng service/tool/data nào|
|**Tenant isolation**|Không lẫn dữ liệu giữa team/customer|
|**PII protection**|Che/lọc thông tin nhạy cảm|
|**Prompt injection defense**|Không để document/tool output điều khiển agent|
|**Tool permission**|Tool nguy hiểm cần approval|
|**Audit log**|Ai hỏi gì, tool nào được gọi|
|**Data retention**|Log/data giữ bao lâu|
|**Policy enforcement**|Rule doanh nghiệp|
|**Secrets management**|API keys/passwords|

Cần hiểu:

```text
Platform càng reusable thì càng phải kiểm soát quyền.
Một lỗi permission ở retrieval/tool có thể leak dữ liệu toàn hệ thống.
```

---

## 14. Runbook

Runbook là tài liệu xử lý sự cố.

Không có runbook thì production fail là mò.

Runbook nên trả lời:

```text
Nếu API chậm thì xem dashboard nào?
Nếu model OOM thì giảm config gì?
Nếu retrieval trả sai thì kiểm tra index nào?
Nếu agent gọi sai tool thì xem trace ở đâu?
Nếu deploy lỗi thì rollback thế nào?
Nếu vector index stale thì re-index ra sao?
Nếu error rate tăng thì ai chịu trách nhiệm?
```

Runbook nên có:

|Mảng|Nội dung|
|---|---|
|**Symptoms**|Dấu hiệu lỗi|
|**Dashboard**|Xem metric nào|
|**Logs/traces**|Tìm theo request_id ở đâu|
|**Likely causes**|Nguyên nhân thường gặp|
|**Immediate mitigation**|Giảm thiệt hại trước|
|**Rollback steps**|Quay lại version cũ|
|**Escalation**|Báo ai|
|**Postmortem**|Ghi nguyên nhân và phòng tránh|

Ví dụ:

```text
Symptom: p95 latency tăng mạnh.
Check:
1. Gateway queue time
2. Agent tool-call count
3. Retrieval latency
4. Model TTFT/TPOT
5. GPU memory
6. Error/timeout rate
Mitigation:
- giảm top_k
- tắt reranker tạm thời
- giảm max_tokens
- route sang model nhỏ hơn
- rollback prompt/model config
```

---

## 15. Feedback loop

Platform cần học từ usage thật.

Feedback có thể là:

```text
user thumbs up/down
user correction
human review
failed queries
low-confidence answers
tool failure
hallucination reports
latency/cost outliers
```

Feedback dùng để:

```text
cập nhật eval set
tạo hard cases
tune retrieval
tune prompt
thêm guardrail
sửa chunking
đổi reranker
fine-tune nếu cần
```

Cần hiểu:

```text
Production logs không chỉ để debug.
Chúng là nguồn tạo eval set và cải thiện hệ thống.
```

---

## 16. Platform failure modes

AI Factory fail có thể do nhiều tầng.

|Tầng|Lỗi|
|---|---|
|**Gateway**|Auth fail, rate limit sai, route sai|
|**Retrieval service**|Index stale, metadata filter sai, permission leak|
|**Model service**|OOM, latency cao, model version sai|
|**Agent service**|Planner sai, tool call sai, loop vô hạn|
|**Tool registry**|Schema sai, permission sai, tool timeout|
|**Evaluation service**|Eval set cũ, metric sai, regression không bắt được lỗi|
|**Monitoring**|Không có trace/log đủ để debug|
|**Deployment**|Config drift, secret thiếu, rollback khó|
|**Security**|Prompt injection, PII leak, tenant leak|
|**Cost**|Context quá dài, tool calls quá nhiều, GPU idle|

Pattern debug platform:

```text
Request sai/chậm
→ lấy request_id
→ xem trace qua gateway/agent/retrieval/model/tool
→ xác định service nào chậm/sai
→ xem version/config của service đó
→ so với eval/benchmark gần nhất
→ rollback hoặc hotfix đúng tầng
```

---

# Bảng tổng hợp kiến thức

|Mảng|Nội dung cần nắm|
|---|---|
|**Platform concept**|Gom RAG, Agent, Model Serving, Eval, Monitoring thành platform reusable|
|**Service boundary**|Tách model service, retrieval service, agent service, tool registry, eval service|
|**API Gateway**|Routing, auth, rate limit, quota, request ID, timeout, versioning|
|**Model Service**|OpenAI-compatible API, streaming, health, metrics, model version, benchmark|
|**Retrieval Service**|ingestion, chunking, vector DB, metadata filter, permission-aware retrieval, citations|
|**Agent Service**|planner, tool calls, state, verifier, trace logs, guardrails|
|**Tool Registry**|tool schema, permission, risk level, approval policy, audit log|
|**Evaluation Service**|RAG eval, agent eval, model benchmark, regression test, eval dataset|
|**Monitoring**|logs, metrics, traces, dashboard, alert, latency/error/cost|
|**Deployment**|Docker, Compose, basic CI/CD, health/readiness, rollback|
|**Config/versioning**|model, prompt, retriever, index, tool schema, eval dataset, deployment version|
|**Security/governance**|auth, RBAC, tenant isolation, PII, prompt injection, audit logs|
|**Runbook**|debug guide, rollback steps, mitigation, escalation, postmortem|
|**Feedback loop**|user feedback, failed queries, human review, eval set update|
|**Failure modes**|gateway/retrieval/model/agent/tool/eval/monitoring/deployment/security/cost failures|

---

# Dòng roadmap chuẩn hóa

> **AI Factory / GenAI Platform:** hiểu cách đóng gói RAG, Agent, LLM Serving, Evaluation, Monitoring và Deployment thành platform reusable với service boundaries rõ ràng: API gateway, model service, retrieval service, agent service, tool registry, evaluation service, observability layer, deployment pipeline, security/governance và runbook.  
> **Mức cần đạt:** thiết kế được một Mini AI Factory nơi nhiều app có thể dùng chung retrieval/model/agent/tools; biết trace một request qua nhiều service, đo latency/cost/error, chạy regression eval, kiểm soát quyền, rollback khi lỗi và viết runbook debug production.

---

# Cấu trúc ghi chú

```text
AI Factory / GenAI Platform

1. Platform overview
   - from apps to reusable AI services
   - RAG + Agent + Model Serving + Eval + Monitoring

2. Service boundaries
   - model service
   - retrieval service
   - agent service
   - tool registry
   - evaluation service
   - monitoring layer

3. API Gateway
   - routing
   - auth
   - rate limit
   - quota
   - timeout
   - request ID
   - versioning

4. Model Service
   - OpenAI-compatible API
   - streaming
   - health/readiness
   - metrics
   - model version
   - benchmark

5. Retrieval Service
   - ingestion
   - chunking/indexing
   - vector search
   - metadata filtering
   - permission-aware retrieval
   - citations
   - freshness

6. Agent Service
   - planner
   - tool calling
   - state
   - verifier
   - guardrails
   - trace logs

7. Tool Registry
   - tool schema
   - tool description
   - permissions
   - risk level
   - approval policy
   - audit log

8. Evaluation Service
   - RAG evaluation
   - agent evaluation
   - model serving benchmark
   - regression test
   - eval dataset versioning

9. Observability
   - logs
   - metrics
   - traces
   - dashboard
   - alerts
   - request-level tracing
   - cost tracking

10. Deployment
   - Docker
   - Docker Compose
   - basic CI/CD
   - health checks
   - config/secrets
   - rollback
   - Kubernetes concept

11. Versioning and registries
   - model version
   - prompt version
   - retriever/index version
   - tool schema version
   - eval dataset version

12. Security and governance
   - authentication
   - authorization/RBAC
   - tenant isolation
   - PII
   - prompt injection
   - audit log
   - secrets management

13. Runbook
   - symptoms
   - dashboards
   - logs/traces
   - likely causes
   - mitigation
   - rollback
   - escalation
   - postmortem

14. Feedback loop
   - user feedback
   - failed queries
   - human review
   - eval set update
   - prompt/retriever/tool improvement

15. Platform failure modes
   - gateway failure
   - retrieval failure
   - model service failure
   - agent failure
   - tool failure
   - eval failure
   - monitoring blind spot
   - deployment/config drift
   - security failure
   - cost explosion
```

Phạm vi ở level **AI Systems Engineer** cần thêm **tool registry rõ schema/quyền, prompt/model/index/tool versioning, request tracing xuyên service, eval regression tự động, security/governance, config drift, feedback loop, rollback/runbook, cost tracking**. Đây là giai đoạn chuyển từ build app AI riêng lẻ sang thiết kế **AI platform**.
