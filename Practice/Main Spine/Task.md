# Main Spine Practice — Bộ đề bài doanh nghiệp

Mục tiêu của file này là giữ danh sách bài toán thực tế để luyện theo Main Spine. Đây chưa phải phase triển khai; mỗi đề chỉ xác định business problem, user, dữ liệu, hành vi hệ thống, metric và stage kiến thức liên quan.

Format mỗi đề:

```text
Tên bài toán
Doanh nghiệp gặp vấn đề gì?
Ai dùng?
Dữ liệu đầu vào là gì?
AI system phải làm gì?
Metric nào chứng minh business value?
Main Spine nào được dùng?
Vì sao đề này thực tế?
```

Nguyên tắc chọn đề: tránh lặp lại nhiều biến thể “chatbot tài liệu”, ưu tiên bài toán có user rõ, dữ liệu rõ, metric rõ và liên hệ trực tiếp tới RAG, Agent, Serving, GPU Inference hoặc AI Factory.

## Dataset Plan

Nguồn dữ liệu, synthetic/public dataset và mock enterprise data cho các đề nằm ở `Datasets.md`.

---

# Nhóm đề bài Main Spine thực tế

## 1. Internal Knowledge Copilot cho doanh nghiệp nhiều tài liệu

### Bài toán

Một công ty có quá nhiều tài liệu nội bộ: HR policy, IT guide, product docs, onboarding docs, SOP, engineering wiki, security policy. Nhân viên mới hoặc nhân viên nội bộ mất nhiều thời gian tìm câu trả lời, hỏi lặp đi lặp lại trên Slack/Teams, và câu trả lời thường không có nguồn rõ ràng.

McKinsey 2025 ghi nhận agentic AI được báo cáo dùng nhiều nhất ở **IT** và **knowledge management**, nên đây là bài toán rất thực tế cho doanh nghiệp. ([McKinsey & Company][1])

### Người dùng

* Nhân viên mới.
* HR.
* IT support.
* Engineering team.
* Internal operations team.

### Dữ liệu

* PDF policy.
* Markdown docs.
* Confluence/Notion export.
* Google Drive docs.
* FAQ.
* Slack knowledge dump nếu có.

### AI system phải làm gì

* Trả lời câu hỏi dựa trên tài liệu nội bộ.
* Dẫn nguồn chính xác.
* Không trả lời nếu không có evidence.
* Tìm đúng policy/version mới nhất.
* Cho người dùng biết câu trả lời đến từ document nào, section nào.
* Ghi log câu hỏi nào bị fail để cải thiện tài liệu.

### Metric chứng minh business value

| Metric                       | Ý nghĩa                                   |
| ---------------------------- | ----------------------------------------- |
| Time-to-answer               | Nhân viên tìm câu trả lời nhanh hơn không |
| Deflection rate              | Bao nhiêu câu hỏi không cần hỏi HR/IT nữa |
| Citation accuracy            | Dẫn nguồn đúng không                      |
| Answer groundedness          | Trả lời có bám tài liệu không             |
| Repeated question reduction  | Câu hỏi lặp trên Slack/Teams giảm không   |
| New employee onboarding time | Onboarding có nhanh hơn không             |

### Main Spine dùng

* RAG Foundation.
* API/backend.
* Evaluation.
* Observability.
* Sau này nối được Agent.

### Vì sao đề này đáng làm

Vì nó đánh đúng nhu cầu gần như công ty nào cũng có:

> “Thông tin có sẵn rồi, nhưng người cần thì không tìm được.”

Đây là bài tập trọng tâm cho **RAG Foundation**.

---

## 2. Customer Support Resolution Assistant

### Bài toán

Một công ty SaaS/e-commerce/fintech có nhiều ticket customer support. Support agent phải đọc FAQ, policy, order status, refund rule, ticket history, product docs. Việc xử lý chậm, câu trả lời không đồng nhất, khách phải chờ lâu.

Các use case customer service AI agent phổ biến gồm order tracking, refund/returns, account management, billing inquiry, troubleshooting, product guidance, IT/HR support và proactive outreach. ([fin.ai][2])

### Người dùng

* Customer support agent.
* Customer success team.
* Support manager.
* End customer, nếu expose ra ngoài.

### Dữ liệu

* FAQ.
* Product docs.
* Refund policy.
* Ticket history.
* Order database mock.
* Customer profile mock.
* CRM notes.

### AI system phải làm gì

* Đọc ticket và hiểu khách đang hỏi gì.
* Retrieve policy/product docs liên quan.
* Tóm tắt lịch sử case.
* Đề xuất câu trả lời cho support agent.
* Gọi tool để check order/refund/account status.
* Nếu case nhạy cảm thì không tự trả lời, phải escalate.
* Ghi lại lý do vì sao đề xuất câu trả lời đó.

### Metric chứng minh business value

| Metric              | Ý nghĩa                          |
| ------------------- | -------------------------------- |
| First response time | Trả lời khách nhanh hơn          |
| Average handle time | Agent xử lý mỗi ticket nhanh hơn |
| Resolution rate     | Tự xử lý được bao nhiêu ticket   |
| Escalation rate     | Bao nhiêu case phải chuyển người |
| CSAT                | Khách hài lòng hơn không         |
| Cost per ticket     | Chi phí xử lý ticket giảm không  |
| Agent edit distance | Agent có phải sửa nhiều không    |

### Main Spine dùng

* RAG.
* Agent tool calling.
* Guardrails.
* LLM serving.
* Observability.

### Vì sao đề này đáng làm

Vì nó không phải demo chatbot đơn giản. Nó gắn với chi phí và chất lượng vận hành:

> Giảm thời gian support, giảm chi phí ticket, tăng chất lượng phản hồi khách hàng.

Đây là đề mạnh để chứng minh AI có tác động trực tiếp tới business operation.

---

## 3. IT Service Desk Agent

### Bài toán

Công ty có rất nhiều ticket IT lặp lại: quên mật khẩu, lỗi VPN, xin quyền truy cập, setup laptop, cài phần mềm, lỗi email, onboarding account. IT team bị quá tải vì task L1 đơn giản nhưng vẫn phải xử lý thủ công.

Gartner có riêng nhóm nghiên cứu về AI use cases cho IT service desk, cho thấy đây là bài toán đủ thực tế trong enterprise IT. ([gartner.com][3])

### Người dùng

* Nhân viên công ty.
* IT helpdesk.
* IT operations.
* Security/admin team.

### Dữ liệu

* IT SOP.
* Troubleshooting docs.
* Access policy.
* Asset inventory mock.
* Ticket system mock.
* User directory mock.
* App permission matrix.

### AI system phải làm gì

* Nhận ticket IT bằng natural language.
* Phân loại loại sự cố.
* Tìm SOP liên quan.
* Hỏi thêm thông tin nếu thiếu.
* Gọi tool mock: check account, check VPN, create ticket, reset password mock, request access.
* Chặn hành động rủi ro.
* Yêu cầu human approval với hành động nhạy cảm.
* Tạo audit log đầy đủ.

### Metric chứng minh business value

| Metric                  | Ý nghĩa                                 |
| ----------------------- | --------------------------------------- |
| L1 auto-resolution rate | Tự xử lý được bao nhiêu ticket đơn giản |
| Mean time to resolve    | Thời gian xử lý ticket giảm không       |
| Escalation accuracy     | Chuyển đúng team không                  |
| Unsafe action blocked   | Có chặn hành động nguy hiểm không       |
| Human approval rate     | Bao nhiêu action cần duyệt              |
| Audit completeness      | Log có đủ để kiểm tra không             |

### Main Spine dùng

* RAG cho SOP.
* Agent planner/tool calling.
* Guardrails.
* Observability.
* Serving gateway.
* AI Factory sau này.

### Vì sao đề này đáng làm

Đây là đề **rất enterprise** vì nó có đủ:

> Knowledge retrieval + workflow automation + tool calling + approval + audit + security.

Đây là bài tập trọng tâm cho **Agentic AI Systems**.

---

## 4. Contract / Policy / RFP Review Copilot

### Bài toán

Legal, procurement, sales engineer hoặc compliance team phải đọc nhiều hợp đồng, vendor proposal, RFP, policy, security questionnaire. Việc review chậm, dễ bỏ sót điều khoản rủi ro, và mất nhiều thời gian copy-paste thông tin từ tài liệu công ty sang form phản hồi.

### Người dùng

* Legal team.
* Procurement.
* Sales engineer.
* Compliance.
* Security team.
* Finance.

### Dữ liệu

* Contract PDF.
* RFP document.
* Company policy.
* Security compliance docs.
* Pricing/SLA docs.
* Previous proposal.
* Approved legal clauses.

### AI system phải làm gì

* Đọc tài liệu dài.
* Extract requirement, clause, deadline, obligation.
* So sánh hợp đồng/RFP với policy nội bộ.
* Flag điều khoản rủi ro.
* Gợi ý câu trả lời cho RFP.
* Dẫn nguồn cho từng claim.
* Xuất structured output: risk, evidence, suggested action.
* Không tự quyết định legal final answer.

### Metric chứng minh business value

| Metric                     | Ý nghĩa                           |
| -------------------------- | --------------------------------- |
| Review time reduction      | Giảm thời gian đọc tài liệu       |
| Clause extraction accuracy | Extract đúng điều khoản không     |
| Risk flag precision        | Flag đúng risk không              |
| Citation coverage          | Mỗi risk có evidence không        |
| Human correction rate      | Người review phải sửa nhiều không |
| RFP completion time        | Nộp proposal nhanh hơn không      |

### Main Spine dùng

* RAG.
* Long document retrieval.
* Reranking.
* Agent verifier.
* Structured output.
* Evaluation.

### Vì sao đề này đáng làm

Đây là bài toán rất rõ business value:

> Giảm thời gian review tài liệu rủi ro cao, giảm bỏ sót requirement, tăng tốc sales/procurement/legal workflow.

Nó khác Knowledge Copilot ở chỗ không chỉ hỏi đáp, mà là **review + compare + flag risk**.

---

## 5. Natural Language Business Data Analyst

### Bài toán

Manager muốn hỏi dữ liệu bằng ngôn ngữ tự nhiên nhưng không biết SQL:

* “Doanh thu tháng này giảm ở khu vực nào?”
* “Ticket nào quá SLA?”
* “Top khách hàng có nguy cơ churn?”
* “Refund rate theo sản phẩm ra sao?”
* “Chiến dịch marketing nào có conversion thấp?”

Data analyst bị quá tải bởi các câu hỏi lặp lại. Business user thì phải chờ quá lâu mới có insight.

### Người dùng

* Operations manager.
* Sales manager.
* Product manager.
* Finance analyst.
* Customer success manager.
* Founder/CEO ở startup.

### Dữ liệu

* SQL database mock.
* Business glossary.
* Metric definitions.
* Dashboard metadata.
* CRM/order/ticket tables.
* API mock.

### AI system phải làm gì

* Hiểu câu hỏi business.
* Retrieve schema và metric definition.
* Sinh SQL read-only.
* Validate SQL trước khi chạy.
* Không cho query nguy hiểm như DELETE/UPDATE.
* Giải thích kết quả bằng ngôn ngữ dễ hiểu.
* Nêu caveat nếu dữ liệu thiếu.
* Lưu query trace để analyst kiểm tra.

### Metric chứng minh business value

| Metric                     | Ý nghĩa                                    |
| -------------------------- | ------------------------------------------ |
| Query success rate         | Câu hỏi có convert đúng thành SQL không    |
| Execution accuracy         | SQL chạy ra kết quả đúng không             |
| Time-to-insight            | Business user có insight nhanh hơn không   |
| Unsafe query blocked       | Có chặn query nguy hiểm không              |
| Analyst workload reduction | Giảm request lặp cho data team không       |
| Metric correctness         | Model có dùng đúng định nghĩa metric không |

### Main Spine dùng

* RAG cho schema/metric glossary.
* Agent tool calling cho SQL/API.
* Guardrails.
* Evaluation.
* Serving.

### Vì sao đề này đáng làm

Đây là đề chứng minh năng lực dùng LLM vượt khỏi chatbot tài liệu:

> Biến natural language thành business insight có kiểm soát.

Phù hợp với profile AI Engineer có nền backend/data/system.

---

## 6. Enterprise LLM Gateway / Model Router

### Bài toán

Khi nhiều team trong công ty bắt đầu dùng LLM, mỗi team tự gọi API riêng. Hậu quả:

* Không kiểm soát được cost.
* Không biết team nào dùng model nào.
* Không có logging chuẩn.
* Không có fallback.
* Không có rate limit.
* Không biết latency p95.
* Không benchmark được cloud model vs local model.
* Không kiểm soát dữ liệu nhạy cảm.

Doanh nghiệp cần một lớp **LLM Gateway** đứng giữa application và model provider.

### Người dùng

* AI engineer.
* Platform engineer.
* MLOps/LLMOps team.
* Engineering manager.
* Finance/CTO muốn theo dõi chi phí.

### Dữ liệu

* Request logs.
* Token usage.
* User/app metadata.
* Model config.
* Prompt templates.
* Cost table.
* Benchmark workload.

### AI system phải làm gì

* Cung cấp OpenAI-compatible API nội bộ.
* Route request tới model phù hợp.
* Hỗ trợ streaming.
* Rate limit theo app/user/team.
* Log token, latency, error.
* Fallback khi model lỗi.
* Chọn model rẻ hơn cho task đơn giản.
* Chọn model mạnh hơn cho task khó.
* Chặn dữ liệu nhạy cảm nếu policy không cho gửi ra cloud.

### Metric chứng minh business value

| Metric                         | Ý nghĩa                      |
| ------------------------------ | ---------------------------- |
| Cost per request               | Kiểm soát chi phí LLM        |
| Token usage by team            | Biết team nào đang tốn nhiều |
| P50/P95 latency                | Theo dõi trải nghiệm thật    |
| Error rate                     | Model/API có ổn định không   |
| Fallback success rate          | Khi lỗi có cứu request không |
| Rate-limit violations          | App nào dùng vượt quota      |
| Local vs cloud cost comparison | Chạy local có đáng không     |

### Main Spine dùng

* LLM Serving.
* API/backend.
* Observability.
* GPU Inference.
* AI Factory.

### Vì sao đề này đáng làm

Đây là đề chuyển trọng tâm từ “build AI app” sang “build AI infrastructure”.

Câu business rất rõ:

> Khi công ty có nhiều AI app, làm sao kiểm soát model, cost, latency, privacy và reliability?

Đây là bài tập trọng tâm cho **LLM Serving**.

---

## 7. GPU Inference Cost & Performance Optimizer

### Bài toán

Công ty muốn chạy model nội bộ vì dữ liệu nhạy cảm hoặc muốn giảm chi phí API. Nhưng họ không biết:

* Chọn model nào.
* Chạy FP16 hay INT4.
* Cần bao nhiêu VRAM.
* Bao nhiêu user concurrent thì chịu được.
* Long-context RAG tốn thêm bao nhiêu.
* Local serving có thật sự rẻ hơn cloud không.
* GPU có bị idle không.

NVIDIA NeMo nhấn mạnh enterprise AI agent cần specialization, optimization, governance và có thể chạy trên cloud/on-prem/hybrid environment; điều này đúng với nhu cầu doanh nghiệp muốn kiểm soát deployment và tối ưu inference. ([NVIDIA][4])

### Người dùng

* AI platform team.
* Infrastructure team.
* CTO.
* MLOps engineer.
* Công ty cần private AI deployment.

### Dữ liệu

* Benchmark prompts.
* RAG workloads.
* Agent workloads.
* Long context prompts.
* Model configs.
* GPU telemetry.
* Latency logs.
* Cost assumptions.

### AI system phải làm gì

* Benchmark nhiều model/precision.
* Đo TTFT, TPOT, tokens/sec.
* Đo VRAM usage.
* Đo throughput theo concurrency.
* So sánh quality drop khi quantization.
* Ước lượng cost/token.
* Đưa recommendation: model nào dùng cho workload nào.

### Metric chứng minh business value

| Metric              | Ý nghĩa                               |
| ------------------- | ------------------------------------- |
| TTFT                | Người dùng chờ token đầu tiên bao lâu |
| TPOT                | Tốc độ sinh token                     |
| Tokens/sec/GPU      | GPU có hiệu quả không                 |
| Max concurrency     | Phục vụ được bao nhiêu user           |
| VRAM usage          | Model có fit GPU không                |
| Cost per 1M tokens  | Có rẻ hơn cloud không                 |
| Quality degradation | Quantization có làm tệ nhiều không    |

### Main Spine dùng

* LLM Serving.
* GPU Inference.
* Benchmarking.
* Cost analysis.
* AI Factory infra.

### Vì sao đề này đáng làm

Đây là đề cực hợp NVIDIA-style:

> Không chỉ chạy model, mà biết đo, tối ưu, và ra quyết định hạ tầng dựa trên số liệu.

Đây là bài tập trọng tâm cho **GPU Inference**.

---

## 8. AI Workflow Evaluation & Monitoring Platform

### Bài toán

Công ty có nhiều AI app: RAG, support agent, IT agent, SQL agent. Nhưng không ai biết hệ thống đang tốt hay tệ:

* RAG retrieve sai.
* Agent gọi nhầm tool.
* Câu trả lời không có citation.
* Latency tăng.
* Cost tăng.
* User feedback xấu.
* Không có regression test trước khi đổi model.
* Không biết app nào đang hallucinate nhiều.

Vấn đề không còn là “làm AI app”, mà là **vận hành AI app an toàn và đo được**.

### Người dùng

* AI engineer.
* QA engineer.
* Platform team.
* Product manager.
* Compliance/security.

### Dữ liệu

* Production query logs.
* Retrieved chunks.
* Model responses.
* Tool traces.
* Human feedback.
* Golden test set.
* Evaluation results.
* Latency/cost logs.

### AI system phải làm gì

* Đánh giá RAG retrieval.
* Đánh giá answer faithfulness.
* Đánh giá tool-call correctness.
* Chạy regression test khi đổi model/prompt.
* Theo dõi latency/cost/error.
* Gom failure cases.
* Tạo dashboard chất lượng theo từng workflow.
* Cảnh báo khi metric tụt.

### Metric chứng minh business value

| Metric                      | Ý nghĩa                              |
| --------------------------- | ------------------------------------ |
| Retrieval recall@k          | RAG có lấy đúng tài liệu không       |
| Faithfulness score          | Câu trả lời có bám source không      |
| Tool-call accuracy          | Agent gọi đúng tool không            |
| Regression failure rate     | Đổi model/prompt có làm hỏng không   |
| P95 latency                 | Trải nghiệm production               |
| Cost drift                  | Chi phí có tăng bất thường không     |
| User negative feedback rate | Người dùng có thấy hệ thống tệ không |

### Main Spine dùng

* RAG Evaluation.
* Agent Evaluation.
* Observability.
* LLM Serving metrics.
* AI Factory.

### Vì sao đề này đáng làm

Đây là đề rất thực tế vì enterprise không thể scale AI nếu không đo được chất lượng.

Câu business:

> Làm sao biết hệ thống AI đang đáng tin, đáng dùng, và không âm thầm hỏng sau mỗi lần update?

Đây là đề rất mạnh cho **AI Factory mindset**.

---

## 9. Multi-Department Enterprise AI Factory

### Bài toán

Một công ty có nhiều phòng ban đều muốn dùng AI:

* HR muốn policy assistant.
* IT muốn service desk agent.
* CS muốn support assistant.
* Sales muốn RFP assistant.
* Ops muốn data analyst agent.
* Legal muốn contract reviewer.

Nếu mỗi team tự làm một app riêng, công ty sẽ có:

* nhiều vector DB rời rạc,
* nhiều prompt không kiểm soát,
* nhiều cách gọi model khác nhau,
* không có cost control,
* không có eval chung,
* không có audit,
* không có permission,
* không có observability.

Cần một **AI Factory mini-platform** để nhiều workflow AI dùng chung hạ tầng.

### Người dùng

* Employee.
* HR.
* IT.
* CS.
* Legal.
* Sales.
* Ops.
* AI engineer.
* Platform engineer.
* Compliance officer.

### Dữ liệu

* Department knowledge bases.
* Business databases.
* Tool/API registry.
* Model configs.
* User/team permissions.
* Evaluation datasets.
* Logs/traces.
* Cost data.

### AI system phải làm gì

* Cho mỗi department có knowledge base riêng.
* Có retrieval service dùng chung.
* Có model gateway dùng chung.
* Có agent orchestrator dùng chung.
* Có tool registry với permission.
* Có evaluation service.
* Có monitoring dashboard.
* Có audit trail.
* Có cost tracking theo team/workflow.
* Có policy để chặn task rủi ro.

### Metric chứng minh business value

| Metric                        | Ý nghĩa                        |
| ----------------------------- | ------------------------------ |
| Number of workflows onboarded | Bao nhiêu phòng ban dùng được  |
| Reuse rate                    | Bao nhiêu component dùng chung |
| Cost per workflow             | Mỗi workflow tốn bao nhiêu     |
| Quality per workflow          | Workflow nào tốt/tệ            |
| Time to launch new workflow   | Tạo workflow mới nhanh không   |
| Audit coverage                | Có trace đủ không              |
| Policy violation blocked      | Có chặn hành vi sai không      |
| Platform uptime               | Hạ tầng có ổn định không       |

### Main Spine dùng

Toàn bộ:

* Base engineering.
* RAG.
* Agentic systems.
* LLM serving.
* GPU inference.
* AI Factory.

### Vì sao đề này đáng làm

Đây là bài tập tổng hợp cuối cho **Main Spine**.

Nó trả lời câu doanh nghiệp cấp cao hơn:

> Làm sao công ty scale từ vài AI demo rời rạc thành một nền tảng AI dùng được cho nhiều phòng ban, có governance, monitoring, cost control và reusable infrastructure?

Đây là đề mạnh nhất để thể hiện năng lực vượt khỏi chatbot và tiến tới vai trò **AI Engineer / GenAI Platform Engineer**.

---

# Danh sách đề nên giữ, không lan man

Nên giữ 9 đề này trong repo và chia vai trò như sau:

| Vai trò                  | Đề                                           |
| ------------------------ | -------------------------------------------- |
| RAG core                 | Internal Knowledge Copilot                   |
| Business ops RAG         | Contract/RFP/Policy Review Copilot           |
| Customer-facing workflow | Customer Support Resolution Assistant        |
| Agent core               | IT Service Desk Agent                        |
| Data agent               | Natural Language Business Data Analyst       |
| Serving core             | Enterprise LLM Gateway                       |
| GPU core                 | GPU Inference Cost & Performance Optimizer   |
| AI Factory ops           | AI Workflow Evaluation & Monitoring Platform |
| Final boss               | Multi-Department Enterprise AI Factory       |

# Nếu chỉ chọn 5 đề mạnh nhất

Nếu cần rút gọn, ưu tiên 5 đề sau:

```text
1. Internal Knowledge Copilot
2. IT Service Desk Agent
3. Enterprise LLM Gateway
4. GPU Inference Cost & Performance Optimizer
5. Multi-Department Enterprise AI Factory
```

# Nhóm đề thực tế nhất theo doanh nghiệp

Thực tế nhất theo doanh nghiệp:

```text
1. Customer Support Resolution Assistant
2. IT Service Desk Agent
3. Internal Knowledge Copilot
```

Vì ba đề này có người dùng rõ, pain rõ, KPI rõ, và công ty nào cũng hiểu.

# Nhóm đề thể hiện năng lực AI Engineer mạnh nhất

Mạnh nhất cho profile AI Engineer:

```text
1. Multi-Department Enterprise AI Factory
2. Enterprise LLM Gateway
3. GPU Inference Cost & Performance Optimizer
4. IT Service Desk Agent
```

Vì các đề này vượt khỏi chatbot và chạm vào **system design, serving, evaluation, monitoring, governance, cost, GPU**.

# Chốt

Bộ đề Main Spine khuyến nghị:

```text
Knowledge Copilot
→ IT Service Desk Agent
→ LLM Gateway
→ GPU Inference Optimizer
→ Enterprise AI Factory
```

Đây không phải 5 project rời rạc. Đây là 5 bài toán doanh nghiệp nối nhau thành một đường:

> **Từ AI app → AI workflow → AI serving → AI infra optimization → AI platform.**

[1]: https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai "The State of AI: Global Survey 2025"
[2]: https://fin.ai/learn/ai-agents-in-customer-service "Benefits of AI Agents for Customer Service (2026)"
[3]: https://www.gartner.com/en/documents/6368911 "AI Use-Case Assessment for IT Service Desk"
[4]: https://www.nvidia.com/en-us/ai-data-science/products/nemo/ "NeMo | Build, monitor, and optimize AI agents"
