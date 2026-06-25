# Main Spine Practice — Dataset Plan

Tất cả project Main Spine có thể dùng **synthetic data**, **public dataset** hoặc **mock enterprise data**. Không cần dùng dữ liệu thật của công ty.

Nguyên tắc chính:

> **Data có thể synthetic, nhưng bài toán, workflow, metric và constraint phải thật.**

Dữ liệu giả phải mô phỏng đúng cách doanh nghiệp vận hành: role, permission, SLA, ticket priority, policy version, customer tier, cost, latency, audit log và failure case.

---

# Tổng quan nguồn dữ liệu


| Project                       | Có thể dùng synthetic data không? |                           Có public data không? |
| ----------------------------- | --------------------------------: | ----------------------------------------------: |
| Internal Knowledge Copilot    |                                Có | Có thể dùng public docs hoặc tự viết policy giả |
| Customer Support Assistant    |                                Có |     Có public/synthetic support ticket datasets |
| IT Service Desk Agent         |                                Có |                 Có synthetic IT ticket datasets |
| Contract/RFP Review           |                                Có |                 Có CUAD/legal contract datasets |
| Natural Language Data Analyst |                                Có |              Có Spider / Spider 2.0 text-to-SQL |
| LLM Gateway                   |                                Có |                 Chủ yếu dùng synthetic workload |
| GPU Inference Optimizer       |                                Có |         Dùng synthetic prompts + benchmark thật |
| Eval/Monitoring Platform      |                                Có |                           Tạo logs/eval set giả |
| Enterprise AI Factory         |                                Có |          Gom toàn bộ mock/synthetic data ở trên |

Không có dữ liệu nội bộ thật không phải blocker cho các project Main Spine.

---

# 1. Internal Knowledge Copilot

Project này synthetic rất dễ.

Có thể tự tạo một công ty giả:

```text
Company: NovaRetail
Departments: HR, IT, Security, Engineering, Sales
Docs:
- Employee handbook
- Remote work policy
- Leave policy
- Password policy
- Incident response policy
- Product onboarding guide
- Engineering deployment SOP
```

Data cần có:

| Loại data            | Cách tạo                                  |
| -------------------- | ----------------------------------------- |
| HR policy            | tự viết bằng LLM                          |
| IT guide             | tự viết bằng LLM                          |
| Security policy      | tự viết bằng LLM                          |
| Engineering docs     | tự viết theo style wiki                   |
| FAQ                  | tự generate từ policy                     |
| Evaluation questions | tự generate nhưng phải có expected source |

Điểm quan trọng là **không chỉ tạo tài liệu**, mà phải tạo cả **ground truth**:

```json
{
  "question": "How many days of annual leave does a full-time employee get?",
  "expected_answer": "18 days",
  "expected_source": "HR Leave Policy v2.1",
  "expected_section": "Annual Leave"
}
```

Project này không cần public dataset. Tự synthetic là đủ.

---

# 2. Customer Support Resolution Assistant

Project này có thể dùng synthetic hoặc public dataset.

Có các dataset dạng customer support tickets trên Kaggle, ví dụ một dataset có **20,000 synthetic customer support tickets** mô phỏng môi trường omnichannel support. ([Kaggle][1]) Ngoài ra còn có các dataset customer support ticket chứa thông tin về customer, product purchased, ticket type, channel, status và các field liên quan. ([Kaggle][2])

Có thể tạo công ty giả:

```text
Company: ShopVerse
Domain: e-commerce
Products: electronics, fashion, home appliances
Support issues:
- refund
- delayed delivery
- damaged product
- warranty
- account login
- payment failed
- subscription cancellation
```

Data cần có:

| Loại data              | Cách tạo                        |
| ---------------------- | ------------------------------- |
| Ticket history         | public dataset hoặc tự generate |
| Product FAQ            | synthetic                       |
| Refund policy          | synthetic                       |
| Order database         | synthetic CSV/SQLite            |
| Customer profile       | synthetic                       |
| Support reply examples | synthetic                       |

Mock tool nên có:

```text
get_order_status(order_id)
check_refund_eligibility(order_id)
get_customer_tier(customer_id)
create_escalation_ticket(...)
```

Không cần dữ liệu nội bộ thật.

---

# 3. IT Service Desk Agent

Project này cũng rất hợp synthetic data.

ServiceNow có bài viết về synthetic data trong Now Assist Data Kit, nói synthetic data có thể dùng để mô phỏng IT incidents, HR requests hoặc customer service tickets, giúp làm nhanh và an toàn hơn. ([ServiceNow][3]) Cũng có dataset synthetic IT support tickets trên Hugging Face, với các field như `short_description`, `description`, `urgency`, `impact`, `category`, `assignment_group`, `resolution`. ([Hugging Face][4])

Có thể tạo công ty giả:

```text
Company: VertexBank
Internal systems:
- VPN
- Email
- Jira
- GitHub Enterprise
- HR Portal
- Finance Portal
- Laptop inventory
```

Data cần có:

| Loại data       | Cách tạo          |
| --------------- | ----------------- |
| IT tickets      | synthetic dataset |
| SOP             | synthetic         |
| Access matrix   | synthetic         |
| User directory  | synthetic         |
| Asset inventory | synthetic         |
| Audit log       | synthetic         |
| Tool outputs    | mock API          |

Ví dụ ticket synthetic:

```json
{
  "ticket_id": "INC-10492",
  "employee": "Minh Tran",
  "department": "Finance",
  "issue": "Cannot connect to VPN after password reset",
  "urgency": "medium",
  "impact": "single_user",
  "category": "network/vpn",
  "expected_assignment_group": "IT Network Support",
  "expected_resolution": "Verify MFA sync, reset VPN profile, ask user to reconnect"
}
```

Dạng dữ liệu này đủ thực tế để chứng minh Agent + tool calling + approval + audit.

---

# 4. Contract / Policy / RFP Review Copilot

Project này có public data rất tốt.

CUAD là dataset cho legal contract review, được The Atticus Project curate, dùng cho NLP research và legal contract review. ([atticus-project][5]) CUAD v1 có hơn **13,000 labels** trên **510 commercial legal contracts**, được annotate để xác định 41 loại clause quan trọng khi review hợp đồng. ([Zenodo][6])

Có thể dùng CUAD cho phần contract clause extraction/risk review. Nếu muốn tránh vấn đề license hoặc complexity legal, có thể dùng synthetic data:

```text
Company: CloudNova
Documents:
- Vendor SaaS Agreement
- Data Processing Agreement
- NDA
- Master Service Agreement
- RFP from enterprise client
- Internal legal playbook
```

Data cần có:

| Loại data            | Cách tạo            |
| -------------------- | ------------------- |
| Contract docs        | CUAD hoặc synthetic |
| Clause labels        | CUAD hoặc tự label  |
| Company policy       | synthetic           |
| Risk checklist       | synthetic           |
| RFP requirements     | synthetic           |
| Approved answer bank | synthetic           |

Ground truth nên có:

```json
{
  "clause": "Limitation of Liability",
  "risk_level": "high",
  "reason": "Liability cap exceeds internal policy",
  "expected_source": "MSA Section 12.2",
  "expected_action": "Require legal review"
}
```

Dữ liệu vẫn mô phỏng enterprise workflow dù là synthetic.

---

# 5. Natural Language Business Data Analyst

Project này có public dataset rất rõ.

Spider là dataset text-to-SQL lớn, cross-domain, gồm **10,181 questions**, **5,693 SQL queries**, **200 databases**, phủ **138 domains**. ([Yale Lily][7]) Spider 2.0 mới hơn, tập trung vào **632 real-world text-to-SQL workflow problems** từ enterprise-level database use cases, với database từ real data applications, có thể chứa hơn 1,000 columns và dùng local/cloud systems như BigQuery và Snowflake. ([Spider 2.0][8])

Có thể dùng public text-to-SQL để học, nhưng portfolio nên có thêm business database synthetic:

```text
Company: RetailOps
Tables:
- customers
- orders
- products
- support_tickets
- refunds
- marketing_campaigns
- sales_regions
```

Data cần có:

| Loại data               | Cách tạo                    |
| ----------------------- | --------------------------- |
| SQL database            | synthetic SQLite/Postgres   |
| Business questions      | synthetic                   |
| Expected SQL            | tự viết/generate rồi review |
| Business glossary       | synthetic                   |
| Metric definitions      | synthetic                   |
| Unsafe query test cases | synthetic                   |

Ví dụ:

```json
{
  "question": "Which product category had the highest refund rate last quarter?",
  "expected_sql_type": "SELECT only",
  "required_tables": ["orders", "refunds", "products"],
  "business_metric": "refund_rate",
  "blocked_actions": ["DELETE", "UPDATE", "DROP"]
}
```

Nguyên tắc: data có thể synthetic, nhưng SQL output phải chạy thật trên DB thật.

---

# 6. Enterprise LLM Gateway / Model Router

Project này gần như **không cần business data thật**.

Chỉ cần synthetic workload:

```text
Workload types:
- short chat
- long RAG answer
- code explanation
- SQL generation
- agent planning
- summarization
- customer support reply
```

Data cần có:

| Loại data            | Cách tạo                          |
| -------------------- | --------------------------------- |
| Prompt workload      | synthetic                         |
| User/team metadata   | synthetic                         |
| Model cost table     | tự config                         |
| API logs             | generate từ request thật khi test |
| Latency logs         | đo thật                           |
| Failure cases        | simulate                          |
| Rate-limit scenarios | synthetic                         |

Điểm quan trọng:

> Prompt có thể synthetic, nhưng latency/cost/token/error phải đo thật.

Không được bịa benchmark.

Ví dụ workload:

```json
{
  "app": "support-agent",
  "team": "customer-success",
  "model_policy": "cheap-for-draft,strong-for-escalation",
  "prompt_type": "ticket_reply",
  "max_budget_usd": 0.02,
  "requires_streaming": true
}
```

---

# 7. GPU Inference Cost & Performance Optimizer

Project này cũng dùng synthetic prompt được.

Nhưng metric phải là đo thật:

* TTFT thật.
* TPOT thật.
* tokens/sec thật.
* VRAM thật.
* concurrency thật.
* model quality test thật.

Data cần có:

| Loại data                | Cách tạo  |
| ------------------------ | --------- |
| Prompt benchmark set     | synthetic |
| Long-context RAG prompts | synthetic |
| Agent planning prompts   | synthetic |
| SQL prompts              | synthetic |
| Model configs            | thật      |
| GPU telemetry            | đo thật   |
| Latency logs             | đo thật   |

Nguyên tắc benchmark:


> Synthetic workload OK. Synthetic benchmark number thì không OK.

---

# 8. AI Workflow Evaluation & Monitoring Platform

Project này synthetic được gần như 100%.

Có thể tự tạo logs:

```json
{
  "workflow": "it-service-agent",
  "query": "I cannot access VPN",
  "retrieved_docs": ["VPN Troubleshooting SOP v1.3"],
  "tool_calls": ["check_user_mfa_status", "create_ticket"],
  "answer": "...",
  "latency_ms": 1840,
  "token_usage": 932,
  "human_feedback": "helpful",
  "eval_result": {
    "retrieval_correct": true,
    "tool_call_correct": true,
    "citation_correct": true,
    "unsafe_action": false
  }
}
```

Data cần có:

| Loại data         | Cách tạo                         |
| ----------------- | -------------------------------- |
| Query logs        | synthetic + logs từ project thật |
| Retrieval results | từ RAG app đã build               |
| Tool traces       | mock                             |
| Feedback          | synthetic hoặc tự label          |
| Golden eval set   | synthetic                        |
| Regression cases  | synthetic                        |

Đây là project rất hợp synthetic vì bản chất là platform đo hệ thống.

---

# 9. Multi-Department Enterprise AI Factory

Boss project cũng không cần data thật.

Có thể tạo một công ty giả hoàn chỉnh:

```text
Company: NovaOne Group

Departments:
- HR
- IT
- Customer Support
- Legal
- Sales
- Operations
- Engineering

AI workflows:
- HR Policy Copilot
- IT Service Desk Agent
- Customer Support Assistant
- Contract Review Copilot
- Business Data Analyst
- LLM Gateway
- Evaluation Dashboard
```

Data cần có:

| Department | Data synthetic                      |
| ---------- | ----------------------------------- |
| HR         | handbook, leave policy, payroll FAQ |
| IT         | SOP, tickets, asset inventory       |
| Support    | tickets, customers, orders          |
| Legal      | contracts, RFPs, risk checklist     |
| Ops        | SQL database, KPI glossary          |
| Platform   | model configs, logs, cost table     |

Project này hoàn toàn làm được bằng synthetic enterprise data.

---

# Cách synthetic data cho đúng, không bị “giả trân”

Đây là phần quan trọng nhất.

Không generate bừa 1 đống text. Làm theo quy tắc này:

## 1. Schema-first

Trước khi tạo data, định nghĩa schema.

Ví dụ IT ticket:

```json
{
  "ticket_id": "string",
  "created_at": "datetime",
  "requester_role": "employee | manager | contractor",
  "department": "HR | Finance | Engineering | Sales",
  "category": "vpn | password | access | hardware | software",
  "urgency": "low | medium | high",
  "impact": "single_user | team | company",
  "description": "string",
  "expected_action": "string",
  "requires_approval": "boolean",
  "expected_assignment_group": "string"
}
```

Schema càng enterprise thì project càng thật.

## 2. Business rules-first

Phải có rule giống vận hành doanh nghiệp thật:

```text
- Finance data access always requires manager approval.
- Password reset can be automated if MFA is verified.
- Refund above $500 requires human review.
- Legal risk high if liability cap > 12 months of fees.
- SQL agent can only run SELECT queries.
- Customer PII cannot be sent to cloud model.
```

Đây là thứ làm project “business-real”.

## 3. Có ground truth

Mỗi data item nên có expected answer/action.

Ví dụ:

```json
{
  "query": "Can I get access to the finance dashboard?",
  "expected_tool": "create_access_request",
  "expected_policy": "Finance Access Policy",
  "requires_approval": true,
  "expected_reason": "Finance data requires manager approval"
}
```

Không có ground truth thì không đánh giá được.

## 4. Có edge cases

Doanh nghiệp không chỉ có case đẹp.

Phải có:

```text
- thiếu thông tin
- thông tin mâu thuẫn
- policy cũ vs policy mới
- user không đủ quyền
- request nguy hiểm
- prompt injection
- ambiguous ticket
- document không có câu trả lời
- tool trả lỗi
- API timeout
```

Đây là phần giúp project nhìn rất thật.

## 5. Có temporal/versioning

Enterprise data phải có version:

```text
Refund Policy v1.0
Refund Policy v2.0
Password Policy updated 2026-06-01
Old SLA vs new SLA
Contract draft v1 vs v2
```

RAG mà không xử lý version thì rất dễ trả lời sai.

## 6. Có user/role/permission

Ví dụ:

```text
Employee
Manager
IT Admin
Legal Reviewer
Support Agent
Finance Analyst
Contractor
```

Sau đó test:

```text
Cùng một câu hỏi, mỗi role có được xem cùng một source không?
```

Đây là enterprise-grade.

---

# Nên dùng data nào: synthetic hay public?

Khuyến nghị dùng **hybrid**.

| Loại                | Khi nào dùng                             |
| ------------------- | ---------------------------------------- |
| Synthetic data      | Khi muốn mô phỏng đúng business workflow |
| Public dataset      | Khi cần benchmark có độ tin cậy          |
| Mock API/DB         | Khi cần agent/tool/data workflow         |
| Real open docs      | Khi cần tài liệu dài và đa dạng          |
| Self-generated logs | Khi làm observability/evaluation         |

Ví dụ tốt nhất:

```text
IT Service Desk Agent:
- Synthetic tickets
- Synthetic SOP
- Mock user directory
- Mock tools
- Synthetic eval set

Contract Review:
- CUAD public contracts
- Synthetic company policy
- Synthetic risk checklist
- Synthetic RFP
```

---

# Cái gì được synthetic, cái gì không nên synthetic?

## Được synthetic

```text
- tickets
- policy docs
- SOP
- FAQ
- contracts giả
- customer profile
- order database
- tool outputs
- business glossary
- eval questions
- user feedback
- logs
```

## Không nên synthetic / phải đo thật

```text
- latency
- TTFT
- TPOT
- tokens/sec
- VRAM usage
- concurrency limit
- error rate khi chạy thật
- cost estimate dựa trên token thật
```

Benchmark mà bịa là mất giá trị.

---

# Data Notice khi trình bày project

Câu trả lời chuẩn khi được hỏi về nguồn dữ liệu:

> “Do policy/privacy, project uses synthetic enterprise data and public datasets. I designed the data schema, business rules, permission model, edge cases, and evaluation ground truth to simulate a real enterprise workflow. The goal is to demonstrate system design, RAG quality, agent safety, serving performance, and observability without exposing confidential company data.”

README của từng project nên ghi rõ:

```text
Data Notice:
This project does not use confidential company data.
All documents, tickets, users, tools, and logs are synthetic or public.
Synthetic data is generated with predefined schemas, business rules, role permissions, and evaluation labels.
```

---

# Kết luận

**Toàn bộ project Main Spine đều làm được bằng synthetic/public/mock data**.

Ba luật cần giữ:

```text
1. Data có thể giả, business problem phải thật.
2. Workflow có thể mock, constraint phải giống doanh nghiệp thật.
3. Prompt/data có thể synthetic, benchmark số đo phải đo thật.
```

Nếu làm đúng, synthetic data không làm project yếu đi. Ngược lại, nó thể hiện năng lực enterprise AI ở mức:

> privacy-aware, policy-aware, evaluation-aware, production-aware.

[1]: https://www.kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset "Customer Support Tickets - CRM dataset"
[2]: https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset "Customer Support Ticket Dataset"
[3]: https://www.servicenow.com/community/now-assist-articles/synthetic-data-in-now-assist-data-kit/ta-p/3299227 "Synthetic Data in Now Assist Data Kit"
[4]: https://huggingface.co/datasets/6StringNinja/synthetic-servicenow-incidents "6StringNinja/synthetic-servicenow-incidents · Datasets at ..."
[5]: https://www.atticusprojectai.org/cuad/ "Contract Understanding Atticus Dataset (CUAD)"
[6]: https://zenodo.org/records/4595826 "Contract Understanding Atticus Dataset (CUAD) v1"
[7]: https://yale-lily.github.io/spider "Spider: Yale Semantic Parsing and Text-to-SQL Challenge"
[8]: https://spider2-sql.github.io/ "Spider 2.0"
