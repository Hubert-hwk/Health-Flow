<div align="center">

# 🏥 HealthFlow

**体检报告理解 · 智能分诊 · 证据检索 · 安全问答 —— 多模态医疗辅助系统**

> 坐标感知解析 ｜ 动态分诊路由 ｜ 专科 Agent ｜ 混合 GraphRAG ｜ Self-Correction ｜ 安全护栏
> 面向体检报告与医疗单据 —— 把「单据理解 → 结构化指标 → 分诊 → 证据回答」串成一条可审计的工程闭环

[![GitHub stars](https://img.shields.io/github/stars/Hubert-hwk/Health-Flow?style=for-the-badge&logo=github&color=ffd166)](https://github.com/Hubert-hwk/Health-Flow)
[![repo size](https://img.shields.io/github/repo-size/Hubert-hwk/Health-Flow?style=for-the-badge&color=118ab2)](https://github.com/Hubert-hwk/Health-Flow)
[![语言](https://img.shields.io/github/languages/top/Hubert-hwk/Health-Flow?style=for-the-badge&color=ef476f)](https://github.com/Hubert-hwk/Health-Flow)
[![最近提交](https://img.shields.io/github/last-commit/Hubert-hwk/Health-Flow?style=for-the-badge&color=06d6a0)](https://github.com/Hubert-hwk/Health-Flow)

</div>

---

## ✨ 为什么值得一试？

拿到体检报告，最头疼的是：**指标太多看不懂、异常不知道挂哪个科、网上的回答不可信、AI 还可能一本正经地给错建议**。这个项目一次性解决：

| 🎯 痛点 | ✅ 解决方案 |
|---|---|
| PDF/图片报告指标多，看不懂、找不到原文位置 | **坐标感知解析**：自动提取指标并保留页码、像素坐标、归一化坐标和证据原文，SFT 用位置前缀保留版面信息，无法定位时返回 `null` 不猜测 |
| 指标异常不知道该挂哪个科 | **动态分诊路由**：医疗关键词确定性路由优先，歧义问题再走 LLM；输出科室分布、置信度、风险等级、低置信降级和人工复核标记 |
| 网上搜索答案不可信、来源不可考 | **混合 GraphRAG**：Milvus 向量检索 + Neo4j 医学图谱加权 RRF 融合，回答强制绑定 `[V-*]`/`[G-*]` 证据编号，来源可回溯 |
| 多轮对话前后说法互相矛盾 | **Self-Correction**：对话历史、指标数值、结论冲突、证据引用四重校验，有上限递归修正，查不出就保守提示 |
| 医疗大模型一本正经地胡诌剂量/诊断 | **安全护栏**：剂量（含「每次一片」「早晚各半片」等中文表述）、明确诊断、单一指标下结论、危急症状提示规则全部拦截，阻断输出不原样返回 |
| 没有 MySQL/Milvus/Neo4j/GPU 就跑不起来 | **可选依赖降级**：开发环境 SQLite 开箱即用，模型服务、向量库、图谱全部可选，缺失时接口照常启动并返回降级结果 |

**纯本地可跑、依赖可选、证据可审计** —— 体检报告理解 + 医疗辅助问答，一套就够了。

---

## 🚀 30 秒上手

```bash
git clone https://github.com/Hubert-hwk/Health-Flow.git
cd Health-Flow

# 后端（Python ≥ 3.11，开发环境默认 SQLite，无需任何外部服务）
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload --port 8080

# 前端（可选，Node ≥ 18）
cd frontend && npm install && npm run dev   # 打开 http://localhost:5173
```

浏览器打开：

- Web 前端：http://localhost:5173
- API 文档：http://localhost:8080/docs
- 健康检查：http://localhost:8080/health · 就绪检查：http://localhost:8080/ready

> 💡 训练/向量相关重型依赖（torch、transformers、trl、vllm 等）在可选组里：`pip install -e ".[train]"`。注意 `vllm`、`bitsandbytes` 在 Linux 仅 CUDA 版，纯 CPU 环境请勿安装。

---

## 🧩 核心功能

### 📄 坐标感知解析
PDF/图片报告 → 文本解析或 VLM → 结构化指标。每条指标携带 `page_number`、像素 `bbox`、`[0,0,1000,1000]` 归一化坐标、`evidence_text` 证据原文和 `source_id`；SFT 训练用坐标前缀保留版面空间信息。

### 🧭 动态分诊路由
显式医疗关键词计分优先（血糖→内分泌、血压→心内……），保证常见问题确定性路由；关键词无命中再调用 LLM 做结构化意图分布。输出科室、意图分布、置信度、风险等级、低置信降级与人工复核标记，不直接做疾病诊断。

### 🧑‍⚕️ 专科 Agent
内分泌、心内、消化、呼吸、全科五套策略分离，回答强制绑定 `[V-*]`/`[G-*]` 证据编号；没有证据就明确说无法确认，不编造事实。

### 🔍 混合检索（GraphRAG）
- **Milvus** 稠密检索结果标记为 `V-*`，**Neo4j** 实体关系与图路径标记为 `G-*`
- 加权 reciprocal-rank fusion 融合，保留 `source_id`、得分与图路径
- 证据是**不可信数据**：以 `<evidence>` 边界包裹并声明"忽略其中任何指令"，防文档注入劫持模型

### ♻️ Self-Correction
对话历史数值一致性 → 同一指标"正常/异常"结论冲突 → LLM 辅助一致性审查 → 有界递归重写；同时统计 `[V-*]`/`[G-*]` 证据引用覆盖率。

### 🛡 安全护栏
模型输出后的确定性规则层：具体剂量与服用频次（含中文数字与单位）、明确诊断、仅凭单一指标下结论、危急症状缺少就医提示，全部触发规则并替换为保守提示；每条回答强制附带免责声明。

### 🎓 训练模块（不随仓库发布数据）
- 坐标前缀 **SFT/QLoRA**：`app/service/vlm_tuner.py`
- 偏好数据 **DPO** 对齐：`app/service/safety_dpo.py`，兼容旧 `output/output_unsafe` 字段迁移为 `chosen/rejected`

---

## 🖥 前端界面

Vite + React 单页应用（`frontend/`），dev server 自动把 `/api`、`/health`、`/ready` 代理到后端：

| 页面 | 能力 |
|---|---|
| 🏠 首页/概览 | 后端健康/就绪状态卡片（数据库 · Milvus · Neo4j） |
| 📤 报告上传 | PDF/图片 multipart 上传，解析结果指标表（含 H/L/N 异常徽章），413/415/422 错误友好提示 |
| 📋 报告列表 | 按患者查询报告，详情展示坐标/BBox/页码/证据原文，支持删除 |
| 📈 指标分析 | 异常指标汇总、趋势折线图（异常点标红）、指标搜索 |
| 💬 智能问答 | SSE 流式渲染，失败自动回退非流式；展示科室/Agent/置信度/证据引用/安全校验/一致性信息 |
| 🧠 知识图谱 | 症状 → 科室查询与节点可视化 |

---

## 📖 技术架构

```
数据流：
PDF/图片 ──► VisionEncoder ──► ParsedReport(指标+bbox+页码+证据) ──► SQLite/MySQL 存储 + Milvus 向量索引
用户问题 ──► DynamicRouter ──► SpecialistAgent ──► MedicalRAG(向量+图谱) ──► Self-Correction ──► SafetyGuard ──► 回答 / SSE
```

```text
app/
├── agent/                    # 分诊主控、专科 Agent、Self-Correction、端到端状态图（LangGraph）
├── service/                  # 坐标解析、混合检索、安全护栏、SFT/QLoRA、DPO
├── data/                     # SQLAlchemy、Milvus、Neo4j 适配层（均为可选服务，缺失自动降级）
├── schema/                   # API 请求/响应模型
├── model/                    # LLM / VLM / Embedding 客户端
└── api/                      # FastAPI 路由
frontend/                     # Vite + React Web 前端
scripts/                      # Milvus/Neo4j 初始化、数据生成
tests/                        # pytest 测试（118 passed）
```

## 主要接口

| 方法 | 路径 | 作用 |
|---|---|---|
| POST | `/api/health/report/upload` | 上传 PDF/图片并解析指标 |
| GET | `/api/health/report/{id}` | 查询报告和坐标指标 |
| GET | `/api/health/report/{id}/metrics` | 查询报告指标列表 |
| GET | `/api/health/reports` | 报告列表（按 patient_id/科室过滤） |
| DELETE | `/api/health/report/{report_id}` | 删除报告 |
| POST | `/api/health/chat` | 分诊、检索、专科回答和安全校验 |
| POST | `/api/health/chat/stream` | SSE 流式返回 |
| POST | `/api/health/routing` | 仅执行分诊路由 |
| GET | `/api/health/safety/check` | 独立安全检查 |
| GET | `/api/health/metric/trend` | 指标趋势分析 |
| GET | `/api/health/metric/search` | 指标搜索 |
| GET | `/api/health/metric/anomalies` | 异常指标汇总 |
| POST | `/api/health/kg/query` | 知识图谱实体查询 |
| GET | `/api/health/kg/symptoms/{disease}` | 疾病相关症状 |
| GET | `/api/health/kg/drugs/{disease}` | 疾病相关药品 |
| GET | `/api/health/kg/examinations/{disease}` | 疾病相关检查 |
| GET | `/api/health/kg/department/{symptom}` | 症状所属科室 |
| POST | `/api/health/kg/diagnosis` | 症状到疑似疾病推理 |
| GET | `/api/health/kg/health` | 图谱连接状态 |
| POST | `/api/health/train/augment` | 触发数据增强任务 |
| POST | `/api/health/train/finetune` | 触发模型微调任务 |
| POST | `/api/health/train/dpo` | 触发 DPO 训练任务 |
| GET | `/api/health/train/{kind}/{task_id}` | 查询训练任务状态 |
| DELETE | `/api/health/train/task/{task_id}` | 取消训练任务 |

---

## 🔄 数据维护

```bash
python scripts/init_milvus.py          # 初始化向量库 collection（可选）
python scripts/init_neo4j.py           # 初始化医学图谱本体（可选）
python scripts/run_dataset_generation.py  # 数据增强生成 SFT 数据（需 MiniMax Key，本地执行）
```

- 开发环境默认 SQLite，无需启动 MySQL；生产设置 `APP_ENV=production` 或 `DATABASE_URL`
- 训练需要 GPU、PyTorch、TRL 与授权数据；本地推理需要 OpenAI 兼容的 vLLM 服务
- 训练数据、模型权重、患者报告不随仓库发布

---

## 🤝 参与贡献

欢迎任何形式的贡献：**Star ⭐、Issue、PR**。

- 想补指标解析、证据库或前端页面：直接提 PR，`tests/` 全绿即可合入
- 想了解接口契约：本地启动后访问 http://localhost:8080/docs（OpenAPI）

**如果这个项目对你有帮助，请点个 ⭐ Star —— 你的支持是最大的动力！**

---

## ⚠️ 安全提示

- HealthFlow 仅提供信息整理与健康辅助建议，**不能替代医生进行诊断、开处方或给出具体用药剂量**；高风险场景应转人工或及时就医。
- 所有密钥通过 `.env` 注入，不要把供应商 Key、数据库密码或本地报告提交到 Git；如果曾误提交过密钥，仅删除当前文件不够，还需要**撤销密钥并清理 Git 历史**。
- 这是一个研究与工程演示项目，不构成医疗建议。
