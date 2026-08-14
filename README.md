# HealthFlow

<p align="center">
  <img src="docs/assets/healthflow-hero.svg" alt="HealthFlow 医疗辅助系统" width="920" />
</p>

<p align="center">
  <strong>简体中文</strong> · <a href="README.en.md">English</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a>
</p>

HealthFlow 是一个面向体检报告和医疗单据的多模态医疗辅助系统原型，目标是把“单据理解、结构化指标、分诊路由、证据检索和安全校验”串成可审计的工程闭环。

> 重要边界：HealthFlow 仅提供信息整理和健康辅助建议，不能替代医生进行诊断、开处方或给出具体用药剂量。高风险场景应转人工或及时就医。

## 项目主线

<p align="center">
  <img src="docs/assets/healthflow-pipeline.svg" alt="HealthFlow 动态处理流程" width="920" />
</p>

```mermaid
flowchart LR
  A[PDF / 图片报告] --> B[文本解析或 VLM]
  B --> C[指标 + 页码 + BBOX]
  C --> D[分诊主控]
  D --> E[专科 Agent]
  E --> F[Milvus 稠密检索]
  E --> G[Neo4j GraphRAG]
  F --> H[带来源证据的回答]
  G --> H
  H --> I[Self-Correction]
  I --> J[安全阻断与免责声明]
```

## 简历/面试手册中的实验指标

以下数字是项目简历和面试手册中的实验口径，用于说明设计目标和历史实验结果；当前公开仓库不包含原始医疗数据集，因此不能声称新环境可以自动复现这些数字。

| 模块 | 实验口径 |
|---|---|
| 坐标感知多模态解析 | 自建 500 份测试集，非结构化数据提取精度 74% → 83%，较 Qwen2.5-VL 通用基线提升 9 个百分点 |
| DPO 安全对齐 | 2,400 对偏好数据，高风险场景幻觉率 31% → 8% |
| 动态分诊路由 | 分发准确率 92% |
| GraphRAG | 召回率较纯向量检索提升 18% |
| Self-Correction | 证据语义一致性 BERTScore > 0.82，多轮逻辑冲突率 24% → 6% |

这些指标应理解为离线实验结果，而不是医疗诊断准确率或线上服务 SLA。正式发布实验复现包时，还应补充数据来源、切分方式、Recall@K、置信区间和人工审核协议。

## 已实现的关键逻辑

- 坐标感知解析：VLM 输出页面像素坐标、`[0, 0, 1000, 1000]` 归一化坐标、页码、证据文本和来源 ID；SFT 文本通过位置前缀保留空间信息。
- 动态路由：显式医疗关键词优先，歧义问题再调用 LLM；输出科室分布、置信度、风险等级、低置信降级和人工复核标记。
- 专科 Agent：内分泌、心内、消化、呼吸和全科策略分离，回答要求绑定 `[V-*]`/`[G-*]` 证据编号。
- 混合检索：Milvus 向量结果与 Neo4j 图谱结果进行加权 RRF 融合，保留 `source_id`、得分和图路径。
- Self-Correction：对话历史、数值一致性、结论冲突和证据引用进行有上限递归校验。
- 安全护栏：剂量、明确诊断、单一指标判断和危急症状未就医提示会触发规则；阻断输出不会原样返回。
- DPO/SFT 训练入口：支持旧的 `output/output_unsafe` 数据字段迁移为 `chosen/rejected`，支持 QLoRA 与坐标前缀；训练数据不随仓库发布。

## 快速启动

开发环境默认使用 SQLite，不需要先启动 MySQL。模型服务、Milvus 和 Neo4j 是可选依赖；没有这些服务时，接口仍可启动，但对应能力会返回空证据或降级结果。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
uvicorn app.main:app --reload --port 8080
```

> 说明：`[project]` 只声明了运行时依赖。训练/向量相关的重型依赖（torch、transformers、trl、vllm 等）在可选组里，按需安装：`pip install -e ".[train]"`（注意 `vllm`、`bitsandbytes` 在 Linux 仅 CUDA 版，纯 CPU 环境请勿安装）。

访问：

- Web 前端：http://localhost:5173（启动方式见下）
- API 文档：http://localhost:8080/docs
- 健康检查：http://localhost:8080/health
- 就绪检查：http://localhost:8080/ready

### 启动前端（可选）

前端位于 `frontend/`，是 Vite + React 单页应用，开发服务器默认把 `/api` 代理到 `http://localhost:8080`：

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
npm run build      # 产物输出到 frontend/dist
```

前端页面：首页/概览（后端状态）、报告上传（PDF/图片解析出指标）、报告列表与详情（坐标/BBox/证据）、指标分析（异常汇总、趋势图、搜索）、智能问答（SSE 流式，失败自动回退非流式）、知识图谱（症状→科室）。

如需生产数据库，设置 `APP_ENV=production` 或显式设置 `DATABASE_URL`。如需向量和图谱能力，分别配置 Milvus 和 Neo4j，并执行：

```bash
python scripts/init_milvus.py
python scripts/init_neo4j.py
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

## 目录结构

```text
app/
├── agent/
│   ├── dynamic_router.py          # 分诊主控
│   ├── specialist_agents.py       # 专科 Agent
│   ├── recursive_feedback.py      # Self-Correction
│   └── graph/medical_graph.py      # 端到端状态图
├── service/
│   ├── vision_encoder.py           # PDF/图片与 BBOX 解析
│   ├── medical_rag.py              # 混合检索与证据上下文
│   ├── safety_guard.py             # 模型外安全护栏
│   ├── vlm_tuner.py                # 坐标前缀 SFT/QLoRA
│   └── safety_dpo.py               # 偏好字段校验与 DPO
├── data/                           # SQLAlchemy、Milvus、Neo4j
└── api/                            # FastAPI 路由
frontend/                           # Vite + React Web 前端
```

## 数据与安全

- 仓库不发布患者报告、真实医疗信息或未确认许可证的数据集。
- 所有密钥必须通过 `.env` 注入；不要把供应商 Key、数据库密码或本地报告提交到 Git。
- 如果曾经误提交过密钥，仅删除当前文件是不够的，还需要撤销密钥并清理 Git 历史。
  （注意：本仓库早期提交曾包含一个 MiniMax API Key，已从当前文件移除，但历史中仍可提取——请立即到供应商控制台撤销该 Key，如需彻底清理历史请使用 `git filter-repo` 并强制推送。）
- 这是研究与工程演示项目，不构成医疗建议。

## 当前限制

训练需要 GPU、PyTorch、Transformers、TRL 和经过授权的数据；本地推理需要兼容 OpenAI API 的 vLLM 服务。模型训练指标、BERTScore 和 500 份测试集的完整复现实验暂未随仓库开放。
