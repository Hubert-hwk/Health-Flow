# HealthFlow Python 架构文档

> 本文档详细描述 HealthFlow Python 版本的项目架构、数据流向、和核心模块逻辑。

**版本:** v1.0
**日期:** 2026-04-23
**GitHub:** `https://github.com/your-org/healthflow-python`

---

## 一、项目概述

HealthFlow Python 是基于多模态大模型（VLM）与 Agent 架构的医疗辅助系统。核心目标是解决医疗场景下：
1. **复杂单据解析难** — 体检报告 PDF/图片自动解析
2. **医学逻辑幻觉** — RAG + 知识图谱增强 + DPO 安全对齐

### 1.1 技术选型

| 层次 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI + Uvicorn | 异步高性能 |
| Agent 框架 | LangGraph | 状态机 Agent |
| LLM 推理 | vLLM (Qwen2-VL-2B) | 本地推理 |
| 数据生成 | MiniMax M2.7 API | SFT 数据批量生成 |
| 向量检索 | Milvus 2.4 (未集成) | 语义相似度检索 |
| 知识图谱 | Neo4j 5.x (未集成) | 医学实体关系 |
| 关系数据库 | MySQL 8 / H2 | 报告、对话存储 |
| SFT 微调 | LLaDA / QLoRA | 4-bit 量化微调 |
| DPO 对齐 | TRL DPOTrainer | 安全回答偏好学习 |

### 1.2 当前完成度

| 模块 | 完成度 | 说明 |
|------|--------|------|
| 后端核心服务 | 95% | DashScope、DocumentParser、MedicalRAG 等 |
| Agent 架构 | 85% | 动态路由、多 Agent 协作、意图识别 |
| 安全机制 | 80% | SafetyDPOTrainer 框架、安全红线检测 |
| 数据层 | 100% | 实体、Repository、H2 数据库 |
| 前端界面 | 90% | React + Vite + Tailwind |
| 多模态解析 | 80% | PDF 文本提取 + VLM 图像解析 |
| 知识图谱 | 30% | 模拟数据，Neo4j 未连接 |
| 向量存储 | 30% | 内存存储，Milvus 未集成 |
| 数据增强 | 48% | 3922/8000 条 SFT 数据 |
| 模型微调 | 框架 | Pipeline 存在，未实际运行 |
| DPO 训练 | 框架 | Pipeline 存在，未实际运行 |

---

## 二、目录结构

```
healthflow-python/
├── app/                          # 主应用包
│   ├── main.py                   # FastAPI 入口，lifespan，CORS，路由注册
│   ├── config.py                # Pydantic Settings，所有配置从 .env 加载
│   │
│   ├── api/                      # API 路由层（FastAPI Router）
│   │   ├── chat.py              # 问诊、分诊、意图检测、safety check
│   │   ├── report.py           # 体检报告上传 / 解析 / 查询
│   │   ├── metric.py           # 指标趋势分析
│   │   ├── kg.py               # 知识图谱诊断查询
│   │   └── train.py            # 训练 pipeline 触发
│   │
│   ├── agent/                    # LangGraph Agent 层
│   │   ├── dynamic_router.py   # 动态路由 Agent：
│   │   │                         #   StateGraph + intent_calculate + reasoning + kg_query
│   │   ├── recursive_feedback.py # 递归反馈 Agent：
│   │   │                         #   矛盾检测 → 修正（最多 3 轮）
│   │   ├── consistency_manager.py # 会话一致性管理器：
│   │   │                         #   MedicalEntity 追踪，实体类型：metric/symptom/disease/drug
│   │   │                         #   context_summary 生成
│   │   └── graph/
│   │       └── medical_graph.py  # 医学知识图谱工具
│   │
│   ├── service/                  # 核心业务服务层
│   │   ├── medical_rag.py        # 混合 RAG：
│   │   │                         #   向量检索(Milvus) + 知识图谱(Neo4j) 权重融合 60:40
│   │   ├── vision_encoder.py     # 体检报告 VLM 解析：
│   │   │                         #   PDF/image → 结构化 ParsedReport(BBOX格式)
│   │   ├── data_augment.py       # SFT 数据增强 Pipeline：
│   │   │                         #   模板生成 / LLM扩展 / 安全过滤 / 去重 / JSONL输出
│   │   ├── llm_expander.py        # LLM 批量数据生成：
│   │   │                         #   MiniMax API 调用，4类别生成，JSON解析，retry
│   │   ├── safety_dpo.py         # DPO 安全对齐训练：
│   │   │                         #   TRL DPOTrainer，安全红线规则
│   │   └── vlm_tuner.py         # VLM SFT/QLoRA 微调：
│   │                             #   TunerConfig (rank=64,lora_alpha=16,lr=2e-4)
│   │
│   ├── model/                    # 模型层
│   │   ├── llm.py               # LLM 客户端：
│   │   │                         #   - vLLMClient (本地推理)
│   │   │                         #   - MiniMaxClient (数据生成)
│   │   │                         #   - DashScopeClient (备选)
│   │   └── embedding.py          # Embedding 模型：
│   │                             #   BAAI/bge-large-zh-v1.5 (1024维)
│   │
│   ├── data/                    # 数据访问层
│   │   ├── models.py            # SQLAlchemy ORM：
│   │   │                         #   MedicalReport, MetricRecord
│   │   │                         #   ChatSession, ChatMessage
│   │   │                         #   RoutingLog
│   │   ├── mysql_client.py       # MySQL 连接管理
│   │   ├── milvus_client.py      # Milvus 客户端 (未集成)
│   │   └── neo4j_client.py      # Neo4j 客户端 (未集成)
│   │
│   └── schema/                  # Pydantic 请求/响应模型
│       ├── chat.py
│       ├── report.py
│       └── train.py
│
├── scripts/                      # 运维 / 数据生成脚本
│   ├── run_dataset_generation.py # SFT 数据生成入口
│   │                             #   usage: python run_dataset_generation.py --api-key KEY --size 8000
│   ├── fill_examination_and_metric.py # 补全体检报告 + 指标问询数据
│   ├── fill_dataset_gaps.py      # 按类别补全数据缺口
│   ├── init_milvus.py           # Milvus Collection 初始化
│   └── init_neo4j.py            # Neo4j Schema 初始化
│
├── data/                        # 数据文件
│   ├── sft/
│   │   ├── training_data.jsonl  # SFT 训练数据
│   │   │                         #   体检报告解读 1495 条
│   │   │                         #   指标异常问询 1142 条
│   │   │                         #   科室分诊建议 284 条
│   │   └── safety_qa.jsonl     # DPO 安全问答（1000 条 safe/unsafe 对）
│   └── 医疗安全问答（DPO 专用）.md  # 豆包原始生成数据
│
├── docs/                        # 设计文档
│   └── superpowers/
│       └── specs/
│           ├── 2026-04-23-sft-dataset-generation-spec.md
│           └── 2026-04-21-healthflow-sft-dataset-design.md
│
├── tests/                       # 测试
│   ├── test_chat_api.py
│   ├── test_dynamic_router.py
│   ├── test_recursive_feedback.py
│   ├── test_medical_rag.py
│   ├── test_vision_encoder.py
│   └── ...
│
├── pyproject.toml              # 项目依赖
├── pytest.ini
├── .env.example               # 环境变量模板
└── README.md                   # 项目概览
```

---

## 三、数据流向

### 3.1 SFT 数据生成流程

```
                    ┌─────────────────────────────────────────────────┐
                    │           数据生成目标：8000 条 SFT 数据          │
                    │  ┌──────────────┐  ┌──────────────┐             │
                    │  │ 体检报告解读 │  │ 指标异常问询 │  ...       │
                    │  │   (目标3000)  │  │   (目标2500)  │             │
                    │  └───────┬──────┘  └───────┬──────┘             │
                    │          │                  │                   │
        ┌─────────────▼──┐      │      ┌─────────────▼──┐             │
        │ llm_expander.py │      │      │ llm_expander.py │             │
        │  (MiniMax API) │      │      │  (MiniMax API)  │             │
        │                 │      │      │                 │             │
        │ expand_examination()    │      │ expand_metric_query()        │
        │ expand_triage()         │      │                 │             │
        └─────────────┬───────────┘      └──────┬──────────┘             │
                      │                         │                        │
                      │   ┌─────────────────────┼────────────────────┐ │
                      │   ▼                     ▼                        │ │
                      │  JSON数组解析 ──► ExpansionResult.to_dict()     │ │
                      │       │                                          │ │
                      │       ▼                                          │ │
                      │  data_augment.py                                │ │
                      │   ├─ deduplicate() [instruction 前缀去重]         │ │
                      │   ├─ filter_by_safety() [红线过滤]                │ │
                      │   └─ save() → JSONL                             │ │
                      │       │                                          │ │
                      │       ▼                                          │ │
                      │  data/sft/training_data.jsonl                    │ │
                      └─────────────────────────────────────────────────┘ │
```

### 3.2 DPO 数据生成流程

```
豆包原始数据 ──► parse_dpo.py ──► clean_and_expand_dpo.py ──► safety_qa.jsonl
                                    │
                                    │ 验证 safe（无剂量 + 有免责）
                                    │ 验证 unsafe（含剂量 OR 替代诊断）
                                    │
                                    ▼
                          8 个模板 × 大量随机值 ──► 合成 DPO 对
                                    │
                                    ▼
                          is_safe() + is_unsafe() 双重验证
                                    │
                                    ▼
                          data/sft/safety_qa.jsonl (1000 条)
```

### 3.3 推理流程（问诊）

```
用户输入 ──► /api/health/chat ──► dynamic_router.py (LangGraph)
                                        │
                                        ├─ intent_calculate (意图识别)
                                        │    └─ 科室分类：心内/消化/呼吸/...
                                        │
                                        ├─ reasoning (医学推理)
                                        │    └─ 结合 MedicalEntity context
                                        │
                                        └─ kg_query (知识图谱)
                                             └─ Hybrid RAG (向量+KG融合)

安全检查 ──► safety_dpo.py 规则 ──► [不通过] → 拒绝回答
                                               ↓
                                          [通过] → 回复用户
```

---

## 四、核心模块详解

### 4.1 LLM 客户端 (`app/model/llm.py`)

```python
# vLLM 本地推理（生产环境）
vllm_client = get_vllm_client()  # http://localhost:8000/v1

# MiniMax 数据生成（开发/数据生成）
minimax_client = MiniMaxClient(api_key="sk-...", model="MiniMax-M2.7")
```

| 客户端 | 用途 | Base URL |
|--------|------|----------|
| vLLMClient | 运行时推理 | `http://localhost:8000/v1` |
| MiniMaxClient | SFT 数据生成 | `https://api.minimax.chat/v1` |
| DashScopeClient | 备选 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |

### 4.2 数据增强 Pipeline (`app/service/data_augment.py`)

```python
# 两种来源模式
config.source = "template"  # 本地模板直接生成
config.source = "llm"       # MiniMax API 扩展生成

pipeline = DataAugmentationPipeline(config)
pipeline.run()   # 生成 → 过滤 → 去重 → 保存
pipeline.save()  # 输出 JSONL
```

关键类：
- `DataAugmentationPipeline` — 主管道
- `InstructionPair` — 单条数据（instruction/input/output/category）
- `AugmentConfig` — 配置（target_size, source, categories, batch_size）

### 4.3 LLM 扩展器 (`app/service/llm_expander.py`)

```python
expander = LLMExpander(config)  # batch_size=3, max_retries=3

# 体检报告解读
expander.expand_examination(template, placeholders, count=10)

# 指标异常问询
expander.expand_metric_query(template, placeholders, count=10)

# 科室分诊
expander.expand_triage(template, symptoms, count=10)

# DPO 安全对
expander.expand_safety_pair(safe_outputs, count=10)
```

每个方法：
1. 构造 prompt（system + user）
2. 调用 MiniMax API with retry
3. 解析 JSON 数组响应
4. 转换为 `ExpansionResult` 列表

### 4.4 DPO 安全训练 (`app/service/safety_dpo.py`)

```python
trainer = SafetyDPOTrainer(config)
trainer.train()  # TRL DPOTrainer + 安全红线规则
```

安全红线规则：
| 红线 | 描述 |
|------|------|
| 红线1 | 不能给出具体用药剂量 |
| 红线2 | 不能替代医生诊断 |
| 红线3 | 危急值必须警示 |
| 红线4 | 必须包含免责提示 |
| 红线5 | 不能基于单一指标下结论 |

### 4.5 Agent 系统 (`app/agent/`)

**dynamic_router.py** — StateGraph：
```
input → intent_calculate → [reasoning + kg_query] → output
              ↑
         consistency_manager (实体追踪)
```

**recursive_feedback.py** — 反馈循环：
```
用户输入 → 推理 → 检测矛盾 → 修正 → 最多3轮
```

**consistency_manager.py**：
```python
MedicalEntity(type="metric", name="空腹血糖", value="6.5", unit="mmol/L", trend="↑")
MedicalEntity(type="symptom", name="多饮多尿", ...)
```

### 4.6 混合 RAG (`app/service/medical_rag.py`)

```python
# 权重融合：60% 向量 + 40% 知识图谱
results = rag.hybrid_search(query, top_k=10)
```

---

## 五、数据集详情

### 5.1 目标规格

| 类别 | 目标 | 现状 | 缺口 |
|------|------|------|------|
| 体检报告解读 | 3000 | 1495 | 1505 |
| 指标异常问询 | 2500 | 1142 | 1358 |
| 科室分诊建议 | 1500 | 284 | 1216 |
| 医疗安全问答 (DPO) | 1000 | 1000 | 0 ✅ |
| **合计** | **8000** | **3922** | **4078** |

### 5.2 数据质量

**体检报告解读 / 指标异常问询：**
- ✅ 字段完整（0缺失）
- ✅ instruction 平均 20-35 字符
- ✅ output 平均 100-320 字符
- ✅ 免责提示覆盖率 ~100%
- ✅ 0 条含具体剂量

**科室分诊建议：**
- ⚠️ 数量严重不足（仅 19%）
- ⚠️ 免责提示覆盖率仅 0.3%
- ⚠️ 科室分布不均（神经内科过度集中）

**医疗安全问答（DPO）：**
- ✅ safe 含剂量：0 条
- ✅ unsafe 无剂量：175 条（通过替代诊断触发）
- ✅ 替代诊断 safe 中 0 次，unsafe 中 638 次
- ✅ 无重复 instruction
- ✅ output 长度 50-115 字符

---

## 六、配置管理

所有配置通过 `app/config.py` 的 `Pydantic Settings` 管理，从 `.env` 文件加载：

```bash
# 必填
MINIMAX_API_KEY=sk-...      # 数据生成
MINIMAX_MODEL=MiniMax-M2.7

# vLLM 推理
VLLM_HOST=localhost
VLLM_PORT=8000
VLLM_MODEL=qwen-vl-plus

# 数据库
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=password
MYSQL_DATABASE=healthflow

# 可选（未集成）
MILVUS_HOST=localhost
MILVUS_PORT=19530
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=password
```

---

## 七、待完成功能（优先级排序）

1. **Milvus 向量库集成** — 当前用内存替代，需要连接真实 Milvus
2. **Neo4j 知识图谱集成** — 当前用模拟数据，需要连接真实 Neo4j
3. **科室分诊数据补全** — 当前仅 284/1500 条，且缺少免责提示
4. **数据增强 Pipeline 完善** — 补全剩余 4078 条数据
5. **实际 SFT 微调运行** — 当前仅有框架，未实际训练
6. **实际 DPO 训练运行** — 当前仅有框架，未实际训练

---

## 八、GitHub 上传检查清单

上传前请确认：

- [ ] `README.md` 完整（项目概览、启动命令、技术栈）
- [ ] `ARCHITECTURE.md` 完整（本文档）
- [ ] `.env.example` 包含所有需要的环境变量
- [ ] `pyproject.toml` 依赖完整
- [ ] `data/sft/` 已加入 `.gitignore`（数据集文件大，不建议提交）
- [ ] API Key、数据库密码等 secrets 未提交（使用 `.env` 而非硬编码）
- [ ] 测试覆盖核心模块
- [ ] `scripts/` 目录包含数据生成和初始化脚本
- [ ] 设计文档在 `docs/` 目录
