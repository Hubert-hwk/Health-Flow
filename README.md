# HealthFlow Python 版本

基于多模态大模型（VLM）与 Agent 架构的医疗辅助系统，针对医疗场景下复杂单据解析难、医学逻辑容易产生幻觉的核心痛点，提供体检报告解析、生化指标趋势分析、多专科智能分诊等服务。

**GitHub 仓库：** `https://github.com/your-org/healthflow-python`

---

## 一、项目概览

### 1.1 核心能力

| 模块 | 功能 | 技术 |
|------|------|------|
| 体检报告解析 | PDF/图片报告自动解析，提取指标 | VLM + 规则解析 |
| 指标趋势分析 | 时序指标可视化，异常预警 | Matplotlib |
| 智能分诊 | 症状 → 科室 routing | LangGraph Agent |
| 混合 RAG | 医学知识检索增强 | Milvus + Neo4j |
| SFT 微调 | 领域适应训练 | QLoRA |
| DPO 安全对齐 | 医疗安全红线学习 | TRL DPO |

### 1.2 技术栈

- **Web 框架**: FastAPI + Uvicorn
- **Agent 框架**: LangGraph
- **LLM 推理**: vLLM (Qwen/Qwen2-VL) / MiniMax API (数据生成)
- **多模态**: Qwen2-VL-2B-Instruct
- **向量数据库**: Milvus (未集成，使用内存存储)
- **知识图谱**: Neo4j (未集成，使用模拟数据)
- **关系数据库**: MySQL / H2
- **训练框架**: TRL (DPO), LLaDA/QLoRA (SFT)

### 1.3 项目结构

```
healthflow-python/
├── app/
│   ├── main.py                    # FastAPI 入口，lifespan 管理
│   ├── config.py                  # Pydantic Settings 配置管理
│   ├── api/                      # API 路由层
│   │   ├── chat.py               # 问诊 / 分诊 / 意图检测
│   │   ├── report.py             # 体检报告上传 / 解析
│   │   ├── metric.py            # 指标趋势
│   │   ├── kg.py               # 知识图谱查询
│   │   └── train.py            # 训练pipeline触发
│   ├── agent/                    # LangGraph Agent 层
│   │   ├── dynamic_router.py    # 动态路由 Agent (意图检测 → 科室)
│   │   ├── recursive_feedback.py # 递归反馈 Agent (矛盾检测 / 修正)
│   │   ├── consistency_manager.py # 会话一致性管理 (实体追踪)
│   │   └── graph/
│   │       └── medical_graph.py # 知识图谱工具
│   ├── service/                  # 核心业务服务
│   │   ├── medical_rag.py       # 混合 RAG (向量 + KG)
│   │   ├── vision_encoder.py     # 体检报告 VLM 解析
│   │   ├── data_augment.py       # SFT 数据增强 Pipeline
│   │   ├── llm_expander.py       # LLM 批量数据生成
│   │   ├── safety_dpo.py         # DPO 安全对齐训练
│   │   └── vlm_tuner.py         # VLM SFT / QLoRA 微调
│   ├── model/                    # 模型层
│   │   ├── llm.py               # vLLM / MiniMax / DashScope 客户端
│   │   └── embedding.py          # Embedding 模型
│   ├── data/                    # 数据访问层
│   │   ├── models.py            # SQLAlchemy ORM 模型
│   │   ├── mysql_client.py       # MySQL 连接管理
│   │   ├── milvus_client.py      # Milvus 向量库客户端
│   │   └── neo4j_client.py      # Neo4j 图数据库客户端
│   └── schema/                  # Pydantic 请求 / 响应模型
│       ├── chat.py
│       ├── report.py
│       └── train.py
├── scripts/                      # 运维 / 数据生成脚本
│   ├── run_dataset_generation.py # SFT 数据生成入口
│   ├── fill_dataset_gaps.py      # 数据缺口补全
│   ├── fill_examination_and_metric.py # 体检报告 / 指标问询补全
│   ├── init_milvus.py            # Milvus Collection 初始化
│   └── init_neo4j.py            # Neo4j Schema 初始化
├── data/                        # 数据文件
│   └── sft/
│       ├── training_data.jsonl   # SFT 训练数据 (体检报告 / 指标问询 / 分诊)
│       └── safety_qa.jsonl       # DPO 安全问答 (safe/unsafe 对)
├── tests/                       # 测试
├── docs/                        # 设计文档
│   └── superpowers/
│       └── specs/
│           ├── 2026-04-23-sft-dataset-generation-spec.md  # 数据生成规范
│           └── 2026-04-21-healthflow-sft-dataset-design.md # 数据设计
├── pyproject.toml
├── pytest.ini
├── .env.example
└── README.md
```

---

## 二、启动命令

### 2.1 后端服务

```bash
cd healthflow-python
uvicorn app.main:app --reload --port 8080
```

### 2.2 API 文档

- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

### 2.3 依赖安装

```bash
pip install -e .
```

---

## 三、数据集生成

### 3.1 数据集规格

| 类别 | 目标数量 | 说明 |
|------|---------|------|
| 体检报告解读 | 3000 | 指标解读、异常分析、复查建议 |
| 指标异常问询 | 2500 | 饮食建议、是否需要检查 |
| 科室分诊建议 | 1500 | 症状 → 正确科室 |
| 医疗安全问答 (DPO) | 1000 | safe/unsafe 回答对 |
| **合计** | **8000** | |

### 3.2 生成入口

```bash
# 完整生成（需 MiniMax API Key）
python scripts/run_dataset_generation.py \
    --api-key YOUR_KEY \
    --size 8000 \
    --output data/sft/training_data.jsonl

# 补全体检报告和指标问询
python scripts/fill_examination_and_metric.py

# 补全特定类别
python scripts/fill_dataset_gaps.py --api-key YOUR_KEY --target examination_report
```

### 3.3 数据格式

#### 标准 SFT 数据
```json
{
  "instruction": "用户指令（问题）",
  "input": "输入上下文（指标数值等）",
  "output": "期望的完整回答",
  "category": "体检报告解读"
}
```

#### DPO 安全数据
```json
{
  "instruction": "空腹血糖 6.5mmol/L 该怎么用药",
  "input": "空腹血糖 6.5mmol/L",
  "output": "您的空腹血糖为6.5mmol/L...仅供参考，请咨询专业医生。",
  "output_unsafe": "就是糖尿病，每天吃二甲双胍 500mg，早晚各一次...",
  "category": "医疗安全问答",
  "is_dpo_pair": true,
  "source": "doubao"
}
```

### 3.4 数据质量要求

- **体检报告解读**: 含指标含义 + 原因 + 建议 + 免责提示
- **指标异常问询**: 饮食建议具体可操作，不提药物剂量
- **科室分诊建议**: 科室名称准确（如"心内科"而非"内科"）
- **医疗安全问答**: safe 完全不含剂量，unsafe 明确违反红线

---

## 四、训练 Pipeline

### 4.1 SFT 微调

```python
from app.service.vlm_tuner import VLMTuner, TunerConfig

config = TunerConfig(
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    data_path="data/sft/training_data.jsonl",
    output_dir="output/sft/checkpoints",
    rank=64,
    lora_alpha=16,
    lr=2e-4,
    epochs=3,
    batch_size=1,
    gradient_accumulation=16
)
tuner = VLMTuner(config)
tuner.train()
```

### 4.2 DPO 安全对齐

```python
from app.service.safety_dpo import SafetyDPOTrainer, DPOConfig

config = DPOConfig(
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    data_path="data/sft/safety_qa.jsonl",
    output_dir="output/dpo/checkpoints",
    beta=0.1,
    epochs=3,
    batch_size=1,
    lr=1e-5
)
trainer = SafetyDPOTrainer(config)
trainer.train()
```

### 4.3 触发训练 API

```bash
# 触发 SFT
curl -X POST http://localhost:8080/api/health/train/finetune \
  -H "Content-Type: application/json" \
  -d '{"category": "体检报告解读", "epochs": 3}'

# 触发 DPO
curl -X POST http://localhost:8080/api/health/train/dpo \
  -H "Content-Type: application/json" \
  -d '{"epochs": 3}'
```

---

## 五、API 参考

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/health/chat` | POST | 智能问诊 |
| `/api/health/chat/stream` | POST | SSE 流式聊天 |
| `/api/health/routing` | POST | 意图检测 + 科室路由 |
| `/api/health/safety/check` | GET | 内容安全检查 |
| `/api/health/report/upload` | POST | 上传体检报告 (PDF/图片) |
| `/api/health/report/{id}` | GET | 获取报告详情 |
| `/api/health/metric/trend` | GET | 指标趋势分析 |
| `/api/health/kg/diagnosis` | POST | 症状诊断推理 |
| `/api/health/train/augment` | POST | 触发数据增强 |
| `/api/health/train/finetune` | POST | 触发 SFT 微调 |
| `/api/health/train/dpo` | POST | 触发 DPO 训练 |

---

## 六、配置说明

环境变量配置 (.env)：

```bash
# MySQL
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=password
MYSQL_DATABASE=healthflow

# Milvus (未集成)
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Neo4j (未集成)
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=password

# vLLM (推理服务)
VLLM_HOST=localhost
VLLM_PORT=8000
VLLM_MODEL=qwen-vl-plus

# MiniMax (数据生成)
MINIMAX_API_KEY=your_api_key
MINIMAX_MODEL=MiniMax-M2.7

# Embedding
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5

# API
API_HOST=0.0.0.0
API_PORT=8080
```

---

## 七、医疗安全红线

1. **不能给出具体用药剂量建议** — 如"每天 500mg"
2. **不能替代医生进行诊断** — 如"你就是糖尿病"
3. **不能基于单一指标下结论** — 如"这个指标说明你肯定是癌症"
4. **危急值必须警示** — 如血压 200/120 需提示立即就医
5. **必须包含免责提示** — "仅供参考，请咨询专业医生"

---

## 八、当前进度

| 模块 | 状态 | 说明 |
|------|------|------|
| 后端 API | 95% | 核心接口已完成 |
| Agent 架构 | 85% | 动态路由、多 Agent 协作 |
| 数据集生成 | 48% | 3922/8000 条 |
| SFT 微调 | 框架 | Pipeline 存在，未实际运行 |
| DPO 训练 | 框架 | Pipeline 存在，未实际运行 |
| Milvus 集成 | 30% | 使用内存存储替代 |
| Neo4j 集成 | 30% | 使用模拟数据替代 |

---

## 九、待完成功能

1. **Milvus 向量库集成** — 当前使用内存存储
2. **Neo4j 知识图谱集成** — 当前使用模拟数据
3. **数据增强 Pipeline** — 逆向指令工程构造 8000 条完整数据集
4. **VLM 微调** — SFT + 投影层优化
5. **DPO 安全对齐** — 医疗安全红线对齐实际训练
