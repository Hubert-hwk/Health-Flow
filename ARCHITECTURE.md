# HealthFlow Python 架构

HealthFlow Python 是当前项目主线。它是医疗辅助系统，不承担独立诊断、处方或具体剂量建议。

## 数据流

```text
PDF/图片
  -> VisionEncoder
  -> ParsedReport(metrics + bbox + page + evidence)
  -> SQLAlchemy 报告存储 / Milvus 报告索引

用户问题
  -> DynamicRouter
  -> SpecialistAgent
  -> MedicalRAG(vector + Neo4j)
  -> evidence-aware Self-Correction
  -> SafetyGuard
  -> ChatResponse / SSE
```

## 1. 坐标感知解析

`app/service/vision_encoder.py` 根据文件类型选择文本 PDF、扫描 PDF 或图片路径。VLM 输出的指标记录包含：

- `page_number`：从 1 开始的页码；
- `bbox`：页面像素坐标 `[x1, y1, x2, y2]`；
- `bbox_normalized`：归一化到 `[0, 0, 1000, 1000]` 的坐标；
- `evidence_text`：与指标对应的原文片段；
- `source_id`：用于回答引用和评测回溯。

坐标前缀由 `app/service/vlm_tuner.py` 的 `coordinate_prefix` 生成，形式为：

```text
[PAGE=1][BBOX=120.00,340.00,280.00,360.00] 空腹血糖
```

无法定位时返回 `null`，不猜测坐标。真正的 OCR 检测器和训练集仍由部署方按授权数据源接入。

## 2. 分诊主控与专科 Agent

`DynamicRouter` 的路由顺序是：

1. 显式医疗关键词计分，保证常见体检问题可确定性路由；
2. 关键词无命中时调用 LLM 做结构化意图分布；
3. 对概率和主次差值做归一化；
4. 置信度过低、出现高风险词或跨科室不明确时设置 `human_review_required`；
5. 将专科策略和风险等级传给 `SpecialistAgent`。

路由不直接做疾病诊断。其输出包含科室、意图分布、置信度、风险等级、解释和图谱提示。

## 3. GraphRAG 与证据

`MedicalRAGService` 将：

- Milvus 的稠密检索结果标记为 `V-*`；
- Neo4j 的实体关系和图路径标记为 `G-*`；
- 使用加权 reciprocal-rank fusion 合并结果；
- 在 Prompt 中保留来源编号和图谱路径。

Neo4j 初始化脚本和运行时查询采用同一关系方向：

```text
(Disease)-[:HAS_SYMPTOM]->(Symptom)
(Disease)-[:TREATED_BY]->(Drug)
(Symptom)-[:BELONGS_TO]->(Department)
(Disease)-[:BELONGS_TO]->(Department)
```

Milvus 运行时 collection 为 `medical_reports`，动态字段中至少包含 `report_id`、`content` 和 `department`。数据库服务不可用时会返回空证据并让上层继续运行，不会伪造检索结果。

## 4. Self-Correction

`recursive_feedback.py` 采用有界递归：

1. 先比较历史 Assistant 消息和当前回答中的指标数值；
2. 检查同一指标附近的“正常/异常”等结论冲突；
3. 再让 LLM 做辅助一致性审查；
4. 发现冲突时重写，超过最大轮数则保守提示并停止；
5. 统计回答对 `[V-*]`/`[G-*]` 证据的引用覆盖率。

语义相似度指标不能替代医学事实校验。BERTScore、数值规则、否定关系校验和人工抽检应在独立评测脚本中完成。

## 5. 安全边界

`app/service/safety_guard.py` 位于模型输出之后，检查：

- 具体剂量和服用频次；
- 明确诊断或确定性疾病判断；
- 仅凭单一指标下结论；
- 危急症状是否给出及时就医提示；
- 是否包含医疗辅助免责声明。

红旗输出不会原样返回，而是转换为保守的人工/就医提示。模型内 DPO 和模型外安全规则是互补层，不能互相替代。

## 6. 训练模块

- `vlm_tuner.py`：支持 QLoRA、坐标前缀和 projector 参数解冻；
- `safety_dpo.py`：将 `output/output_unsafe` 兼容迁移为 `chosen/rejected`，在训练前校验空字段和缺失文件，并支持通用域混采配置；
- 训练数据、模型权重和真实医疗数据不随仓库发布。

训练任务应该通过独立 CLI 或受控异步队列执行，不能在 Web 进程中直接长时间阻塞。

## 7. 部署

开发环境默认：

```text
FastAPI + SQLite + 可选 vLLM
```

生产环境建议：

```text
FastAPI + MySQL/PostgreSQL + Milvus + Neo4j + vLLM + Redis/任务队列
```

所有外部服务通过环境变量配置。`.env.example` 只包含占位值，禁止提交真实 Key、数据库密码和患者文件。
