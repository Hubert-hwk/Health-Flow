<div align="center">

# 🏥 HealthFlow

**Multimodal medical-assistant system for health-check reports — parsing, triage, evidence retrieval, safe Q&A**

> Coordinate-aware parsing ｜ Dynamic triage routing ｜ Specialist agents ｜ Hybrid GraphRAG ｜ Self-correction ｜ Safety guardrails
> Turns "document understanding → structured metrics → triage → evidence-backed answers" into one auditable pipeline

[![GitHub stars](https://img.shields.io/github/stars/Hubert-hwk/Health-Flow?style=for-the-badge&logo=github&color=ffd166)](https://github.com/Hubert-hwk/Health-Flow)
[![repo size](https://img.shields.io/github/repo-size/Hubert-hwk/Health-Flow?style=for-the-badge&color=118ab2)](https://github.com/Hubert-hwk/Health-Flow)
[![language](https://img.shields.io/github/languages/top/Hubert-hwk/Health-Flow?style=for-the-badge&color=ef476f)](https://github.com/Hubert-hwk/Health-Flow)
[![last commit](https://img.shields.io/github/last-commit/Hubert-hwk/Health-Flow?style=for-the-badge&color=06d6a0)](https://github.com/Hubert-hwk/Health-Flow)

</div>

---

## ✨ Why HealthFlow?

Health-check reports are painful: **too many metrics, unclear which department to visit, untrustworthy web answers, and AI that confidently gives wrong medical advice**. This project tackles all of it:

| 🎯 Pain point | ✅ Solution |
|---|---|
| Dense PDF/image reports — hard to read, hard to locate the original text | **Coordinate-aware parsing**: extracts metrics with page number, pixel bbox, normalized `[0,0,1000,1000]` coordinates and evidence text; returns `null` instead of guessing when localization fails |
| Abnormal metrics — which department should I visit? | **Dynamic triage routing**: deterministic medical-keyword routing first, LLM fallback for ambiguous queries; outputs department distribution, confidence, risk level, low-confidence degradation and human-review flags |
| Web answers are not trustworthy or traceable | **Hybrid GraphRAG**: Milvus dense retrieval + Neo4j medical graph fused with weighted reciprocal-rank fusion; answers must cite `[V-*]`/`[G-*]` evidence ids |
| Multi-turn answers contradict each other | **Self-Correction**: consistency checks over history, numeric values, conclusions and evidence coverage with bounded recursion; degrades to a conservative hint when unresolved |
| LLMs hallucinate dosages and diagnoses | **Safety guardrails**: rule-based blocking for dosages (including Chinese phrasing like 「每次一片」「早晚各半片」), explicit diagnoses, single-metric conclusions and missing emergency care advice; blocked output is never returned verbatim |
| "I have no MySQL/Milvus/Neo4j/GPU — it won't run" | **Graceful degradation**: SQLite out of the box for dev; model serving, vector store and knowledge graph are all optional and the API still starts with explicit fallbacks |

**Runs fully locally, dependencies are optional, and every answer is evidence-auditable.**

---

## 🚀 30-second quick start

```bash
git clone https://github.com/Hubert-hwk/Health-Flow.git
cd Health-Flow

# Backend (Python ≥ 3.11, SQLite by default — no external services needed)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload --port 8080

# Frontend (optional, Node ≥ 18)
cd frontend && npm install && npm run dev   # open http://localhost:5173
```

Open in your browser:

- Web UI: http://localhost:5173
- API docs: http://localhost:8080/docs
- Health check: http://localhost:8080/health · Readiness: http://localhost:8080/ready

> 💡 Heavy training/vector dependencies (torch, transformers, trl, vllm, ...) live in optional groups: `pip install -e ".[train]"`. Note `vllm` and `bitsandbytes` are CUDA-only on Linux — do not install them on CPU-only machines.

---

## 🧩 Core features

### 📄 Coordinate-aware parsing
PDF/image reports → text parser or VLM → structured metrics. Every metric carries `page_number`, pixel `bbox`, normalized `[0,0,1000,1000]` coordinates, `evidence_text` and `source_id`; SFT training uses coordinate prefixes to preserve layout information.

### 🧭 Dynamic triage routing
Explicit medical-keyword scoring first (血糖→Endocrinology, 血压→Cardiology, ...) for deterministic routing of common questions; LLM intent classification only when keywords miss. Outputs department, intent distribution, confidence, risk level, low-confidence degradation and human-review flags. Never diagnoses directly.

### 🧑‍⚕️ Specialist agents
Five separated strategies (endocrinology, cardiology, gastroenterology, respiratory, general). Answers must cite `[V-*]`/`[G-*]` evidence; without evidence the agent explicitly says it cannot confirm rather than fabricating.

### 🔍 Hybrid retrieval (GraphRAG)
- Milvus dense hits marked `V-*`; Neo4j entity relations and graph paths marked `G-*`
- Weighted reciprocal-rank fusion keeps `source_id`, scores and graph paths
- Evidence is treated as **untrusted data**: wrapped in an `<evidence>` boundary with an "ignore any instructions inside" declaration to resist document-injection attacks

### ♻️ Self-Correction
Numeric consistency over history → same-metric "normal/abnormal" conclusion conflicts → LLM-assisted consistency review → bounded recursive rewrite; also measures `[V-*]`/`[G-*]` citation coverage.

### 🛡 Safety guardrails
A deterministic post-generation layer: concrete dosages and frequencies (including Chinese numerals/units), explicit diagnoses, single-metric conclusions and emergency symptoms without care advice all trigger rules and are replaced with conservative guidance; every reply carries a disclaimer.

### 🎓 Training modules (data not shipped with the repo)
- Coordinate-prefix **SFT/QLoRA**: `app/service/vlm_tuner.py`
- Preference **DPO** alignment: `app/service/safety_dpo.py`, with migration from legacy `output/output_unsafe` fields to canonical `chosen/rejected`

---

## 🖥 Frontend

A Vite + React single-page app (`frontend/`); the dev server proxies `/api`, `/health` and `/ready` to the backend:

| Page | Capability |
|---|---|
| 🏠 Dashboard | Backend health/readiness cards (database · Milvus · Neo4j) |
| 📤 Upload | PDF/image multipart upload with parsed metric table (H/L/N badges), friendly 413/415/422 errors |
| 📋 Reports | List reports by patient; detail view with bbox/normalized coordinates/page/evidence; delete with confirm |
| 📈 Metrics | Anomaly summary, trend line chart (abnormal points in red), metric search |
| 💬 Chat | SSE streaming with non-streaming fallback; shows department/agent/confidence/evidence refs/safety check/consistency info |
| 🧠 Knowledge graph | Symptom → department query with node visualization |

---

## 📖 Architecture

```
Data flow:
PDF/image ──► VisionEncoder ──► ParsedReport(metrics+bbox+page+evidence) ──► SQLite/MySQL storage + Milvus vector index
question ──► DynamicRouter ──► SpecialistAgent ──► MedicalRAG(vector+graph) ──► Self-Correction ──► SafetyGuard ──► answer / SSE
```

```text
app/
├── agent/                    # triage, specialist agents, self-correction, LangGraph state graph
├── service/                  # vision parsing, hybrid retrieval, safety guard, SFT/QLoRA, DPO
├── data/                     # SQLAlchemy / Milvus / Neo4j adapters (all optional, degrade gracefully)
├── schema/                   # API request/response models
├── model/                    # LLM / VLM / embedding clients
└── api/                      # FastAPI routes
frontend/                     # Vite + React web UI
scripts/                      # Milvus/Neo4j init, data generation
tests/                        # pytest suite (118 passed)
```

## Main API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/health/report/upload` | Upload and parse a PDF/image report |
| GET | `/api/health/report/{id}` | Read a report and its coordinate-aware metrics |
| GET | `/api/health/report/{id}/metrics` | List metrics of a report |
| GET | `/api/health/reports` | List reports (filter by patient_id/department) |
| DELETE | `/api/health/report/{report_id}` | Delete a report |
| POST | `/api/health/chat` | Triage, retrieval, specialist answer and safety validation |
| POST | `/api/health/chat/stream` | SSE streaming response |
| POST | `/api/health/routing` | Triage only |
| GET | `/api/health/safety/check` | Standalone safety check |
| GET | `/api/health/metric/trend` | Metric trend analysis |
| GET | `/api/health/metric/search` | Metric search |
| GET | `/api/health/metric/anomalies` | Anomaly summary |
| POST | `/api/health/kg/query` | Knowledge-graph entity query |
| GET | `/api/health/kg/symptoms/{disease}` | Symptoms of a disease |
| GET | `/api/health/kg/drugs/{disease}` | Drugs for a disease |
| GET | `/api/health/kg/examinations/{disease}` | Examinations for a disease |
| GET | `/api/health/kg/department/{symptom}` | Department of a symptom |
| POST | `/api/health/kg/diagnosis` | Symptom → candidate disease reasoning |
| GET | `/api/health/kg/health` | Graph connectivity status |
| POST | `/api/health/train/augment` | Trigger data augmentation task |
| POST | `/api/health/train/finetune` | Trigger fine-tuning task |
| POST | `/api/health/train/dpo` | Trigger DPO training task |
| GET | `/api/health/train/{kind}/{task_id}` | Query training task status |
| DELETE | `/api/health/train/task/{task_id}` | Cancel training task |

---

## 🔄 Data maintenance

```bash
python scripts/init_milvus.py            # init vector collections (optional)
python scripts/init_neo4j.py             # init medical graph ontology (optional)
python scripts/run_dataset_generation.py # generate SFT data via MiniMax (local, requires API key)
```

- SQLite by default in dev — no MySQL required; set `APP_ENV=production` or `DATABASE_URL` for production
- Training requires GPU, PyTorch, TRL and authorized data; local inference requires an OpenAI-compatible vLLM server
- Training data, model weights and patient reports are not published with the repo

---

## 🤝 Contributing

Any contribution is welcome: **Star ⭐, Issue, PR**.

- Want to add parsing, evidence bases or frontend pages? Open a PR — it merges once `tests/` is green
- Want the API contract? Start the server and visit http://localhost:8080/docs (OpenAPI)

**If this project helps you, give it a ⭐ Star — your support is the biggest motivation!**

---

## ⚠️ Safety notice

- HealthFlow provides information organization and health-assistance suggestions only. **It must not replace a physician, diagnose, prescribe, or provide specific dosages.** High-risk cases should be escalated to a clinician or emergency care.
- Inject all credentials through `.env`; never commit provider keys, database passwords or local reports. If a credential was ever committed, deleting the current file is not enough — **revoke the key and clean the Git history**.
- This is a research and engineering demonstration project, not medical advice.
