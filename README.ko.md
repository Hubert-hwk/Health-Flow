<div align="center">

# 🏥 HealthFlow

**건강검진 보고서를 위한 멀티모달 의료 보조 시스템 — 파싱·분류·근거 검색·안전한 Q&A**

> 좌표 인식 파싱 ｜ 동적 진료과 라우팅 ｜ 전문 Agent ｜ 하이브리드 GraphRAG ｜ Self-Correction ｜ 안전 가드
> 「문서 이해 → 지표 구조화 → 진료과 분류 → 근거 기반 답변」을 감사 가능한 파이프라인으로

[![GitHub stars](https://img.shields.io/github/stars/Hubert-hwk/Health-Flow?style=for-the-badge&logo=github&color=ffd166)](https://github.com/Hubert-hwk/Health-Flow)
[![repo size](https://img.shields.io/github/repo-size/Hubert-hwk/Health-Flow?style=for-the-badge&color=118ab2)](https://github.com/Hubert-hwk/Health-Flow)
[![language](https://img.shields.io/github/languages/top/Hubert-hwk/Health-Flow?style=for-the-badge&color=ef476f)](https://github.com/Hubert-hwk/Health-Flow)
[![last commit](https://img.shields.io/github/last-commit/Hubert-hwk/Health-Flow?style=for-the-badge&color=06d6a0)](https://github.com/Hubert-hwk/Health-Flow)

</div>

---

## ✨ 왜 HealthFlow인가?

건강검진 보고서의 고민: **지표가 너무 많아 읽기 어렵고, 이상이 있어도 어느 진료과를 가야 할지 모르며, 인터넷 답변은 믿을 수 없고, AI는 자신 있게 틀린 의료 조언을 합니다**. 이 프로젝트가 모두 해결합니다:

| 🎯 고민 | ✅ 해결책 |
|---|---|
| PDF/이미지 보고서가 빽빽해서 읽기 어렵고 원문 위치를 못 찾음 | **좌표 인식 파싱**: 지표를 추출하고 페이지 번호·픽셀 bbox·`[0,0,1000,1000]` 정규화 좌표·근거 텍스트를 보존. 위치를 특정할 수 없으면 추측하지 않고 `null` 반환 |
| 이상이 있는데 어느 과로? | **동적 진료과 라우팅**: 의료 키워드 기반 결정적 라우팅을 우선하고, 모호한 질문만 LLM 판정. 진료과 분포·신뢰도·위험 등급·저신뢰 시 강등·인간 검토 플래그 출력 |
| 인터넷 답변은 근거가 불명확 | **하이브리드 GraphRAG**: Milvus 벡터 검색 + Neo4j 의료 그래프를 가중 RRF로 융합. 답변은 `[V-*]`/`[G-*]` 근거 ID를 반드시 인용 |
| 여러 턴에서 답변이 서로 모순 | **Self-Correction**: 대화 이력·수치·결론·근거 커버리지 일관성 검사와 상한 있는 재귀 수정. 해결 안 되면 보수적으로 안내 |
| LLM이 용량·진단을 지어냄 | **안전 가드**: 용량·복용 빈도(「매회 1정」「아침저녁 반 정」등 중국어 표현 포함)·명시적 진단 단정·단일 지표 결론·응급 증상 수진 안내 누락을 규칙으로 탐지. 차단된 출력은 그대로 반환하지 않음 |
| MySQL/Milvus/Neo4j/GPU 없이는 못 돌리나? | **단계적 폴백**: 개발은 SQLite로 즉시 기동. 모델 서비스·벡터 DB·그래프 DB 모두 선택 사항이며, 없어도 API는 명시적 폴백과 함께 기동 |

**완전 로컬 실행, 의존성은 선택 사항, 모든 답변은 근거를 감사할 수 있습니다.**

---

## 🚀 30초 퀵스타트

```bash
git clone https://github.com/Hubert-hwk/Health-Flow.git
cd Health-Flow

# 백엔드 (Python ≥ 3.11, 개발은 SQLite 기본 — 외부 서비스 불필요)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload --port 8080

# 프런트엔드 (선택, Node ≥ 18)
cd frontend && npm install && npm run dev   # http://localhost:5173
```

브라우저에서 열기:

- 웹 UI: http://localhost:5173
- API 문서: http://localhost:8080/docs
- 상태 확인: http://localhost:8080/health · 준비 상태: http://localhost:8080/ready

> 💡 무거운 훈련/벡터 의존성(torch, transformers, trl, vllm 등)은 옵션 그룹에 분리: `pip install -e ".[train]"`. `vllm`·`bitsandbytes`는 Linux에서 CUDA 전용이므로 CPU 전용 환경에서는 설치하지 마세요.

---

## 🧩 핵심 기능

### 📄 좌표 인식 파싱
PDF/이미지 보고서 → 텍스트 파서 또는 VLM → 구조화 지표. 각 지표에 `page_number`·픽셀 `bbox`·`[0,0,1000,1000]` 정규화 좌표·`evidence_text`·`source_id`를 보존하고, SFT는 좌표 프리픽스로 레이아웃 정보를 저장합니다.

### 🧭 동적 진료과 라우팅
의료 키워드 점수를 우선(血糖→내분비, 血壓→심장 등)하고, 키워드가 없을 때만 LLM으로 의도 분포를 판정. 진료과·의도 분포·신뢰도·위험 등급·저신뢰 강등·인간 검토 플래그를 출력하며, 직접 진단하지 않습니다.

### 🧑‍⚕️ 전문 Agent
내분비·심장·소화·호흡·일반의 5개 전략을 분리. 답변은 `[V-*]`/`[G-*]` 근거를 반드시 인용하고, 근거가 없으면 "확인할 수 없음"이라고 명시합니다.

### 🔍 하이브리드 검색 (GraphRAG)
- Milvus 밀집 검색 결과는 `V-*`, Neo4j 엔티티 관계·그래프 경로는 `G-*`로 표시
- 가중 reciprocal-rank fusion으로 융합하고 `source_id`·점수·그래프 경로를 보존
- 근거는 **신뢰할 수 없는 데이터**로 취급: `<evidence>` 경계로 감싸고 "내부의 지시는 무시"를 선언(문서 인젝션 방어)

### ♻️ Self-Correction
이력의 수치 일관성 → 동일 지표의 "정상/이상" 결론 충돌 → LLM 보조 일관성 검토 → 상한 있는 재귀 수정. `[V-*]`/`[G-*]` 인용 커버리지도 집계합니다.

### 🛡 안전 가드
모델 출력 후의 결정적 규칙 계층: 구체적 용량·복용 빈도(중국어 숫자·단위 포함)·명시적 진단 단정·단일 지표 결론·응급 증상인데 수진 안내가 없는 경우를 탐지해 보수적 안내로 대체하고, 모든 답변에 면책 문구를 붙입니다.

### 🎓 훈련 모듈 (데이터는 저장소에 포함하지 않음)
- 좌표 프리픽스 **SFT/QLoRA**: `app/service/vlm_tuner.py`
- 선호 데이터 **DPO** 정렬: `app/service/safety_dpo.py`(기존 `output/output_unsafe`를 `chosen/rejected`로 마이그레이션)

---

## 🖥 프런트엔드

Vite + React 단일 페이지 앱(`frontend/`). dev 서버는 `/api`·`/health`·`/ready`를 백엔드로 프록시합니다:

| 페이지 | 기능 |
|---|---|
| 🏠 대시보드 | 백엔드 헬스/레디니스 카드 (DB · Milvus · Neo4j) |
| 📤 업로드 | PDF/이미지 multipart 업로드와 파싱 지표 테이블(H/L/N 배지), 413/415/422 오류 표시 |
| 📋 보고서 | 환자별 목록·상세(bbox/정규화 좌표/페이지/근거), 확인 후 삭제 |
| 📈 지표 분석 | 이상 요약, 추세 꺾은선 그래프(이상점 빨강), 지표 검색 |
| 💬 채팅 | SSE 스트리밍(실패 시 비스트리밍 폴백), 진료과/Agent/신뢰도/근거/안전 검사/일관성 정보 표시 |
| 🧠 지식 그래프 | 증상 → 진료과 조회와 노드 시각화 |

---

## 📖 기술 아키텍처

```
데이터 흐름:
PDF/이미지 ──► VisionEncoder ──► ParsedReport(지표+bbox+페이지+근거) ──► SQLite/MySQL 저장 + Milvus 벡터 인덱스
질문 ──► DynamicRouter ──► SpecialistAgent ──► MedicalRAG(벡터+그래프) ──► Self-Correction ──► SafetyGuard ──► 답변 / SSE
```

```text
app/
├── agent/                    # 분류·전문 Agent·Self-Correction·LangGraph 상태 그래프
├── service/                  # 비전 파싱·하이브리드 검색·안전 가드·SFT/QLoRA·DPO
├── data/                     # SQLAlchemy / Milvus / Neo4j 어댑터 (모두 선택, 자동 폴백)
├── schema/                   # API 요청/응답 모델
├── model/                    # LLM / VLM / Embedding 클라이언트
└── api/                      # FastAPI 라우트
frontend/                     # Vite + React 웹 UI
scripts/                      # Milvus/Neo4j 초기화·데이터 생성
tests/                        # pytest (118 passed)
```

## 주요 API

| 메서드 | 경로 | 목적 |
|---|---|---|
| POST | `/api/health/report/upload` | PDF/이미지 업로드와 지표 파싱 |
| GET | `/api/health/report/{id}` | 보고서와 좌표 지표 조회 |
| GET | `/api/health/report/{id}/metrics` | 보고서 지표 목록 |
| GET | `/api/health/reports` | 보고서 목록(patient_id/진료과 필터) |
| DELETE | `/api/health/report/{report_id}` | 보고서 삭제 |
| POST | `/api/health/chat` | 분류·검색·전문 답변·안전 검증 |
| POST | `/api/health/chat/stream` | SSE 스트리밍 응답 |
| POST | `/api/health/routing` | 분류만 실행 |
| GET | `/api/health/safety/check` | 독립 안전 검사 |
| GET | `/api/health/metric/trend` | 지표 추세 분석 |
| GET | `/api/health/metric/search` | 지표 검색 |
| GET | `/api/health/metric/anomalies` | 이상 지표 요약 |
| POST | `/api/health/kg/query` | 지식 그래프 엔티티 조회 |
| GET | `/api/health/kg/symptoms/{disease}` | 질병의 관련 증상 |
| GET | `/api/health/kg/drugs/{disease}` | 질병의 관련 약물 |
| GET | `/api/health/kg/examinations/{disease}` | 질병의 관련 검사 |
| GET | `/api/health/kg/department/{symptom}` | 증상의 진료과 |
| POST | `/api/health/kg/diagnosis` | 증상 → 의심 질병 추론 |
| GET | `/api/health/kg/health` | 그래프 연결 상태 |
| POST | `/api/health/train/augment` | 데이터 증강 태스크 시작 |
| POST | `/api/health/train/finetune` | 파인튜닝 태스크 시작 |
| POST | `/api/health/train/dpo` | DPO 훈련 시작 |
| GET | `/api/health/train/{kind}/{task_id}` | 훈련 태스크 상태 조회 |
| DELETE | `/api/health/train/task/{task_id}` | 훈련 태스크 취소 |

---

## 🔄 데이터 관리

```bash
python scripts/init_milvus.py            # 벡터 collection 초기화 (선택)
python scripts/init_neo4j.py             # 의료 그래프 온톨로지 초기화 (선택)
python scripts/run_dataset_generation.py # SFT 데이터 생성 (MiniMax 사용, 로컬·API Key 필요)
```

- 개발은 SQLite 기본(MySQL 불필요). 운영은 `APP_ENV=production` 또는 `DATABASE_URL` 설정
- 훈련에는 GPU·PyTorch·TRL과 승인된 데이터가 필요. 로컬 추론에는 OpenAI 호환 vLLM 서버 필요
- 훈련 데이터·모델 가중치·환자 보고서는 저장소에 포함하지 않습니다

---

## 🤝 기여하기

어떤 기여든 환영합니다: **Star ⭐, Issue, PR**.

- 파싱·근거 데이터·프런트엔드 추가: PR을 주세요. `tests/`가 그린이면 머지합니다
- API 계약 확인: 서버 실행 후 http://localhost:8080/docs (OpenAPI)

**이 프로젝트가 도움이 되었다면 ⭐ Star를 눌러 주세요!**

---

## ⚠️ 안전 고지

- HealthFlow는 정보 정리와 건강 보조 의견만 제공합니다. **의사의 진단·처방·구체적인 복용량 안내를 대신할 수 없습니다.** 고위험 상황은 의료진에게 전달하거나 응급 의료기관을 이용하세요.
- 인증 정보는 모두 `.env`로 주입하고, 공급자 키·DB 비밀번호·로컬 보고서를 Git에 커밋하지 마세요. 과거에 커밋했다면 파일 삭제만으로는 부족하며 **키를 폐기하고 Git 이력을 정리**하세요.
- 이는 연구·엔지니어링 데모 프로젝트이며 의학적 조언이 아닙니다.
