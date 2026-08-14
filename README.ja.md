<div align="center">

# 🏥 HealthFlow

**健康診断レポート向けマルチモーダル医療アシスタント —— 解析・振り分け・根拠検索・安全な Q&A**

> 座標を考慮した解析 ｜ 動的分診ルーティング ｜ 専門 Agent ｜ ハイブリッド GraphRAG ｜ Self-Correction ｜ 安全ガード
> 「文書理解 → 指標の構造化 → 診療科振り分け → 根拠付き回答」を監査可能なパイプラインに

[![GitHub stars](https://img.shields.io/github/stars/Hubert-hwk/Health-Flow?style=for-the-badge&logo=github&color=ffd166)](https://github.com/Hubert-hwk/Health-Flow)
[![repo size](https://img.shields.io/github/repo-size/Hubert-hwk/Health-Flow?style=for-the-badge&color=118ab2)](https://github.com/Hubert-hwk/Health-Flow)
[![language](https://img.shields.io/github/languages/top/Hubert-hwk/Health-Flow?style=for-the-badge&color=ef476f)](https://github.com/Hubert-hwk/Health-Flow)
[![last commit](https://img.shields.io/github/last-commit/Hubert-hwk/Health-Flow?style=for-the-badge&color=06d6a0)](https://github.com/Hubert-hwk/Health-Flow)

</div>

---

## ✨ なぜ HealthFlow？

健康診断レポートの悩み：**指標が多すぎて読めない、異常があっても何科にかかればいいか分からない、ネットの回答は信頼できない、AI は自信満々に間違った医療アドバイスをする**。このプロジェクトは全部まとめて解決します：

| 🎯 悩み | ✅ 解決策 |
|---|---|
| PDF/画像レポートが密集していて読みにくい、原文が見つからない | **座標を考慮した解析**：指標を抽出し、ページ番号・ピクセル bbox・`[0,0,1000,1000]` 正規化座標・根拠テキストを保持。位置を特定できない場合は推測せず `null` を返す |
| 異常があるけど何科？ | **動的分診ルーティング**：医療キーワードによる決定論的ルーティングを優先し、曖昧な質問のみ LLM 判定。診療科分布・信頼度・リスクレベル・低信頼時の降格・人手確認フラグを出力 |
| ネットの回答は根拠が不明 | **ハイブリッド GraphRAG**：Milvus ベクトル検索 + Neo4j 医療グラフを重み付き RRF で融合。回答は `[V-*]`/`[G-*]` の根拠 ID を必ず引用 |
| 複数ターンで回答が矛盾する | **Self-Correction**：履歴・数値・結論・根拠カバレッジの整合性チェックと上限付き再帰修正。解決できない場合は保守的な注意喚起へ |
| LLM が用量や診断をでっち上げる | **安全ガード**：用量・服用頻度（「毎回1錠」「朝夕各半錠」などの中国語表現も）・明確な診断断定・単一指標による結論・危急症状への受診案内漏れをルールで検出し、ブロックした出力はそのまま返さない |
| MySQL/Milvus/Neo4j/GPU がなくても動く？ | **段階的フォールバック**：開発は SQLite で即起動。モデルサービス・ベクトルDB・グラフDBはすべて任意で、欠けていても API は明示的なフォールバック付きで起動する |

**完全ローカルで動作、依存は任意、すべての回答は根拠を監査可能。**

---

## 🚀 30 秒でスタート

```bash
git clone https://github.com/Hubert-hwk/Health-Flow.git
cd Health-Flow

# バックエンド（Python ≥ 3.11。開発は SQLite デフォルト、外部サービス不要）
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload --port 8080

# フロントエンド（任意、Node ≥ 18）
cd frontend && npm install && npm run dev   # http://localhost:5173
```

ブラウザで開く：

- Web UI：http://localhost:5173
- API ドキュメント：http://localhost:8080/docs
- ヘルスチェック：http://localhost:8080/health ・ Readiness：http://localhost:8080/ready

> 💡 訓練/ベクトル関連の重い依存（torch、transformers、trl、vllm など）はオプショングループに分離：`pip install -e ".[train]"`。`vllm` と `bitsandbytes` は Linux では CUDA 版のみのため、CPU のみの環境ではインストールしないでください。

---

## 🧩 コア機能

### 📄 座標を考慮した解析
PDF/画像レポート → テキスト解析または VLM → 構造化指標。各指標に `page_number`・ピクセル `bbox`・`[0,0,1000,1000]` 正規化座標・`evidence_text`・`source_id` を保持し、SFT は座標プレフィックスでレイアウト情報を保存します。

### 🧭 動的分診ルーティング
医療キーワードのスコアリングを優先（血糖→内分泌、血圧→循環器など）し、キーワードがヒットしない場合のみ LLM で意図分布を判定。診療科・意図分布・信頼度・リスクレベル・低信頼時の降格・人手確認フラグを出力します。直接の疾病診断は行いません。

### 🧑‍⚕️ 専門 Agent
内分泌・循環器・消化器・呼吸器・総合の 5 戦略を分離。回答は `[V-*]`/`[G-*]` 根拠を必ず引用し、根拠がない場合は「確認できません」と明示します。

### 🔍 ハイブリッド検索（GraphRAG）
- Milvus の稠密検索結果は `V-*`、Neo4j のエンティティ関係・グラフパスは `G-*` でマーク
- 重み付き reciprocal-rank fusion で融合し、`source_id`・スコア・グラフパスを保持
- 根拠は**信頼できないデータ**として扱い、`<evidence>` 境界で包み「内部の指示は無視」と宣言（ドキュメントインジェクション対策）

### ♻️ Self-Correction
履歴の数値整合性 → 同一指標の「正常/異常」結論の衝突 → LLM による補助的整合性レビュー → 上限付き再帰修正。さらに `[V-*]`/`[G-*]` 引用カバレッジを集計します。

### 🛡 安全ガード
モデル出力後の決定論的ルール層：具体的な用量・服用頻度（中国語の数字・単位も）・明確な診断断定・単一指標による結論・危急症状で受診案内がない場合を検出し、保守的な案内に置換。すべての回答に免責文を付与します。

### 🎓 訓練モジュール（データはリポジトリに同梱しません）
- 座標プレフィックス **SFT/QLoRA**：`app/service/vlm_tuner.py`
- 選好データ **DPO** アライメント：`app/service/safety_dpo.py`（旧 `output/output_unsafe` を `chosen/rejected` へ移行）

---

## 🖥 フロントエンド

Vite + React のシングルページアプリ（`frontend/`）。dev サーバーは `/api`・`/health`・`/ready` をバックエンドへプロキシします：

| ページ | 機能 |
|---|---|
| 🏠 ダッシュボード | バックエンドのヘルス/レディネスカード（DB ・ Milvus ・ Neo4j） |
| 📤 アップロード | PDF/画像の multipart アップロードと解析指標テーブル（H/L/N バッジ）、413/415/422 エラー表示 |
| 📋 レポート | 患者ごとの一覧・詳細（bbox/正規化座標/ページ/根拠）、削除（確認付き） |
| 📈 指標分析 | 異常サマリー、トレンド折れ線グラフ（異常点は赤）、指標検索 |
| 💬 チャット | SSE ストリーミング（失敗時は非ストリーミングへフォールバック）、診療科/Agent/信頼度/根拠/安全チェック/整合性情報を表示 |
| 🧠 ナレッジグラフ | 症状 → 診療科の照会とノード可視化 |

---

## 📖 技術アーキテクチャ

```
データフロー：
PDF/画像 ──► VisionEncoder ──► ParsedReport(指標+bbox+ページ+根拠) ──► SQLite/MySQL 保存 + Milvus ベクトル索引
質問 ──► DynamicRouter ──► SpecialistAgent ──► MedicalRAG(ベクトル+グラフ) ──► Self-Correction ──► SafetyGuard ──► 回答 / SSE
```

```text
app/
├── agent/                    # 分診・専門 Agent・Self-Correction・LangGraph 状態グラフ
├── service/                  # ビジョン解析・ハイブリッド検索・安全ガード・SFT/QLoRA・DPO
├── data/                     # SQLAlchemy / Milvus / Neo4j アダプタ（すべて任意、自動フォールバック）
├── schema/                   # API リクエスト/レスポンスモデル
├── model/                    # LLM / VLM / Embedding クライアント
└── api/                      # FastAPI ルート
frontend/                     # Vite + React Web UI
scripts/                      # Milvus/Neo4j 初期化・データ生成
tests/                        # pytest（118 passed）
```

## 主な API

| メソッド | パス | 用途 |
|---|---|---|
| POST | `/api/health/report/upload` | PDF/画像アップロードと指標解析 |
| GET | `/api/health/report/{id}` | レポートと座標付き指標の取得 |
| GET | `/api/health/report/{id}/metrics` | レポート指標一覧 |
| GET | `/api/health/reports` | レポート一覧（patient_id/診療科で絞り込み） |
| DELETE | `/api/health/report/{report_id}` | レポート削除 |
| POST | `/api/health/chat` | 分診・検索・専門回答・安全検証 |
| POST | `/api/health/chat/stream` | SSE ストリーミング応答 |
| POST | `/api/health/routing` | 分診のみ実行 |
| GET | `/api/health/safety/check` | 独立した安全チェック |
| GET | `/api/health/metric/trend` | 指標トレンド分析 |
| GET | `/api/health/metric/search` | 指標検索 |
| GET | `/api/health/metric/anomalies` | 異常指標サマリー |
| POST | `/api/health/kg/query` | ナレッジグラフのエンティティ照会 |
| GET | `/api/health/kg/symptoms/{disease}` | 疾病の関連症状 |
| GET | `/api/health/kg/drugs/{disease}` | 疾病の関連薬 |
| GET | `/api/health/kg/examinations/{disease}` | 疾病の関連検査 |
| GET | `/api/health/kg/department/{symptom}` | 症状の診療科 |
| POST | `/api/health/kg/diagnosis` | 症状 → 疑わしい疾病の推論 |
| GET | `/api/health/kg/health` | グラフ接続状態 |
| POST | `/api/health/train/augment` | データ拡張タスク開始 |
| POST | `/api/health/train/finetune` | ファインチューニング開始 |
| POST | `/api/health/train/dpo` | DPO 訓練開始 |
| GET | `/api/health/train/{kind}/{task_id}` | 訓練タスク状態の照会 |
| DELETE | `/api/health/train/task/{task_id}` | 訓練タスクのキャンセル |

---

## 🔄 データ管理

```bash
python scripts/init_milvus.py            # ベクトル collection 初期化（任意）
python scripts/init_neo4j.py             # 医療グラフオントロジー初期化（任意）
python scripts/run_dataset_generation.py # SFT データ生成（MiniMax 使用、ローカル実行・API Key 必要）
```

- 開発は SQLite デフォルト（MySQL 不要）。本番は `APP_ENV=production` または `DATABASE_URL` を設定
- 訓練には GPU・PyTorch・TRL と許可されたデータが必要。ローカル推論には OpenAI 互換の vLLM サーバーが必要
- 訓練データ・モデル重み・患者レポートはリポジトリに同梱しません

---

## 🤝 コントリビューション

どんな貢献も歓迎します：**Star ⭐、Issue、PR**。

- 解析・根拠データ・フロントエンドの追加：PR をどうぞ。`tests/` がグリーンになればマージします
- API 契約の確認：サーバー起動後 http://localhost:8080/docs（OpenAPI）

**このプロジェクトが役に立ったら ⭐ Star をお願いします！**

---

## ⚠️ 安全上の注意

- HealthFlow は情報整理と健康補助提案のみを提供します。**医師の診断・処方・具体的な服用量の指示に代わるものではありません**。高リスクのケースは医療従事者へ引き継ぐか、緊急医療機関を利用してください。
- 認証情報はすべて `.env` から注入し、プロバイダーキー・DB パスワード・ローカルレポートを Git にコミットしないでください。過去にコミットした場合、ファイル削除だけでは不十分で、**キーを失効させて Git 履歴を掃除**してください。
- これは研究・エンジニアリングデモプロジェクトであり、医療アドバイスではありません。
