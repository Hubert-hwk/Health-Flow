<p align="center">
  <img src="docs/assets/healthflow-hero.svg" alt="HealthFlow" width="920" />
</p>

<p align="center">
  <a href="README.md">简体中文</a> · <a href="README.en.md">English</a> · <strong>日本語</strong> · <a href="README.ko.md">한국어</a>
</p>

# HealthFlow

HealthFlow は、健康診断レポートや医療文書を対象としたマルチモーダル医療アシスタントのプロトタイプです。文書理解、検査指標の構造化、動的な診療科振り分け、根拠検索、専門 Agent、安全性検証を、監査可能な一連のワークフローとして統合します。

<p align="center">
  <img src="docs/assets/healthflow-pipeline.svg" alt="HealthFlow の処理パイプライン" width="920" />
</p>

> 安全上の制約：HealthFlow は情報整理と健康に関する補助的な提案のみを提供します。医師の診断、処方、具体的な服用量の指示に代わるものではありません。高リスクのケースは医療従事者へ引き継ぐか、緊急医療機関を利用してください。

## HealthFlow の特徴

- 座標を考慮した解析：抽出した指標とページ番号、正規化した BBOX、根拠テキストを一緒に保持します。
- LangGraph によるオーケストレーション：ルーティング、検索、専門 Agent の回答生成、上限付き Self-Correction を型付き状態グラフで実行します。
- ハイブリッド GraphRAG：Milvus のベクトル検索と、Neo4j の制約付き医療グラフパス・出典情報を融合します。
- 安全性を優先した回答生成：決定論的ルール、根拠参照、不確実性、人手確認フラグを組み合わせます。
## アーキテクチャ

```mermaid
flowchart LR
  A[PDF / 画像レポート] --> B[テキスト解析または VLM]
  B --> C[指標 + ページ + BBOX]
  C --> D[LangGraph ルーター]
  D --> E[専門 Agent]
  E --> F[Milvus ベクトル検索]
  E --> G[Neo4j GraphRAG]
  F --> H[根拠コンテキスト]
  G --> H
  H --> I[Self-Correction]
  I --> J[安全ガード + 注意書き]
  J --> K[アシスタント回答 / 人手確認]
```

メインのグラフは [`app/agent/graph/medical_graph.py`](app/agent/graph/medical_graph.py) に実装され、次の 4 ノードで構成されます。

```text
route → retrieve → generate → validate
```

専門 Agent はグラフから選択される通常の Python サービスです。独立したリモート Agent ではありません。この構成により、プロトタイプを決定論的かつテストしやすく保ちながら、将来的にマルチ Agent 実行基盤へ置き換えられます。

## 報告されているオフライン指標

以下の数値はプロジェクトの履歴書と面接資料に記載された実験結果です。公開リポジトリには元の医療データセットが含まれていないため、ここに示す数値は自動的に再現できるベンチマークではありません。

| 領域 | 報告された結果 |
|---|---|
| 座標対応マルチモーダル解析 | 独自 500 件のレポートテストセットで 74% → 83%。Qwen2.5-VL の汎用ベースラインより 9 ポイント向上 |
| DPO 安全性アライメント | 2,400 件の選好ペア。高リスク幻覚率 31% → 8% |
| 動的振り分け | ルーティング精度 92% |
| GraphRAG | 純粋なベクトル検索より再現率 18% 向上 |
| Self-Correction | BERTScore > 0.82。複数ターンの論理矛盾 24% → 6% |

これらは診断精度や本番 SLA を表すものではありません。再現可能な実験パッケージには、データの出典、分割方法、Recall@K、信頼区間、人手確認の手順も必要です。

## 実装済みの主な機能

- ページ内ピクセル座標、`[0, 0, 1000, 1000]` の正規化座標、ページ番号、根拠テキスト、source ID を含む PDF/画像解析。
- 明示的な医療キーワードを優先し、曖昧な質問では LLM にフォールバックする診療科振り分け。診療科分布、信頼度、リスクレベル、低信頼時の縮退、人手確認フラグを返します。
- 内分泌科、循環器科、消化器科、呼吸器科、一般支援の専門戦略。回答には `[V-*]` / `[G-*]` の根拠参照を付与します。
- ベクトル検索とグラフ検索の結果を重み付き Reciprocal Rank Fusion で統合し、スコアとグラフパスを保持します。
- 数値の矛盾、結論の矛盾、会話履歴、根拠カバレッジを対象にした上限付き整合性検証。
- 服用量の要求、明確な診断断定、単一指標による結論、危険症状、受診案内の欠落を検出する安全ルール。
- SFT/QLoRA と DPO の実行入口。旧 `output/output_unsafe` を正規の `chosen/rejected` 選好形式へ移行できます。

## クイックスタート

開発環境では SQLite を標準で使用します。モデルサービス、Milvus、Neo4j はオプションです。未設定の場合も API は起動し、該当機能は明示的なフォールバック結果を返します。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
uvicorn app.main:app --reload --port 8080
```

アクセス先：

- Web フロントエンド：http://localhost:5173（下記参照）
- API ドキュメント：http://localhost:8080/docs
- ヘルスチェック：http://localhost:8080/health
- Readiness：http://localhost:8080/ready

### フロントエンド（任意）

`frontend/` は Vite + React のシングルページアプリです。開発サーバーは `/api` を `http://localhost:8080` へプロキシします。

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
npm run build      # frontend/dist に出力
```

ページ：ダッシュボード（バックエンド状態）、レポートアップロード、レポート一覧/詳細、指標分析（異常・トレンド・検索）、チャット（SSE ストリーミング）、ナレッジグラフ（症状→診療科）。

本番データベースを使う場合は `APP_ENV=production` または `DATABASE_URL` を設定してください。ベクトル検索とグラフ検索を有効にする場合は Milvus と Neo4j を設定してから、次を実行します。

```bash
python scripts/init_milvus.py
python scripts/init_neo4j.py
```

## 主な API

| メソッド | パス | 用途 |
|---|---|---|
| POST | `/api/health/report/upload` | PDF/画像レポートをアップロードして解析 |
| GET | `/api/health/report/{id}` | レポートと座標付き指標を取得 |
| POST | `/api/health/chat` | 振り分け、検索、専門回答、安全性検証 |
| POST | `/api/health/chat/stream` | SSE ストリーム応答 |
| POST | `/api/health/routing` | 振り分けのみ実行 |
| GET | `/api/health/safety/check` | 独立した安全性チェック |

## プロジェクト構成

```text
app/
├── agent/
│   ├── dynamic_router.py          # 診療科振り分け
│   ├── specialist_agents.py       # 専門戦略
│   ├── recursive_feedback.py      # Self-Correction
│   └── graph/medical_graph.py      # LangGraph StateGraph
├── service/
│   ├── vision_encoder.py          # PDF/画像と BBOX の解析
│   ├── medical_rag.py              # ハイブリッド検索と根拠コンテキスト
│   ├── safety_guard.py             # モデル外の安全ガード
│   ├── vlm_tuner.py                # 座標プレフィックス SFT/QLoRA
│   └── safety_dpo.py               # 選好検証と DPO
├── data/                           # SQLAlchemy、Milvus、Neo4j アダプター
└── api/                            # FastAPI ルート
```

## データ、セキュリティ、現在の制約

- 患者レポート、実在する医療情報、ライセンスが確認されていないデータセットは公開しません。
- すべての認証情報は `.env` から注入し、プロバイダーキー、データベースパスワード、ローカルレポートを Git にコミットしないでください。
- 認証情報を過去にコミットした場合、現在のファイルを削除するだけでは不十分です。キーを失効させ、公開前に Git 履歴も清掃してください。
- 学習には GPU、PyTorch、Transformers、TRL、承認済みデータが必要です。完全な学習・BERTScore・500 件レポートの再現パッケージは含めていません。
- 本プロジェクトは研究・エンジニアリング用のデモであり、医療 advice ではありません。
