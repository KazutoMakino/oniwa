# 🌿 ONIWA ラズパイ引越し＆コンテキスト引き継ぎガイド (Handover Guide)

> **対象**: PC (`km-gpd`) から **Raspberry Pi** 実機へのプロジェクト移行、および AI エージェントとのやりとり・文脈の完全引き継ぎ

<p align="left">
  <a href="HANDOVER_RASPBERRY_PI.md">English</a> | <b>日本語</b>
</p>


---

## 1. プロジェクトの現状と設計思想

- **プロジェクト名**: `oniwa` (小さな庭・箱庭AI)
- **コア思想**:
  - **脱・巨大GPU / 脱・ブラックボックス**: PyTorchや外部深層学習フレームワークに依存せず、**Pure Rust（標準ライブラリ＋純粋な数学）**でゼロから自作したスクラッチ言語モデル。
  - **盆栽・観葉植物を愛でるエッジAI**: ラズパイの数W〜十数Wの電力、70℃前後の安全な温度で、数千〜数万ステップを静かに回し、言葉の芽吹き（知能の発達）を観葉植物のように育てる。
  - **透明性と再現性**: 学習データは著作権保護に配慮してGitに含めず、外部レシピ（`pipelines/config/recipes.json`）と監査台帳（`logs/ledger_index.jsonl`）で100%再構築可能。
- **現在の学習ステータス**:
  - **コーパス規模**: 青空文庫パブリックドメイン28作品、約143.5万文字（語彙4,201文字）。
  - **モデル規模**: Transformer Decoder (2 layers, 2 heads, dim 64, seq_len 128, パラメータ数約62万)。
  - **最新到達ステップ**: Step 2900 (Train Loss: 3.3786 / Val Loss: 3.8179)。
  - **観測方式**: 4大プローブ巡回方式（「その時、」「メロスは、」「吾輩は、」「私は、」）。

---

## 2. ラズパイへの引越し手順

### Step 1: Git リポジトリの同期（コード・レシピ・台帳）
ラズパイのターミナルにて：
```bash
# 既存クローンがある場合
cd ~/Github/oniwa   # または任意の配置パス
git pull origin main

# 新規クローンの場合
git clone https://github.com/KazutoMakino/oniwa.git
cd oniwa
```

### Step 2: 学習済み重みとデータの転送（オプション）
`.gitignore` されている学習済みモデル重み（約2.5MB）やトークナイズ済みバイナリをPC（現在のマシン）から直接転送する場合：

```bash
# PC側（km-gpd）からラズパイへ rsync で転送する例
# （※ raspberrypi.local やユーザー名・IPアドレスは環境に合わせて変更）
rsync -avzP \
  crates/oniwa-lm/checkpoints/ \
  pi@raspberrypi.local:~/Github/oniwa/crates/oniwa-lm/checkpoints/

rsync -avzP \
  crates/oniwa-lm/data/ \
  pi@raspberrypi.local:~/Github/oniwa/crates/oniwa-lm/data/
```
> **備考**: もし重みを引き継がずゼロから育て直す場合、またはラズパイ側でデータを再取得する場合は以下で完結します：
> ```bash
> cargo run --release -p pipelines --bin ingest
> cargo run --release -p oniwa-lm --bin train -- --reset --steps 100
> ```

### Step 3: ラズパイ上でのビルドとテスト
```bash
# 全ワークスペースのテスト確認
cargo test --workspace

# 対話チャットの起動テスト（現在の知能状態を確認）
cargo run --release -p oniwa-lm --bin chat

# 学習の継続（自動で最新チェックポイントから再開）
cargo run --release -p oniwa-lm --bin train -- --steps 500 --interval 25
```

---

## 3. ラズパイ側で AI エージェント（Antigravity）とのやりとりを引き継ぐ方法

ラズパイ上で Antigravity（または VS Code Remote-SSH）を起動した際、最初のプロンプトとして以下をそのまま入力してください。  
**AIが過去の文脈・決定事項・設計思想を100%把握した状態で即座に再開します。**

```markdown
@docs/HANDOVER_RASPBERRY_PI.md を読み込んでください。
PC環境からラズパイ実機への引越しが完了しました。
これまでの設計思想（Pure Rust・外部非依存・盆栽的エッジAI・143万文字コーパス・複数プローブ巡回観測）を引き継ぎ、開発と学習の継続をサポートしてください。
```

---

## 4. 主なディレクトリ構成

```
oniwa/
├── crates/
│   ├── oniwa-core/         # テンソル・自動微分・基礎数学演算
│   ├── oniwa-nn/           # LayerNorm, MultiHeadAttention, TransformerBlock
│   └── oniwa-lm/           # トークナイザ、自己回帰推論、学習ループ
│       ├── src/bin/train.rs# 訓練バイナリ（複数プローブ巡回・安全トラッカー内蔵）
│       ├── src/bin/chat.rs # 対話チャットバイナリ
│       └── checkpoints/    # best / latest 重み（ローカル保持）
├── pipelines/              # 青空文庫データ収集・クレンジング・監査
│   ├── config/recipes.json # 収集対象作家・作品リスト（宣言的レシピ）
│   └── src/                # 高速スクレイピング・正規化パイプライン
├── logs/                   # 生育観察日記・監査台帳
│   ├── growth_journal.md   # 言葉の芽吹き（マルチプローブ観測記録）
│   └── ledger_index.jsonl  # SHA-256監査ログ台帳
└── docs/
    └── HANDOVER_RASPBERRY_PI.md # 本引越しガイド
```
