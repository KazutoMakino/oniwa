# ONIWA Data Pipelines (土壌づくり)

<p align="left">
  <a href="README.md">English</a> | <b>日本語</b>
</p>

ONIWA プロジェクトにおける「クリーンな土壌づくり（出自が100%追跡可能なオープンデータ収集・前処理・トークナイザー構築）」を担当する専用クレート（`oniwa-pipeline`）です。

---

## 主な役割
1. **法・コンプライアンス遵守の自動収集**:
   - **青空文庫**: 公式拡張全作品インデックスから、**著作権保護期間満了（パブリックドメイン / CC0相当）**の作品のみを厳格にフィルタリング。著作権存続作品の混入をゼロにします。
   - **e-Gov 基本法令**: **著作権法第13条（権利の目的とならない著作物＝パブリックドメイン）**に基づき、日本国憲法・刑法・著作権法・民法・裁判所法等の基本法規を公式APIから取得・クレンジング。
2. **Webエチケット・過負荷防止**:
   - 明確な User-Agent の設定、リクエスト間隔のウェイト（1秒）、ローカルキャッシュ（`data/raw/`）による再ダウンロード防止。
3. **高精度テキストクレンジング**:
   - ルビ記法（`｜親文字《るび》`、`漢字《るび》`）、入力者注記（`［＃...］`）、法令XMLタグの自動除去と正規化。
4. **透明な監査台帳（Provenance Ledger）記録**:
   - 取得した全作品・法令の名称、原典URL、法的根拠、SHA-256ハッシュ、文字数を `logs/ledger_index.jsonl` に完全記録。
5. **マルチソース統合コーパス生成 & BPEトークナイザー学習**:
   - クレンジング済みテキスト群を一括結合し、統合テキストコーパス（`data/corpus_combined.txt`）を生成。
   - 外部依存ゼロの Pure Rust による **Byte-level BPE（語彙サイズ 4,096）** 学習ツール（`train-bpe`）により、語彙マージテーブルを構築・保存。

---

## 使い方

### 1. データセット収集・コーパス生成 (`oniwa-dataset`)

```bash
# ① 青空文庫プリセット名作＋e-Gov基本法令を一括取得し、統合コーパスを再生成
cargo run --release -p oniwa-pipeline -- --all

# ② e-Gov 基本法令オープンデータのみを一括取得
cargo run --release -p oniwa-pipeline -- --laws

# ③ レシピ指定による名作群の一括収集（56作品）
cargo run --release -p oniwa-pipeline -- --recipe pipelines/config/recipes.json --build

# ④ 特定の著者の著作権満了作品を片っ端から検索・追加収集
cargo run --release -p oniwa-pipeline -- --author "夏目漱石" --limit 5 --build

# ⑤ 現在の収集状況と監査台帳の確認
cargo run --release -p oniwa-pipeline -- --status
```

### 2. Pure Rust Byte-level BPE トークナイザー学習 (`train-bpe`)

文字単位（Char-level）によるコンテキスト長不足を解決するため、統合コーパスから 4,096 語彙の BPE トークナイザーを学習・出力します：

```bash
# 統合コーパスから 4,096 語彙の BPE 語彙ファイルを生成
cargo run --release -p oniwa-pipeline --bin train-bpe -- --input data/corpus_combined.txt --vocab-size 4096 --output data/bpe_vocab.json
```

### 3. キラーパターン対照変異生成器 (`gen-killer-patterns`)

Rust、Python、技術/法務文書、文学コーパスから、正常スライスと変異破壊スライスの1:1対照ペアを自動合成し、System 1 の異常検知ヘッド（`Noul`）および複雑度スコアヘッドの学習データを生成します：

```bash
# 5,000 ペア（計 10,000 件）の対照データセットを合成
cargo run --release -p oniwa-pipeline --bin gen-killer-patterns -- --pairs 5000 --output data/killer_patterns.jsonl
```
