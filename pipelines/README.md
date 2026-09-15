# ONIWA Data Pipelines (土壌づくり)

ONIWA プロジェクトにおける「クリーンな土壌づくり（出自が100%追跡可能なオープンデータ収集・前処理）」を担当する専用クレート（`oniwa-pipeline`）です。

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
5. **マルチソース統合コーパス生成**:
   - クレンジング済みテキスト群を一括結合し、モデル学習用の `data/tokens.bin` と `data/vocab.json` を再生成。

---

## 使い方 (`oniwa-dataset`)

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
