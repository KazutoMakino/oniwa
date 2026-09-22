# 成果・振り返りドキュメント (Walkthrough): 10_system_integration_roadmap に基づくドキュメント体系の整合・最新化

Issue: #87
タスク: 最新実装（Pure Rust Byte-level BPE 4,096語彙、MLMマルチタスク学習、100MBメモリ制約、クォータニオン代数）に基づくリポジトリ内ドキュメント体系の整合・最新化および計算並行安全運用の確立。

## 実施内容サマリー

### 1. 主要ルートおよびパイプラインドキュメントの改訂
- **`README.md` / `README.ja.md`**:
  - 日英完全同期で全面改訂。
  - 100MBメモリバジェット制約バッジおよび最新アーキテクチャ概要を追加。
  - `docs/10_system_integration_roadmap.ja.md` / `10_system_integration_roadmap.md` を最新正本（SSoT）として明記。
  - System 1（`oniwa-decide`）の非自己回帰決定、MLM、キラーパターン、クォータニオン代数（Hamilton積）の確定仕様を反映。
- **`pipelines/README.md` / `pipelines/README.ja.md`**:
  - `train-bpe` ツール（`cargo run -p oniwa-pipeline --bin train-bpe -- --input ... --vocab-size 4096 --output ...`）の解説を追加。
  - 統合コーパス生成とBPE語彙学習のワークフローを整理。

### 2. クレートドキュメントの整備
- **`crates/oniwa-decide/README.md` / `crates/oniwa-decide/README.ja.md`（新規作成）**:
  - これまで未配置だった `oniwa-decide` に詳細な解説ドキュメントを日英で新設。
  - 3大決定プリミティブ（`Choice`, `Noul`, `Score`）、温度較正、損失関数、フラットメモリ100MBバジェット制約、CLIコマンド（`audit`, `train`）を明文化。
- **`crates/oniwa-lm/docs/README.md` / `crates/oniwa-lm/docs/README.ja.md`**:
  - ドキュメントインデックスに `10_system_integration_roadmap`（正本）および `09_roadmap`（クォータニオン研究書）を追加・更新。

### 3. 過去設計書への誘導Banner注記の挿入
- 以下の過去・初期設計ドキュメント群の冒頭に、正本（`10_system_integration_roadmap`）への誘導Banner注記を日英双方で統一配置：
  - `docs/00_oniwa-project-manifesto.{md,ja.md}`
  - `docs/01_llm_c_architecture.{md,ja.md}`
  - `docs/02_modern_primitives.{md,ja.md}`
  - `docs/03_rust_memory_model.{md,ja.md}`
  - `docs/04_development_handoff.{md,ja.md}`
  - `docs/05_reproducibility_and_logging.{md,ja.md}`
  - `docs/06_thermal_and_end_to_end_provenance.{md,ja.md}`
  - `docs/07_data_ingestion_charter.{md,ja.md}`
  - `docs/08_green_energy_and_power_tracking.{md,ja.md}`
  - `docs/09_roadmap.{md,ja.md}`
  - `docs/HANDOVER_RASPBERRY_PI.{md,ja.md}`
- 物理アーカイブディレクトリ（`old/`等）を作成せず、最新1本を維持しつつ過去の版はGit履歴に委ねる運用方針を徹底。

### 4. 計算安全優先ルールの遵守
- ホスト上で稼働中の長時間計算・学習プロセス保護のため、`cargo test` や `cargo build` などの高負荷コマンドは一切実行せず完了。
- `git add .` を避け、更新対象のMarkdownファイル群および `.agent/task/` 配下のみを明示的に指定してステージング。

## 検証結果
- 全対象Markdownドキュメント内の相対リンクをPythonスクリプトにより網羅検証（リンク切れ0件）。
- 日本語・英語版のファイル構成および内容が完全に対称であることを確認。
