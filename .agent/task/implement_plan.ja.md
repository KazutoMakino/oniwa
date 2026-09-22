# 実装計画書: 10_system_integration_roadmap に基づくドキュメント体系の整合・最新化

Issue: #87
タスク: 最新実装（BPE 4K, MLM, 100MBメモリ制約, クォータニオン代数）に基づくリポジトリ内ドキュメント体系の整合・最新化および計算並行安全運用の徹底。

## 背景と目的
1. Phase 1（`train-bpe` 4,096語彙トークナイザー）および Phase 2（MLMマルチタスク、キラーパターン注入、フラットメモリ100MB制約計算）の実装がコードベース側で完了。
2. ルート `README` や各クレート、パイプラインのドキュメントに旧仕様（文字単位トークナイザー前提や旧極小モデル構成）の記述が残っているため是正。
3. 日英ドキュメントを完全同期して改訂。
4. `docs/10_system_integration_roadmap.*` を実用化・統合ロードマップの正本（SSoT）とし、`docs/09_roadmap.*` を学術クォータニオン研究書として位置づけを明確化。
5. 過去の設計書（`docs/00`〜`09`、`docs/HANDOVER_RASPBERRY_PI.*`）の冒頭に、正本への誘導Banner注記を追加。
6. `crates/oniwa-decide/README.md` および `README.ja.md` を新規作成。
7. 計算安全優先: 稼働中計算プロセス保護のため `cargo test` / `cargo build` などの高負荷コマンドは実行せず、`logs/` や `checkpoints/` を誤ってステージングしない。

## 対象ファイル
- ルート文書: `README.md`, `README.ja.md`
- パイプライン文書: `pipelines/README.md`, `pipelines/README.ja.md`
- クレート文書:
  - `crates/oniwa-decide/README.md`, `crates/oniwa-decide/README.ja.md` (新規)
  - `crates/oniwa-lm/docs/README.md`, `crates/oniwa-lm/docs/README.ja.md`
- 誘導Banner付与:
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
- 進行・記録文書:
  - `.agent/task/task.md`
  - `.agent/task/implement_plan.md`
  - `.agent/task/implement_plan.ja.md`
  - `.agent/task/walkthrough.md`
  - `.agent/task/walkthrough.ja.md`

## 実施手順
1. `.agent/task/` へのタスク定義・計画書配置（完了）。
2. 各既存ファイルの調査。
3. ルート `README.md` / `README.ja.md` の改訂。
4. `pipelines/README.md` / `pipelines/README.ja.md` の改訂。
5. `crates/oniwa-decide/README.md` / `crates/oniwa-decide/README.ja.md` の新規作成。
6. `crates/oniwa-lm/docs/README.md` / `crates/oniwa-lm/docs/README.ja.md` の改訂。
7. 過去設計書への誘導Banner挿入。
8. フォーマット適用（Pythonファイル等の整合性）、リンク検証。
9. `walkthrough.md` / `walkthrough.ja.md` の作成。
10. 対象ファイルのみのステージング、コミット、プッシュ、PR作成、スカッシュマージ、ブランチクリーンアップ。
