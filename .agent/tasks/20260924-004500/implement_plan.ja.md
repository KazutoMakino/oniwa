# 実装計画: ログ・標準出力への短縮UTC日時付与および温度単位degCへの統一

## 背景と目的
`oniwa` リポジトリでは、構造化台帳（JSON Lines）で ISO 8601 UTC タイムスタンプが記録されている一方で、学習およびベンチマーク実行時の標準出力（コンソール表示）やプレーンテキストログ（`*train*.log`）には日時情報が出力されていませんでした。これにより、長時間のバックグラウンド学習監視や時系列での照合が困難でした。また、コンソールおよびログにおける温度表記（`C` や `℃`）を工学的な標準である `degC` に統一します。

本計画の内容:
1. `crates/oniwa-lm/src/logger.rs` に外部クレート依存なし（Pure Rust）で `YYYY-MM-DD HH:MM:SS UTC` を生成する `current_utc_display()` および `format_utc_display(epoch_secs: u64)` を実装。
2. `crates/oniwa-lm/src/bin/train.rs` および `crates/oniwa-decide/src/bin/train.rs` の進捗ログ行の先頭に `[YYYY-MM-DD HH:MM:SS UTC]` を付与。
3. 標準出力およびプレーンテキストログ内の温度表示を `degC`（例: `72.4 degC`）に統一。機械可読な JSON/JSONL キー（`cpu_temp_c`, `thermal_celsius` 等）は完全維持。
4. 単体テストの追加とエッジケースの検証。

---

## 影響ファイル
- `crates/oniwa-lm/src/logger.rs`: 日時フォーマッタ関数とテスト
- `crates/oniwa-lm/src/bin/train.rs`: 進捗ログへの日時付与と温度表記更新
- `crates/oniwa-decide/src/bin/train.rs`: 進捗ログへの日時付与と温度表記更新
- `crates/oniwa-decide/src/bin/bench.rs`: 出力確認・温度表記更新
- `.agent/tasks/20260924-004500/*`: 計画・振り返りドキュメント

---

## 検証計画
- `cargo test --workspace` による全テスト通過
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo run --release -p oniwa-lm --bin train -- --steps 2`
- `cargo run --release -p oniwa-decide --bin train -- --steps 2`
- セルフチェックリスト照合
