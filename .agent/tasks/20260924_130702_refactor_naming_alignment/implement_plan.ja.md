# 実装計画 - Step 2: Rust標準命名規則・慣例への全面整合およびドキュメント同期

## 概要
`oniwa` ワークスペース全体を Rust 公式 API ガイドライン（RFC 430）に準拠させるための全面的なリファクタリングを実施します:
- バイナリターゲット名・ファイル名を `kebab-case` に統一（`compress_eval.rs` -> `compress-eval.rs`）。
- 単純ゲッターの `get_` プレフィックス排除（例: `get_extension` -> `extension`）。
- コンストラクタ・変換トレイトの慣例遵守（`new`, `as_`, `to_`, `into_`）。
- ルートおよび各クレートの README、解説文書内の実行コマンド表記を改訂後バイナリ名と 100% 同期（`docs/benchmarks/*` 内の静的実測記録は温存）。
- 全自動テスト Green、Clippy 警告ゼロの確認。

## 変更内容

### 1. バイナリ名・ファイル名リネーム
- `crates/oniwa-decide/src/bin/compress_eval.rs` を `crates/oniwa-decide/src/bin/compress-eval.rs` にリネーム。
- `crates/oniwa-decide/Cargo.toml` の `[[bin]]` 定義を `name = "compress-eval", path = "src/bin/compress-eval.rs"` に更新。

### 2. Rust API ガイドライン（RFC 430）準拠監査
- ワークスペース全体の `fn get_` を検索し、Rust 慣行に合致しない単純ゲッターを改修（例: `get_extension` -> `extension` 等）。
- メソッド名、構造体、Enum 命名の整合性確認。

### 3. ドキュメント同期
- `README.md`, `README.ja.md`, 各サブクレート README の `compress_eval` を `compress-eval` に同期。
- その他コマンド記述の整合性確認。

## 検証計画
1. `cargo test --workspace`
2. `cargo clippy --workspace --all-targets -- -D warnings`
3. `cargo fmt --all -- --check`
