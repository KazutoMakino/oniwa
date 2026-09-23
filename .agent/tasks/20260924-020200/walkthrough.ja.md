# ウォークスルー: Gatekeeper CLI 実装 (#91)

## 変更概要

`oniwa-decide` クレート内にスタンドアロンの **Gatekeeper CLI** バイナリ（`gatekeeper.rs`）を実装。学習済み `DecisionEngine` を用いてミリ秒オーダーで Git ステージング差分を走査する。

### 変更ファイル一覧

| ファイル | 操作 | 説明 |
|---------|------|------|
| `crates/oniwa-decide/src/bin/gatekeeper.rs` | **新規** | スタンドアロン門番バイナリ (543行) |
| `crates/oniwa-decide/tests/decision_test.rs` | **変更** | 差分パース・ハンク抽出の単体テスト9件 + テストヘルパーモジュール追加 |
| `.agent/tasks/20260924-020200/implement_plan.md` | **新規** | 実装計画書（英語） |
| `.agent/tasks/20260924-020200/implement_plan.ja.md` | **新規** | 実装計画書（日本語） |
| `.agent/tasks/20260924-020200/walkthrough.md` | **新規** | ウォークスルー（英語） |
| `.agent/tasks/20260924-020200/walkthrough.ja.md` | **新規** | 本ファイル |

### 実装した主要機能

1. **Unified Diff パーサー**: `git diff --cached` またはパイプされた unified-diff テキストを解析
2. **拡張子フィルタ**: `.rs`, `.py`（コード）、`.md`, `.txt`（文書）、その他はスキップ
3. **ハンク抽出**: 追加行＋コンテキストから 128〜256 トークンウィンドウを構築
4. **厳格判定ポリシー**:
   - コード異常 (Noul==true, 確信度 ≥ 80%) → **BLOCK** (exit 1)
   - 文書異常 → **WARNING** (exit 0)
   - 異常なし → **PASS** (exit 0)
5. **ベンチマークモード** (`--bench`): `logs/benchmarks/gatekeeper_eval.jsonl` に JSONL 記録
6. **チェックポイント自動検出**: フォールバック付きローディング
7. **差分ゼロ処理**: 対象差分がない場合は exit 0 で即時正常終了

## テスト結果

### `gatekeeper.rs` 内ユニットテスト (9件)
- 全件 ✅ PASS

### `decision_test.rs` 統合テスト (新規9件)
- 全件 ✅ PASS

### ワークスペース全体テスト
- **合計: 90テスト通過、0失敗** ✅

## 検証結果

| チェック項目 | 結果 |
|------------|------|
| `cargo fmt --all -- --check` | ✅ フォーマット差分なし |
| `cargo clippy --workspace --all-targets -- -D warnings` | ✅ 警告なし |
| `cargo test --workspace` | ✅ 90/90 通過 |
| `cargo run --release -p oniwa-decide --bin gatekeeper -- --help` | ✅ ヘルプ表示、exit 0 |
| `cargo run --release -p oniwa-decide --bin gatekeeper -- --bench` | ✅ ステージング差分なし、exit 0 |

## AI セルフチェック（セクション7準拠）

- [x] **仕様準拠**: `gatekeeper.rs` がスタンドアロンで動作し、コードと文書で適切なブロック/警告制御が行われている
- [x] **スコープ厳守**: `.githooks/pre-commit` や既存の推論・学習パイプラインを破壊・改変していない
- [x] **影響範囲の一致**: セクション3の「影響ファイル一覧」以外の無関係なファイルを変更していない
- [x] **Pure Rust 原則順守**: `Cargo.toml` に外部MLクレート等の不要な新規依存が追加されていない
- [x] **フォーマット順守**: `cargo fmt --all` 実行済み、フォーマット差分なし
- [x] **テスト通過**: `cargo test --workspace` が 100% エラーなく成功（90 tests Green）
- [x] **ドキュメント網羅**: 日英両方の `implement_plan`（.md / .ja.md）および `walkthrough`（.md / .ja.md）が指定ディレクトリ内に生成済み
- [x] **Git対象の完全性**: ソースファイルに加えて `.agent/tasks/20260924-020200/` 配下の全ファイルが `git add` の対象に含まれる
