# 実装計画: oniwa-decide Gatekeeper CLI

## 概要

`oniwa-decide` クレート内にスタンドアロンの **Gatekeeper CLI** バイナリ（`gatekeeper.rs`）を実装し、学習済み `DecisionEngine` を用いてミリ秒オーダーで Git ステージング差分を走査する。ONIWA プロジェクトの論文実証用ツールとして、System 1 意思決定エンジンのコードレビュー実務への適用可能性を実証する。

## アーキテクチャ

```
git diff --cached → Unified Diff パーサー → ハンク抽出 → DecisionEngine → 判定 (PASS/WARNING/BLOCK)
                                                                          ↓ (--bench モード)
                                                            logs/benchmarks/gatekeeper_eval.jsonl
```

## 影響ファイル

### [新規] `crates/oniwa-decide/src/bin/gatekeeper.rs`
- **CLI引数解析**: `--bench`, `--checkpoint <パス>`, `--stdin`, `--help`
- **Unified Diff パーサー**: `git diff --cached` またはstdinからハンクを抽出
- **拡張子フィルタ**: `.rs`, `.py`（コード）、`.md`, `.txt`（文書）、バイナリ/lockファイルはスキップ
- **ハンク抽出**: 追加行＋コンテキストから128〜256トークンウィンドウを構築
- **DecisionEngine連携**: チェックポイントからロードし、ハンクごとに `audit_text()` 実行
- **判定ポリシー**:
  - コード（`.rs`, `.py`）: Noul==true かつ確信度 ≥ 80% → BLOCK (exit 1)
  - 文書（`.md`, `.txt`）: 異常 → WARNING (exit 0)
  - 異常なし → PASS (exit 0)
- **ベンチマークモード**（`--bench`）: 推論時間・エントロピー・確信度をJSONL記録

### [変更] `crates/oniwa-decide/tests/decision_test.rs`
- 差分パースおよびハンク抽出の単体テスト9件を追加

## 検証計画

1. `cargo fmt --all -- --check` — フォーマット差分なし
2. `cargo clippy --workspace --all-targets -- -D warnings` — Clippy警告なし
3. `cargo test --workspace` — 全テスト通過（100% Green）
4. `cargo run --release -p oniwa-decide --bin gatekeeper -- --help` — ヘルプ表示
5. `cargo run --release -p oniwa-decide --bin gatekeeper -- --bench` — ベンチマークモード動作

## Issue ドリブンワークフロー

- **Issue**: #91
- **ブランチ**: `91/feat/gatekeeper-cli-standalone-engine`
- **コミットメッセージ**: `feat: implement Gatekeeper CLI for paper-grade standalone gate-keeping engine (#91)`
