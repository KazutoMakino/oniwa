# 成果・振り返り - Step 1: キラーパターン対照変異生成器（`gen-killer-patterns`）の実装

## 概要
System 1（`oniwa-decide`）の異常検知ヘッド（`Noul`）および複雑度スコアヘッドの弁別性能を高めるため、統合コーパスから正常スライス（`Noul=0.0`）と破壊スライス（`Noul=1.0`）の1:1対照ペアを決定論的に自動合成するCLIツール `pipelines/src/bin/gen-killer-patterns.rs` を実装・検証しました。

## 変更内容
- **`pipelines/src/bin/gen-killer-patterns.rs`**:
  - 引数 `--corpus`, `--output`, `--vocab`, `--bpe`, `--pairs`, `--seed` に対応した CLI 実装。
  - 4つのカテゴリ（Rust、Python、技術/法務文書、文学）から均等にサンプリングし、以下の対照ペアを生成:
    - 正常スライス: `Noul = 0.0`, `Score in 0.0..=1.0`, `mask_indices = []`
    - 破壊スライス: `Noul = 1.0`, `Score in 0.70..=0.90` (3.5〜4.5相当), `mask_indices = [変異位置]`
  - トークン長・エンコード結果と `mask_indices` の整合性を厳密に検証し、境界外エラーを完全排除。
- **`crates/oniwa-decide/tests/decision_test.rs`**:
  - `test_generated_killer_patterns_jsonl_roundtrip` を追加し、生成された `data/killer_patterns.jsonl` の読み込みおよび `KillerPatternDataset::write_batch` 往復検証をパス。

## 検証・セルフチェック結果
- [x] **仕様準拠**: 5,000 ペア（10,000 件）の対照変異データを `data/killer_patterns.jsonl` に生成完了。
- [x] **フォーマット順守**: `cargo fmt --all` を適用し、フォーマット差分なし。
- [x] **Clippy 検証通過**: `cargo clippy --workspace --all-targets -- -D warnings` が警告 0 件で通過。
- [x] **テスト通過**: `cargo test --workspace` が 100% 成功（Green）。
- [x] **Git 対象の完全性**: `.agent/tasks/20260924_125029_feat_gen_killer_patterns/` 配下の全ドキュメント（task.md, implement_plan.*, walkthrough.*）が `git add` 対象に含まれることを確認。
