# 実装計画 - Step 1: キラーパターン対照変異生成器（`gen-killer-patterns`）の実装

## 概要
決定論的なキラーパターン対照変異データセット生成CLI（`pipelines/src/bin/gen-killer-patterns.rs`）を実装し、`crates/oniwa-decide` の `KillerPatternDataset` による往復検証テストを実施します。

## ユーザー確認事項
本ステップでの破壊的変更はありません。生成するデータセットフォーマットは `KillerPatternDataset::from_jsonl_reader` のバリデーション制約に完全準拠します。

## 変更内容

### `pipelines/Cargo.toml` および `pipelines/src/bin/gen-killer-patterns.rs`
- CLI引数:
  - `--corpus <DIR>`: デフォルト `data/corpus`
  - `--output <FILE>`: デフォルト `data/killer_patterns.jsonl`
  - `--pairs <COUNT>`: デフォルト 5000（合計 10,000 件）
  - `--seed <U64>`: デフォルト 42
- 生成ロジック:
  - 4つのカテゴリ（Rust、Python、技術/法務文書、文学）からコーパスをロード。
  - `DeterministicRng` により決定論的・再現可能な抽出。
  - 各ペアの生成:
    - 正常スライス（`Noul = 0.0`, `Score in 0.0..=1.0`, `mask_indices = []`）
    - 破壊スライス（`Noul = 1.0`, `Score in 0.0..=1.0`, `mask_indices = [変異位置]`）
    - トークンレベルまたは文字境界を正確に考慮し、`mask_indices` は昇順・重複なし・トークン長未満であることを保証。
- 決定論的 JSONL 出力。

### `crates/oniwa-decide/tests/decision_test.rs`
- 生成データセットの読み込み・`write_batch` 往復検証テストを追加。

## 検証計画
1. `cargo check --workspace`
2. `cargo run --release -p oniwa-pipeline --bin gen-killer-patterns -- --pairs 5000 --output data/killer_patterns.jsonl`
3. `cargo test --workspace`
4. `cargo fmt --all -- --check`
5. `cargo clippy --workspace --all-targets -- -D warnings`
