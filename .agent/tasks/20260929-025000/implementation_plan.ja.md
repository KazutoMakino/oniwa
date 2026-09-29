# 実装計画書: oniwa-decide レイヤー別マイクロベンチマーク計測とホットスポット特定

## 概要
Phase 4 の目標である Gatekeeper 即時応答（~100ms 台）に向け、当てずっぽうの最適化を避け実測数値に基づいてボトルネックを特定するため、`oniwa-decide` にレイヤー別マイクロベンチマーク計測機能を導入します。
既存のホットループ（`forward` や `train.rs`）は一切改変せず、隔離された `DecisionModel::forward_with_profile` を新設し、`profiler.rs` へのテレメトリ統合、および `bench.rs` への `--profile-layers` CLI オプション（ASCII テーブル描画 ＆ JSONL 出力）を追加します。

## 変更内容

### 1. `crates/oniwa-decide/src/model.rs`
- `ForwardBreakdown` 構造体の定義:
  ```rust
  #[derive(Clone, Debug, Default, Serialize, Deserialize)]
  pub struct ForwardBreakdown {
      pub embedding_ms: f64,
      pub layer_ms: Vec<f64>,
      pub pooling_ms: f64,
      pub heads_ms: f64,
      pub total_ms: f64,
  }
  ```
- `DecisionModel::forward_with_profile(&self, tokens: &[u16], b: usize, t: usize) -> (ForwardCache, ForwardBreakdown)` の実装。
- 既存の `forward` メソッドは完全温存し、オーバーヘッドをゼロに維持。

### 2. `crates/oniwa-decide/src/profiler.rs`
- ブレークダウン情報を含む記録構造体 `LayerProfileRecord` / `HardwareProfileRecord` への統合またはシリアライズ定義。

### 3. `crates/oniwa-decide/src/bin/bench.rs`
- `--profile-layers` / `--breakdown` フラグの解析処理の追加。
- `forward_with_profile` を実行し、平均所要時間および構成割合（%）を算出。
- ターミナルへ ASCII 内訳テーブルを描画し、`--output` 指定時は JSONL にブレークダウンを反映。

### 4. `crates/oniwa-decide/tests/decision_test.rs`
- 単体テスト `test_forward_with_profile` を追加し、各計測値が非負であり、レイヤー数と設定が合致し、合計時間が構成要素と整合していることを検証。

## 検証計画
- `cargo check -p oniwa-decide`
- `cargo test -p oniwa-decide --test decision_test test_forward_with_profile`
- `cargo run --release -p oniwa-decide --bin bench -- --iters 1 --profile-layers`
- `cargo test --workspace`
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
