# [実装計画書]: 差分並列走査・NEON Softmax最適化 ＆ チェックポイント調整・学習評価

## 1. 目的と概要
1. **差分ハンク並列走査（Rayon）**: `gatekeeper.rs` において、`rayon::prelude::*` による並列イテレータを適用し、複数ハンクが存在する場合でも全CPUコアを活用して並列推論を行い、コミット待ち時間の線形増加を防止。
2. **NEON SIMD による Attention Softmax ベクトル化**: `oniwa-lm::simd` に `max_element_simd`（`vmaxvq_f32`）および `sum_slice_simd`（`vaddvq_f32`）を追加し、`attention.rs` および `quaternion_attention.rs` 内の Softmax 最大値・総和リダクションを高速化。
3. **学習実行と推論精度・レイテンシ評価**: トレーニングを実行し、更新されたパスの安定性を検証するとともに、複数ハンクでの実推論時間および精度を評価。次点のアクションを提示する。

## 2. 変更対象ファイル
- `crates/oniwa-lm/src/simd.rs`: `max_element_simd`, `sum_slice_simd` の実装と単体等価性テスト。
- `crates/oniwa-decide/src/simd/mod.rs`: 再エクスポートとテスト。
- `crates/oniwa-decide/src/layers/attention.rs` / `quaternion_attention.rs`: Softmax 行リダクションの SIMD 化。
- `crates/oniwa-decide/src/bin/gatekeeper.rs`: Rayon 並列走査への切り替え。

## 3. 検証手順
- `cargo test -p oniwa-lm simd::tests`
- `cargo test -p oniwa-decide`
- `cargo test --workspace`
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo run --release -p oniwa-decide --bin bench -- --iters 5`
- `cargo run --release -p oniwa-decide --bin gatekeeper -- --stdin --bench`
