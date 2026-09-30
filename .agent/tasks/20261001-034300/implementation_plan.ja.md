# [実装計画書]: oniwa-decide 推論専用パス分離 ＆ 可変長NEON最適化によるエッジPre-commitレイテンシ短縮

## 1. 目的と概要
- `oniwa-decide`（System 1）の推論パスを完全分離（`DecisionModel::forward_inference`）し、決定ヘッド（Choice, Noul, Score）に不要な MLM 語彙射影（$V \times C = 4096 \times 256$ 内積ループ）を完全にスキップ。中間 `LayerCache` / `ForwardCache` の不要確保を排除します。
- Mean Pooling 処理に ARM NEON SIMD ベクトル累積加算（`accumulate_slice_simd`）を導入し、`float32x4_t` による 4 並列演算とスケーリングを適用します。
- `DecisionEngine::audit_text` および `gatekeeper` において、実効トークン長 $T_{\text{valid}}$（クォータニオン及び RoPE の 4 の倍数アライン、下限 4、上限 `seq_len`）に適応させて Transformer 層を駆動し、短い差分コード走査時の演算量を大幅に削減します。
- 既存の学習用 `forward` と `forward_inference` の決定ヘッド出力（Choice, Noul, Score）が単精度丸め誤差内（$< 10^{-5}$）で完全一致することを検証します。

## 2. 変更対象ファイルと差分設計

### `crates/oniwa-lm/src/simd.rs`
- `accumulate_slice_simd(acc: &mut [f32], x: &[f32])` を新設:
  - `aarch64` 環境: `vld1q_f32`, `vaddq_f32`, `vst1q_f32` による 4 並列累積加算 + 端数スカラー処理。
  - 非 aarch64 環境: スカラー加算ループ。
- 単体等価性テスト `test_accumulate_slice_simd_equivalence` を追加。

### `crates/oniwa-decide/src/simd/mod.rs`
- `oniwa_lm::simd::accumulate_slice_simd` を再エクスポート。
- 等価性テストを追加。

### `crates/oniwa-decide/src/model.rs`
- `DecisionModel::forward_inference(&self, tokens: &[u16], b: usize, t: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)` を実装:
  - 中間アクティベーションキャッシュ（`LayerCache`）の保持を省略し、層ごとのバッファ再利用でアロケーションを抑制。
  - MLM 語彙射影を完全にスキップ。
  - Mean Pooling に `accumulate_slice_simd` と `scale_slice_simd` を適用。
- `DecisionModel::decide` を更新:
  - 実効トークン長 $T$ を 4 の倍数に切り上げ（下限 4、上限 `seq_len`）。
  - パディングトークン列を用意し、`forward_inference(&padded, 1, t_aligned)` を呼び出し。
- 既存の `forward`, `forward_with_profile`, `train.rs` は学習・ベンチマーク互換性のため無変更で維持。

### `crates/oniwa-decide/src/lib.rs`
- `DecisionEngine::audit_text(&self, text: &str)`:
  - トークナイズ後、適応長で `self.model.decide(&tokens)` を実行。

### `crates/oniwa-decide/src/bin/gatekeeper.rs`
- 差分ハンク走査時に高速化された適応長推論を活用。

### `crates/oniwa-decide/tests/decision_test.rs`
- 単体テスト `test_forward_inference_equivalence` を追加:
  - `forward` と `forward_inference` の Choice ロジット、Noul ロジット、Score 予測値が誤差 $< 10^{-5}$ 以内で完全一致することを検証。

## 3. 検証手順（Verify コマンド）
- Phase 2: `cargo test -p oniwa-lm simd::tests`
- Phase 3: `cargo test -p oniwa-decide --test decision_test test_forward_inference_equivalence`
- Phase 4: `cargo test -p oniwa-decide --test regression_snippets`
- Phase 5:
  - `cargo fmt --all -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
- Phase 6:
  - ベンチマーク実行: `cargo run --release -p oniwa-decide --bin bench -- --iters 5 --profile-layers`
  - スモーク実行: `cargo run --release -p oniwa-decide --bin gatekeeper -- --bench`
