# 実装計画書: ARM NEON SIMD による oniwa-decide レイヤー演算の直接ベクトル化と推論高速化

## 概要
`oniwa-decide` の各レイヤー（RmsNorm, SwiGlu, Attention, Quaternion Attention）における推論・ホットループを、ARM NEON SIMD（`float32x4_t`）および高精度スカラーフォールバックを用いてベクトル化し、数値的一貫性と推論高速化を両立する。

## 変更スコープと作業項目
1. **SIMD プリミティブ拡充 (`crates/oniwa-lm/src/simd.rs`)**:
   - `sum_squares_simd(x: &[f32]) -> f32`: `float32x4_t` を用いたスライスの二乗和計算と余りスカラー処理。
   - `mul_slices_simd(out: &mut [f32], a: &[f32], b: &[f32])`: 4 並列 `vmulq_f32` による要素ごとの積。
   - `scale_slice_simd(out: &mut [f32], a: &[f32], scalar: f32)`: 4 並列 `vmulq_n_f32` によるスカラー乗算。
   - 単体等価性テスト（`test_sum_squares_simd_equivalence`, `test_mul_slices_simd_equivalence`, `test_scale_slice_simd_equivalence`）の追加（許容誤差 $< 10^{-5}$）。
2. **再エクスポート確認 (`crates/oniwa-decide/src/simd/mod.rs`)**:
   - `oniwa-decide` 内から `oniwa-lm` の新設プリミティブが直接呼び出し可能であることを確認。
3. **レイヤー演算への適用 (`crates/oniwa-decide/src/layers/`)**:
   - `rmsnorm.rs`: 二乗和平均とスケーリングループに `sum_squares_simd` / SIMD積を適用。
   - `mlp.rs`: SwiGlu の `act_h = silu(g) * u` の要素積を SIMD 化。
   - `quaternion_mlp.rs`: 四元数 SwiGlu の `act_h = silu(g) * u` の要素積を SIMD 化。
   - `attention.rs`: QK 内積計算および Attention 出力加算ループを SIMD 化。
   - `quaternion_attention.rs`: クォータニオン内積および Attention 出力加算ループの最適化確認。
4. **検証と品質担保**:
   - `cargo test --workspace` による全テスト通過確認。
   - `cargo fmt --all -- --check`
   - `cargo clippy --workspace --all-targets -- -D warnings`
   - スモークテスト（`gatekeeper --help`, `audit`）の実行。
5. **ドキュメント作成と完了フロー**:
   - `walkthrough.md` および `walkthrough.ja.md` の作成。
   - セルフチェックリストの検証記録。
   - コミット、プルリクエスト、スカッシュマージ、main同期。
