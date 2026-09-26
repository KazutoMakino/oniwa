# 実装計画書: refactor: Rust API Guidelines (RFC 430) 準拠に伴う頭字語・公開シグネチャ・命名規則の全面整合

Rust API Guidelines (RFC 430: 頭字語を CamelCase の1単語として扱う、自己文書化された引数名) に準拠し、`oniwa-decide` および `oniwa-lm` の型名・公開シグネチャを整合させます。チェックポイントの JSON シリアライズ互換性は 100% 維持します。

## 変更計画

### 1. `crates/oniwa-decide`
- `src/layers/rmsnorm.rs`:
  - `RMSNorm` -> `RmsNorm`
  - 公開引数名: `c: usize` -> `dim: usize`
- `src/layers/mlp.rs`:
  - `SwiGLU` -> `SwiGlu`
  - 公開引数名: `c: usize` -> `dim: usize`, `ffn: usize` -> `ffn_dim: usize`
- `src/layers/attention.rs`:
  - `BidirectionalSelfAttention` -> `Attention`
  - 公開引数名: `c: usize` -> `dim: usize`, `nh: usize` -> `num_heads: usize`, `d_h: usize` -> `head_dim: usize`
- `src/layers/quaternion_attention.rs`:
  - `QuaternionSelfAttention` -> `QuaternionAttention`
  - `Attention::apply_rope` 呼出への追従
  - 公開引数名: `c: usize` -> `dim: usize`, `nh: usize` -> `num_heads: usize`
- `src/layers/quaternion_mlp.rs`:
  - `QuaternionSwiGLU` -> `QuaternionSwiGlu`
  - 公開引数名: `c: usize` -> `dim: usize`, `ffn: usize` -> `ffn_dim: usize`
- `src/layers/quaternion_linear.rs`:
  - 引数名の整合性確認
- `src/layers/mod.rs`:
  - 再エクスポート定義の更新 (`Attention`, `SwiGlu`, `QuaternionAttention`, `QuaternionLinear`, `QuaternionSwiGlu`, `RmsNorm`)
- `src/model.rs`:
  - レイヤー型名および公開シグネチャ変更に伴う呼出追従
- `tests/decision_test.rs`:
  - テストコードの追従

### 2. `crates/oniwa-lm`
- `src/layers/rmsnorm.rs`:
  - `RMSNorm` -> `RmsNorm`
  - 公開引数名: `d: usize` -> `dim: usize`
- `src/layers/mlp.rs`:
  - `SwiGLU` -> `SwiGlu`
  - 公開引数名: `c: usize` -> `dim: usize`, `ffn: usize` -> `ffn_dim: usize`
- `src/layers/attention.rs`:
  - `CausalSelfAttention` (公開引数名: `c: usize` -> `dim: usize`, `nh: usize` -> `num_heads: usize`, `d_h: usize` -> `head_dim: usize`)
- `src/layers/quaternion_head.rs`:
  - 公開引数名: `n: usize` -> `num_tokens: usize`, `d: usize` -> `dim: usize`, `v: usize` -> `vocab_size: usize`
- `src/model.rs`:
  - レイヤー呼出の追従
- 各種バイナリ (`train.rs`, `chat.rs` 等):
  - コメントや呼出箇所の追従

## 検証計画
- `cargo test --workspace` による全テスト通過確認
- `cargo fmt --all -- --check` によるフォーマット検証
- `cargo clippy --workspace --all-targets -- -D warnings` による Lint 検証
- スモークテスト:
  - `cargo run --release -p oniwa-decide --bin gatekeeper -- --help`
  - `cargo run --release -p oniwa-lm --bin chat -- --help`
