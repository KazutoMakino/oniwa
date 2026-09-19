# Walkthrough: QuaternionLMHead & Weight Tying in oniwa-lm

## 概要
タスク仕様書 `task-001-quaternion-lm-head.md` に基づき、`oniwa-lm` にクォータニオン自己回帰生成レイヤー（`QuaternionLMHead`）および厳密な重み共有（Strict Weight Tying: $E \in \mathbb{H}^{V \times (D/4)}$）を備えた `QuaternionModelWeights` を導入しました。既存の実数モデル（`ModelWeights`）や既存チェックポイント（`checkpoints/best`）との完全な後方互換性を担保し、ARM NEON SIMDによる高速演算を実現しています。

---

## 変更内容と構成

### 1. 循環依存の解消と基盤モジュール
- `oniwa-decide` が `oniwa-lm` に依存するクレート構成であったため、純粋クォータニオン数理（`Quaternion`）および ARM NEON SIMD プリミティブ（`accumulate_hamilton_simd`, `dot_product_4d_simd` 等）を `crates/oniwa-lm/src/quaternion.rs` および `crates/oniwa-lm/src/simd.rs` に配置。
- `oniwa-decide` では上記モジュールを再エクスポート（`pub mod quaternion`, `pub mod simd`, `pub use simd::*`）することで、相互循環依存を起こさずに既存の全コード・単体テストの後方互換性を 100% 維持。

### 2. `QuaternionLMHead` 層（`crates/oniwa-lm/src/layers/quaternion_head.rs`）
- 隠れ状態 $h \in \mathbb{R}^{B \times T \times D}$ を $\mathbb{H}^{B \times T \times (D/4)}$ として解釈。
- Forward: エルミート共役ハミルトン積の実部抽出（4Dドット積）として $\text{logits}_{b,t,v} = \sum_{k=1}^{D/4} \text{dot\_product\_4d}(E[v, k], h_{b,t}[k])$ を NEON SIMD で高速計算。
- Backward: GHR微積分に基づき、$dh$ および $dE$ への解析的勾配を逆伝播。
- 数値微分（Finite Differences）との勾配チェックテスト（相対誤差 $< 5 \times 10^{-3}$）をパス。

### 3. `QuaternionModelWeights` & `LanguageModel` Trait（`crates/oniwa-lm/src/model.rs`）
- 共通トレイト `LanguageModel` を導入（`ModelWeights` と `QuaternionModelWeights` が実装）。
- `QuaternionModelWeights` では、語彙埋め込み行列と `QuaternionLMHead` の重み $E$ を同一バッファとして共有（Strict Weight Tying）。
- 逆伝播（`forward_backward`）で Embedding と LM Head の双対勾配を $E$ に累積加算。
- チェックポイント保存時に `"model_type": "quaternion_tied"` を付与し、実数ローダー・クォータニオンローダー相互の取り違えをバリデーションガード。

### 4. CLI連携（`train.rs` & `chat.rs`）
- `crates/oniwa-lm/src/bin/train.rs`:
  - `--quaternion` フラグを追加。指定時は `QuaternionModelWeights` を初期化し、チェックポイントを `checkpoints/quaternion/` 配下に隔離保存。
- `crates/oniwa-lm/src/bin/chat.rs`:
  - `--quaternion` フラグを追加。指定時は `checkpoints/quaternion/best` からモデルを読み込んで対話推論・テキスト生成を実行。フラグなし時は既存の `checkpoints/best`（実数モデル）を安全にロード。

---

## 検証結果

### 1. ワークスペース自動テスト
```bash
cargo test --workspace
```
- `oniwa-decide`: 23 tests passed (13 lib + 10 integration)
- `oniwa-lm`: 34 tests passed (包括的単体テスト、クォータニオン勾配チェック、チェックポイント往復テスト、実数モデルによるクォータニオン誤ロード遮断テストを含む)
- `oniwa_dataset`: 9 tests passed
- **結果**: 100% Green (全 66 テスト成功)

### 2. 静的解析・フォーマット
```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```
- **結果**: 警告・フォーマット違反ゼロ（Pass）

### 3. 実機動作検証（スモークテスト）
- **学習テスト**:
  ```bash
  cargo run --release -p oniwa-lm --bin train -- --quaternion --reset --steps 10
  ```
  - Mode: `QuaternionLMHead` (Total parameters: 1,260,800)
  - 10ステップの学習および検証、チェックポイント `checkpoints/quaternion/best`（`"model_type": "quaternion_tied"`）の正常保存を確認。
- **推論テスト**:
  ```bash
  cargo run --release -p oniwa-lm --bin chat -- --quaternion
  ```
  - クォータニオンチェックポイントをロードし、プロンプト「その時、」に対する自己回帰生成が正常に動作することを確認。
  - `--quaternion` なしの `cargo run --release -p oniwa-lm --bin chat` で既存の実数チェックポイント（Step 5800, Loss 3.3931）がそのままロード・生成できることを確認。
