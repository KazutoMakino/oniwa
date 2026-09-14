# Phase 3: 他PC（16GB RAM）向け 開発・実行引き継ぎガイド

本ドキュメントは、別マシン（RAM 16GB環境）で `niwa-lm` の実装・テスト・学習を引き継いで実行するための完全ガイドです。

---

## 1. 前提環境と推奨スペック

* **対象マシン**: RAM 16GB 搭載の PC (Linux / macOS / WSL2)
* **Rust バージョン**: 1.75 以上（`rustup update` 推奨）
* **外部依存**: **ゼロ**（Python、PyTorch、CUDAドライバ不要。`cargo` だけで完結します）

---

## 2. 引き継ぎ後のクイックスタート（コマンド手順）

別PCでリポジトリをクローンまたは pull した後、以下の手順で進めます。

```bash
# 1. oniwa-lm ディレクトリへ移動
cd apps/oniwa-lm

# 2. 全単体テスト（数値微分勾配チェック・乱数再現性）の実行
cargo test -- --nocapture

# 3. リセットしてシード0からサクッと動作確認 (150ステップ)
cargo run --release --bin train -- --reset

# 4. 対話チャットで動作確認
cargo run --release --bin chat
```

---

## 3. 16GB RAM でのモデル規模・ハイパーパラメータ設定指針

16GB RAM マシンで「スワップゼロ・快適なオンメモリ学習」を実現するための設定値です。

```rust
pub struct Config {
    pub vocab_size: usize,   // 8,192 (青空文庫/日本語BPEに最適)
    pub seq_len: usize,      // 512 トークン
    pub dim: usize,          // 512 (隠れ層次元)
    pub num_layers: usize,   // 8 レイヤー
    pub num_heads: usize,    // 8 (Query ヘッド数)
    pub num_kv_heads: usize, // 2 (KV ヘッド数, GQA G=4)
    pub ffn_dim: usize,      // 1,365 (SwiGLU 黄金比: 512 * 8/3)
}
```

### メモリ配分表（所要メモリ合計：約 910 MB）
* **パラメータ重み**: 約 140 MB (`f32`)
* **勾配バッファ**: 約 140 MB (`f32`)
* **AdamW 状態バッファ**: 約 280 MB (`m` と `v` で重みの2倍)
* **中間活性化値 (B=4, T=512)**: 約 350 MB
* **OS / バックグラウンド余力**: **15 GB 以上の空き**（他作業と並行しても極めて安全）

---

## 4. 引き継ぎ後の開発タスク（実装順ロードマップ）

別PCで開発を再開する際は、以下の順番で `src/layers/` を埋めていきます。

### Step 1: 残りの Modern Primitives の実装
1. **`src/layers/rope.rs`**:
   - `02_modern_primitives.md` の第2節を参照。
   - 順伝播回転と、逆回転（符号反転）による逆伝播。
2. **`src/layers/swiglu.rs`**:
   - `02_modern_primitives.md` の第3節を参照。
   - $\text{Swish}'(u)$ を用いた逆伝播。
3. **`src/layers/matmul.rs`**:
   - `matrixmultiply::sgemm` を用いた CPU 行列積の順伝播・逆伝播。
   - $C = A B \implies dA = dC B^T, \quad dB = A^T dC$

### Step 2: フラットアリーナバッファの実装 (`src/buffer.rs`)
- `03_rust_memory_model.md` に基づき、単一の `Vec<f32>` から `split_at_mut` で各レイヤーのスライスを切り出す安全な構造体を定義。

### Step 3: 学習ループ・オプティマイザの結合 (`src/model.rs`, `src/optim.rs`)
- `01_llm_c_architecture.md` に基づき、全レイヤーを逆順に呼び出す手動 `backward` と、1次元ループの `AdamW` を結合。

---

## 5. ドキュメント相互参照マップ

* **構造分解の基礎**: [01_llm_c_architecture.md](file:///home/multi/GitHub/my-mono-repo/apps/niwa-lm/docs/01_llm_c_architecture.md)
* **数式と導出**: [02_modern_primitives.md](file:///home/multi/GitHub/my-mono-repo/apps/niwa-lm/docs/02_modern_primitives.md)
* **メモリ・ライフタイム設計**: [03_rust_memory_model.md](file:///home/multi/GitHub/my-mono-repo/apps/niwa-lm/docs/03_rust_memory_model.md)
* **先行実装コード**: [src/layers/rmsnorm.rs](file:///home/multi/GitHub/my-mono-repo/apps/niwa-lm/src/layers/rmsnorm.rs) （※数値微分テスト付き）
