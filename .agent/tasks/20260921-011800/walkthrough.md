# [Walkthrough Report]: oniwa-lm データ拡充および中規模スケールアップ（Phase 2: Scale v2 & Provenance Logging）

## 1. 概要
- **GitHub Issue**: #80 (`feat: scale v2 architecture, dataset enrichment and provenance timestamping`)
- **ブランチ**: `80/feat/scale-v2-and-provenance-logging`
- **目的**:
  1. `TrainingStepLog` および `growth_journal.jsonl` への UTC タイムスタンプ付与（出自追跡性の強化）。
  2. Raspberry Pi 4 向け中規模スケール `scale_v2`（dim: 192, layers: 6, heads: 6, head_dim: 32, ffn_dim: 384, seq_len: 256）および `QuaternionLMHead`（48クォータニオン、ヘッドあたり8クォータニオン）のプリセット新設。
  3. `checkpoints/scale_v2/` への完全隔離チェックポイント管理と、チェックポイント読み込み時の `dim` / `seq_len` 不整合ガード。
  4. 早期停止機構（Early Stopping: 500ステップ猶予）および NaN/Inf 発散検出安全停止機構の実装。
  5. コーパス再ビルドおよび実機スモーク学習（Step 0〜10）・チャット推論テストの完遂。

---

## 2. 変更差分詳細

### (1) `crates/oniwa-lm/src/logger.rs`
- `TrainingStepLog` に `#[serde(default)] pub timestamp_utc: String` を追加。
- 既存ログ読み込み時の後方互換性を保証。
- 単体テスト `test_full_lifecycle_provenance_ledger` にてタイムスタンプ記録および旧形式 JSON デシリアライズの互換性を検証。

### (2) `crates/oniwa-lm/src/model.rs`
- `ModelConfig::scale_v2(vocab_size: usize) -> Self` を追加。
  - `dim: 192`, `num_layers: 6`, `num_heads: 6`, `head_dim: 32`, `ffn_dim: 384`, `seq_len: 256`, `weight_tying: true`。
- `QuaternionModelWeights::load_checkpoint` にて `dim` および `seq_len` の整合性チェックを追加。旧世代（128次元）等の誤ロードを防止。
- 単体テスト `test_scale_v2_preset_and_quaternion` を追加し、4分割アサーションやアテンションヘッド次元の整合性を自動テスト。

### (3) `crates/oniwa-lm/src/bin/train.rs`
- `--scale-v2` CLI オプションを新設。
  - 指定時は `checkpoints/scale_v2/latest` および `checkpoints/scale_v2/best` に保存先を隔離。
  - モデル未初期化時は `ModelConfig::scale_v2` 構成で自動生成。
- `--epochs <N>` 引数によるエポック駆動ステップ数計算に対応。
- `TrainingStepLog` 生成時に `timestamp_utc` および計算された `current_epoch` を自動設定。
- `growth_journal.jsonl`（Step 0 および各評価ステップ）に `timestamp_utc` を記録。
- **安全停止・早期停止機構**:
  - `loss.is_nan() || loss.is_infinite()` を検出した際、即座に安全脱出し最良モデルを保全。
  - `val_loss` が 500 ステップ連続で改善しない場合に `🛑 [Early Stopping]` を発動して正常終了。

### (4) `crates/oniwa-lm/src/bin/chat.rs`
- `--scale-v2` オプションおよび `--checkpoint-dir` オプションを追加。
- 指定パス（絶対パス・相対パス問わず）から柔軟にチェックポイントを解決・ロード可能に改修。

---

## 3. 検証結果

### 自動テスト
- `cargo test --workspace`
  - 全 35 個の `oniwa-lm` テスト、全 23 個の `oniwa-decide` テスト、全 9 個の `oniwa-pipeline` テストが **100% Green** でパス。

### データセット再ビルド
- `cargo run --release -p oniwa-pipeline -- --build`
  - 統合作品数: 91作品
  - 総文字数: 3,672,453文字（約 10.3 MB）
  - 語彙サイズ: 4,721文字
  - `vocab.json` および `tokens.bin` を正常に再生成。

### スモーク学習検証
- コマンド: `cargo run --release -p oniwa-lm --bin train -- --quaternion --scale-v2 --reset --steps 10`
  - パラメータ数: 3,120,768 パラメータ（約 3.12M パラメータ）
  - 初期 Validation Loss: 8.4618（Step 0）
  - Step 10: Train Loss 8.4128, Val Loss 8.3852（Top-5 Accuracy 9.3%）
  - チェックポイント: `crates/oniwa-lm/checkpoints/scale_v2/best/` に `meta.json`, `weights.bin`（36MB）が正常保存。
  - ログ検証: `growth_journal.jsonl` に `timestamp_utc: "2026-09-20T16:37:24Z"` 等の ISO 8601 UTC タイムスタンプが正常記録されたことを確認。

### チャット推論テスト
- コマンド: `cargo run --release -p oniwa-lm --bin chat -- --quaternion --checkpoint-dir crates/oniwa-lm/checkpoints/scale_v2/best`
  - Scale v2 チェックポイント（dim: 192, seq_len: 256, params: 3.12M）を正常にロードし、テキスト生成動作を確認。

### 静的解析・フォーマット
- `cargo fmt --all -- --check` -> **Pass**
- `cargo clippy --workspace --all-targets -- -D warnings` -> **Pass (Warnings: 0)**
