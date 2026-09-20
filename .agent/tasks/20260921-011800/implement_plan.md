# [Implementation Plan]: oniwa-lm データ拡充および中規模スケールアップ（Phase 2: Scale v2 & Provenance Logging）

Issue: #80

## 1. 目的 & 概要
現在稼働中のモデル（dim: 128, layers: 4, heads: 4, ffn: 256, seq_len: 128）の学習プラトー（Val Loss ~3.55、過学習傾向）を解消するため、以下の4点を同時に引き上げる Scale v2 への移行を実施する:
1. **Provenance ログ基盤の拡張**:
   - `TrainingStepLog`（`StepLog`）に `timestamp_utc: String`（ISO 8601 UTC）を追加し、後方互換性（`#[serde(default)]`）を担保。
   - `growth_journal.jsonl` にも `timestamp_utc` を付与。
2. **Scale v2 モデル構成プリセット**:
   - `ModelConfig::scale_v2(vocab_size: usize)`（`dim: 192`, `layers: 6`, `heads: 6`, `head_dim: 32`, `ffn_dim: 384`, `seq_len: 256`）を新設。
   - `QuaternionLMHead`（48クォータニオン、ヘッドあたり8クォータニオン）の4分割アサーションを検証・維持。
3. **学習制御・安全停止機構（Early Stopping & NaN/Inf Guard）**:
   - `train.rs` に `--scale-v2` オプションを追加。
   - チェックポイント保存先を `checkpoints/scale_v2/` 配下に完全隔離。
   - Validation Loss 早期停止（patience: 500 steps）および NaN/Inf 検出安全停止を実装。
   - トークン予算（2エポック等）と連動した Cosine LR Decay スケジュール。
   - `chat.rs` でも `--scale-v2` および `--checkpoint-dir` オプションの対応（Scale v2 チェックポイントが自然に読み込めるように整備）。
4. **データセット再構築・動作検証**:
   - `pipelines` によるコーパス生成と単体テスト・スモーク学習検証。

---

## 2. 変更対象ファイルと詳細計画

### Phase 1: ログ基盤の拡張（タイムスタンプ追加）
- `crates/oniwa-lm/src/logger.rs`
  - `TrainingStepLog` に `#[serde(default)] pub timestamp_utc: String` を追加。
  - テストコードを更新し、タイムスタンプのシリアライズ・デシリアライズを検証。

### Phase 2: モデル構成プリセット（Scale v2）と安全停止機構の実装
- `crates/oniwa-lm/src/model.rs`
  - `ModelConfig::scale_v2(vocab_size: usize) -> Self` を追加。
  - `ModelConfig::from_meta_json` の互換性確認。
- `crates/oniwa-lm/src/bin/train.rs`
  - `--scale-v2` CLI オプションの追加。
  - チェックポイント保存先を `checkpoints/scale_v2/latest` および `checkpoints/scale_v2/best` に分岐。
  - `timestamp_utc` を `TrainingStepLog` および `growth_journal.jsonl`（Step 0 および各ステップ）に記録。
  - 早期停止コントローラー: 最良 Val Loss が 500 ステップ更新されない場合、または NaN/Inf 発生時にブレークして最良チェックポイントを保存し安全終了。
  - 学習率スケジュールの調整。
- `crates/oniwa-lm/src/bin/chat.rs`
  - `--scale-v2` および `--checkpoint-dir` オプションの対応。

### Phase 3: データパイプライン・テスト・実機スモーク検証
- `cargo test --workspace`
- `cargo run --release -p oniwa-pipeline -- --build`（または必要に応じて `--all`）
- スモーク学習検証:
  `cargo run --release -p oniwa-lm --bin train -- --quaternion --scale-v2 --reset --steps 10`
- `growth_journal.jsonl` 内の `timestamp_utc` 検証。
- チャット推論テスト:
  `cargo run --release -p oniwa-lm --bin chat -- --quaternion --checkpoint-dir crates/oniwa-lm/checkpoints/scale_v2/best`
- `cargo fmt --all` および `cargo clippy --workspace --all-targets -- -D warnings`
