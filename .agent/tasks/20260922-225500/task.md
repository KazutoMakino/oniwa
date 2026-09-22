# [Task Plan / Issue Spec]: System 1 MLM 学習パイプラインおよびキラーパターン統合 (#84)

## 1. Issue & 目的定義
- **背景と課題**:
  Issue #84 および ロードマップ (`docs/10_system_integration_roadmap.md`) Phase 2 に基づき、`oniwa-decide` (System 1) における意味的文脈理解の向上とエッジ (Raspberry Pi 4) 上での厳格な 100MB メモリバジェット順守が求められています。
  従来の文字単位/静的構文決定モデルから、Pure Rust の BPE 語彙拡張 (4,096 vocab) と BERT スタイルの Masked Language Modeling (MLM) 射影・マルチタスク逆伝播を統合し、キラーパターン (論理・意味論的デッドロック、セマンティックリーク等) の直接バッチ書き込み (Zero-allocation ingestion) を確立する必要があります。
- **解決方針とスコープ**:
  1. **フラットメモリレイアウト算定と 100MB 制約保証**:
     - `DecisionConfig` または専用の `MemoryBudget` 計算モジュールにより、モデルパラメータ、勾配バッファ、AdamW 1次・2次モーメント、および順伝播・逆伝播のアクティベーションバッファの総メモリフットプリントを厳密に計算し、100MB バジェットの超過検出・アサーションを実装する。
  2. **MLM マスキング・Tied 語彙射影・重み付きマルチタスク損失の統合**:
     - `oniwa-decide` に BERT スタイル (15% トークンマスキング) の MLM 処理を追加。
     - Embedding と語彙射影の Weight Tying (共有) をサポートした MLM 射影層 (順伝播・逆伝播) を実装。
     - `LossConfig` / `LossCalculator` に `mlm_weight` および MLM Cross-Entropy 損失計算を追加し、マルチタスク損失 $\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{MLM}} + \beta \mathcal{L}_{\text{choice}} + \gamma \mathcal{L}_{\text{noul}} + \delta \mathcal{L}_{\text{score}}$ を統合。
  3. **Killer Pattern JSONL 高速インジェスチョン**:
     - 呼び出し側が提供するバッチスライス (caller-provided batch slice) にエンコード済みトークン・ラベルをゼロアロケーションで直接書き込むバリデーション付き JSONL パーサー/ローダーを実装。
  4. **テスト・後方互換性保証**:
     - 既存の Pure Rust BPE 実装 (`BpeTokenizer`) を温存しつつ、新規 MLM およびバジェット計算の単体テスト・勾配チェック・スモークテストを追加。
- **やらないこと (Non-Goals)**:
  - 既存の Pure Rust BPE トークナイザーの再実装。
  - 外部 ML フレームワーク (PyTorch, Candle 等) の依存追加 (Pure Rust 原則維持)。

## 2. 影響ファイル一覧
- `.agent/tasks/20260922-225500/task.md`: 本タスク指示書の永続化
- `.agent/tasks/20260922-225500/implement_plan.md`: メインエージェントの作業計画書
- `.agent/tasks/20260922-225500/walkthrough.md`: 検証結果レポート
- `crates/oniwa-decide/src/loss.rs`: MLM 損失計算関数の追加、`LossConfig` 拡張
- `crates/oniwa-decide/src/model.rs`: フラットメモリレイアウト算定 (`calculate_training_memory_bytes`, `assert_memory_budget`)、MLM マスキング & 語彙射影 (Forward/Backward) の統合
- `crates/oniwa-decide/src/dataset.rs`: Killer Pattern JSONL インジェスチョン (`ingest_killer_patterns_jsonl` / caller-provided batch slice 書き込み)
- `crates/oniwa-decide/src/lib.rs`: 新規型・API のエクスポート
- `crates/oniwa-decide/tests/decision_test.rs`: メモリバジェット・MLM 勾配チェック・JSONL インジェスチョンのテスト

## 3. 実装ステップ
1. **Phase 1: メモリレイアウト算定と 100MB バジェット検証**:
   - `DecisionConfig` に `calculate_training_memory_bytes(batch_size: usize) -> usize` を実装。
   - 100MB バジェット (`<= 100 * 1024 * 1024`) の検証関数とテストを追加。
2. **Phase 2: MLM 損失および Tied 語彙射影の実装**:
   - `LossConfig` に `mlm_weight` を追加 (`#[serde(default)]` 対応)。
   - `LossCalculator::mlm_loss` を実装。
   - `DecisionModel` に MLM 語彙射影 (`forward_mlm` / `backward_mlm`、または統合 forward/backward) を実装。
3. **Phase 3: Killer Pattern JSONL インジェスチョンの実装**:
   - JSONL 形式のキラーパターンデータセット定義とパーサーを実装。
   - 呼び出し元の事前割り当てバッファ (`&mut [u16]`, `&mut [usize]`, `&mut [bool]`, `&mut [f32]`) への直接書き込み。
4. **Phase 4: 単体テスト・統合テスト・Lint・CLI 動作確認**:
   - `cargo test --workspace`
   - `cargo clippy --workspace --all-targets -- -D warnings`
   - `cargo fmt --all -- --check`
