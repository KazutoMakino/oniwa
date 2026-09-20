# [Task Plan / Issue Spec]: oniwa-lm データ拡充および中規模スケールアップ（Phase 2: Scale v2 & Provenance Logging）

## 1. Issue & 目的定義
- **背景と課題**: 
  現在実行中の学習（PID 2447341 / Step 13,000+）において、Validation Loss が約 3.55、Top-5 Accuracy が約 57.8% で完全にプラトー化（サチり）しています[cite: 1]。分析の結果、データセット（`tokens.bin` 約 7.1MB / 350万トークン）を既に 3.5〜4 周以上周回（過学習・記憶の飽和）している点、および極小パラメータ構成（dim: 128, layers: 4, heads: 4, ffn: 256）と短い系列長（seq_len: 128）により、コード構文や多ドメイン知識を保持するキャパシティが枯渇していることが判明しました[cite: 1]。さらに、学習ログ（`growth_journal.jsonl`）にタイムスタンプが記録されておらず、時系列解析やエッジ熱プロファイルの追跡性が不足しています[cite: 1]。
- **解決方針とスコープ**: 
  データ・モデル容量・学習ハイパーパラメータ・ログ追跡性の4者を同時に整合させる「ハイブリッド・スケールアップ」を実施します。
  1. `pipelines` クレートを活用し、青空文庫（レシピ56作品）、e-Gov基本法令、オープンソース基本アルゴリズムコード（Python/Rust）を一括統合・拡充し、新規語彙 `vocab.json` と 1,500万〜2,000万トークン規模の `tokens.bin` を新規ビルドします[cite: 1]。
  2. モデル構成を Raspberry Pi 4 の L2 キャッシュ（1MB）および熱制約に最適化した中規模スケール（`dim: 192`, `layers: 6`, `heads: 6`, `ffn_dim: 384`, `seq_len: 256`）へ更新し、`QuaternionLMHead`（48クォータニオン）の制約を遵守します[cite: 1]。
  3. 既存の `checkpoints/quaternion/` や `checkpoints/best` との混同を防ぐため、チェックポイント保存先を `checkpoints/scale_v2/` 配下に完全隔離して Step 0 から再育成します[cite: 1]。
  4. 学習率スケジュールを新規トークン総数の最大 2 エポック分に連動させ、コサイン減衰を適用します。また、Validation Loss が 500 ステップ改善しない場合、または NaN/Inf 発生時の早期停止（Early Stopping）機構を実装します。
  5. `StepLog` および `growth_journal.jsonl` に ISO 8601 形式の UTC タイムスタンプ（`timestamp_utc`）フィールドを導入し、完全な出自追跡性を確立します[cite: 1]。
- **やらないこと（Non-Goals）**: 
  - 既存の旧世代チェックポイント（dim 128 / seq_len 128）からの重み部分移植やパディング（完全リセットによりProvenenceを維持）[cite: 1]。
  - Byte-pair encoding (BPE) 等の別トークナイザー実装への置換（文字単位 UTF-8 トークナイザーの枠組みを維持）[cite: 1]。
  - 外部深層学習フレームワーク（PyTorch/Candle等）への依存追加（Pure Rust 原則を厳格維持）[cite: 1]。

## 2. エージェント実行体制・開発運用ルール
本指示を受け取ったエージェントは、以下の「Issueドリブン ＆ 階層型エージェント体制」を厳格に遵守して自走すること[cite: 1]。

### 体制と責務
- **メインエージェント（司令塔）**:
  - 本指示書および関連する既存ソースコードを深く理解し、全体の進行管理と統合テストを担当する[cite: 1]。
  - リポジトリの `.agent/tasks/{yyyymmdd-hhmmss}/` ディレクトリに `implement_plan.md` を作成・管理する[cite: 1]。
  - サブタスクごとに作業スコープを切り出し、サブエージェントへ明確な指示を渡す[cite: 1]。
  - 実装完了後、全体の変更差分と動作検証結果をまとめた `walkthrough.md` を同ディレクトリに作成する[cite: 1]。
- **サブエージェント（実装担当）**:
  - メインエージェントから指示された特定モジュールの実装・テストコード作成に忠実に従う[cite: 1]。
  - 自身が担当した実装の詳細ステップと検証手順をサブエージェント視点のプラン/ログとして記録・更新する[cite: 1]。
  - コードフォーマット規律: 静的解析（Lint/型チェック）は実行せず、変更・新規作成したファイルに対して isort と ruff format によるフォーマットのみを適用すること（Rustファイルについては `cargo fmt --all` を適用）[cite: 1]。

## 3. 影響ファイル一覧
- `.agent/tasks/{yyyymmdd-hhmmss}/task.md`: [新規作成] 本タスク指示書の永続化配置[cite: 1]
- `.agent/tasks/{yyyymmdd-hhmmss}/implement_plan.md`: [新規作成] メインエージェントの作業実行計画書[cite: 1]
- `.agent/tasks/{yyyymmdd-hhmmss}/walkthrough.md`: [新規作成] 実装差分および動作検証結果レポート[cite: 1]
- `crates/oniwa-lm/src/logger.rs`: `StepLog` 構造体への `timestamp_utc` フィールドの追加、およびシリアライズ対応[cite: 1]
- `crates/oniwa-lm/src/model.rs`: デフォルト構成または `ModelConfig::scale_v2()` プリセット（dim: 192, layers: 6, heads: 6, ffn: 384, seq_len: 256）の追加、および `QuaternionModelWeights` の 4 の倍数アサーション維持[cite: 1]
- `crates/oniwa-lm/src/bin/train.rs`: 
  - `--scale-v2` CLI フラグの追加（デフォルト構成の上書き、チェックポイント保存先を `checkpoints/scale_v2/` に設定）
  - エポック上限連動コサイン減衰計算ロジックの実装
  - Validation Loss 早期停止（Early Stopping: 500ステップ猶予）および NaN/Inf 検出安全停止の実装
  - ログ出力時の `timestamp_utc` 付与
- `pipelines/src/main.rs`: 統合データセット再構築コマンド（全青空文庫レシピ ＋ e-Gov法令 ＋ コード群）の検証と再生成パスの整備[cite: 1]

## 4. 実装ステップ（サブエージェントへの作業分割案）
1. Phase 1: ログ基盤の拡張（タイムスタンプ追加）
   - 対象ファイル: `crates/oniwa-lm/src/logger.rs`
   - 具体指示:
     - `StepLog` 構造体に `pub timestamp_utc: String` を追加する。
     - `current_timestamp_utc()` ユーティリティを利用して、各ステップのログ生成時に現在時刻（ISO 8601 形式）を自動注入する[cite: 1]。
     - 既存ログファイル読み込み時の後方互換性のため、`#[serde(default)]` を設定する[cite: 1]。
     - 単体テストで `StepLog` のシリアライズ・デシリアライズにタイムスタンプが含まれることを確認する。
2. Phase 2: モデル構成プリセット（Scale v2）と安全停止機構の実装
   - 対象ファイル: `crates/oniwa-lm/src/model.rs`, `crates/oniwa-lm/src/bin/train.rs`
   - 具体指示:
     - `ModelConfig` に `scale_v2(vocab_size: usize) -> Self` 関数を定義（`dim: 192`, `num_layers: 6`, `num_heads: 6`, `head_dim: 32`, `ffn_dim: 384`, `seq_len: 256`）。
     - `train.rs` に `--scale-v2` フラグを新設し、有効時は本構成で初期化を行う。チェックポイント保存先を `checkpoints/scale_v2/` 配下に設定する。
     - 早期停止コントローラーを実装: 最良 Validation Loss が 500 ステップ更新されない場合、または Loss が NaN/Inf に発散した場合に即座にループを脱出し、最良モデルを保存して正常終了する。
     - 学習率スケジュールを引数またはトークン総数から算定される総ステップ数に合わせて正規化減衰させる。
3. Phase 3: データパイプライン実行・テスト・実機スモーク検証
   - 対象ファイル: `pipelines/`, `crates/oniwa-lm/src/bin/train.rs`, `crates/oniwa-lm/src/bin/chat.rs`
   - 具体指示:
     - `cargo run --release -p oniwa-pipeline -- --all` を実行し、全コーパスから `data/vocab.json` と `data/tokens.bin` を再生成する[cite: 1]。
     - `cargo test --workspace` を実行し、既存の全単体テスト（NEON SIMD、クォータニオン勾配チェック、RMSNorm等）が Green であることを確認する[cite: 1]。
     - スモークテスト実行:
       `cargo run --release -p oniwa-lm --bin train -- --quaternion --scale-v2 --reset --steps 10`
     - ログファイルに `timestamp_utc` が正しく記録され、`checkpoints/scale_v2/` にチェックポイントが正常保存されることを確認する。
     - チャット推論テスト:
       `cargo run --release -p oniwa-lm --bin chat -- --quaternion --checkpoint-dir crates/oniwa-lm/checkpoints/scale_v2/best`

## 5. エッジケースと制約事項
- **クォータニオン次元アサーション**: `dim: 192` は $192 / 4 = 48$ クォータニオン、各ヘッド次元は $192 / 6 = 32$（8クォータニオン）となり、クォータニオン代数の 4 分割要件を厳密に満たすこと[cite: 1]。初期化時に明示的にアサートすること[cite: 1]。
- **チェックポイントの完全隔離**: 以前の 128 次元チェックポイント（`checkpoints/best` や `checkpoints/quaternion/`）を絶対に上書き・混同して読み込ませないこと[cite: 1]。メタデータ内の `dim` および `"model_type": "quaternion_tied"` の検証によって安全に弾くこと[cite: 1]。
- **熱暴走ガード**: `ThermalController` のスロットリング（70℃/78℃）は引き続き学習ループの各ステップ末尾で呼び出し、計算負荷増に伴うハードウェア損傷を未然に防止すること[cite: 1]。
- **ゼロアロケーション維持**: 系列長が 256 に伸長しても、1ステップあたりのアリーナバッファ事前確保規約を維持し、ループ内でのヒープ動的確保（malloc）を発生させないこと[cite: 1]。

## 6. 完了判定コマンド（検証・フォーマット）
- コードフォーマット適用:
  - uv run isort .
  - uv run ruff format .
  - cargo fmt --all
- 自動テスト実行:
  - cargo test --workspace
- スモーク学習検証:
  - cargo run --release -p oniwa-lm --bin train -- --quaternion --scale-v2 --reset --steps 10
- ログ検証:
  - head -n 5 logs/runs/.../growth_journal.jsonl で timestamp_utc の存在を確認