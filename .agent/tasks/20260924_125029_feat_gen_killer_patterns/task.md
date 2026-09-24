# [Task Plan / Issue Spec]: oniwa 連続自走タスク: キラーパターン対照変異生成器の実装 ＆ Rust標準命名規則・慣例への全面リファクタリング

## 1. Issue & 目的定義
- **背景と課題**: 
  1. `oniwa-decide`（System 1）を用いた Gatekeeper CLI の実機検証において、構文破壊コードに対する異常検知ヘッド（`Noul`）の確信度が約 2.1%（迷い状態）にとどまり、異常コードが素通し（PASS）される課題が確認されました。これはモデルアーキテクチャの欠陥ではなく、学習データ中に「何が異常なコード・文章構造なのか」を示す明確な教師信号（`Noul=true` の負例データ）が不足していることが根本原因です。四元数バックボーン（Full Q-Transformer）の本格訓練に進む前に、同一の高品質な対照データセットを自動合成・固定する仕組みが不可欠です。
  2. また、コードベースおよびドキュメント全体において、Rust 公式 API ガイドライン（RFC 430）に準拠しない命名や慣例の揺らぎ（バイナリ名の `kebab-case` / `snake_case` 混在、単純ゲッターの命名、記載コマンドの乖離等）が存在しています。過去の学習済みチェックポイント互換性を無理に維持しようとするとコードが複雑化するため、完全リセット前提で全レイヤーの命名・慣例を全面的に整理し、洗練された Rust 実装へと統一する必要があります。
  3. これらを別々のブランチ・Issue で安全に分離しつつ、連続して自律的に完遂させる必要があります。
- **解決方針とスコープ**: 
  本タスクは以下の **2段階シーケンシャル・ワークフロー** により、独立したブランチと PR を経て順次実行します。
  - **[Step 1: feat] キラーパターン対照変異生成器（`gen-killer-patterns`）の実装**:
    - `pipelines/src/bin/gen-killer-patterns.rs` を新設し、統合コーパスから正常スライス（`Noul=false`）と1箇所変異させた破壊スライス（`Noul=true`）の 1:1 対照ペアを決定論的に自動合成（デフォルト 10,000 件 / 5,000 ペア）。
    - トークンレベルで変異を適用し、`mask_indices`（昇順ソート・重複なし）を正確に追跡して `KillerPatternDataset` バリデーション往復テストをパス。
    - 独立ブランチにて実装、テスト、コミット、PR作成、スカッシュマージ、main同期を完了。
  - **[Step 2: refactor] Rust 標準命名規則・慣例への全面整合およびドキュメント同期**:
    - Step 1 完了後の最新 `main` から新ブランチを作成。
    - バイナリターゲット名およびファイル名を `kebab-case` に統一（例: `compress_eval.rs` -> `compress-eval.rs` 等）。
    - 単純ゲッターの `get_` プレフィックス排除、コンストラクタ・変換トレイトの慣例遵守。
    - 過去チェックポイント互換性は考慮せず完全リセット（クリーンな最新型定義へ刷新）。
    - ルートおよび各クレートの README、解説文書内の実行コマンド表記を改訂後バイナリ名と 100% 同期（`docs/benchmarks/*` の静的実測記録は歴史的温存）。
    - テスト全件通過（Green）および Clippy 警告ゼロを確認し、独立ブランチにてコミット、PR作成、スカッシュマージ、main同期を完了。
- **やらないこと（Non-Goals）**: 
  - 2つのタスクを同一ブランチで混同して実装・コミットすること（必ず 1 タスクずつ独立して完結させる）。
  - 外部 LLM API を用いた合成データ生成（他社プロプライエタリモデル排除の原則維持）。
  - 過去チェックポイントの後方互換維持（完全リセット）。
  - `.githooks/pre-commit` の改変（既存フックは温存）。
  - `docs/benchmarks/*` 内の歴史的実測数値の書き換え。

## 2. エージェント実行体制・開発運用ルール
本指示を受け取ったエージェントは、以下の「Issueドリブン ＆ 階層型エージェント体制」を厳格に遵守して自走すること。

### 保存ディレクトリとドキュメント作成ルール
各ステップのドキュメントは、それぞれ専用のタイムスタンプディレクトリ（`.agent/tasks/{タイムスタンプ}/`）に集約・保存すること。
- **タスク指示書**: 実行開始時に本仕様を `task.md` として配置。
- **計画ドキュメント**: `implement_plan.md` および日本語版 `implement_plan.ja.md` を作成・更新。
- **成果・振り返りドキュメント**: 実装完了後、全体の変更差分、セルフチェック結果、検証結果を `walkthrough.md` および日本語版 `walkthrough.ja.md` として作成。

### 体制と責務
- **メインエージェント（司令塔）**:
  - 本指示書を理解し、Step 1 と Step 2 を順番に実行・管理する。
  - 各ステップごとに Issue 作成 -> ブランチ作成 -> 実装 -> テスト -> セルフチェック -> コミット -> PR作成 -> マージ -> main同期 のライフサイクルを完遂する。
  - **Git コミット＆プッシュの責務**:
    - ソースコードの変更差分だけでなく、**`.agent/tasks/{タイムスタンプ}/` 配下に生成された全ドキュメント（task.md, implement_plan.*, walkthrough.*）を漏れなく `git add` の対象に含めること**。
    - 各ステップ完了ごとに一括コミット・プッシュを実行すること。
- **サブエージェント（実装担当）**:
  - 指示されたモジュールの実装・テストコード作成に忠実に従う。
  - **コードフォーマット規律**: 静的解析（Lint/型チェック）は実行せず、`cargo fmt --all`（Pythonファイルがある場合は `uv run isort .` と `uv run ruff format .`）によるフォーマットのみを適用すること。

## 3. 影響ファイル一覧

### Step 1 (feat: キラーパターン対照変異生成器)
- `pipelines/src/bin/gen-killer-patterns.rs`: [新規作成] 対照変異データセットを合成する CLI ツール
- `crates/oniwa-decide/tests/decision_test.rs`: [変更] 生成された JSONL の `KillerPatternDataset` 読み込みおよび `write_batch` 往復検証テスト
- `.agent/tasks/{タイムスタンプ_step1}/*`: [新規作成] Step 1 用タスク文書群

### Step 2 (refactor: Rust標準命名規則・慣例の全面整合)
- `crates/oniwa-decide/src/bin/compress-eval.rs`: [`compress_eval.rs` からリネーム改修]
- `crates/oniwa-decide/Cargo.toml`: [バイナリターゲット名の更新・整合]
- `crates/oniwa-decide/src/**/*.rs`: [メソッド名・型名・構造体フィールドの慣例準拠改修]
- `crates/oniwa-lm/src/**/*.rs`: [メソッド名・型名・構造体フィールドの慣例準拠改修]
- `pipelines/src/**/*.rs`: [命名規約・慣習の準拠改修]
- `crates/**/tests/**/*.rs`: [テスト呼び出しコードの整合]
- `README.md`, `README.ja.md`: [最新コマンド・バイナリ名の同期更新]
- `crates/oniwa-decide/README.md`, `crates/oniwa-decide/README.ja.md`: [最新コマンド表記の同期更新]
- `pipelines/README.md`, `pipelines/README.ja.md`: [最新コマンド表記の同期更新]
- `.agent/tasks/{タイムスタンプ_step2}/*`: [新規作成] Step 2 用タスク文書群

## 4. 実装ステップ（サブエージェントへの作業分割案）

### 【Step 1 実行フェーズ】
1. **Phase 1-1: 準備 & ブランチ作成**
   - GitHub Issue を作成: `gh issue create --title "feat: キラーパターン対照変異生成器（gen-killer-patterns）の実装"`
   - ブランチ作成: `git checkout -b <issue番号>/feat/gen-killer-patterns`
   - `.agent/tasks/{タイムスタンプ}/` ディレクトリを作成し、タスク指示書および `implement_plan.*` を配置。
2. **Phase 1-2: 変異生成ロジックの実装**
   - `pipelines/src/bin/gen-killer-patterns.rs` を実装。
   - 引数: `--corpus`, `--output`, `--pairs` (デフォルト5000), `--seed` (デフォルト42)。
   - トークンレベルで 1 箇所変異を適用し、`Noul=true, Score=3.5~4.5`、`mask_indices` に変異位置トークンインデックスを昇順・ユニークで記録。
3. **Phase 1-3: 検証テスト & データ生成**
   - `crates/oniwa-decide/tests/decision_test.rs` に `KillerPatternDataset::from_jsonl_reader` および `write_batch` 往復テストを追加。
   - スモーク実行: `cargo run --release -p oniwa-pipeline --bin gen-killer-patterns -- --pairs 5000 --output data/killer_patterns.jsonl`
   - `cargo test --workspace` が Green であることを確認。
4. **Phase 1-4: フォーマット・セルフチェック・コミット & マージ**
   - `cargo fmt --all` を適用。
   - セルフチェックを実施し、`walkthrough.*` を作成。
   - `git add pipelines/src/bin/gen-killer-patterns.rs crates/oniwa-decide/tests/decision_test.rs .agent/tasks/{タイムスタンプ}/`
   - `git commit -m "feat: キラーパターン対照変異生成器（gen-killer-patterns）の実装 (#<issue番号>)"`
   - `git push -u origin <ブランチ名>`
   - `gh pr create` -> `gh pr merge --squash --delete-branch`
   - `git checkout main && git pull origin main && git fetch --prune` で最新同期。

---

### 【Step 2 実行フェーズ】
1. **Phase 2-1: 準備 & ブランチ作成**
   - GitHub Issue を作成: `gh issue create --title "refactor: Rust標準命名規則・慣例への全面整合およびドキュメント同期"`
   - ブランチ作成: `git checkout -b <issue番号>/refactor/rust-idiomatic-naming-alignment`
   - 新規タイムスタンプディレクトリ `.agent/tasks/{新タイムスタンプ}/` を作成し、タスク指示書および `implement_plan.*` を配置。
2. **Phase 2-2: バイナリ名・ファイル名リネーム**
   - `crates/oniwa-decide/src/bin/compress_eval.rs` -> `compress-eval.rs`
   - `Cargo.toml` のバイナリ設定を `kebab-case` に整合。
3. **Phase 2-3: Rust API ガイドライン準拠リファクタリング**
   - 全クレートの関数、メソッド（`get_` プレフィックス排除等）、列挙子、構造体フィールドの命名を標準規約に改修。
   - 過去チェックポイント互換性は考慮せず完全リセット。
4. **Phase 2-4: テスト & ドキュメント同期**
   - テストコードの呼び出しを整合し、`cargo test --workspace` を全件パス。
   - `cargo clippy --workspace --all-targets -- -D warnings` が警告 0 件であることを確認。
   - 全 README およびマニュアルのコマンド記載を最新バイナリ名に同期。
5. **Phase 2-5: フォーマット・セルフチェック・コミット & マージ**
   - `cargo fmt --all` を適用。
   - セルフチェックを実施し、`walkthrough.*` を作成。
   - `git add <変更ファイル群> .agent/tasks/{新タイムスタンプ}/`
   - `git commit -m "refactor: Rust標準命名規則・慣例への全面整合およびドキュメント同期 (#<issue番号>)"`
   - `git push -u origin <ブランチ名>`
   - `gh pr create` -> `gh pr merge --squash --delete-branch`
   - `git checkout main && git pull origin main && git fetch --prune` でクリーンアップ。

## 5. エッジケースと制約事項
- **ブランチの厳格な分離**: Step 1 の PR がマージされて `main` に反映されるまで、絶対に Step 2 のブランチ作業を開始しないこと。
- **インデックス外クラッシュの完全防止**: 変異適用後のトークン長に対し、`mask_indices` の全値が `index < written_tokens` かつ昇順・ユニークであることを厳密に保証すること。
- **静的レポートの保全**: `docs/benchmarks/` 配下の実測数値は改ざんせず、README やマニュアル側の実行コマンド記述のみを改訂すること。
- **Pure Rust 原則の維持**: 外部の Python スクリプトや機械学習フレームワークを追加しないこと。

## 6. 完了判定・デプロイコマンド（検証・フォーマット・Git）
- 各ステップでのコードフォーマット適用（rustソースを含む場合のみ）:
  - `cargo fmt --all`
- 各ステップでのコードフォーマット適用（pythonソースを含む場合のみ）:
  - `uv run isort .`
  - `uv run ruff format .`
- 各ステップでのテスト実行:
  - `cargo test --workspace`
- Step 2 での静的解析:
  - `cargo clippy --workspace --all-targets -- -D warnings`
- 最終完了状態:
  - `main` ブランチが最新かつクリーンであり、差分なし（`git status` が clean）であること。

## 7. AIセルフチェックリスト（各ステップの実装完了判定）
エージェントは各ステップのコミット前に以下のチェックを自律的に実施し、すべてクリアしていることを確認すること（walkthrough にチェック結果を記載すること）。
- [ ] **独立ライフサイクル順守**: Step 1 と Step 2 が別々の Issue、ブランチ、PR でマージされているか？
- [ ] **仕様準拠**: `gen-killer-patterns` が対照ペアを生成し、Step 2 でバイナリ名やメソッド名が Rust ガイドラインに準拠しているか？
- [ ] **ドキュメント同期**: README や解説文書内の実行コマンドが改訂後のバイナリ名と 100% 一致しているか？
- [ ] **スコープ厳守**: 過去ベンチマークレポートの歴史的実測数値を不当に書き換えていないか？
- [ ] **Pure Rust 原則順守**: `Cargo.toml` に不要な外部依存が追加されていないか？
- [ ] **フォーマット順守**: `cargo fmt --all` を実行し、差分がない状態になっているか？
- [ ] **Clippy 検証通過**: `cargo clippy --workspace --all-targets -- -D warnings` が警告 0 件で通過しているか？
- [ ] **テスト通過**: `cargo test --workspace` が 100% エラーなく成功（Green）しているか？
- [ ] **ドキュメント網羅**: 両ステップの `implement_plan`（.md / .ja.md）および `walkthrough`（.md / .ja.md）がそれぞれ指定ディレクトリ内に生成されているか？
- [ ] **Git対象の完全性**: ソースファイルに加えて各ステップの `.agent/tasks/{タイムスタンプ}/` 配下の全ファイルが `git add` の対象に含まれているか？
