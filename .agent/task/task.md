# [Task Plan / Issue Spec]: 最新仕様（10_system_integration_roadmap）に基づくドキュメント体系の整合・最新化および計算並行安全運用の確立

## 1. Issue & 目的定義
- **背景と課題**: 
  現在、コードベース側ではロードマップ（`docs/10_system_integration_roadmap.ja.md`）の Phase 1（Pure Rust BPE 4,096語彙トークナイザー `train-bpe`）および Phase 2（MLM統合、キラーパターンインジェスチョン、フラットメモリ100MB制約計算）の実装が完了しています。しかし、ルートの `README` や各クレートのドキュメントには「文字単位（Char-level）トークナイザー前提」や「初期の極小モデル構成」の記述が残っており、リポジトリ内外の認識に齟齬が生じています。
  また、今後のドキュメント再整理頻発を防ぐため「物理アーカイブ（old/等）を作らず、最新版を1本のみ維持して履歴はGit履歴/タグに委ねる」方針を確立しつつ、現在ホスト上で稼働中の長時間計算・学習プロセスへ一切干渉しない安全な実行手順を敷く必要があります。
- **解決方針とスコープ**: 
  1. プロジェクトの「顔」となる主要ドキュメント（ルート `README.md` / `README.ja.md`、`pipelines/README.md` / `pipelines/README.ja.md`、`crates/oniwa-lm/docs/README.md` / `crates/oniwa-lm/docs/README.ja.md`）を日英完全同期で全面改訂し、BPE 4K、MLM、100MBバジェット、クォータニオン等の確定仕様と実行コマンドを反映します。
  2. 現在 README が存在しない `crates/oniwa-decide/` に、System 1 の役割・MLM・100MB制約・キラーパターン・クォータニオン判定エンジンを解説する `README.md` および `README.ja.md` を新設します。
  3. 既存の相対リンク切れを防ぐため現行ファイル名を維持しつつ、`docs/10_system_integration_roadmap.ja.md` を「最新実用化・システム統合ロードマップの正本（Single Source of Truth）」、`docs/09_roadmap.*` を「学術クォータニオン研究書」として位置づけを明確化します。
  4. 過去の設計書（`docs/00`〜`08`、`docs/09_roadmap.*`、`docs/HANDOVER_RASPBERRY_PI.*`）の冒頭に、最新ロードマップ（`docs/10_system_integration_roadmap.ja.md`）への誘導Banner注記を統一配置し、歴史的経緯を温存しながら仕様の混乱を防止します。
  5. **計算安全優先ルール**: 稼働中の計算プロセスへのCPU・メモリ競合やプロセス停止を防ぐため、`cargo test` や `cargo build` などの高負荷ビルド・テストコマンドは一切実行せず、純粋なMarkdown更新・フォーマット検証・限定的なGitステージングのみで完結させます。
- **やらないこと（Non-Goals）**: 
  - Rustコード（`*.rs`）やテストコードの機能変更・リファクタリング。
  - `cargo test --workspace` や学習実行などの高負荷コマンド実行（計算プロセス保護）。
  - `docs/old/` や `docs/archive/` などの物理アーカイブディレクトリの新設およびファイル移動（Git履歴とタグで管理）。
  - 過去の実測検証レポート（`docs/benchmarks/*`）や設計仮説書（`docs/design/*`）の本文改変。
  - 歴史的設計ドキュメント（`docs/00`〜`08`）の本文全体の書き換え（ヘッダー注記のみ付与）。

## 2. エージェント実行体制・開発運用ルール
本指示を受け取ったエージェントは、以下の「Issueドリブン ＆ 階層型エージェント体制」を厳格に遵守して自走すること。

### 保存ディレクトリとドキュメント作成ルール
本タスクの指示内容および進行記録は、すべて **`.agent/task/`** 配下に保存すること。
- **タスク指示書**: 実行開始時に本仕様を `.agent/task/task.md` として保存する。
- **計画ドキュメント**: `.agent/task/implement_plan.md` および日本語版 `.agent/task/implement_plan.ja.md` を作成・更新する。
- **成果・振り返りドキュメント**: 実装完了後、全体の変更差分と検証結果を `.agent/task/walkthrough.md` および日本語版 `.agent/task/walkthrough.ja.md` として作成する。

### 体制と責務
- **メインエージェント（司令塔）**:
  - 本指示書および関連する既存ドキュメントを深く理解し、全体の進行管理と整合性確認を担当する。
  - 上記の計画書（`implement_plan`）と振り返り（`walkthrough`）の日英両バージョンを確実に `.agent/task/` 内に作成・管理する。
  - サブタスクごとに作業スコープを切り出し、サブエージェントへ明確な指示を渡す。
  - **Git コミット＆プッシュの責務**:
    - **計算プロセス保護**: 現在出力されているログファイル（`logs/` 等）やチェックポイントファイル（`checkpoints/` 等）を誤ってステージングしないよう、更新対象のMarkdownファイル群および `.agent/task/` のみを明示的・ピンポイントに `git add` すること。
    - 編集・フォーマット・ドキュメント作成がすべて完了した後、一括してコミットおよびリモートへのプッシュを実行すること。
- **サブエージェント（実装担当）**:
  - メインエージェントから指示された特定ドキュメント群の更新・作成作業に忠実に従う。
  - 自身が担当した実装の詳細ステップと検証手順をサブエージェント視点のプラン/ログとして記録・更新する。
  - **コードフォーマット規律**: 静的解析（Lint/型チェック）は実行せず、変更・新規作成したファイルに対して `isort` と `ruff format` によるフォーマットのみを適用すること（Markdownファイルについては構文崩れのないように整形）。

## 3. 影響ファイル一覧
- `README.md`: [改訂] BPE 4K、MLM、100MBメモリ制約、クォータニオン、Phase 1〜2完了状況の反映（英語）
- `README.ja.md`: [改訂] 同上（日本語）
- `pipelines/README.md`: [改訂] BPE 4,096 語彙構築（`train-bpe`）フローおよび最新コーパス収集手順の反映（英語）
- `pipelines/README.ja.md`: [改訂] 同上（日本語）
- `crates/oniwa-decide/README.md`: [新規作成] System 1 のMLM、キラーパターン、フラットメモリ100MB制約、クォータニオン判定エンジンの技術解説（英語）
- `crates/oniwa-decide/README.ja.md`: [新規作成] 同上（日本語）
- `crates/oniwa-lm/docs/README.md`: [改訂] `10_system_integration_roadmap` を含む全体ドキュメントインデックスの更新（英語）
- `crates/oniwa-lm/docs/README.ja.md`: [改訂] 同上（日本語）
- `docs/00_oniwa-project-manifesto.md` / `.ja.md`: [改訂] 冒頭に最新ロードマップ誘導Banner注記を追加
- `docs/01_llm_c_architecture.md` / `.ja.md`: [改訂] 冒頭に最新ロードマップ誘導Banner注記を追加
- `docs/02_modern_primitives.md` / `.ja.md`: [改訂] 冒頭に最新ロードマップ誘導Banner注記を追加
- `docs/03_rust_memory_model.md` / `.ja.md`: [改訂] 冒頭に最新ロードマップ誘導Banner注記を追加
- `docs/04_development_handoff.md` / `.ja.md`: [改訂] 冒頭に最新ロードマップ誘導Banner注記を追加
- `docs/05_reproducibility_and_logging.md` / `.ja.md`: [改訂] 冒頭に最新ロードマップ誘導Banner注記を追加
- `docs/06_thermal_and_end_to_end_provenance.md` / `.ja.md`: [改訂] 冒頭に最新ロードマップ誘導Banner注記を追加
- `docs/07_data_ingestion_charter.md` / `.ja.md`: [改訂] 冒頭に最新ロードマップ誘導Banner注記を追加
- `docs/08_green_energy_and_power_tracking.md` / `.ja.md`: [改訂] 冒頭に最新ロードマップ誘導Banner注記を追加
- `docs/09_roadmap.md` / `.ja.md`: [改訂] 冒頭に最新ロードマップ誘導Banner注記を追加（学術クォータニオン研究書として位置づけ明記）
- `docs/HANDOVER_RASPBERRY_PI.md` / `.ja.md`: [改訂] 冒頭に最新ロードマップ誘導Banner注記を追加
- `.agent/task/*`: [新規作成] タスク指示書（task.md）、計画書（implement_plan.*）、振り返り（walkthrough.*）

## 4. 実装ステップ（サブエージェントへの作業分割案）
1. Phase 1: タスク・計画ドキュメントの初期配置
   - `.agent/task/` ディレクトリを作成。
   - 本仕様書を `.agent/task/task.md` として配置。
   - `implement_plan.md` および `implement_plan.ja.md` を作成。
2. Phase 2: 主要ドキュメントの改訂および新規作成
   - ルート `README.md` / `README.ja.md` の改訂: BPE 4K、MLM、100MBバジェット、クォータニオン代数、現在の進捗ステータスを同期。最新ロードマップ正本として `docs/10_system_integration_roadmap.ja.md` へのリンクを明記。
   - `pipelines/README.md` / `pipelines/README.ja.md` の改訂: `cargo run -p oniwa-pipeline --bin train-bpe -- --input ...` の解説を追加。
   - `crates/oniwa-decide/README.md` / `crates/oniwa-decide/README.ja.md` の新設: アーキテクチャ、マルチタスク損失、100MBバジェット検証を解説。
   - `crates/oniwa-lm/docs/README.md` / `crates/oniwa-lm/docs/README.ja.md` のインデックス更新。
3. Phase 3: 過去設計書への誘導Banner注記の挿入
   - `docs/00`〜`09` および `docs/HANDOVER_RASPBERRY_PI`（日英両ファイル）の先頭に、以下の共通Bannerを挿入。
     - 日本語版:
       ```markdown
       > [!NOTE]
       > **最新仕様について**: 本ドキュメントはプロジェクト初期の設計・構想書です。現在の確定アーキテクチャ（BPEトークナイザー、MLMマルチタスク学習、100MBメモリ制約、クォータニオン代数統合）および開発マイルストーンについては、最新の正本である [10. ONIWA 実用化ロードマップ仕様書](10_system_integration_roadmap.ja.md) を参照してください。過去の版や経緯はGitのコミット履歴・タグにて追跡されています。
       ```
     - 英語版:
       ```markdown
       > [!NOTE]
       > **Current Specification**: This document reflects the foundational/historical design. For current production specifications (Byte-level BPE, MLM multi-task learning, 100MB memory budget, and quaternion algebra integration), please refer to the latest primary specification: [10. ONIWA Production Roadmap Specification](10_system_integration_roadmap.md). Historical revisions are tracked via Git commit history and release tags.
       ```
4. Phase 4: フォーマット適用・walkthrough 作成 ＆ 計算安全 Git コミット・プッシュ
   - `isort` と `ruff format` を実行（Pythonスクリプトが存在する場合の整合性維持）。
   - `.agent/task/walkthrough.md` および `.agent/task/walkthrough.ja.md` を作成。
   - 変更したMarkdownファイル群および `.agent/task/` 配下を明示的に指定して `git add`、コミット、プッシュを実行。

## 5. エッジケースと制約事項
- **計算プロセスの不可侵**: `cargo test`, `cargo build`, `cargo run` 等のコンパイル・実行コマンドは並行稼働中の計算プロセス保護のため一切実行しない。
- **物理アーカイブ運用の禁止**: 古いロードマップや設計書を `docs/old/` や `archive/` に退避させず、最新1本（正本）を更新し、履歴はGitに委ねるルールを厳守する。
- **動的ファイルのステージング除外**: `git add .` や `git add -A` を禁止し、計算プロセスが出力中の `logs/` や `checkpoints/` をコミットに巻き込まない。
- **ベンチマーク・仮説書の除外**: `docs/benchmarks/` 配下および `docs/design/system-one-hypotheses.*` にはBannerを挿入せず、当時の実測値・理論仮説としての独立性を保持する。
- **相対リンクの整合性**: 挿入するBannerやREADME内のリンクパス（`../`, `../../` 等の相対階層）が破損しないよう厳密に検証する。
- **日英完全一致**: 日本語版のみを更新して英語版を放置しないよう、すべての改訂・新設・注記付与をペアで実施する。

## 6. 完了判定・デプロイコマンド（検証・フォーマット・Git）
- コードフォーマット適用:
  - `uv run isort .`
  - `uv run ruff format .`
- 計算安全 Git コミット ＆ プッシュ:
  - `git add README.md README.ja.md pipelines/README.md pipelines/README.ja.md crates/oniwa-decide/README.md crates/oniwa-decide/README.ja.md crates/oniwa-lm/docs/README.md crates/oniwa-lm/docs/README.ja.md docs/00_oniwa-project-manifesto.md docs/00_oniwa-project-manifesto.ja.md docs/01_llm_c_architecture.md docs/01_llm_c_architecture.ja.md docs/02_modern_primitives.md docs/02_modern_primitives.ja.md docs/03_rust_memory_model.md docs/03_rust_memory_model.ja.md docs/04_development_handoff.md docs/04_development_handoff.ja.md docs/05_reproducibility_and_logging.md docs/05_reproducibility_and_logging.ja.md docs/06_thermal_and_end_to_end_provenance.md docs/06_thermal_and_end_to_end_provenance.ja.md docs/07_data_ingestion_charter.md docs/07_data_ingestion_charter.ja.md docs/08_green_energy_and_power_tracking.md docs/08_green_energy_and_power_tracking.ja.md docs/09_roadmap.md docs/09_roadmap.ja.md docs/HANDOVER_RASPBERRY_PI.md docs/HANDOVER_RASPBERRY_PI.ja.md .agent/task/`
  - `git commit -m "docs: 最新仕様（10_system_integration_roadmap）に基づくドキュメント体系の整合・最新化 (#<issue番号>)"`
  - `git push origin <現在のブランチ名>`