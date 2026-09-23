# [Task Plan / Issue Spec]: ログ・標準出力への短縮UTC日時付与および温度単位degCへの統一

## 1. Issue & 目的定義
- **背景と課題**:
  現在、`oniwa` リポジトリの構造化台帳（`growth_journal.jsonl` や `ledger_index.jsonl`）では ISO 8601 UTC（`timestamp_utc`）が記録されている一方で、学習・ベンチマーク実行時の標準出力（コンソール表示）やプレーンテキストログ（`*train*.log`）には日時情報が出力されていませんでした[cite: 1]。これにより、長時間のバックグラウンド学習監視や熱・損失プロファイルの時系列照合に支障をきたしていました[cite: 1]。
  また、コンソールおよびログにおける温度表記に `C` や `℃` が混在しており、工学的な単位表記としての厳密性と視認性の向上（`degC` への統一）が求められていました[cite: 1]。
- **解決方針とスコープ**:
  1. **Pure Rust による短縮 UTC 日時生成関数の提供**:
     - 外部クレート（`chrono` や `time` 等）を追加せず、`crates/oniwa-lm/src/logger.rs` に `std::time::SystemTime` からエポック秒・閏年を計算し `YYYY-MM-DD HH:MM:SS UTC` を返すフォーマッタ関数（`current_utc_display()`）を追加します[cite: 1]。
  2. **学習・ベンチマークの進捗ログへの日時プレフィックス付与**:
     - `oniwa-lm`（`train.rs`）および `oniwa-decide`（`train.rs`, `bench.rs` 等）の進捗ログ行の先頭に `[YYYY-MM-DD HH:MM:SS UTC]` を付与します[cite: 1]。
     - プレーンテキストログ（`*train*.log`）への書き込み時にも同プレフィックスを適用します[cite: 1]。
  3. **温度単位の `degC` 統一**:
     - 標準出力（コンソール表示）およびプレーンテキストログ内の温度単位表示（例: `72.4 C` や `[72.4℃]`）をすべて `degC`（例: `72.4 degC`）に統一します[cite: 1]。
     - JSON/JSONL などの機械可読な構造化ログのキー名（`thermal_celsius` や `cpu_temp_celsius`）は、既存のスキーマ・パーサー互換性を保つため維持します[cite: 1]。
- **やらないこと（Non-Goals）**:
  - `chrono` や `time` などの外部時刻ライブラリの追加（Pure Rust / ゼロ外部依存原則を厳格維持）[cite: 1]。
  - `chat` バイナリの対話ストリーミング出力や、`audit`、CLI パイプラインツールの全標準出力への一律日時付与（対話表示やパイプライン処理の汚染を防止）[cite: 1]。
  - JSON/JSONL の構造化キー名の変更（破壊的変更の防止）[cite: 1]。
  - 過去の完了タスク仕様書やベンチマーク静的レポート本文の過去ログ改変[cite: 1]。

## 2. エージェント実行体制・開発運用ルール
本指示を受け取ったエージェントは、以下の「Issueドリブン ＆ 階層型エージェント体制」を厳格に遵守して自走すること[cite: 1]。

### 保存ディレクトリとドキュメント作成ルール
本タスクのドキュメントは、対象 `task.md` と同階層に集約・保存すること[cite: 1]。
- **タスク指示書**: 既に保存され共有される想定[cite: 1]
- **計画ドキュメント**: `implement_plan.md` および日本語版 `implement_plan.ja.md` を作成・更新する[cite: 1]。
- **成果・振り返りドキュメント**: 実装完了後、全体の変更差分、セルフチェック結果、および検証結果を `walkthrough.md` および日本語版 `walkthrough.ja.md` として作成する[cite: 1]。

### 体制と責務
- **メインエージェント（司令塔）**:
  - 本指示書および関連する既存ソースコードを深く理解し、全体の進行管理と統合テストを担当する[cite: 1]。
  - タイムスタンプディレクトリ（`.agent/tasks/{yyyymmdd-hhmmss}/`）を作成し、計画書（`implement_plan.*`）と振り返り（`walkthrough.*`）の日英両バージョンを確実に作成・管理する[cite: 1]。
  - サブタスクごとに作業スコープを切り出し、サブエージェントへ明確な指示を渡す[cite: 1]。
  - **自己検証（セルフチェック）の責務**:
    - 全実装・テスト完了後、セクション7の「AIセルフチェックリスト」を1項目ずつ突き合わせて検証すること。
    - すべての項目を満たしていることを確認した結果を `walkthrough.md` にチェックログとして記録すること。未達の項目があれば完了とみなさず、サブエージェントに再指示を出して修正すること。
  - **Git コミット＆プッシュの責務**:
    - ソースコードの変更差分だけでなく、**`task.md` のディレクトリ `.agent/tasks/{yyyymmdd-hhmmss}/` 配下に生成されたすべてのドキュメント（task.md, implement_plan.*, walkthrough.*）を漏れなく `git add` の対象に含めること**。
    - 実装・テスト・フォーマット・セルフチェック・ドキュメント作成がすべて完了した後、一括してコミットおよびリモートへのプッシュを実行すること。
- **サブエージェント（実装担当）**:
  - メインエージェントから指示された特定モジュールの実装・テストコード作成に忠実に従う[cite: 1]。
  - 自身が担当した実装の詳細ステップと検証手順をサブエージェント視点のプラン/ログとして記録・更新する[cite: 1]。
  - **コードフォーマット規律**: 静的解析（Lint/型チェック）は実行せず、変更・新規作成したファイルに対して `isort` と `ruff format` によるフォーマットのみを適用すること（Rustファイルについては `cargo fmt --all` を適用）[cite: 1]。

## 3. 影響ファイル一覧
- `crates/oniwa-lm/src/logger.rs`: 短縮 UTC 文字列生成関数 `current_utc_display()` の実装、エクスポート、単体テスト追加[cite: 1]
- `crates/oniwa-lm/src/bin/train.rs`: 画面出力およびログ出力への短縮 UTC 日時プレフィックス付与、温度単位表示の `degC` 統一[cite: 1]
- `crates/oniwa-decide/src/bin/train.rs`: 画面出力およびログ出力への短縮 UTC 日時プレフィックス付与、温度単位表示の `degC` 統一[cite: 1]
- `crates/oniwa-decide/src/bin/bench.rs`: ベンチマーク画面出力における温度単位表示の `degC` 統一[cite: 1]
- `.agent/tasks/{yyyymmdd-hhmmss}/*`: [新規作成] タスク指示書（task.md）、計画書（implement_plan.*）、振り返り（walkthrough.*）

## 4. 実装ステップ（サブエージェントへの作業分割案）
1. Phase 1: タイムスタンプディレクトリの作成と初期ドキュメント配置
   - `.agent/tasks/{yyyymmdd-hhmmss}/` ディレクトリを作成。
   - 本仕様書を `task.md` として永続化配置。
   - `implement_plan.md` および `implement_plan.ja.md` を作成。
2. Phase 2: Pure Rust 短縮 UTC フォーマッタの実装
   - 対象ファイル: `crates/oniwa-lm/src/logger.rs`
   - 具体指示:
     - `std::time::SystemTime::now()` から UNIX エポック秒を取得し、外部依存なしで UTC 年・月・日・時・分・秒を算定する。
     - `YYYY-MM-DD HH:MM:SS UTC` 形式の `String` を返すパブリック関数 `current_utc_display() -> String` を実装。
     - 既知のエポック秒（例: 2026年9月の固定値）に対する単体テストを追加し、閏年や時刻計算の正確性を担保。
3. Phase 3: 標準出力・テキストログの日時付与 ＆ 温度単位 `degC` 統一
   - 対象ファイル: `crates/oniwa-lm/src/bin/train.rs`, `crates/oniwa-decide/src/bin/train.rs`, `crates/oniwa-decide/src/bin/bench.rs`
   - 具体指示:
     - `train.rs` の各ステップ・エポック進捗ログ出力に `[YYYY-MM-DD HH:MM:SS UTC]` を付与。
     - コンソール表示中の温度表記 `C` / `℃` をすべて `degC` に置換（例: `{:.1} degC`）。
     - `bench.rs` のテーブルヘッダーや出力における温度表記を `degC` に統一。
     - JSON/JSONL のシリアライズキー（`thermal_celsius` 等）は一切変更せず維持。
4. Phase 4: テスト実行・フォーマット適用
   - `cargo test --workspace` を実行し、全件 Green であることを確認。
   - 必要に応じてスモーク実行（`cargo run --release -p oniwa-lm --bin train -- --steps 2` 等）を行い、出力形式を目視検証。
   - `cargo fmt --all` を適用。Python ファイルに変更がある場合は `uv run isort .` および `uv run ruff format .` を適用。
5. Phase 5: AIセルフチェック ＆ walkthrough 作成
   - セクション7のセルフチェックリストを照合し、`walkthrough.md` および `walkthrough.ja.md` に結果を記録。
6. Phase 6: Git コミット・プッシュ
   - 変更したソースコードと `.agent/tasks/{yyyymmdd-hhmmss}/` 配下の全ドキュメントをステージングしてコミット・プッシュ。

## 5. エッジケースと制約事項
- **外部クレート追加の禁止**: `Cargo.toml` に `chrono` や `time` などを追加してはならない（Pure Rust 原則の維持）[cite: 1]。
- **ゼロアロケーション・高速計算**: ログフォーマット計算による学習ループへのオーバーヘッドを最小限に抑えること。
- **後方互換性の担保**: `TrainingStepLog` や `HardwareProfileRecord` などの構造化ログフィールド名（`*_celsius`）は改名しないこと[cite: 1]。
- **出力の可読性**: プレフィックスのフォーマット崩れや改行崩れを起こさないよう、既存の桁揃え・出力マクロと整合させること。

## 6. 完了判定・デプロイコマンド（検証・フォーマット・Git）
- コードフォーマット適用:
  - `cargo fmt --all`
  - `uv run isort .`
  - `uv run ruff format .`
- テスト実行:
  - `cargo test --workspace`
- 実機動作確認（スモークテスト）:
  - `cargo run --release -p oniwa-lm --bin train -- --steps 2`
- Git コミット ＆ プッシュ:
  - `git add crates/oniwa-lm/src/logger.rs crates/oniwa-lm/src/bin/train.rs crates/oniwa-decide/src/bin/train.rs crates/oniwa-decide/src/bin/bench.rs .agent/tasks/<生成されたタイムスタンプ>/`
  - `git commit -m "feat: ログ・標準出力への短縮UTC日時付与および温度単位degCへの統一 (#<issue番号>)"`
  - `git push origin <現在のブランチ名>`

## 7. AIセルフチェックリスト（実装完了判定）
エージェントはコミット前に以下のチェックを自律的に実施し、すべてクリアしていることを確認すること（walkthrough にチェック結果を記載すること）。
- [ ] **仕様準拠**: 標準出力および `*train*.log` の各行に短縮 UTC（`YYYY-MM-DD HH:MM:SS UTC`）が付与され、温度表示が `degC` に統一されているか？
- [ ] **スコープ厳守**: `chat` や CLI パイプライン、あるいは JSON のデータフィールド名（`*_celsius`）を誤って変更していないか？
- [ ] **影響範囲の一致**: セクション3の「影響ファイル一覧」以外の無関係なファイルを変更していないか？
- [ ] **Pure Rust 原則順守**: `Cargo.toml` に `chrono` や `time` などの新規依存が追加されていないか？
- [ ] **フォーマット順守**: `cargo fmt --all`（および必要に応じて `uv run isort .` / `uv run ruff format .`）を実行し、差分がない状態になっているか？
- [ ] **テスト通過**: `cargo test --workspace` が 100% エラーなく成功（Green）しているか？
- [ ] **ドキュメント網羅**: 日英両方の `implement_plan`（.md / .ja.md）および `walkthrough`（.md / .ja.md）がすべて指定ディレクトリ内に生成されているか？
- [ ] **Git対象の完全性**: ソースファイルに加えて `.agent/tasks/{yyyymmdd-hhmmss}/` 配下の全ファイルが `git add` の対象に含まれているか？