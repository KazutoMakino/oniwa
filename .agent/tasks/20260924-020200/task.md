# [Task Plan / Issue Spec]: oniwa-decide 論文実証用スタンドアロン門番エンジン（Gatekeeper CLI）の実装

## 1. Issue & 目的定義
- **背景と課題**: 
現在 `oniwa-decide`（System 1）では 8,000 steps の学習が完了し、非自己回帰意思決定や MLM、キラーパターン検知の能力が獲得されています。しかし、これを開発現場の実用タスク（Git 差分走査による構文・論理異常の即時検知）として動かすためのインターフェースが存在しませんでした。また、先行の商用ツール（Jev等）と異なり、ONIWA は「100%出自追跡可能なクリーンデータ」「Pure Rust」「完全オフライン」「四元数（クォータニオン）代数による非可換表現」を特徴としており、これらの優位性を arXiv 論文等の学術データとして実証・記録する仕組みが不可欠です。
- **解決方針とスコープ**: 
1. `crates/oniwa-decide/src/bin/gatekeeper.rs` を新設し、Git ステージング差分（`git diff --cached`）または指定差分パッチをミリ秒オーダーで走査するスタンドアロン門番バイナリを実装します。
2. 対象拡張子を `.rs`, `.py`（ソースコード）および `.md`, `.txt`（文書・仕様書）に限定し、バイナリファイルや lock ファイル等を安全に除外します。
3. 差分の追加行（`+`）とその周辺コンテキストからハンクを抽出し、ウィンドウサイズ（128〜256トークン）に切り出して `DecisionEngine` で走査します。
4. 判定ポリシーは「厳格モード」とし、コード差分における重大異常（`Noul == true` かつ確信度 $\ge 80\%$）は Exit 1 でブロック判定、文書差分における異常は警告メッセージと品質スコア表示にとどめて Exit 0 とします。
5. 論文実証モード（`--bench`）を実装し、判定結果、推論レイテンシ、出力エントロピー、確信度を `logs/benchmarks/gatekeeper_eval.jsonl` に構造化記録します。
- **やらないこと（Non-Goals）**: 
- 既存の `.githooks/pre-commit` への強制組み込み（開発者のコミットを阻害しないため、実証段階ではフック側は一切変更せず温存）。
- 外部 ML ランタイム（Python、PyTorch、ONNX Runtime 等）の追加（Pure Rust 原則を厳格維持）。
- クレートの新設（`oniwa-gatekeeper` のような新規クレートは作らず、`oniwa-decide` 内のバイナリとして実装）。

## 2. エージェント実行体制・開発運用ルール
本指示を受け取ったエージェントは、以下の「Issueドリブン ＆ 階層型エージェント体制」を厳格に遵守して自走すること。

### 保存ディレクトリとドキュメント作成ルール
本タスクのドキュメントは、対象 `task.md` と同階層に集約・保存すること。
- (**タスク指示書**: 既に保存され共有される想定)
- **計画ドキュメント**: `implement_plan.md` および日本語版 `implement_plan.ja.md` を作成・更新する。
- **成果・振り返りドキュメント**: 実装完了後、全体の変更差分、セルフチェック結果、および検証結果を `walkthrough.md` および日本語版 `walkthrough.ja.md` として作成する。

### 体制と責務
- **メインエージェント（司令塔）**:
  - 本指示書および関連する既存ソースコードを深く理解し、全体の進行管理と統合テストを担当する。
  - 上記のタイムスタンプディレクトリを作成し、計画書（`implement_plan`）と振り返り（`walkthrough`）の日英両バージョンを確実に作成・管理する。
  - サブタスクごとに作業スコープを切り出し、サブエージェントへ明確な指示を渡す。
  - **自己検証（セルフチェック）の責務**:
    - 全実装・テスト完了後、セクション7の「AIセルフチェックリスト」を1項目ずつ突き合わせて検証すること。
    - すべての項目を満たしていることを確認した結果を `walkthrough.md` にチェックログとして記録すること。未達の項目があれば完了とみなさず、サブエージェントに再指示を出して修正すること。
  - **Git コミット＆プッシュの責務**:
    - ソースコードの変更差分だけでなく、**`task.md` のディレクトリ `.agent/tasks/{タイムスタンプ}/` 配下に生成されたすべてのドキュメント（task.md, implement_plan.*, walkthrough.*）を漏れなく `git add` の対象に含めること**。
    - 実装・テスト・フォーマット・セルフチェック・ドキュメント作成がすべて完了した後、一括してコミットおよびリモートへのプッシュを実行すること。
- **サブエージェント（実装担当）**:
  - メインエージェントから指示された特定モジュールの実装・テストコード作成に忠実に従う。
  - 自身が担当した実装の詳細ステップと検証手順をサブエージェント視点のプラン/ログとして記録・更新する。
  - **コードフォーマット規律**: 静的解析（Lint/型チェック）は実行せず、変更・新規作成したファイルに対して `cargo fmt --all`（Pythonファイルがある場合は `isort` と `ruff format`）によるフォーマットのみを適用すること。

## 3. 影響ファイル一覧
- `crates/oniwa-decide/src/bin/gatekeeper.rs`: [新規作成] Git差分走査・判定・ベンチマーク出力を行う門番CLIバイナリ
- `crates/oniwa-decide/tests/decision_test.rs`: [変更] 差分パース処理およびハンク抽出ロジックの単体テスト追加
- `.agent/tasks/{タイムスタンプ}/*`: [新規作成] タスク指示書、計画書、walkthrough等の各ドキュメント

## 4. 実装ステップ（サブエージェントへの作業分割案）
1. Phase 1: [タイムスタンプディレクトリの作成と初期ドキュメント配置]
   - `.agent/tasks/{タイムスタンプ}/` を作成し、`task.md`、`implement_plan.md`、`implement_plan.ja.md` を配置。
2. Phase 2: [差分パース ＆ ハンク抽出ロジックの実装]
   - 対象ファイル: `crates/oniwa-decide/src/bin/gatekeeper.rs`
   - 具体指示:
     - `git diff --cached` または標準入力/引数から渡された Unified Diff テキストを解析するパーサーを実装。
     - 拡張子フィルタ（`.rs`, `.py`, `.md`, `.txt` のみを抽出、その他やバイナリはスキップ）。
     - 純粋な削除（`-` のみ）はスキップし、追加行（`+`）とその文脈から128〜256トークン以内のテキスト片（ハンク）を構築。
3. Phase 3: [DecisionEngine 連携 ＆ 判定・ベンチマーク出力ロジックの実装]
   - 対象ファイル: `crates/oniwa-decide/src/bin/gatekeeper.rs`
   - 具体指示:
     - `checkpoints/best` または指定チェックポイントから `DecisionEngine` をロード。
     - 各ハンクを走査し、`Choice`、`Noul`、`Score`、および確信度を算出。
     - コード拡張子（`.rs`, `.py`）で `Noul == true` かつ確信度 $\ge 0.80$ の場合は「BLOCK（Exit 1）」、それ以外は「PASS / WARNING（Exit 0）」とする終了コード制御を実装。
     - `--bench` オプション指定時、各ハンクの推論時間（ms）、エントロピー（bits）、判定結果を `logs/benchmarks/gatekeeper_eval.jsonl` に追記。
4. Phase 4: [テスト作成・フォーマット適用]
   - `crates/oniwa-decide/tests/decision_test.rs` に正常系・異常系（バイナリスキップ、削除のみ差分、コード異常検知）のテストを追加。
   - `cargo test --workspace` を実行し、全件 Green であることを確認。
   - `cargo fmt --all` を適用。
5. Phase 5: [AIセルフチェック ＆ walkthrough 作成]
   - セクション7のセルフチェックリストを照合し、`walkthrough.md` および `walkthrough.ja.md` に結果を記録。
6. Phase 6: [Git コミット・プッシュ]
   - 変更したソースファイル群および `.agent/tasks/{タイムスタンプ}/` 配下の全ドキュメントをステージングしてコミット・プッシュ。

## 5. エッジケースと制約事項
- **`.githooks/pre-commit` の不可侵**: 既存のコミットフックは絶対に改変しないこと。開発中の Git 操作に予期せぬブロッキングを起こさないため、門番機能は `cargo run --release -p oniwa-decide --bin gatekeeper` で明示的に呼び出すスタンドアロン設計とする。
- **バイナリ・巨大ファイルの自動除外**: `.png`, `.jpg`, `.bin`, `Cargo.lock` 等の差分は検出しても即時スキップし、トークナイザーやアロケータを保護する。
- **差分ゼロ時の挙動**: ステージングされた対象差分が存在しない場合は即座に Exit 0 で正常終了する。
- **メモリバジェット厳守**: 差分走査時も 100MB バジェットの枠内で実行し、大規模な一時文字列の過剰なアロケーションを避ける。

## 6. 完了判定・デプロイコマンド（検証・フォーマット・Git）
- コードフォーマット適用:
  - `cargo fmt --all`
  - `uv run isort .`（※Pythonファイルに触れた場合のみ）
  - `uv run ruff format .`（※Pythonファイルに触れた場合のみ）
- テスト実行:
  - `cargo test --workspace`
- 動作検証（スモークテスト）:
  - `cargo run --release -p oniwa-decide --bin gatekeeper -- --help`
  - `cargo run --release -p oniwa-decide --bin gatekeeper -- --bench`
- Git コミット ＆ プッシュ:
  - `git add crates/oniwa-decide/src/bin/gatekeeper.rs crates/oniwa-decide/tests/decision_test.rs .agent/tasks/<生成されたタイムスタンプ>/`
  - `git commit -m "feat: oniwa-decide 論文実証用スタンドアロン門番エンジン（Gatekeeper CLI）の実装 (#<issue番号>)"`
  - `git push origin <現在のブランチ名>`

## 7. AIセルフチェックリスト（実装完了判定）
エージェントはコミット前に以下のチェックを自律的に実施し、すべてクリアしていることを確認すること（walkthrough にチェック結果を記載すること）。
- [ ] **仕様準拠**: `gatekeeper.rs` がスタンドアロンで動作し、コードと文書で適切なブロック/警告制御が行われているか？
- [ ] **スコープ厳守**: `.githooks/pre-commit` や既存の推論・学習パイプラインを破壊・改変していないか？
- [ ] **影響範囲の一致**: セクション3の「影響ファイル一覧」以外の無関係なファイルを変更していないか？
- [ ] **Pure Rust 原則順守**: `Cargo.toml` に外部MLクレート等の不要な新規依存が追加されていないか？
- [ ] **フォーマット順守**: `cargo fmt --all` を実行し、フォーマット差分がない状態になっているか？
- [ ] **テスト通過**: `cargo test --workspace` が 100% エラーなく成功（Green）しているか？
- [ ] **ドキュメント網羅**: 日英両方の `implement_plan`（.md / .ja.md）および `walkthrough`（.md / .ja.md）がすべて指定ディレクトリ内に生成されているか？
- [ ] **Git対象の完全性**: ソースファイルに加えて `.agent/tasks/{タイムスタンプ}/` 配下の全ファイルが `git add` の対象に含まれているか？