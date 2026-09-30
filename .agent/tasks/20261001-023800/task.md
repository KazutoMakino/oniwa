# [Task Plan / Issue Spec]: System 1 精度改善・Gitフック組込・System 2 カスケード統合（B -> C -> A）

## 1. Issue & 目的定義
- **背景と課題**:
  - 現在の System 1（oniwa-decide）は、短縮コードスニペット等で「過信による誤分類」を発生させるリスクがある。
  - Git コミット時のリアルタイム検査（500ms台）を日常開発に組み込むためのレイテンシ短縮（100ms台目標）と、ブロック時の適切な案内が必要。
  - ラズパイ4の限られたリソース下で、System 1 の判定結果に応じた System 2（oniwa-lm）へのオンデマンドエスカレーション基盤が必要。
- **解決方針とスコープ**:
  - **Phase B（精度・キャリブレーション）**:
    - 学習後温度スケーリング（Post-hoc Temperature Scaling）を導入し、温度係数 $T$ を設定メタデータ（`config.json` 等）で動的管理。
    - 短コードスニペットおよび曖昧サンプルのデータ拡張を実施し、ECE 閾値達成と回帰テスト全件パスを担保。
  - **Phase C（Gitフック / CI組み込み）**:
    - `git diff --cached` からの変更チャンク抽出による推論対象絞り込みでレイテンシを短縮。
    - ハードブロック＋判定根拠および `--no-verify` スキップ案内を出力するフックを構築。
  - **Phase A（System 1/2 カスケード統合）**:
    - `EscalateToSystemTwo` 時に System 2 をオンデマンド起動・推論・即時解放するメモリ保護構成を実装。
    - 構造化プロンプト（コードチャンク＋System 1 判定理由・スコア）を標準入力/引数経由で伝達する疎結合インターフェースを構築。
- **やらないこと（Non-Goals）**:
  - Choice ヘッドのカテゴリ定義自体の破壊的変更（既存カテゴリ体系を維持）。
  - 常駐型バックグラウンド IPC デーモンの構築（プロセスの複雑性・死活監視の排除）。
  - StateEmbedding（潜在中間テンソル）のバイナリ直接伝達によるモデル間の密結合。

## 2. エージェント実行体制・開発運用ルール（グラフ的自走規律）
本指示を受け取ったエージェントは、以下の「ゲート制 ＆ 検証主導ループ ＆ 安全規律」を厳格に遵守して自走すること。

### 保存ディレクトリとドキュメント作成ルール
本タスクのドキュメントは、対象 `task.md` と同階層に **`.agent/tasks/{YYYYMMDD_HHMMSS}/`**（タスク開始時のタイムスタンプディレクトリ）を作成し、すべてその配下に集約・保存すること。
- **タスク指示書**: 実行開始時に本仕様を `.agent/tasks/{タイムスタンプ}/task.md` として保存・配置する。
- **計画ドキュメント**: 実装着手前に `.agent/tasks/{タイムスタンプ}/implementation_plan.md` および日本語版 `implementation_plan.ja.md` を作成する。
- **成果・振り返りドキュメント**: 実装完了後、全体の変更差分、セルフチェック結果、および検証ログを `.agent/tasks/{タイムスタンプ}/walkthrough.md` および日本語版 `walkthrough.ja.md` として作成する。

### ゲートフェーズ（初動の必須フロー）
1. コードをいきなり書き換えてはならない。
2. まず本仕様と対象コードを読み解き、変更差分計画として `implementation_plan.md` / `implementation_plan.ja.md` を作成すること。
3. 計画書作成完了後、メインエージェントからサブエージェントへフェーズごとの作業指示を渡し、実装に着手すること。

### 体制と責務
- **メインエージェント（司令塔）**:
  - 全体の進行管理と統合テストを担当する。
  - タイムスタンプディレクトリおよび各Artifacts（`implementation_plan` / `walkthrough`）の日英両バージョンを作成・管理する。
  - セクション4の各ステップにおける「Verify コマンド」がPASSしたことを確認してから次のステップへ遷移させる。
  - **検証ループ制限（Circuit Breaker）**: 同一ステップのVerifyに**3回連続で失敗した場合は自己修正ループを中断**し、エラー内容・試行履歴・現状差分を `walkthrough.md` / `walkthrough.ja.md` に記録して作業を停止し、人間にエスカレーションすること。
  - 全実装完了後、セクション7のセルフチェックリストを照合し、未達があればサブエージェントに再指示を出す。
  - **Git コミット＆プッシュの安全責務**:
    - **`main` / `master` ブランチへの直接プッシュは厳禁**。必ず各タスクの専用作業ブランチ（例: `feature/sys1-calibration`, `feature/git-hook-gatekeeper`, `feature/sys1-sys2-cascade`）上であることを確認してからプッシュすること。
    - ソースコードの変更差分だけでなく、**`.agent/tasks/{タイムスタンプ}/` 配下に生成されたすべてのドキュメント（task.md, implementation_plan.*, walkthrough.*）を漏れなく `git add` の対象に含めること**。
    - 実装・テスト・フォーマット・セルフチェック・ドキュメント作成がすべて完了した後、一括してコミットおよびリモートへのプッシュを実行すること。
- **サブエージェント（実装担当）**:
  - メインエージェントから指示された特定モジュールの実装・テストコード作成に忠実に従う。
  - **コードフォーマット規律**: 静的解析（Lint/型チェック）は実行しない。リポジトリ全体ではなく、**今回変更・新規作成したファイルのみ**に対して `isort` と `ruff format`（または Rust ファイルに対する `cargo fmt`）を適用すること。

## 3. 影響ファイル一覧
- `crates/oniwa-decide/src/config.rs`: 温度パラメータ $T$ の読み込み・保持ロジック追加
- `crates/oniwa-decide/src/model.rs`: ロジットに対する温度スケーリング除算の組み込み
- `crates/oniwa-decide/src/bin/gatekeeper.rs`: `git diff --cached` 変更チャンク抽出・ハードブロック出力の実装
- `crates/oniwa-cascade/src/lib.rs` (または新規統合モジュール): System 2 オンデマンド起動および構造化プロンプト伝達パイプライン
- `.githooks/pre-commit`: 差分抽出型ルーター実行およびスキップ案内スクリプト
- `tests/test_calibration_ece.py` / `tests/regression_snippets.rs`: ECE 検証および短コード回帰テスト集
- `.agent/tasks/{タイムスタンプ}/*`: `task.md`, `implementation_plan.*`, `walkthrough.*`

## 4. グラフ的実装ステップ（Verify駆動ステートマシン）

- [ ] **Phase 1: ゲート（計画策定と初期配置）**
  - Target: `.agent/tasks/{タイムスタンプ}/`
  - Action: ディレクトリ作成、`task.md` の配置、および `implementation_plan.md` / `implementation_plan.ja.md` の作成
  - Verify: `ls -la .agent/tasks/<生成されたタイムスタンプ>/` でファイルの存在を確認
  - Rule: 計画書が揃うまで Phase 2 へ遷移しないこと

- [ ] **Phase 2: 候補 B（温度キャリブレーション ＆ データセット強化）**
  - Target: `crates/oniwa-decide/src/config.rs`, `crates/oniwa-decide/src/model.rs`, 回帰テスト
  - Action: 
    - メタデータからの温度パラメータ $T$ 読み込み・ロジット除算の実装
    - 短コードスニペット回帰テスト集の追加
  - Verify: `cargo test -p oniwa-decide --test regression_snippets` が全件PASSすること
  - Rule: VerifyがPASSするまで Phase 3 へ遷移しないこと（3回失敗時は中断しエスカレーション）

- [ ] **Phase 3: 候補 C（差分抽出型プリコミットフック ＆ ハードブロック整備）**
  - Target: `crates/oniwa-decide/src/bin/gatekeeper.rs`, `.githooks/pre-commit`
  - Action: 
    - `git diff --cached` からの変更チャンク抽出機能の実装
    - 遮断時の判定理由・スコア・`--no-verify` 解除方法のターミナル出力装飾
  - Verify: テストリポジトリ環境でのプリコミットフック実行（変更チャンク抽出および終了ステータス検証）
  - Rule: フック検証がPASSするまで Phase 4 へ遷移しないこと（3回失敗時は中断しエスカレーション）

- [ ] **Phase 4: 候補 A（オンデマンド System 2 カスケード統合）**
  - Target: `crates/oniwa-cascade/` またはカスケード呼び出し部
  - Action: 
    - `EscalateToSystemTwo` 判定検知時の System 2 プロセスオンデマンド起動・終了制御
    - 構造化コンテキスト（差分チャンク＋確信度スコア）の引き渡し処理
  - Verify: `cargo test -p oniwa-cascade` （モック/実プロセス起動およびメモリ解放確認）
  - Rule: 結合テストがPASSするまで Phase 5 へ遷移しないこと（3回失敗時は中断しエスカレーション）

- [ ] **Phase 5: コードフォーマット ＆ 全体結合テスト**
  - Target: 今回変更した全ファイル
  - Action: 対象ファイルへの個別フォーマット適用、全体リグレッションテストの実行
  - Verify:
    - 変更された Rust ファイルへの `cargo fmt -- <変更ファイル>` または Python ファイルへの `uv run isort` / `uv run ruff format`
    - `cargo test --all` がオールGreenであること

- [ ] **Phase 6: AIセルフチェック ＆ walkthrough 作成**
  - Target: `.agent/tasks/{タイムスタンプ}/walkthrough.md`, `walkthrough.ja.md`
  - Action: セクション7のセルフチェックリストを照合し、日英両方の walkthrough に結果・検証ログを記録
  - Verify: `cat .agent/tasks/<タイムスタンプ>/walkthrough.ja.md` でチェック項目が全て埋まっていることを確認

- [ ] **Phase 7: Git コミット・プッシュ（ブランチ保護確認付き）**
  - Target: リポジトリ全体
  - Action: 作業ブランチの確認、ステージング、コミット、プッシュ
  - Verify: 
    - `current_branch=$(git branch --show-current) && [ "$current_branch" != "main" ] && [ "$current_branch" != "master" ]`
    - `git status` でワーキングツリーがクリーンであること

## 5. エッジケースと制約事項
- **空コミット / 非コード変更**:
  - `git diff --cached` でコード差分が存在しない場合（ドキュメントのみの変更など）は判定を安全にバイパスし、無駄な推論コストを発生させないこと。
- **ラズパイ4 メモリ上限保護**:
  - System 2 プロセス起動時にメモリ割り当て失敗を検知した場合、システムクラッシュを避けフォールバックエラーを開発者に安全に返すこと。
- **緊急脱出ハッチ**:
  - フック遮断時、開発者が作業を継続できるよう `git commit --no-verify` または環境変数 `ONIWA_BYPASS=1` による明示的バイパス手段を必ず案内すること。

## 6. 完了判定・デプロイコマンド（検証・フォーマット・Git）
- 変更ファイルへのフォーマット適用:
  - `cargo fmt`（または Python 対象ファイルへの `uv run ruff format`）
- 結合テスト実行:
  - `cargo test --all`
- Git コミット ＆ プッシュ（main/master 直接push禁止）:
  - `git branch --show-current`（作業ブランチであることを確認）
  - `git add <変更したソースファイル群> .agent/tasks/<生成されたタイムスタンプ>/`
  - `git commit -m "feat: System 1キャリブレーション・Gitフック組込・オンデマンドカスケード統合の実装"`
  - `git push origin $(git branch --show-current)`

## 7. AIセルフチェックリスト（実装完了判定）
エージェントはコミット前に以下のチェックを自律的に実施し、すべてクリアしていることを確認すること（walkthrough にチェック結果を記載すること）。
- [ ] **仕様準拠**: セクション1の解決方針（温度係数メタデータ化、差分チャンク抽出、オンデマンド起動）が漏れなく反映されているか？
- [ ] **スコープ厳守**: セクション1の「やらないこと（Non-Goals）」に抵触する不要なカテゴリ変更や常駐デーモン化が含まれていないか？
- [ ] **影響範囲の一致**: セクション3の「影響ファイル一覧」以外の無関係なファイルを変更していないか？
- [ ] **フォーマット順守**: 今回変更したファイルに対してのみフォーマッタを実行し、不要な差分を広げていないか？
- [ ] **テスト通過**: 指定テストコマンドがエラーなく成功（Green）しているか？（※3回連続失敗による中断フラグが立っていないか？）
- [ ] **ブランチ保護**: 現在のブランチが `main` または `master` でないことを確認したか？
- [ ] **ドキュメント網羅**: 日英両方の `implementation_plan`（.md / .ja.md）および `walkthrough`（.md / .ja.md）がすべて指定ディレクトリ内に生成されているか？
- [ ] **Git対象の完全性**: ソースファイルに加えて `.agent/tasks/{タイムスタンプ}/` 配下の全ファイルが `git add` の対象に含まれているか？