# [Task Plan / Issue Spec]: oniwa-decide レイヤー別マイクロベンチマーク計測とホットスポット特定

## 1. Issue & 目的定義
- **背景と課題**: 
PR #102（ARM NEON SIMD ベクトル化）の完了により推論レイテンシが 755ms から 639ms へと短縮されたものの、ロードマップ Phase 4 が目標とする「Gatekeeper 即時応答（~100ms 台）」には依然として約6倍の乖離が存在する[cite: 1]。当てずっぽうの最適化や不要なコード作り込み（YAGNI）を避けるため、推論パス（639ms）の内訳（Embedding、各Transformer層、Pooling、Heads等）を正確に計測し、最大のボトルネックがどこにあるのかを実測数値に基づいて特定する必要がある[cite: 1]。
- **解決方針とスコープ**: 
1. **専用プロファイリングメソッドの分離**: 既存の `DecisionModel::forward` および学習ループ・既存推論パスのホットループは一切変更せず、`forward_with_profile(&self, tokens: &[u16], batch_size: usize, seq_len: usize) -> (ForwardCache, ForwardBreakdown)` を新設する。
2. **フェーズ別ブロック計測**: レイヤー深部への侵入的改変を避け、「トークン埋め込み（Embedding）」「Transformer各層全体（Layer 0..N）」「プーリング（Pooling）」「決定ヘッド群（Heads: Choice, Noul, Score）」のフェーズ単位で所要時間を計測する。
3. **二重出力（ASCII 比較表 ＆ 構造化 JSONL）**: `crates/oniwa-decide/src/bin/bench.rs` に `--profile-layers`（または `--breakdown`）オプションを追加し、コンソールに見やすい ASCII 内訳テーブル（各ブロックの所要時間 ms および割合 %）を出力するとともに、`profiler.rs` 経由で構造化 JSONL に追記可能とする[cite: 1]。
4. **客観的検証とスモーク実行**: 単体テストでプロファイル内訳値の整合性を担保しつつ、CLI 検証は `--iters 1` のスモーク実行により高速・安全に検証可能とする。
- **やらないこと（Non-Goals）**: 
- 既存の `DecisionModel::forward` のシグネチャ変更や学習ループ（`train.rs`）へのオーバーヘッド混入[cite: 1]。
- 超越関数（SiLU, RoPE 等）や QKV 射影ループ内部への侵入的な細分化インストルメンテーション（YAGNI原則遵守）。
- 新規外部ベンチマーククレート（Criterion 等）の追加（Pure Rust 原則・Zero-Dep の厳格維持）[cite: 1]。
- この段階での投機的なコード最適化（まずプロファイル計測を完了してエビデンスを確定させる）。

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
    - **`main` / `master` ブランチへの直接プッシュは厳禁**。必ず feature/fix などの作業ブランチ上であることを確認してからプッシュすること。
    - ソースコードの変更差分だけでなく、**`.agent/tasks/{タイムスタンプ}/` 配下に生成されたすべてのドキュメント（task.md, implementation_plan.*, walkthrough.*）を漏れなく `git add` の対象に含めること**。
    - 実装・テスト・フォーマット・セルフチェック・ドキュメント作成がすべて完了した後、一括してコミットおよびリモートへのプッシュを実行すること。
- **サブエージェント（実装担当）**:
  - メインエージェントから指示された特定モジュールの実装・テストコード作成に忠実に従う。
  - **コードフォーマット規律**: 静的解析（Lint/型チェック）は実行しない。リポジトリ全体ではなく、**今回変更・新規作成したファイルのみ**に対して `cargo fmt --` を適用すること（Pythonファイルに触れた場合のみ `isort` と `ruff format` を適用）。

## 3. 影響ファイル一覧
- `crates/oniwa-decide/src/model.rs`: [変更] `ForwardBreakdown` 構造体の定義および `DecisionModel::forward_with_profile` の新設（既存 `forward` は完全温存）
- `crates/oniwa-decide/src/profiler.rs`: [変更] レイヤー別ブレークダウンを含む `LayerProfileRecord` / `HardwareProfileRecord` 拡張およびシリアライズ対応[cite: 1]
- `crates/oniwa-decide/src/bin/bench.rs`: [変更] `--profile-layers`（または `--breakdown`）CLIオプションの追加、ASCII内訳テーブル描画、JSONL出力連携[cite: 1]
- `crates/oniwa-decide/tests/decision_test.rs`: [変更] `forward_with_profile` の所要時間整合性およびブレークダウン取得の単体テスト追加
- `.agent/tasks/{タイムスタンプ}/*`: [新規作成] `task.md`, `implementation_plan.md`, `implementation_plan.ja.md`, `walkthrough.md`, `walkthrough.ja.md`

## 4. グラフ的実装ステップ（Verify駆動ステートマシン）
エージェントは各ステップで指定された「Verify」コマンドを実行し、PASS（Green）になるまで修正ループを回すこと（※上限3回まで）。

- [ ] **Phase 1: ゲート（計画策定と初期配置）**
  - Target: `.agent/tasks/{タイムスタンプ}/`
  - Action: ディレクトリ作成、`task.md` の配置、および `implementation_plan.md` / `implementation_plan.ja.md` の作成
  - Verify: `ls -la .agent/tasks/<生成されたタイムスタンプ>/` でファイルの存在を確認
  - Rule: 計画書が揃うまで Phase 2 へ遷移しないこと

- [ ] **Phase 2: 基盤・データ構造およびプロファイリングメソッドの実装**
  - Target: `crates/oniwa-decide/src/model.rs`, `crates/oniwa-decide/src/profiler.rs`
  - Action: 
    - `ForwardBreakdown`（embedding_ms, layer_ms: Vec<f64>, pooling_ms, heads_ms, total_ms）を定義。
    - `DecisionModel::forward_with_profile` を実装（各ブロックの前後で高精度タイム計測を実施し、既存 `forward` と同等の `ForwardCache` とともに返却）。
    - `profiler.rs` にブレークダウン記録用の構造体を統合。
  - Verify: `cargo check -p oniwa-decide`
  - Rule: コンパイルが正常に通るまで Phase 3 へ遷移しないこと（3回失敗時は中断しエスカレーション）

- [ ] **Phase 3: CLI 連携 ＆ ASCII テーブル / JSONL 出力実装**
  - Target: `crates/oniwa-decide/src/bin/bench.rs`
  - Action: 
    - `--profile-layers` フラグの解析処理を追加。
    - `forward_with_profile` を呼び出し、各フェーズの所要時間（ms）と全体に対する割合（%）を計算。
    - ASCII テーブル形式でターミナルに出力し、指定時は JSONL へ追記。
  - Verify: `cargo run --release -p oniwa-decide --bin bench -- --iters 1 --profile-layers`
  - Rule: スモーク実行が正常終了し、ASCIIテーブルが出力されるまで Phase 4 へ遷移しないこと（3回失敗時は中断しエスカレーション）

- [ ] **Phase 4: テスト拡充 ＆ コードフォーマット**
  - Target: `crates/oniwa-decide/tests/decision_test.rs` および変更ファイル群
  - Action: 
    - `decision_test.rs` に `forward_with_profile` の各ブロック値（非負値、合計時間との整合）を検証する単体テストを追加。
    - 今回変更したファイルに対してのみフォーマットを適用。
  - Verify: 
    - `cargo test -p oniwa-decide --test decision_test test_forward_with_profile`
    - `cargo fmt --all -- --check`
  - Rule: 全テスト・フォーマットがGreenになるまで Phase 5 へ遷移しないこと（3回失敗時は中断しエスカレーション）

- [ ] **Phase 5: AIセルフチェック ＆ walkthrough 作成**
  - Target: `.agent/tasks/{タイムスタンプ}/walkthrough.md`, `walkthrough.ja.md`
  - Action: 
    - セクション7のセルフチェックリストを照合。
    - `cargo run --release -p oniwa-decide --bin bench -- --iters 5 --profile-layers` を実行し、得られた実測ブレークダウン（ミリ秒・割合%）を記録。
    - 日英両方の walkthrough に結果・検証エビデンスを記録。
  - Verify: `cat .agent/tasks/<タイムスタンプ>/walkthrough.ja.md` でチェック項目および実測結果が全て埋まっていることを確認

- [ ] **Phase 6: Git コミット・プッシュ（ブランチ保護確認付き）**
  - Target: リポジトリ全体
  - Action: 作業ブランチの確認、ステージング、コミット、プッシュ
  - Verify: 
    - `current_branch=$(git branch --show-current) && [ "$current_branch" != "main" ] && [ "$current_branch" != "master" ]` （保護ブランチでないことの確認）
    - `git status` でワーキングツリーがクリーンであること

## 5. エッジケースと制約事項
- **タイマー分解能と最小イテレーション**: ナノ秒オーダーの短時間処理では環境依存のジッターが生じる可能性があるため、`std::time::Instant` を用い、スモーク検証時は `--iters 1`、実測エビデンス取得時は複数回（5回以上）の平均値を採用すること。
- **既存学習ループへの非侵入**: `DecisionModel::forward` は学習時（`train.rs`）の最頻ループであるため、計測ロジックは必ず `forward_with_profile` に完全に隔離し、学習速度・メモリ消費に 1 バイト・1 サイクルも影響を与えないこと[cite: 1]。
- **Pure Rust 原則の厳守**: 外部のプロファイリングクレートを追加せず、標準ライブラリのプリミティブのみで完結させること[cite: 1]。

## 6. 完了判定・デプロイコマンド（検証・フォーマット・Git）
- 変更ファイルへのフォーマット適用:
  - `cargo fmt --all`
- 単体テストおよびスモーク実行:
  - `cargo test -p oniwa-decide --test decision_test`
  - `cargo run --release -p oniwa-decide --bin bench -- --iters 1 --profile-layers`
- Git コミット ＆ プッシュ（main/master 直接push禁止）:
  - `git branch --show-current`（作業ブランチであることを確認）
  - `git add crates/oniwa-decide/src/model.rs crates/oniwa-decide/src/profiler.rs crates/oniwa-decide/src/bin/bench.rs crates/oniwa-decide/tests/decision_test.rs .agent/tasks/<生成されたタイムスタンプ>/`
  - `git commit -m "perf: レイヤー別マイクロベンチマーク計測機能の実装とホットスポット特定 (#<issue番号>)"`
  - `git push origin $(git branch --show-current)`

## 7. AIセルフチェックリスト（実装完了判定）
エージェントはコミット前に以下のチェックを自律的に実施し、すべてクリアしていることを確認すること（walkthrough にチェック結果を記載すること）。
- [ ] **仕様準拠**: `forward_with_profile` が新設され、Embedding、各層、Pooling、Heads のブレークダウンが正しく取得できているか？
- [ ] **スコープ厳守**: 既存の `DecisionModel::forward` や学習ループ（`train.rs`）に不要な改変やオーバーヘッドを加えていないか？[cite: 1]
- [ ] **影響範囲の一致**: セクション3の「影響ファイル一覧」以外の無関係なファイルを変更していないか？
- [ ] **フォーマット順守**: `cargo fmt --all -- --check` を実行し、フォーマット差分がない状態になっているか？
- [ ] **テスト通過**: 指定テストコマンドがエラーなく成功（Green）し、`bench --iters 1` が正常終了しているか？（※3回連続失敗による中断フラグが立っていないか？）
- [ ] **ブランチ保護**: 現在のブランチが `main` または `master` でないことを確認したか？
- [ ] **ドキュメント網羅**: 日英両方の `implementation_plan`（.md / .ja.md）および `walkthrough`（.md / .ja.md）がすべて指定ディレクトリ内に生成され、実測テーブルが記録されているか？
- [ ] **Git対象の完全性**: ソースファイルに加えて `.agent/tasks/{タイムスタンプ}/` 配下の全ファイルが `git add` の対象に含まれているか？