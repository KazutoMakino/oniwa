# [Task Plan / Issue Spec]: oniwa-decide 推論専用パス分離 ＆ 可変長NEON最適化によるエッジPre-commitレイテンシ短縮

## 1. Issue & 目的定義
- **背景と課題**: 
現在、`oniwa-decide`（System 1）は Git プリコミットフック（`gatekeeper`）に組み込まれ、コミット差分の安全走査を担っていますが、実機推論に約 500ms（プロファイリング上、ボトルネックの 62.09% / 約 137ms が Mean Pooling ＆ MLM 語彙射影）を要しています。また、差分が短い場合でも固定長 $T=128$ のパディング領域全体を計算しているため、日常的な Git コミットに組み込むための目標レイテンシ（100ms〜150ms 台）を達成できていません。
- **解決方針とスコープ**: 
 1. **推論パスの完全分離（`forward_inference`）**:
    学習用 `forward`（逆伝播・中間アクティベーションキャッシュ・MLM 損失計算用）は一切変更せず温存し、推論専用メソッド `DecisionModel::forward_inference` を新設。決定ヘッド（Choice, Noul, Score）に不要な MLM 語彙射影（$V \times C = 4,096 \times 256$ 内積ループ）を完全にスキップします。
 2. **Mean Pooling の ARM NEON SIMD ベクトル化**:
    トークン系列全体の隠れ状態の平均を集約する Mean Pooling 処理を `float32x4_t` による 4 並列ベクトル加算および高速スケーリング（`scale_slice_simd`）で実装し、末端集約コストを最小化します。
 3. **有効トークン長への適応的スライシング（Adaptive Sequence Length）**:
    `gatekeeper` および `DecisionEngine::audit_text` において、入力トークン列の実効長 $T_{\text{valid}}$（クォータニオンおよび RoPE の 4 の倍数アラインメントに切り上げ、上限 `seq_len`）に適応させて Transformer 層およびアテンション計算を実行し、短スニペット時の演算量を物理的に削減します。
 4. **推論等価性と回帰テストの担保**:
    `forward_inference` による決定ヘッド出力（Choice, Noul, Score）が、従来の `forward` 出力と単精度丸め誤差内（$< 10^{-5}$）で完全一致することを検証する単体テストを追加します。
- **やらないこと（Non-Goals）**: 
 - 学習ループ（`train.rs`）や逆伝播（`backward`）の挙動変更・破壊。
 - `ForwardCache` のシグネチャ変更（既存のマイクロベンチマーク `bench.rs` や監査ロジックの後方互換性を維持）。
 - 外部 ML フレームワークや C/アセンブリの依存追加（Pure Rust / Zero-Dep 原則を厳守）。
 - Choice ヘッドのカテゴリ定義（`DocCategory`）の変更。

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
    - **`main` / `master` ブランチへの直接プッシュは厳禁**。必ず専用作業ブランチ（例: `perf/sys1-inference-simd-adaptive`）上であることを確認してからプッシュすること。
    - ソースコードの変更差分だけでなく、**`.agent/tasks/{タイムスタンプ}/` 配下に生成されたすべてのドキュメント（task.md, implementation_plan.*, walkthrough.*）を漏れなく `git add` の対象に含めること**。
    - 実装・テスト・フォーマット・セルフチェック・ドキュメント作成がすべて完了した後、一括してコミットおよびリモートへのプッシュを実行すること。
- **サブエージェント（実装担当）**:
  - メインエージェントから指示された特定モジュールの実装・テストコード作成に忠実に従う。
  - **コードフォーマット規律**: 静的解析（Lint/型チェック）は実行しない。リポジトリ全体ではなく、**今回変更・新規作成したファイルのみ**に対して `cargo fmt --` を適用すること。

## 3. 影響ファイル一覧
- `crates/oniwa-lm/src/simd.rs`: [変更] Mean Pooling 高速化用 SIMD ヘルパー関数（スライス配列の累積加算 `accumulate_slice_simd` 等）の追加と単体等価性テスト
- `crates/oniwa-decide/src/simd/mod.rs`: [変更] `oniwa-lm` 新設 SIMD プリミティブの再エクスポート
- `crates/oniwa-decide/src/model.rs`: [変更] `DecisionModel::forward_inference` の新設（MLM 射影スキップ、中間キャッシュ確保省略、SIMD Mean Pooling 適用）、および `decide` の `forward_inference` 呼び出しへの切り替え
- `crates/oniwa-decide/src/lib.rs`: [変更] `DecisionEngine::audit_text` での実効トークン長 $T_{\text{valid}}$（4の倍数アライン）切り出しと適応長推論の呼び出し
- `crates/oniwa-decide/src/bin/gatekeeper.rs`: [変更] 差分ハンク走査時の適応長推論連携
- `crates/oniwa-decide/tests/decision_test.rs`: [変更] `forward` と `forward_inference` の出力ロジット完全等価性検証テスト（誤差 $< 10^{-5}$）の追加
- `.agent/tasks/{タイムスタンプ}/*`: [新規作成] `task.md`, `implementation_plan.md`, `implementation_plan.ja.md`, `walkthrough.md`, `walkthrough.ja.md`

## 4. グラフ的実装ステップ（Verify駆動ステートマシン）
エージェントは各ステップで指定された「Verify」コマンドを実行し、PASS（Green）になるまで修正ループを回すこと（※上限3回まで）。

- [ ] **Phase 1: ゲート（計画策定と初期配置）**
  - Target: `.agent/tasks/{タイムスタンプ}/`
  - Action: ディレクトリ作成、`task.md` の配置、および `implementation_plan.md` / `implementation_plan.ja.md` の作成
  - Verify: `ls -la .agent/tasks/<生成されたタイムスタンプ>/` でファイルの存在を確認
  - Rule: 計画書が揃うまで Phase 2 へ遷移しないこと

- [ ] **Phase 2: SIMD プリミティブ拡張 ＆ 単体検証**
  - Target: `crates/oniwa-lm/src/simd.rs`, `crates/oniwa-decide/src/simd/mod.rs`
  - Action: 
    - 隠れ状態ベクトルを高速に要素加算する `accumulate_slice_simd(acc: &mut [f32], x: &[f32])`（NEON `vaddq_f32`）を追加。
    - スカラー参照実装との数値等価性テスト（許容誤差 $< 10^{-6}$）を実装。
  - Verify: `cargo test -p oniwa-lm simd::tests`
  - Rule: テストがPASSするまで Phase 3 へ遷移しないこと（3回失敗時は中断しエスカレーション）

- [ ] **Phase 3: `forward_inference` の実装 ＆ 等価性テスト**
  - Target: `crates/oniwa-decide/src/model.rs`, `crates/oniwa-decide/tests/decision_test.rs`
  - Action: 
    - `DecisionModel::forward_inference(&self, tokens: &[u16], b: usize, t: usize) -> (Vec<f32>, Choice, Noul, Score)`（またはプーリング済み表現と決定ヘッド出力）を実装。
    - 内部で不要な `ForwardCache`（全トークンの中間正規化バッファ等）の確保を排除し、MLM 語彙射影を完全にスキップ。
    - Mean Pooling に `accumulate_slice_simd` と `scale_slice_simd` を適用。
    - `DecisionModel::decide` を `forward_inference` 呼び出しに切り替え。
    - `decision_test.rs` に既存 `forward` と `forward_inference` の Choice/Noul/Score 出力等価性テストを追加。
  - Verify: `cargo test -p oniwa-decide --test decision_test test_forward_inference_equivalence`
  - Rule: 等価性テストがPASSするまで Phase 4 へ遷移しないこと（3回失敗時は中断しエスカレーション）

- [ ] **Phase 4: 適応長トークンスライシング ＆ Gatekeeper 連携**
  - Target: `crates/oniwa-decide/src/lib.rs`, `crates/oniwa-decide/src/bin/gatekeeper.rs`
  - Action: 
    - トークナイズ後の有効長 $T_{\text{valid}}$ を検出し、4 の倍数（最低 4、上限 `config.seq_len`）に切り詰めたスライス長で `forward_inference` を呼び出す。
    - 短コードスニペット回帰テスト集（`regression_snippets.rs`）を実行し、短系列での動作を検証。
  - Verify: `cargo test -p oniwa-decide --test regression_snippets`
  - Rule: 回帰テストが全件PASSするまで Phase 5 へ遷移しないこと（3回失敗時は中断しエスカレーション）

- [ ] **Phase 5: コードフォーマット ＆ ワークスペース全体テスト**
  - Target: 今回変更した全ファイル
  - Action: 対象ファイルへの個別フォーマット適用、および全体結合テストの実行。
  - Verify: 
    - `cargo fmt --all -- --check`
    - `cargo clippy --workspace --all-targets -- -D warnings`
    - `cargo test --workspace` （全テストPASS）
  - Rule: フォーマット・Clippy・全テストがGreenになるまで Phase 6 へ遷移しないこと（3回失敗時は中断しエスカレーション）

- [ ] **Phase 6: 実機レイテンシ計測 ＆ AIセルフチェック ＆ walkthrough 作成**
  - Target: `.agent/tasks/{タイムスタンプ}/walkthrough.md`, `walkthrough.ja.md`
  - Action: 
    - `cargo run --release -p oniwa-decide --bin bench -- --iters 5 --profile-layers`（または短系列長ベンチマーク）を実行し、推論レイテンシが **150ms 以内** を達成していることを実測確認。
    - セクション7のセルフチェックリストを照合し、日英両方の walkthrough に結果・検証エビデンスを記録。
  - Verify: `cat .agent/tasks/<タイムスタンプ>/walkthrough.ja.md` でチェック項目および実測レイテンシが全て埋まっていることを確認

- [ ] **Phase 7: Git コミット・プッシュ（ブランチ保護確認付き）**
  - Target: リポジトリ全体
  - Action: 作業ブランチの確認、ステージング、コミット、プッシュ
  - Verify: 
    - `current_branch=$(git branch --show-current) && [ "$current_branch" != "main" ] && [ "$current_branch" != "master" ]` （保護ブランチでないことの確認）
    - `git status` でワーキングツリーがクリーンであること

## 5. エッジケースと制約事項
- **クォータニオン代数および RoPE の 4 の倍数アライメント**:
  適応的トークン長 $T_{\text{valid}}$ の切り詰めを行う際、系列長および隠れ次元は必ず 4 の倍数でなければならない。端数トークンが存在する場合は末尾をパディングトークン（0）で埋め、クォータニオン SIMD 演算および RoPE 回転の境界外アクセスを未然に防止すること。
- **短系列（$T < 4$）のガード**:
  空文字列や 1〜3 トークンの極小入力に対しては、最低 $T=4$ を確保してゼロ除算やスライスパニックを防止すること。
- **学習時モデルの不可侵**:
  `DecisionModel::forward` および `train.rs` の逆伝播経路には一切手を加えないこと。MLM 射影のスキップはあくまで `forward_inference` 内部に閉じること。

## 6. 完了判定・デプロイコマンド（検証・フォーマット・Git）
- 変更ファイルへのフォーマット適用:
  - `cargo fmt --all`
- 静的解析およびテスト実行:
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
- 動作検証（レイテンシ実測スモーク）:
  - `cargo run --release -p oniwa-decide --bin bench -- --iters 5 --profile-layers`
  - `cargo run --release -p oniwa-decide --bin gatekeeper -- --bench`
- Git コミット ＆ プッシュ（main/master 直接push禁止）:
  - `git branch --show-current`（作業ブランチであることを確認）
  - `git add <変更したソースファイル群> .agent/tasks/<生成されたタイムスタンプ>/`
  - `git commit -m "perf: 推論専用パス分離と適応長NEON最適化によるPre-commitレイテンシ短縮 (#<issue番号>)"`
  - `git push origin $(git branch --show-current)`

## 7. AIセルフチェックリスト（実装完了判定）
エージェントはコミット前に以下のチェックを自律的に実施し、すべてクリアしていることを確認すること（walkthrough にチェック結果を記載すること）。
- [ ] **仕様準拠**: `forward_inference` による MLM 射影スキップ、SIMD Mean Pooling、適応長スライシングが漏れなくコードに反映されているか？
- [ ] **等価性確認**: `forward` と `forward_inference` の決定ヘッド出力（Choice, Noul, Score）が誤差 $< 10^{-5}$ で完全一致することがテストで証明されているか？
- [ ] **性能目標達成**: 実機レイテンシ計測で目標とする 100ms〜150ms 台（または旧構成比で大幅短縮）が達成されているか？
- [ ] **スコープ厳守**: 既存の学習用 `forward` や `train.rs` の挙動に不要な変更を加えていないか？
- [ ] **影響範囲の一致**: セクション3の「影響ファイル一覧」以外の無関係なファイルを変更していないか？
- [ ] **フォーマット順守**: `cargo fmt --all -- --check` を実行し、フォーマット差分がない状態になっているか？
- [ ] **テスト通過**: `cargo test --workspace` が 100% エラーなく成功（Green）しているか？
- [ ] **ブランチ保護**: 現在のブランチが `main` または `master` でないことを確認したか？
- [ ] **ドキュメント網羅**: 日英両方の `implementation_plan`（.md / .ja.md）および `walkthrough`（.md / .ja.md）がすべて指定ディレクトリ内に生成されているか？
- [ ] **Git対象の完全性**: ソースファイルに加えて `.agent/tasks/{タイムスタンプ}/` 配下の全ファイルが `git add` の対象に含まれているか？