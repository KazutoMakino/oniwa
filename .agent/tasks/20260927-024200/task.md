# [Task Plan / Issue Spec]: ARM NEON SIMD による oniwa-decide レイヤー演算の直接ベクトル化と推論高速化

## 1. Issue & 目的定義
- **背景と課題**: 
  現在、`oniwa-decide`（Full Q-Transformer バックボーン ＋ 四元数決定ヘッド）は 8,000 ステップの学習を完遂し、言語カテゴリ識別精度 97.0%、構文異常検知における安定した確信度を獲得しています[cite: 1]。しかし、ハミルトン積に SIMD 実装が入っている一方で、推論ループ全体（Attention, RmsNorm, SwiGlu 等）にスカラー演算が残っており、実機推論レイテンシが約 755ms を要しています[cite: 1]。このままではロードマップ Phase 4 のカスケード型ルーター（FastPath / Gatekeeper）としての実用要求（~100ms 台）を満たせません[cite: 1]。
- **解決方針とスコープ**: 
  1. `crates/oniwa-lm/src/simd.rs` に、汎用的な 4D/スライスベクトル演算プリミティブ（内積、要素ごと加減算・乗算、二乗和アキュムレート等）を追加し、`#[cfg(target_arch = "aarch64")]` の NEON SIMD（`float32x4_t`）および非 aarch64 用のスカラーフォールバックを完備します[cite: 1]。
  2. 数値的一貫性とチェックポイント推論精度を厳密に保つため、内積・FMA積和算・要素ごと乗除算を SIMD 化し、$\exp$ や $\sqrt{}$ 等の超越関数は標準数学関数（`f32::exp` / `f32::sqrt`）による安全な処理を維持します。
  3. `crates/oniwa-decide/src/layers/` 配下の各レイヤー（`rmsnorm.rs`, `mlp.rs`, `quaternion_mlp.rs`, `attention.rs`, `quaternion_attention.rs` 等）において、既存の関数シグネチャやバッファ構造を維持したまま、ホットループ内の演算を新設 SIMD プリミティブへ置き換えます[cite: 1]。
  4. `oniwa-lm/src/simd.rs` 内に、NEON SIMD 実装とスカラー参照実装の数値的一致（誤差 $< 10^{-5}$）を検証する単体テストを追加します[cite: 1]。
- **やらないこと（Non-Goals）**: 
  - 初等超越関数（$\exp$, $\sqrt{}$）の多項式近似による無理な SIMD 化（数値誤差・学習済みモデルの判定狂いを防止）。
  - 各レイヤーの関数シグネチャの大規模変更や、事前確保メモリプール構造体の全面新設（差分最小化と後方互換性維持）。
  - 外部 SIMD クレートや C/Assembly の新規依存追加（Pure Rust 原則を厳格維持）[cite: 1]。

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
  - **コードフォーマット規律**: 静的解析（Lint/型チェック）は実行せず、`cargo fmt --all` によるフォーマットのみを適用すること（Pythonファイルがある場合は `uv run isort .` と `uv run ruff format .` を適用）。

## 3. 影響ファイル一覧
- `crates/oniwa-lm/src/simd.rs`: [変更] 要素ごと乗算、二乗和（L2ノルム）、スライス内積等の NEON SIMD プリミティブ追加、および等価性単体テストの追加[cite: 1]
- `crates/oniwa-decide/src/simd/mod.rs`: [変更] `oniwa-lm` から新設プリミティブの再エクスポート確認[cite: 1]
- `crates/oniwa-decide/src/layers/rmsnorm.rs`: [変更] 二乗和算出ループおよびスケール乗算ループの SIMD 化[cite: 1]
- `crates/oniwa-decide/src/layers/quaternion_mlp.rs`: [変更] SwiGlu 要素ごと積の SIMD 化[cite: 1]
- `crates/oniwa-decide/src/layers/mlp.rs`: [変更] 実数 SwiGlu 要素ごと積の SIMD 化[cite: 1]
- `crates/oniwa-decide/src/layers/attention.rs`: [変更] QKスコアおよび Attention Output 算出ループの SIMD 化[cite: 1]
- `crates/oniwa-decide/src/layers/quaternion_attention.rs`: [変更] QK内積スコア算出ループの SIMD 化[cite: 1]
- `crates/oniwa-decide/tests/decision_test.rs`: [変更] 既存の推論・勾配テストが Green であることの確認[cite: 1]
- `.agent/tasks/{タイムスタンプ}/*`: [新規作成] タスク指示書、計画書、walkthrough等の各ドキュメント

## 4. 実装ステップ（サブエージェントへの作業分割案）
1. Phase 1: [タイムスタンプディレクトリの作成と初期ドキュメント配置]
   - `.agent/tasks/{タイムスタンプ}/` を作成し、`task.md`、`implement_plan.md`、`implement_plan.ja.md` を配置。
2. Phase 2: [SIMD プリミティブの拡充と単体等価性テスト]
   - 対象ファイル: `crates/oniwa-lm/src/simd.rs`[cite: 1]
   - 具体指示:
     - `sum_squares_simd(x: &[f32]) -> f32`: スライスの二乗和（$x_i^2$）を `float32x4_t` で 4 並列積算し、端数はスカラー加算[cite: 1]。
     - `mul_slices_simd(out: &mut [f32], a: &[f32], b: &[f32])`: 4 並列 `vmulq_f32` による要素ごとの積[cite: 1]。
     - `scale_slice_simd(out: &mut [f32], a: &[f32], scalar: f32)`: 4 並列 `vmulq_n_f32` によるスカラー乗算。
     - スカラー参照実装との突き合わせテスト（`test_sum_squares_simd_equivalence`, `test_mul_slices_simd_equivalence` 等）を追加し、誤差 $< 10^{-5}$ を確認[cite: 1]。
3. Phase 3: [レイヤー演算への SIMD 適用]
   - 対象ファイル: `crates/oniwa-decide/src/layers/*.rs`[cite: 1]
   - 具体指示:
     - `rmsnorm.rs`: 二乗和平均とスケーリングループに `sum_squares_simd` / `scale_slice_simd` を適用[cite: 1]。
     - `quaternion_mlp.rs` & `mlp.rs`: `act_h[idx] = silu(g) * u` のうち要素積ループを SIMD 化（`silu` 計算自体は標準数学関数を維持）[cite: 1]。
     - `attention.rs` & `quaternion_attention.rs`: クォータニオン内積や実数内積の計算ループを SIMD プリミティブで集約[cite: 1]。
4. Phase 4: [テスト作成・フォーマット適用]
   - `cargo test --workspace` を実行し、全件 Green（勾配チェック、推論テスト、チェックポイント往復等）を確認[cite: 1]。
   - `cargo fmt --all` を適用[cite: 1]。
5. Phase 5: [AIセルフチェック ＆ walkthrough 作成]
   - セクション7のセルフチェックリストを照合し、`walkthrough.md` および `walkthrough.ja.md` に結果を記録。
6. Phase 6: [Git コミット・プッシュ]
   - 変更したソースファイル群および `.agent/tasks/{タイムスタンプ}/` 配下の全ドキュメントをステージングしてコミット・プッシュ。

## 5. エッジケースと制約事項
- **アライメントとスライス端数処理**: 入力次元 $D$ や系列長 $T$ が 4 の倍数でない場合（あるいは 4 で割り切れない端数要素が存在する場合）でもバッファオーバーランを起こさないよう、`chunks_exact(4)` と `remainder()` による安全な端数処理を徹底すること。
- **初等超越関数の数値安定性**: $\text{SiLU}(x) = x / (1 + \exp(-x))$ の $\exp$ や $1/\sqrt{x + \epsilon}$ の計算でゼロ除算やオーバーフローを起こさないよう、既存のイプシロン値（$1e-5$）と標準関数呼び出しの振る舞いを維持すること。
- **Pure Rust 原則の厳守**: 外部の SIMD クレートやインラインアセンブリに依存せず、Rust 標準の `core::arch::aarch64` 組み込み関数のみを使用すること[cite: 1]。

## 6. 完了判定・デプロイコマンド（検証・フォーマット・Git）
- コードフォーマット適用:
  - `cargo fmt --all`[cite: 1]
- テスト実行:
  - `cargo test --workspace`[cite: 1]
- 動作検証（スモークテスト）:
  - `cargo run --release -p oniwa-decide --bin gatekeeper -- --help`[cite: 1]
  - `cargo run --release -p oniwa-decide --bin audit -- "pub fn test() {}"`[cite: 1]
- Git コミット ＆ プッシュ:
  - `git add crates/oniwa-lm/src/simd.rs crates/oniwa-decide/src/ .agent/tasks/<生成されたタイムスタンプ>/`
  - `git commit -m "perf: ARM NEON SIMD による推論パス全体のベクトル化と高速化 (#<issue番号>)"`
  - `git push origin <現在のブランチ名>`

## 7. AIセルフチェックリスト（実装完了判定）
エージェントはコミット前に以下のチェックを自律的に実施し、すべてクリアしていることを確認すること（walkthrough にチェック結果を記載すること）。
- [ ] **仕様準拠**: `oniwa-lm/src/simd.rs` にプリミティブが追加され、`oniwa-decide` の各レイヤー（RmsNorm, SwiGlu, Attention）で適切にベクトル化されているか？[cite: 1]
- [ ] **スコープ厳守**: 関数シグネチャの破壊的変更や初等関数の多項式近似などの不要な改修を行っていないか？
- [ ] **影響範囲の一致**: セクション3の「影響ファイル一覧」以外の無関係なファイルを変更していないか？
- [ ] **フォーマット順守**: `cargo fmt --all` を実行し、差分がない状態になっているか？[cite: 1]
- [ ] **テスト通過**: `cargo test --workspace` が 100% エラーなく成功（Green）し、SIMD 等価性テストがパスしているか？[cite: 1]
- [ ] **ドキュメント網羅**: 日英両方の `implement_plan`（.md / .ja.md）および `walkthrough`（.md / .ja.md）がすべて指定ディレクトリ内に生成されているか？
- [ ] **Git対象の完全性**: ソースファイルに加えて `.agent/tasks/{タイムスタンプ}/` 配下の全ファイルが `git add` の対象に含まれているか？