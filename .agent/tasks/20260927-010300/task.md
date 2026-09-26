# [Task Plan / Issue Spec]: refactor: Rust API Guidelines (RFC 430) 準拠に伴う頭字語・公開シグネチャ・命名規則の全面整合

## 1. Issue & 目的定義
- **背景と課題**: 
現在コードベース（`crates/oniwa-decide`, `crates/oniwa-lm` 等）において、`RMSNorm`、`SwiGLU`、`MLM` といった頭字語を含む型名のケースの揺らぎや、`BidirectionalSelfAttention` と `QuaternionSelfAttention` のような冗長なプレフィックス、関数・メソッドの公開シグネチャにおける極端な略語引数名（`nh`, `d_h`, `c_quat`, `ffn` 等）が存在します。Rust API ガイドライン（RFC 430: 頭字語は 1 単語として CamelCase 化、自己文書化された引数名）に準拠した洗練されたインターフェースへ全面的に是正する必要があります。
- **解決方針とスコープ**: 
 1. **公開型名・列挙型の RFC 430 適合**:
    - `RMSNorm` → `RmsNorm`
    - `SwiGLU` / `QuaternionSwiGLU` → `SwiGlu` / `QuaternionSwiGlu`
    - `BidirectionalSelfAttention` / `QuaternionSelfAttention` → `Attention` / `QuaternionAttention`（モジュールスコープを活用し、冗長な語を整理）
    - MLM 関連型・定数・関数のケース整理（`MaskedTokens`, `apply_mlm_mask` 等の整合）
 2. **公開シグネチャ（関数・メソッド）の自己文書化**:
    - `c` → `dim`, `nh` → `num_heads`, `d_h` → `head_dim`, `ffn` → `ffn_dim` 等、公開インターフェースの引数名を略語から標準的名称へ変更（計算カーネル内部のタイトな局所変数は数式との対照性を尊重して過度な破壊を避ける）。
 3. **チェックポイントおよび設定のシリアライズ互換性維持**:
    - `ModelConfig`、`DecisionConfig`、`meta.json` などの JSON シリアライズ定義では、既存のフィールド名・キー名を `#[serde(rename = "...")]` または現状維持とすることで、保存済みチェックポイントの入出力を破壊しない。
 4. **ファイル・モジュール構成の維持**:
    - ファイルの過度な移動・階層化は行わず、現行ファイル構成（`layers/attention.rs`, `layers/quaternion_attention.rs` 等）を維持して差分を最小限に制御する。
- **やらないこと（Non-Goals）**: 
 - 計算カーネル（SIMD・内積計算・テンソル演算ループ）内部の局所変数の過剰な完全展開（数式・GHR微積分の対照性を優先）。
 - チェックポイント `meta.json` 形式の破壊的変更（完全リセットではなく JSON 互換を維持）。
 - ディレクトリ階層の大規模な移動（`layers/quaternion/` などの新設は行わない）。

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
  - **コードフォーマット規律**: 静的解析（Lint/型チェック）は実行せず、変更・新規作成したファイルに対して `cargo fmt --all` によるフォーマットのみを適用すること（Pythonファイルがある場合は `isort` と `ruff format` を適用）。

## 3. 影響ファイル一覧
- `crates/oniwa-decide/src/layers/rmsnorm.rs`: `RMSNorm` → `RmsNorm` の改修および公開引数名の整理
- `crates/oniwa-decide/src/layers/mlp.rs`: `SwiGLU` → `SwiGlu` の改修および公開引数名の整理
- `crates/oniwa-decide/src/layers/attention.rs`: `BidirectionalSelfAttention` → `Attention` の改修および公開引数名の整理
- `crates/oniwa-decide/src/layers/quaternion_linear.rs`: 公開関数シグネチャの引数名整理
- `crates/oniwa-decide/src/layers/quaternion_attention.rs`: `QuaternionSelfAttention` → `QuaternionAttention` の改修および公開引数名整理
- `crates/oniwa-decide/src/layers/quaternion_mlp.rs`: `QuaternionSwiGLU` → `QuaternionSwiGlu` の改修および公開引数名整理
- `crates/oniwa-lm/src/layers/rmsnorm.rs`: `RMSNorm` → `RmsNorm` の改修
- `crates/oniwa-lm/src/layers/mlp.rs`: `SwiGLU` → `SwiGlu` の改修
- `crates/oniwa-lm/src/layers/attention.rs`: `CausalSelfAttention` 等の型名・公開引数名の整理
- `crates/oniwa-lm/src/layers/quaternion_head.rs`: 公開引数名の整理
- `crates/oniwa-decide/src/model.rs`: 改修したレイヤー型名および公開メソッドシグネチャの呼出追従
- `crates/oniwa-lm/src/model.rs`: 改修したレイヤー型名および公開メソッドシグネチャの呼出追従
- `crates/oniwa-decide/tests/decision_test.rs`: 変更された型名・公開シグネチャのテスト追従
- `crates/oniwa-decide/src/bin/*.rs`: 各バイナリでの呼出追従
- `crates/oniwa-lm/src/bin/*.rs`: 各バイナリでの呼出追従
- `.agent/tasks/{タイムスタンプ}/*`: タスク指示書、計画書、walkthrough等の各ドキュメント

## 4. 実装ステップ（サブエージェントへの作業分割案）
1. Phase 1: [タイムスタンプディレクトリの作成と初期ドキュメント配置]
   - `.agent/tasks/{タイムスタンプ}/` を作成し、`task.md`、`implement_plan.md`、`implement_plan.ja.md` を配置。
2. Phase 2: [レイヤー型名および公開引数シグネチャの改修]
   - 対象ファイル: `crates/oniwa-decide/src/layers/*.rs`, `crates/oniwa-lm/src/layers/*.rs`
   - 具体指示:
     - 型名を RFC 430（頭字語は単語扱い CamelCase: `RmsNorm`, `SwiGlu`, `QuaternionAttention` 等）へリネーム。
     - `forward`, `backward` などの公開関数のシグネチャで略語（`c` → `dim`, `nh` → `num_heads`, `d_h` → `head_dim`, `ffn` → `ffn_dim`）を明瞭な名称に修正。
3. Phase 3: [モデル本体および利用側コードの追従]
   - 対象ファイル: `crates/oniwa-decide/src/model.rs`, `crates/oniwa-lm/src/model.rs`, `src/bin/*.rs`
   - 具体指示:
     - 各レイヤーの呼び出し箇所を新インターフェースへ追従。
     - チェックポイントのシリアライズ（`serde`）フィールドに影響がないか確認し、必要に応じてタグや属性を保護。
4. Phase 4: [テストコード追従・フォーマット適用]
   - 対象ファイル: `crates/**/tests/**/*.rs`, `crates/**/src/layers/*.rs`
   - 具体指示:
     - 単体テスト内のレイヤー呼び出しを追従修正。
     - `cargo test --workspace` を実行し、全件 Green であることを確認。
     - `cargo fmt --all` を適用。
5. Phase 5: [AIセルフチェック ＆ walkthrough 作成]
   - セクション7のセルフチェックリストを照合し、`walkthrough.md` および `walkthrough.ja.md` に結果を記録。
6. Phase 6: [Git コミット・プッシュ]
   - 変更したソースファイル群および `.agent/tasks/{タイムスタンプ}/` 配下の全ドキュメントをステージングしてコミット・プッシュ。

## 5. エッジケースと制約事項
- **チェックポイント互換性の維持**:
  JSON シリアライズ対象の構造体フィールド名（`meta.json` 出力）は既存の形式を壊さないこと。フィールド名を変える場合は `#[serde(rename = "...")]` を適用すること。
- **過剰展開の抑制**:
  テンソル演算ループや SIMD 内積ループ内部の局所変数は、数理論文や GHR 微積分コードの対照性を損なわない範囲に留めること。
- **Pure Rust の厳守**:
  新規外部依存（クレート）は一切追加しないこと。

## 6. 完了判定・デプロイコマンド（検証・フォーマット・Git）
- コードフォーマット適用:
  - `cargo fmt --all`
  - `uv run isort .`（※Pythonファイルに触れた場合のみ）
  - `uv run ruff format .`（※Pythonファイルに触れた場合のみ）
- テスト実行:
  - `cargo test --workspace`
- 動作検証（スモークテスト）:
  - `cargo run --release -p oniwa-decide --bin gatekeeper -- --help`
  - `cargo run --release -p oniwa-lm --bin chat -- --help`
- Git コミット ＆ プッシュ:
  - `git add <変更したソースファイル群> .agent/tasks/<生成されたタイムスタンプ>/`
  - `git commit -m "refactor: Rust API Guidelines (RFC 430) 準拠に伴う頭字語・公開シグネチャ・命名規則の全面整合 (#<issue番号>)"`
  - `git push origin <現在のブランチ名>`

## 7. AIセルフチェックリスト（実装完了判定）
エージェントはコミット前に以下のチェックを自律的に実施し、すべてクリアしていることを確認すること（walkthrough にチェック結果を記載すること）。
- [ ] **仕様準拠**: `RmsNorm`, `SwiGlu`, `QuaternionAttention` 等の RFC 430 準拠型名および公開引数名が漏れなくコードに反映されているか？
- [ ] **スコープ厳守**: チェックポイントの JSON 互換性が維持されており、不要なファイル移動や過剰な内部変数展開が行われていないか？
- [ ] **影響範囲の一致**: セクション3の「影響ファイル一覧」以外の無関係なファイルを変更していないか？
- [ ] **フォーマット順守**: `cargo fmt --all` を実行し、差分がない状態になっているか？
- [ ] **テスト通過**: `cargo test --workspace` が 100% エラーなく成功（Green）しているか？
- [ ] **ドキュメント網羅**: 日英両方の `implement_plan`（.md / .ja.md）および `walkthrough`（.md / .ja.md）がすべて指定ディレクトリ内に生成されているか？
- [ ] **Git対象の完全性**: ソースファイルに加えて `.agent/tasks/{タイムスタンプ}/` 配下の全ファイルが `git add` の対象に含まれているか？