# [Task Plan / Issue Spec]: oniwa-lm へのクォータニオン自己回帰生成レイヤー（QuaternionLMHead & Weight Tying）の導入

## 1. Issue & 目的定義
- **背景と課題**: 
  現在、`oniwa-lm` は実数空間（dim 128）の自己回帰Transformerとして実装されていますが、語彙射影層（LM Head）における語順の非可換性（方向性）の表現力不足や、エッジデバイス（Raspberry Pi 4）でのパラメータ効率が課題となっていました。一方、`oniwa-decide`（System One）において、ARM NEON SIMDで高速化されたクォータニオン線形層およびGHR微積分が単体テスト・勾配チェック100%パスで完成しており、この数理資産を生成モデルへ還元することが求められています。
- **解決方針とスコープ**: 
  既存の実数モデル構造体（`ModelConfig` / `ModelWeights`）を破壊することなく、`oniwa-lm` 内にクォータニオン代数を用いた双対構造を持つ新モジュール（`QuaternionLMHead`）およびクォータニオン版モデル構成（`QuaternionModelWeights`）を分離新設します。
  語彙埋め込み（Embedding）と出力射影（LM Head）の間で厳密な重み共有（Strict Weight Tying: $E \in \mathbb{H}^{V \times (D/4)}$）を行い、トークン $v$ のロジット計算をエルミート共役ハミルトン積の実部抽出（実質的な成分ドット積）として実装します。数理演算には `oniwa-decide` の検証済み NEON SIMD プリミティブを再利用します。
  本タスク仕様書および関連ドキュメントは `.agent/tasks/{yyyymmdd-hhmmss}/` に一元配置し、Git管理対象として追跡します。
- **やらないこと（Non-Goals）**: 
  - 既存の実数空間用 `ModelWeights`、チェックポイント読み書きロジック（`checkpoints/best` 等）の破壊的変更。
  - バックボーン（中間層の `Attention` や `SwiGLU`）の即時クォータニオン化（まず最上流 Embedding と最下流 LM Head の結合・生成能力を検証する）。
  - 語彙サイズ $V$ へのパディング追加やデータ前処理パイプライン（`prepare_data.py`）の改変。
  - クレートの再編（`oniwa-core` 新設などのワークスペース全体の変更は行わず、Cargo 依存の追加にとどめる）。

## 2. エージェント実行体制・開発運用ルール
本指示を受け取ったエージェントは、以下の「Issueドリブン ＆ 階層型エージェント体制」を厳格に遵守して自走すること。

### 体制と責務
- **メインエージェント（司令塔）**:
  - 本指示書および関連する既存ソースコードを深く理解し、全体の進行管理と統合テストを担当する。
  - リポジトリの `.agent/tasks/{yyyymmdd-hhmmss}/` ディレクトリに `implement_plan.md` を作成・管理する。
  - サブタスクごとに作業スコープを切り出し、サブエージェントへ明確な指示を渡す。
  - 実装完了後、全体の変更差分と動作検証結果をまとめた `walkthrough.md` を同ディレクトリ（`.agent/tasks/{yyyymmdd-hhmmss}/walkthrough.md`）に作成する。
- **サブエージェント（実装担当）**:
  - メインエージェントから指示された特定モジュールの実装・テストコード作成に忠実に従う。
  - 自身が担当した実装の詳細ステップと検証手順をサブエージェント視点のプラン/ログとして記録・更新する。
  - コードフォーマット規律: 静的解析（Lint/型チェック）は実行せず、変更・新規作成したファイルに対して isort と ruff format によるフォーマットのみを適用すること（Rustファイルについては `cargo fmt --all` を適用）。

## 3. 影響ファイル一覧
- `.agent/tasks/{yyyymmdd-hhmmss}/task.md`: [新規作成] 本タスク指示書の永続化配置
- `.agent/tasks/{yyyymmdd-hhmmss}/implement_plan.md`: [新規作成] メインエージェントの作業実行計画書
- `.agent/tasks/{yyyymmdd-hhmmss}/walkthrough.md`: [新規作成] 実装差分および動作検証結果レポート
- `crates/oniwa-lm/Cargo.toml`: `oniwa-decide` への依存パス定義追加
- `crates/oniwa-decide/src/lib.rs`: `simd` モジュールおよびクォータニオン基本関数のパブリック公開（`pub mod simd` / `pub use simd::*`）
- `crates/oniwa-lm/src/layers/mod.rs`: `quaternion_head` モジュールのエクスポート追加
- `crates/oniwa-lm/src/layers/quaternion_head.rs`: [新規作成] クォータニオン Weight Tying 語彙射影層（Forward / Backward）および数値微分テスト
- `crates/oniwa-lm/src/model.rs`: クォータニオン生成を司る `QuaternionModelWeights` の新設（既存 `ModelWeights` は温存）
- `crates/oniwa-lm/src/bin/train.rs`: `--quaternion` フラグの追加と、クォータニオンモデルの分岐初期化・学習パスの追加
- `crates/oniwa-lm/src/bin/chat.rs`: クォータニオンチェックポイントからの対話テキスト自己回帰生成対応

## 4. 実装ステップ（サブエージェントへの作業分割案）
1. Phase 1: 基盤・インターフェース作成とクォータニオンLMヘッド層の実装
   - 対象ファイル: `crates/oniwa-decide/src/lib.rs`, `crates/oniwa-lm/Cargo.toml`, `crates/oniwa-lm/src/layers/quaternion_head.rs`, `crates/oniwa-lm/src/layers/mod.rs`
   - 具体指示:
     - `crates/oniwa-decide/src/lib.rs` で `pub mod simd;` をエクスポートし、`accumulate_hamilton_simd`, `dot_product_4d_simd` をクレート外から呼び出せるようにする。
     - `crates/oniwa-lm/Cargo.toml` に `oniwa-decide = { path = "../oniwa-decide" }` を追加する。
     - `crates/oniwa-lm/src/layers/quaternion_head.rs` を新規作成する。
       - 入力隠れ状態 $h \in \mathbb{R}^{B \times T \times D}$ を $\mathbb{H}^{B \times T \times (D/4)}$ として解釈。
       - 語彙重み $E \in \mathbb{H}^{V \times (D/4)}$（実数換算で $V \times D$ パラメータ）を保持し、$\mathcal{N}(0, 1/\sqrt{D})$ で初期化。
       - Forward: 各トークン $v$ に対し、$\text{logits}_{b,t,v} = \sum_{k=1}^{D/4} \text{Re}(E[v, k]^* \otimes h_{b,t}[k]) = \sum_{k=1}^{D/4} \text{dot\_product\_4d}(E[v, k], h_{b,t}[k])$ を追加スケーリングなしで計算。
       - Backward: 上流勾配 $d\text{logits}$ に対し、隠れ状態への勾配 $dh_{b,t} = \sum_v d\text{logits}_{b,t,v} \cdot E[v]$、および語彙重みへの勾配 $dE[v] += \sum_{b,t} d\text{logits}_{b,t,v} \cdot h_{b,t}$ を計算。
       - ユニットテストとして、乱数入力を用いた数値微分（Finite Differences）との勾配比較テストを実装し、誤差 $< 5 \times 10^{-3}$ をパスさせる。
2. Phase 2: クォータニオン自己回帰モデル構造体（QuaternionModelWeights）の実装
   - 対象ファイル: `crates/oniwa-lm/src/model.rs`
   - 具体指示:
     - 既存の `ModelWeights` は一切変更せず、新たに `QuaternionModelWeights` 構造体を定義する。
     - トークン埋め込み（Embedding lookup）は `QuaternionLMHead` の重み $E$ をそのまま参照し、入力 ID に対応するベクトルを取得する。
     - Transformer 中間層（RMSNorm, RoPE, Causal Attention, SwiGLU）を通った後、最終 RMSNorm 出力を `QuaternionLMHead` に通して語彙ロジットを出力する。
     - 逆伝播（`backward`）およびパラメータ更新（`AdamW`）において、共有語彙重み $E$ に Embedding 側と LM Head 側の双対勾配が正しく累積されるように配線する。
3. Phase 3: テスト作成・CLI検証と生成動作確認
   - 対象ファイル: `crates/oniwa-lm/src/bin/train.rs`, `crates/oniwa-lm/src/bin/chat.rs`
   - 具体指示:
     - `train.rs` の引数パーサーに `--quaternion` フラグを追加する。
     - `--quaternion` 指定時は `QuaternionModelWeights` を初期化し、チェックポイント保存先を `checkpoints/quaternion/` 配下に設定して既存の実数チェックポイント（`checkpoints/best`）の上書き事故を完全に隔離する。
     - `chat.rs` にも同様に `--quaternion` フラグを追加し、クォータニオン重みをロードして温度付きサンプリング（Temperature / Top-p）による自己回帰逐次生成ができるようにする。
     - 4大プローブ（「その時、」「メロスは、」「吾輩は、」「私は、」）に対する生成テキストの出力を確認する。

## 5. エッジケースと制約事項
- **隠れ次元のアサーション**: 隠れ次元 $D$（dim）は必ず 4 の倍数（32クォータニオン＝128次元等）でなければならない。初期化時にアサーションで明示的にガードすること。
- **語彙サイズと端数処理**: 語彙サイズ $V$ はアラインメント変更やパディングを行わず、既存の `vocab.json` に記載されたユニーク文字数をそのまま使用する。
- **ロジットスケーリングの数値挙動**: ロジットに対して余分なスケーリング定数を挟まず、生の内積値（分散およそ 1.0）として出力することで、既存の学習率や Cross-Entropy 設定との整合性を保つ。
- **チェックポイントの形式識別**: `meta.json` 内に `"model_type": "quaternion_tied"` を付与し、実数モデルローダーが誤ってクォータニオン重みをロードしようとした際に安全にエラーを返すこと。

## 6. 完了判定コマンド（検証・フォーマット）
- コードフォーマット適用:
  - uv run isort .
  - uv run ruff format .
  - cargo fmt --all
- テスト実行:
  - cargo test -p oniwa-lm -- --nocapture
- 実機動作検証（スモークテスト）:
  - 学習 10 ステップの実行:
    `cargo run --release -p oniwa-lm --bin train -- --quaternion --reset --steps 10`
  - 対話生成テスト:
    `cargo run --release -p oniwa-lm --bin chat -- --quaternion`