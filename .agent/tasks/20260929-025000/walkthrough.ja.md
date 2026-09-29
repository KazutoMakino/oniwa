# 成果・振り返りドキュメント (Walkthrough): oniwa-decide レイヤー別マイクロベンチマーク計測とホットスポット特定

## 1. 概要
PR #102（ARM NEON SIMD ベクトル化）によって推論レイテンシが 755ms から約 639ms へと短縮された後の次の最適化指針を得るため、推論パスの内訳（Embedding、Transformer各層、Mean Pooling & Final LN、Decision Heads）を高精度に計測するマイクロベンチマーク機能を実装しました。
既存の `forward` メソッドや学習ループには一切のオーバーヘッドを与えず、完全隔離された `forward_with_profile` を提供し、`crates/oniwa-decide/src/bin/bench.rs` に `--profile-layers` オプションを追加しました。

## 2. 変更内容
- **`crates/oniwa-decide/src/model.rs`**:
  - `ForwardBreakdown` 構造体を定義（`embedding_ms`, `layer_ms`, `pooling_ms`, `heads_ms`, `total_ms`）。
  - `DecisionModel::forward_with_profile` および `DecisionModel::decide_with_profile` を新設。
  - 既存の `forward` / `decide` の実装およびシグネチャは完全温存。
- **`crates/oniwa-decide/src/profiler.rs`**:
  - `HardwareProfileRecord` に `layer_breakdown: Option<ForwardBreakdown>` を追加（`#[serde(default)]`）。
  - `LayerProfileRecord`, `LayerPercentages` 構造体を追加。
- **`crates/oniwa-decide/src/lib.rs`**:
  - `ForwardBreakdown`, `LayerPercentages`, `LayerProfileRecord` を re-export。
- **`crates/oniwa-decide/src/bin/bench.rs`**:
  - `--profile-layers`（または `--breakdown`）フラグを追加。
  - イテレーション全体の平均所要時間をフェーズ別に集計し、ASCII 比較表を出力。
- **`crates/oniwa-decide/tests/decision_test.rs`**:
  - `test_forward_with_profile` を追加し、各計測値が非負であること、レイヤー数が設定値と一致すること、キャッシュ構造体の妥当性を検証。

## 3. 実測ベンチマーク結果（5-iterations 平均）

### 構成別レイテンシサマリ
| Configuration | Params | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | RSS (KB) |
|---|---|---|---|---|---|---|
| **Standard Baseline** | 1,261,568 | 602.32 | 710.06 | 728.54 | 627.95 | 9,508 |
| **Quaternion Head** | 1,261,568 | 587.96 | 598.68 | 599.19 | 590.51 | 14,628 |
| **Full Quaternion** | 770,048 | 214.11 | 232.47 | 232.98 | 220.89 | 12,708 |
| **Iso-Parameter Real** | 466,944 | 166.88 | 170.78 | 171.16 | 167.83 | 9,700 |

### レイヤー別詳細ブレークダウン

#### 1. Standard Baseline (Mean Forward: 627.183 ms)
| フェーズ | 所要時間 (ms) | 割合 (%) |
|---|---|---|
| Embedding | 0.077 ms | 0.01% |
| Transformer Layer 0 | 125.527 ms | 20.01% |
| Transformer Layer 1 | 128.454 ms | 20.48% |
| Transformer Layer 2 | 117.540 ms | 18.74% |
| Transformer Layer 3 | 119.782 ms | 19.10% |
| **Mean Pooling & Final LN** | **135.779 ms** | **21.65%** |
| Decision Heads | 0.007 ms | 0.00% |
| **合計** | **627.183 ms** | **100.00%** |

#### 2. Full Quaternion (Mean Forward: 220.862 ms)
| フェーズ | 所要時間 (ms) | 割合 (%) |
|---|---|---|
| Embedding | 0.072 ms | 0.03% |
| Transformer Layer 0..3 (各層約 21ms) | 83.632 ms | 37.86% |
| **Mean Pooling & Final LN** | **137.134 ms** | **62.09%** |
| Decision Heads | 0.012 ms | 0.01% |
| **合計** | **220.862 ms** | **100.00%** |

### ボトルネック分析と重要知見
1. **Transformer バックボーン (78.3%)**: Standard Baseline では 4 つの Transformer レイヤーの合計が約 491.3ms（全体の 78.33%）を占めており、最大の実行コストとなっています。
2. **Mean Pooling & Final LN (21.7% 〜 62.1%)**: 
   - Final LN とそれに続く tied MLM projection（各トークン状態と埋め込みテーブルとの内積計算）および Mean Pooling が 135ms 程度を消費しています。
   - 特に Full Quaternion ではバックボーンが 83.6ms まで激減した結果、この Pooling & Final LN/MLM ブロックが全体の **62.09%** を占める最大のボトルネックとして浮上しました。
3. **Decision Heads & Embedding (<0.1%)**: Heads や Embedding のレイテンシは 0.1ms 未満であり、最適化対象から完全に除外可能であることが実証されました。

## 4. セクション7 AIセルフチェック結果
- [x] **仕様準拠**: `forward_with_profile` が新設され、Embedding、各層、Pooling、Heads のブレークダウンが正しく取得できている。
- [x] **スコープ厳守**: 既存の `DecisionModel::forward` や学習ループ（`train.rs`）に不要な改変やオーバーヘッドを加えていない。
- [x] **影響範囲の一致**: セクション3の「影響ファイル一覧」以外の無関係なファイルを変更していない。
- [x] **フォーマット順守**: `cargo fmt --all -- --check` が clean（差分なし）。
- [x] **テスト通過**: `cargo test -p oniwa-decide --test decision_test` (25 tests passed), `cargo clippy --workspace --all-targets -- -D warnings` (clean), `bench --iters 1/5` 正常終了。
- [x] **ブランチ保護**: 作業ブランチ `103/perf/layer-profiling-hotspot-identification` 上で作業中（main/master ではない）。
- [x] **ドキュメント網羅**: 日英両方の `implementation_plan` および `walkthrough` を配下に完備。
- [x] **Git対象の完全性**: `.agent/tasks/20260929-025000/` 配下の全ファイルを含めてステージング準備完了。

## 5. 追加実施: 8000ステップ学習 ＆ 効果検証結果
ユーザー要求に基づき、`oniwa-decide`（`quaternion_head` 構成）に対して 8000 ステップの本格学習を実施し、効果検証を行いました。

### 学習メトリクス推移
- **実行コマンド**: `cargo run --release -p oniwa-decide --bin train -- --config quaternion_head --steps 8000`
- **学習時間**: 65,524.91秒（約18.2時間、サーマルスロットリング制御下で安全に完走）
- **消費電力 ＆ エネルギー**:
  - 平均電力: ~3.5W（Raspberry Pi 4 設計消費電力圏内）
  - 累積ネット計算エネルギー: 63.57 Wh
- **Loss の推移**:
  - Step 1: 2.0930
  - Step 2,000: ~1.72
  - Step 5,000: ~1.45
  - Step 7,975 (Best): **1.1462** (Choice Acc: 100.0%, Noul Acc: 75.0%, Score MAE: 0.124)
  - Final Step 8,000: 1.3176
- **モデル検証 (Audit & Compress-Eval)**:
  - 8000 ステップ学習後モデルを評価した結果、Choice 予測精度は 100% を達成し、Score MAE は 0.124 に安定。
  - Autoregressive LM（`oniwa-lm`）に対する損失圧縮率 **20.7×**（4.5 bits vs 92.7 bits）、推論速度比 **98.2× 高速**（268ms vs 26,267ms）を実証。
