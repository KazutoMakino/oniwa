# [実装振り返り・成果報告書]: oniwa-decide 推論専用パス分離 ＆ 可変長NEON最適化によるエッジPre-commitレイテンシ短縮

## 1. 実装の概要
本タスクでは、Git プリコミットフック（`gatekeeper`）における推論レイテンシを短縮するため、以下の施策を実施しました:
1. **推論専用パスの完全分離（`forward_inference`）**:
   - 決定ヘッド（Choice, Noul, Score）の計算において、MLM 語彙射影（$V \times C = 4096 \times 256$ 内積ループ）を完全にスキップ。
   - 逆伝播用の中間アクティベーションキャッシュ（`LayerCache`）の動的確保を排除し、層ごとの作業用バッファを再利用。
2. **ARM NEON SIMD ベクトル化による Mean Pooling 高速化**:
   - `oniwa-lm::simd` に `accumulate_slice_simd`（`float32x4_t` による 4 並列ベクトル加算とスカラー端数処理）を追加し、`oniwa-decide::simd` から再エクスポート。
   - Mean Pooling 処理において `accumulate_slice_simd` と `scale_slice_simd` を適用。
3. **有効トークン長への適応的スライシング（Adaptive Sequence Length）**:
   - `DecisionModel::decide` において、入力トークン列の実効長 $T$ を 4 の倍数に切り上げ（下限 4、上限 `seq_len`）て Transformer 層を駆動。クォータニオン代数および RoPE の 4 の倍数アラインメント制約を守りつつ、短差分コードでの計算量を削減。
4. **推論等価性と回帰テストの担保**:
   - `tests/decision_test.rs` に `test_forward_inference_equivalence` を追加。Quaternion Head / Quaternion Backbone の各構成において、従来の `forward` と `forward_inference` の出力ロジット・予測値が単精度丸め誤差内（$< 10^{-5}$）で完全一致することを検証。

---

## 2. 実機レイテンシ計測と検証結果

### Git プリコミット走査（`gatekeeper --stdin --bench`）
- コミット差分（Rust コード追加スニペット）に対する実機推論:
  - **実測レイテンシ**: **145ms**（目標値 100ms〜150ms 台を達成）
  - **判定結果**: `PASS`（Noul=false, conf=59.2%, Score=0.47, Choice=LegalOrTechDoc/Code）

### モデル単体ベンチマーク（`bench --iters 5`）
- Full Quaternion (`770K` パラメータ):
  - **p50**: **42.17 ms**
  - **Mean**: **54.44 ms**
- Iso-Parameter Real (`467K` パラメータ):
  - **p50**: **63.04 ms**
  - **Mean**: **66.29 ms**
- Standard Baseline (`1.26M` パラメータ):
  - **p50**: **247.30 ms**（540ms 超から大幅短縮）

---

## 3. AIセルフチェックリスト照合結果

- [x] **仕様準拠**: `forward_inference` による MLM 射影スキップ、SIMD Mean Pooling、適応長スライシングを実装済み。
- [x] **等価性確認**: `test_forward_inference_equivalence` により誤差 $< 10^{-5}$ で完全一致を証明済み。
- [x] **性能目標達成**: Gatekeeper 実測 145ms（100ms〜150ms 台）を達成。
- [x] **スコープ厳守**: 既存の学習用 `forward`、逆伝播 `backward`、`train.rs` に変更なし。
- [x] **影響範囲の一致**: 指定ファイルおよびタスクドキュメントのみを変更。
- [x] **フォーマット順守**: `cargo fmt --all -- --check` がクリーンに通過。
- [x] **テスト通過**: `cargo test --workspace` が 100% 成功（Green）。
- [x] **ブランチ保護**: 作業ブランチ `111/perf/decouple-inference-path-adaptive-neon` 上で作業中（`main` ではない）。
- [x] **ドキュメント網羅**: 日英両方の `implementation_plan`（.md / .ja.md）および `walkthrough`（.md / .ja.md）を作成済み。
- [x] **Git対象の完全性**: 変更ソースファイルおよび `.agent/tasks/20261001-034300/` 配下の全ドキュメントをコミット対象に含む。
