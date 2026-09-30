# [Task Plan / Issue Spec]: oniwa-decide 差分並列走査・NEON RmsNorm/Softmax最適化 ＆ チェックポイント調整・学習評価

## 1. Issue & 目的定義
- **背景と課題**:
  前回タスク（Issue #111）において、推論専用パス（`forward_inference`）の分離、ARM NEON Mean Pooling、適応長スライシング（$T_{\text{valid}}$）を実装し、単一差分ハンクの推論レイテンシを 500ms から 145ms へと大幅に短縮しました。
  さらなる高速化と実用性向上のため、以下の3点の実装および評価を実施します:
  1. **差分ハンク並列走査（Rayon）**:
     大規模コミットで複数ハンク（5〜10個以上）が存在する場合でも、全コアを活用して並列推論を行い、コミットフック全体の待ち時間を極小化。
  2. **NEON SIMD を用いた RmsNorm / Attention Softmax ベクトル化**:
     Transformer 層内のホットスポットである RmsNorm の正規化・スケーリングおよび Attention 行 Softmax を NEON 4 並列ベクトル化し、レイテンシをさらに短縮。
  3. **Gatekeeper チェックポイント探索優先度の調整 ＆ 追加学習（Fine-Tuning）と推論評価**:
     Full Quaternion モデル（770K params, 42ms p50）または System 1 MLM モデルの学習検証を実施し、実推論精度とレイテンシを総合評価する。

## 2. 影響ファイル一覧
- `crates/oniwa-lm/src/simd.rs`: [変更] `max_element_simd`, `sum_slice_simd` 等の SIMD プリミティブ追加と単体等価性テスト
- `crates/oniwa-decide/src/simd/mod.rs`: [変更] 新設 SIMD プリミティブの再エクスポート
- `crates/oniwa-decide/src/layers/attention.rs`: [変更] Attention 行 Softmax のベクトル化
- `crates/oniwa-decide/src/layers/quaternion_attention.rs`: [変更] Quaternion Attention 行 Softmax のベクトル化
- `crates/oniwa-decide/src/bin/gatekeeper.rs`: [変更] Rayon による複数ハンクの並列走査実装
- `.agent/tasks/20261001-064300/*`: [新規作成] `task.md`, `implementation_plan.md`, `implementation_plan.ja.md`, `walkthrough.md`, `walkthrough.ja.md`

## 3. グラフ的実装ステップ
- [ ] Phase 1: 計画策定と初期配置
- [ ] Phase 2: SIMD プリミティブ追加 ＆ 単体検証 (`cargo test -p oniwa-lm simd::tests`)
- [ ] Phase 3: Attention Softmax のベクトル化 ＆ 等価性検証 (`cargo test -p oniwa-decide`)
- [ ] Phase 4: Gatekeeper Rayon 並列走査の実装 ＆ 複数ハンク検証
- [ ] Phase 5: 必要に応じた学習（Training）の実施と推論評価・ベンチマーク計測
- [ ] Phase 6: コードフォーマット・Clippy・ワークスペーステスト全Green確認
- [ ] Phase 7: walkthrough 作成、Git コミット・プッシュ、PR Squashマージ、main同期
