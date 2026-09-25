# [Task Plan / Issue Spec]: System 1 (oniwa-decide) 8,000ステップ学習 ＆ 構文異常検知実機検証

## 1. Issue & 目的定義
- **背景と課題**:
  - `oniwa-decide`（System 1）を用いた Gatekeeper CLI の実機検証において、構文破壊コードに対する異常検知ヘッド（`Noul`）の確信度が約 2.1%（迷い状態）にとどまり、異常コードが素通し（PASS）される課題が確認された。
  - 前ステップ（Issue #93, #95）にて、1:1対照変異データセット `data/killer_patterns.jsonl`（5,000ペア / 10,000件）を生成し、バイナリ命名規則を統一。
  - 本ステップでは、`data/killer_patterns.jsonl` を用いて `oniwa-decide`（Full Q-Transformer）を 8,000 ステップ学習させ、以前迷いが生じていた構文破壊コードに対して実際に推論を実行し、`Noul` 確信度が大幅に向上して正しく異常検知できるかを実機検証する。
- **解決方針とスコープ**:
  - `audit.rs` / `gatekeeper.rs` のチェックポイント探索パスに `checkpoints/full_quaternion/best` を追加。
  - `oniwa-decide` の 8,000 ステップ学習を実行（`--steps 8000 --config full_quaternion --reset`）。
  - 学習済みモデルを用いて正常コードおよび構文破壊コードに対する推論（Audit / Gatekeeper）を実測検証。
  - 結果を `walkthrough.md` / `walkthrough.ja.md` にまとめ、コミット・PR作成・スカッシュマージ・main同期を完遂。
