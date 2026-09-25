# 実装計画 - System 1 (oniwa-decide) 8,000ステップ学習 ＆ 構文異常検知実機検証

## 概要
対照変異データセット（`data/killer_patterns.jsonl`）を用いて `oniwa-decide`（Full Q-Transformer）を 8,000 ステップ学習させ、以前確信度 2.1% で素通しされていた構文破壊コードに対する異常検知ヘッド（`Noul`）の性能を推論・実機検証します。

## 変更内容
1. **チェックポイント探索パスの整合**:
   - `crates/oniwa-decide/src/bin/audit.rs`: `checkpoints/full_quaternion/best` を優先探索に追加。
   - `crates/oniwa-decide/src/bin/gatekeeper.rs`: `checkpoints/full_quaternion/best` を優先探索に追加。
2. **8,000 ステップ学習の実行**:
   - `data/killer_patterns.jsonl` を用いた学習コマンド実行。
3. **実機推論・弁別検証**:
   - 正常コード推論: `Noul = false`（高確信度）の確認。
   - 破壊コード推論: `Noul = true`（確信度80%以上目標）の確認。
   - Gatekeeper CLI での BLOCK 動作検証。
4. **CI・フォーマット確認**:
   - `cargo test --workspace`
   - `cargo clippy --workspace --all-targets -- -D warnings`
   - `cargo fmt --all`
