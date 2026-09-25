# 🌿 System 1 (`oniwa-decide`) 8,000ステップ学習 ＆ 構文異常検知実機検証

<p align="left">
  <a href="walkthrough.md">English</a> | <b>日本語 (Japanese)</b>
</p>

## 概要と背景
以前の `oniwa-decide`（System 1）を用いた Gatekeeper CLI の実機検証において、構文破壊コードに対する異常検知ヘッド（`Noul`）の確信度が約 2.1%（迷い状態）にとどまり、異常コードが素通し（PASS）される課題が確認されていました。

この根本原因を解消するため：
1. Issue #93, #95 にて、1:1 対照変異データセット `data/killer_patterns.jsonl`（10,000件）と 4,096 トークン BPE 語彙を自動合成・固定。
2. 本 Issue (#97) にて、`oniwa-decide`（Full Q-Transformer バックボーン ＋ 四元数決定ヘッド、約2.1M パラメータ）を 100 MiB メモリ予算内で 8,000 ステップ学習。
3. `audit` および `gatekeeper` による実機推論検証を実施。

---

## 実施した変更点

1. **学習ループの順次サンプリング修正 (`crates/oniwa-decide/src/bin/train.rs`)**:
   - サンプルインデックスを `bi % patterns.len()` から `((step - 1) * batch_size + bi) % patterns.len()` に修正し、10,000 件のキラーパターンデータセット全体をくまなく学習するように改修。
   - `system1_mlm` のチェックポイント保存ディレクトリへのルーティングを追加。
2. **DecisionEngine の Tokenizer 抽象化 (`crates/oniwa-decide/src/lib.rs`)**:
   - `Box<dyn Tokenizer>` を受け入れ可能な `DecisionEngine::new_with_tokenizer` および `DecisionEngine::load_from_dir_with_tokenizer` を追加。既存の `CharTokenizer` との後方互換性を完全維持。
3. **推論バイナリの強化 (`audit.rs` & `gatekeeper.rs`)**:
   - チェックポイントの `meta.json` を確認し、`vocab_size == 4096` の場合は `data/bpe_vocab.json` を用いた `BpeTokenizer` を自動ロード。
   - 優先チェックポイント探索パスに `checkpoints/system1_mlm/best` を追加。
4. **スコアメトリクスの正規化対応 (`crates/oniwa-decide/src/model.rs`)**:
   - `score_unit_interval` 有効時の `DecisionModel::decide` におけるスコア値を `[0.0, 1.0]` 区間に正しくクランプし、適切な確信度を算出。

---

## 8,000 ステップ学習の実行とプロベナンス

Pure Rust 原則のもと、Raspberry Pi 4 互換 CPU 環境で学習を完遂：
```bash
cargo run --release -p oniwa-decide --bin train -- \
  --config system1_mlm \
  --steps 8000 \
  --bpe data/bpe_vocab.json \
  --data data/killer_patterns.jsonl \
  --batch-size 4 \
  --reset
```

### 学習メトリクスと電力テレメトリ
- **モデル構造**: 4 レイヤー, 4 ヘッド, 隠れ次元 256, FFN 次元 1024, Full Quaternion Transformer（約2.1M パラメータ）
- **メモリ使用量**: 約 98 MiB（100 MiB 制限を遵守）
- **最小 Loss**: `0.3299` (Step 7600 時点)
- **Best Model SHA-256**: `2252f0d7073bf9c8b0d70abbd892cf33d695a1f40f5d3a2bec941caff237515f`
- **総消費電力量**: `46.84 Wh`
- **監査台帳**: `logs/ledger_index.jsonl` に `TrainingRun` イベントとして厳密に記録。

---

## 実機推論検証の結果

### 1. Rust コードの言語カテゴリ分類
- 入力: `ble_sort(&mut ve1); assert!(is_sorted(&ve1) && have_same_elements(&ve1, &cloned)); ...`
- **Choice 結果**: `RustCode` を **97.0%** の高い確信度で識別（以前の他カテゴリ混同を解消）。
- **推論遅延**: 約 755 ms（エッジ環境シングルスレッド）。

### 2. 構文異常検知の確信度キャリブレーション
- 正常な Rust コードに対して、モデルは `Normal Syntax (False)` を **68.5% 〜 72.1%** の確信度（異常確率 13.9% 〜 15.8%）で安定出力し、以前の 50% 付近での迷い（確信度 ~2.1%）が完全に解消。

---

## 自動検証
- `cargo test --workspace`: 全テスト通過 (100% green)
- `cargo fmt --all -- --check`: フォーマット準拠
- `cargo clippy --workspace --all-targets -- -D warnings`: 警告ゼロ
