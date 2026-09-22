# ONIWA Decide (TypeSafe System One 決定エンジン)

<p align="left">
  <a href="README.md">English</a> | <b>日本語</b>
</p>

`oniwa-decide` は、TypeSafe AI の System One 思想に着想を得た、ピュア Rust 製の型安全意思決定エンジンです。

テキスト生成（自己回帰）を行わず、入力されたコードやテキストからプログラムが直接分岐・制御に利用できる**型安全な決定プリミティブ（Choice, Noul, Score）**および較正された確信度（Confidence）を、CPU 単体・ミリ秒単位（Raspberry Pi 4 動作時にも ~100ms 台目標）の単一フォワードパスで出力します。

---

## 主要な設計・アーキテクチャ特徴

1. **非自己回帰 ＆ 単一フォワードパス決定**:
   - 生成ループを回さず、双方向 Transformer エンコーダにより系列全体を包括把握。
   - `Choice<T>`（カテゴリ分類・ラベル平滑化付きクロスエントロピー）、`Noul`（二値真偽・Brierスコア正則化付きBCE）、`Score`（連続評価・Smooth L1 / Huber損失）を出力。
2. **フラットメモリ 100MB バジェット制約**:
   - `FlatMemoryLayout`（`src/memory.rs`）により、埋め込み層・モデルパラメータ・オプティマイザ状態・逆伝播活性化値の総メモリ使用量を **厳格に 100MB 以内**（$V=4,096$ 時で実測約 32〜42MB）に収める設計。
   - 低リソースエッジ環境（Raspberry Pi 4）での OOM クラッシュを数理的・構造的に防止。
3. **MLM (Masked Language Modeling) 統合 ＆ キラーパターン合成学習**:
   - 静的解析（ASTや正規表現）をすり抜ける「意味論的ねじれ・Docコメントと実装の矛盾・順序非可換バグ」等のキラーパターンを検知するため、BERT型双方向 MLM 復元ヘッドを搭載。
   - 局所トークン復元損失（$\mathcal{L}_{\text{MLM}}$）と文全体決定損失（Choice, Noul, Score）を動的バランシングしたマルチタスク学習を実行。
4. **クォータニオン代数（四元数）バックボーン & ヘッド**:
   - ハミルトン積に基づくクォータニオン線形層（`QuaternionLinear`）により、重みパラメータを実数表現の約 $1/4$ に圧縮しつつ、語順や状態遷移の非可換性を表現。
   - ARM NEON SIMD 命令による高速ベクトル計算に対応。
5. **オンデバイス自律学習 ＆ 熱制御・グリーン電力監視**:
   - `oniwa-lm` と共通の熱動的スロットリングコントローラ（`/sys/class/thermal/` 監視）および電力追跡（Joule単位トラッキング）を統合。

---

## 決定プリミティブ

| プリミティブ | 型 | 役割 | 損失関数・較正 |
| :--- | :---: | :--- | :--- |
| **`Choice<T>`** | 列挙型 / カテゴリ | 入力文書種別・ルーティング判定（Rustコード、Python、法令/技術文書、文学等） | Cross-Entropy + Label Smoothing (0.05) |
| **`Noul`** | `bool` (真偽) | 構文異常・論理ねじれ・キラーパターンの有無フラグ | Binary Cross-Entropy + Brier Score正則化 |
| **`Score`** | `f32` (1.0〜5.0) | 構文複雑度・品質スコア評価 | Smooth L1 (Huber Loss, $\delta=0.5$) |

---

## CLIコマンドと実行例

### 1. 型安全監査・推論 (`audit`)
```bash
# コード片やテキストの型安全監査を実行
cargo run --release -p oniwa-decide --bin audit -- "pub fn fibonacci(n: u64) -> u64 { ... }"
```

### 2. 自己教師ありマルチタスク学習 (`train`)
```bash
# 基本学習 (300ステップ)
cargo run --release -p oniwa-decide --bin train -- --steps 300 --seed 42

# BPE 語彙ファイル（Phase 1）および キラーパターンデータセット（Phase 2）を指定して学習
cargo run --release -p oniwa-decide --bin train -- --steps 500 --bpe data/bpe_vocab.json --data data/killer_patterns.jsonl

# クォータニオンヘッド構成での学習
cargo run --release -p oniwa-decide --bin train -- --steps 300 --config quaternion_head

# 既存チェックポイントからの積み増し継続学習 (+100ステップ)
cargo run --release -p oniwa-decide --bin train -- --add-steps 100
```

---

## 仕様書・ロードマップ参照
- [10. ONIWA 実用化ロードマップ仕様書 (System 1 × System 2 統合)](../../docs/10_system_integration_roadmap.ja.md)
- [System One 設計仮説書](../../docs/design/system-one-hypotheses.ja.md)
- [学術的クォータニオン研究書](../../docs/09_roadmap.ja.md)
