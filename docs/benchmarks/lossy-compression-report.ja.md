# ⚖️ 非可逆圧縮ベンチマーク実測レポート (Lossy Compression Report)

<p align="left">
  <a href="lossy-compression-report.md">English</a> | <b>日本語</b>
</p>

> **検証ステータス**: 実機実測データ検証済み  
> **ハードウェア**: Raspberry Pi 4 (Quad-Core ARM Cortex-A72 @ 1.5GHz, Pure Rust, GPU不使用)  
> **根拠設計仮説**: [System One 設計ドキュメント (仮説 H3: Lossy Compression)](../design/system-one-hypotheses.ja.md)

---

## 1. エグゼクティブサマリー

本ベンチマークは、System One 設計仮説 H3: **「意思決定は不可逆な非可逆情報圧縮であり、それゆえ超高速である」** に対する具体的な実機実験エビデンスを提供するものです。

- **情報圧縮率**: System One 意思決定の平均出力エントロピーは **5.0 bits** であり、自己回帰 LLM (50トークン生成: **110.2 bits**) と比較して **22.1倍の情報圧縮** を達成。
- **推論速度比**: System One は平均 **500 ms**（0.5秒）で判定を完了し、逐次自己回帰生成（**23,744 ms** = 約23.7秒）と比較して **47.4倍の高速化** を実証。

## 2. 実機測定結果一覧

| 分野 (Domain) | サンプル内容 | S2 レイテンシ (LLM生成) | S1 レイテンシ (Decide判定) | S2 エントロピー $H(Y)$ | S1 エントロピー $H(D)$ | 情報圧縮率 | 速度向上比 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rust コード** | フィボナッチ関数 | 24,186 ms | 519 ms | 93.5 bits | 5.4 bits | **17.4×** | **46.6×** |
| **Python コード** | 二分探索関数 | 24,056 ms | 496 ms | 58.0 bits | 4.6 bits | **12.5×** | **48.5×** |
| **技術・法規文書** | 暗号プロトコル仕様 | 24,237 ms | 491 ms | 113.9 bits | 5.6 bits | **20.4×** | **49.4×** |
| **文学 (青空文庫)** | 走れメロス冒頭 | 22,497 ms | 496 ms | 175.2 bits | 4.3 bits | **40.3×** | **45.4×** |
| **平均 / 総合** | - | **23,744 ms** | **500 ms** | **110.2 bits** | **5.0 bits** | **22.1×** | **47.4×** |

## 3. サンプル別判定詳細

### サンプル 1: Rust コード (フィボナッチ関数)
- **System One 判定結果**: `RustCode (確信度=59.4%), 構文異常=true, 複雑度=3.29`
- **System Two 生成文抜粋**: `        >>> if is a condirection and a surrecursin`

### サンプル 2: Python コード (二分探索関数)
- **System One 判定結果**: `PythonCode (確信度=88.7%), 構文異常=true, 複雑度=2.99`
- **System Two 生成文抜粋**: `.          //    .. methowePurePosixPath` popowing`

### サンプル 3: 技術・法規文書 (暗号プロトコル仕様)
- **System One 判定結果**: `LegalOrTechDoc (確信度=55.4%), 構文異常=true, 複雑度=1.13`
- **System Two 生成文抜粋**: ` RIn with on the it rent an will be retriallec:檀忘 `

### サンプル 4: 文学 (走れメロス冒頭)
- **System One 判定結果**: `Literature (確信度=93.6%), 構文異常=false, 複雑度=1.10`
- **System Two 生成文抜粋**: ` 　との武右衛門のは、私の一緒に、またベッテかに頬をかけて、動きに、中年の少年生の影を見たようにして`

---

## 4. 考察と意義

1. **不可逆性 (Irreversibility)**:
   決定情報 $[Choice, Noul, Score]$（合計約 5.0 bits）から元の入力文章を復元することは熱力学的に不可能です。しかしエッジデバイスにおけるタスクルーティングや制御判定には、テキスト生成は不要であり、5.0 bits の決定こそが必要十分な情報です。
2. **圧倒的レイテンシ差**:
   自己回帰ループ（1トークンごとにKVキャッシュと全層順伝播を回す）を廃止し、Mean Pooling と 1回の非可換回転（Quaternion Linear）で射影することで、Raspberry Pi 4 の CPU 上でも 24秒 $\to$ 0.5秒への劇的な短縮（47.4倍）が実現しました。

---

*ローカルでの再現コマンド: `cargo run --release -p oniwa-decide --bin compress_eval`*
