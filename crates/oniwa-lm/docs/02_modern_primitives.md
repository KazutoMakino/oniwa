# Phase 2: 最新仕様（Modern Transformer Primitives）への適合と手動微分設計

`llm.c` の原型である GPT-2 は、2019年当時の古典的な Transformer アーキテクチャに基づいています。
2026年現在のオープンモデルの潮流（Gemma 4, Llama 3系など）では、計算効率・表現力・メモリ帯域のボトルネックを解消するために、より洗練された **「Modern Transformer Primitives」** がデファクトスタンダードとなっています。

本ドキュメントでは、`niwa-lm` に採用する4大コンポーネント（**RMSNorm**, **RoPE**, **SwiGLU**, **GQA**）の数学的定義（順伝播）と、手動逆伝播（Backward）に必要な偏微分の導出式を完全整理します。

---

## 1. RMSNorm (Root Mean Square Normalization)

LayerNormの「平均の減算」を省き、二乗平均平方根（RMS）のみで正規化する手法です。

### 1.1 順伝播（Forward）
入力ベクトル $x \in \mathbb{R}^d$、スケーリングパラメータ $\gamma \in \mathbb{R}^d$（重み）：

$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{j=1}^d x_j^2 + \epsilon}$$

$$\hat{x}_i = \frac{x_i}{\text{RMS}(x)}$$

$$y_i = \hat{x}_i \cdot \gamma_i$$

### 1.2 手動微分の導出（Backward）
上流から流れてきた出力勾配を $\frac{\partial L}{\partial y_i}$ とします。

1. **パラメータ勾配 $\frac{\partial L}{\partial \gamma_i}$**:
   $$\frac{\partial L}{\partial \gamma_i} = \sum_{B, T} \frac{\partial L}{\partial y_i} \cdot \hat{x}_i$$

2. **入力勾配 $\frac{\partial L}{\partial x_i}$**:
   連鎖律より、
   $$\frac{\partial L}{\partial x_i} = \frac{\gamma_i}{\text{RMS}(x)} \cdot \frac{\partial L}{\partial y_i} + \sum_{j=1}^d \left( \frac{\partial L}{\partial y_j} \cdot \gamma_j \cdot x_j \right) \cdot \frac{\partial (1/\text{RMS}(x))}{\partial x_i}$$

   ここで $\frac{\partial (1/\text{RMS}(x))}{\partial x_i} = -\frac{x_i}{d \cdot \text{RMS}(x)^3}$ であるため、整理すると：

   $$\frac{\partial L}{\partial x_i} = \frac{1}{\text{RMS}(x)} \left[ \gamma_i \frac{\partial L}{\partial y_i} - \frac{\hat{x}_i}{d} \sum_{j=1}^d \left( \frac{\partial L}{\partial y_j} \cdot \gamma_j \cdot \hat{x}_j \right) \right]$$

> [!TIP]
> **手動Backwardにおける圧倒的な利点**  
> LayerNormでは「平均 $\mu$」と「分散 $\sigma^2$」の2つの統計量を逆伝播まで保持し、複雑な平均減算の連鎖律を解く必要がありました。  
> RMSNormなら、スカラー内積 $S = \sum_j (\frac{\partial L}{\partial y_j} \cdot \gamma_j \cdot \hat{x}_j)$ を1回計算するだけで、入力勾配を一撃で算出できます。

---

## 2. RoPE (Rotary Position Embedding)

絶対位置埋め込みテーブルを足すのではなく、QueryとKeyの各チャネルペアを複素平面上で回転させることで、内積計算時に「相対的な位置関係」を自然に反映させる手法です。

### 2.1 順伝播（Forward）
位置 $m \in [0, T-1]$ におけるベクトル（Query または Key）の $2i$ 番目と $2i+1$ 番目のチャネルペアに対し：

$$\theta_i = b^{-2i/d} \quad (b = 10000 \text{ または } 500000)$$

$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

### 2.2 手動微分の導出（Backward）
回転行列 $R_m = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$ は**直交行列（Orthogonal Matrix）**です。

直交行列の逆行列は転置行列に等しいため（$R_m^T = R_m^{-1} = R_{-m}$）、上流の勾配 $\frac{\partial L}{\partial q'}$ に対する入力勾配は、**「逆回転（角度の符号反転）」を掛けるだけ**で求まります。

$$\begin{pmatrix} \frac{\partial L}{\partial q_{2i}} \\ \frac{\partial L}{\partial q_{2i+1}} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & \sin(m\theta_i) \\ -\sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} \frac{\partial L}{\partial q'_{2i}} \\ \frac{\partial L}{\partial q'_{2i+1}} \end{pmatrix}$$

> [!NOTE]
> RoPEには**学習可能なパラメータが一切存在しません**。
> そのためパラメータ勾配の計算やオプティマイザの更新バッファが不要で、順伝播とほぼ同一のコード（$\sin$ の符号を変えるだけ）でインプレースに逆伝播を実装できます。

---

## 3. SwiGLU (Swish Gated Linear Unit)

従来の GPT-2 で使われていた `GELU(x W_1) W_2` の代わりに、2本のプロジェクションの要素積をとるゲーティング機構です。現代の最高性能LLMでほぼ標準採用されています。

### 3.1 順伝播（Forward）
入力 $x \in \mathbb{R}^d$、隠れ層次元 $d_{\text{ffn}}$ に対し、3つの重み行列 $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ffn}}}$, $W_{\text{down}} \in \mathbb{R}^{d_{\text{ffn}} \times d}$ を使用：

1. $u = x W_{\text{gate}}$ （ゲート側）
2. $v = x W_{\text{up}}$ （アップ側）
3. $h = \text{Swish}(u) \odot v = \big( u \cdot \sigma(u) \big) \odot v$ （$\sigma$ はシグモイド関数）
4. $\text{out} = h W_{\text{down}}$

### 3.2 手動微分の導出（Backward）
上流勾配を $\frac{\partial L}{\partial \text{out}}$ とします。

1. **Down Proj の逆伝播**:
   $$\frac{\partial L}{\partial h} = \frac{\partial L}{\partial \text{out}} W_{\text{down}}^T, \quad \frac{\partial L}{\partial W_{\text{down}}} = h^T \frac{\partial L}{\partial \text{out}}$$

2. **SwiGLU 要素積の逆伝播**:
   積の微分法則より、
   $$\frac{\partial L}{\partial v} = \frac{\partial L}{\partial h} \odot \text{Swish}(u)$$

   $$\frac{\partial L}{\partial u} = \frac{\partial L}{\partial h} \odot v \odot \text{Swish}'(u)$$

   ここで $\text{Swish}(u) = u \sigma(u)$ の導関数は：
   $$\text{Swish}'(u) = \sigma(u) + u \sigma(u)(1 - \sigma(u)) = \sigma(u) \big[ 1 + u (1 - \sigma(u)) \big]$$

3. **Gate / Up Proj の逆伝播**:
   $$\frac{\partial L}{\partial W_{\text{gate}}} = x^T \frac{\partial L}{\partial u}, \quad \frac{\partial L}{\partial W_{\text{up}}} = x^T \frac{\partial L}{\partial v}$$
   $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial u} W_{\text{gate}}^T + \frac{\partial L}{\partial v} W_{\text{up}}^T$$

---

## 4. GQA (Grouped-Query Attention)

Multi-Head Attention (MHA) では Query と同じ数だけ Key/Value ヘッドを持ちますが、GQA では KV ヘッドをグループ化して削減します。

```text
Query Heads:    [Q0] [Q1]   [Q2] [Q3]   ... (HQ 個)
                  \   /       \   /
KV Heads:         [KV0]       [KV1]     ... (HKV 個, HQ / HKV = G)
```

### 4.1 順伝播・逆伝播における差分
* **順伝播**:
  グループ比率 $G = H_Q / H_{KV}$。
  Query ヘッド $q$ がアテンションスコアを計算する際、対応する KV ヘッド $k = \lfloor q / G \rfloor$ をブロードキャスト（参照）して使用します。
* **逆伝播**:
  $q$ ごとに逆伝播で計算された $\frac{\partial L}{\partial K_q}$ および $\frac{\partial L}{\partial V_q}$ は、**同じ KV ヘッドを参照していた $G$ 個の Query ヘッドの勾配の総和（Reduction / Add）** となります。

$$\frac{\partial L}{\partial K_k} = \sum_{g=0}^{G-1} \frac{\partial L}{\partial K_{k \cdot G + g}}, \quad \frac{\partial L}{\partial V_k} = \sum_{g=0}^{G-1} \frac{\partial L}{\partial V_{k \cdot G + g}}$$

---

## 5. まとめ：GPT-2からniwa-lmへの構造進化

| 項目 | GPT-2 (`llm.c`) | Modern (`niwa-lm`) | Rust実装時のポイント |
| :--- | :--- | :--- | :--- |
| **Normalizer** | LayerNorm | **RMSNorm** | 逆伝播の計算パスが単純化され、キャッシュ局所性が向上 |
| **Positional Encoding** | Absolute (WPE) | **RoPE** | 埋め込みテーブル不要、インプレース逆回転で完結 |
| **Feed-Forward** | 2-layer MLP (GELU) | **SwiGLU (Gate+Up+Down)** | 中間バッファが1本増えるが、シグモイド計算をSIMD化しやすい |
| **Attention** | MHA | **GQA** | KVメモリを劇的に節約し、小規模ハードウェアでの学習に有利 |
| **Bias項** | 全Linear/Normにあり | **No Bias（バイアス全廃）** | パラメータ数とBackwardループを大幅に削減（現代の主流） |
