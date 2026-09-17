# Phase 2: Adaptation to Modern Transformer Primitives & Manual Derivative Derivations

GPT-2, the archetype of `llm.c`, was based on the classical Transformer architecture of 2019.
In contemporary open model designs (e.g. Gemma, Llama 3), **Modern Transformer Primitives** have become the de facto standard to eliminate computational bottlenecks, maximize representation capacity, and optimize memory bandwidth.

This document formulates the mathematical definitions (forward pass) and derives the analytical partial derivatives (backward pass) for the four major components adopted in `oniwa-lm`: **RMSNorm**, **RoPE**, **SwiGLU**, and **GQA**.

---

## 1. RMSNorm (Root Mean Square Normalization)

RMSNorm simplifies LayerNorm by eliminating mean centering, normalizing inputs solely by their root mean square.

### 1.1 Forward Pass
For input vector $x \in \mathbb{R}^d$ and scaling parameter $\gamma \in \mathbb{R}^d$ (weight):

$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{j=1}^d x_j^2 + \epsilon}$$

$$\hat{x}_i = \frac{x_i}{\text{RMS}(x)}$$

$$y_i = \hat{x}_i \cdot \gamma_i$$

### 1.2 Analytical Derivative Derivation (Backward Pass)
Let the incoming gradient from upstream be $\frac{\partial L}{\partial y_i}$.

1. **Parameter Gradient $\frac{\partial L}{\partial \gamma_i}$**:
   $$\frac{\partial L}{\partial \gamma_i} = \sum_{B, T} \frac{\partial L}{\partial y_i} \cdot \hat{x}_i$$

2. **Input Gradient $\frac{\partial L}{\partial x_i}$**:
   By the chain rule:
   $$\frac{\partial L}{\partial x_i} = \frac{\gamma_i}{\text{RMS}(x)} \cdot \frac{\partial L}{\partial y_i} + \sum_{j=1}^d \left( \frac{\partial L}{\partial y_j} \cdot \gamma_j \cdot x_j \right) \cdot \frac{\partial (1/\text{RMS}(x))}{\partial x_i}$$

   Since $\frac{\partial (1/\text{RMS}(x))}{\partial x_i} = -\frac{x_i}{d \cdot \text{RMS}(x)^3}$, simplifying yields:

   $$\frac{\partial L}{\partial x_i} = \frac{1}{\text{RMS}(x)} \left[ \gamma_i \frac{\partial L}{\partial y_i} - \frac{\hat{x}_i}{d} \sum_{j=1}^d \left( \frac{\partial L}{\partial y_j} \cdot \gamma_j \cdot \hat{x}_j \right) \right]$$

> [!TIP]
> **Key Advantage for Manual Backward Passes**  
> In LayerNorm, both mean $\mu$ and variance $\sigma^2$ had to be cached, requiring complex nested reductions during backpropagation.  
> With RMSNorm, a single scalar dot product $S = \sum_j (\frac{\partial L}{\partial y_j} \cdot \gamma_j \cdot \hat{x}_j)$ allows computing input gradients in a single vector pass.

---

## 2. RoPE (Rotary Position Embedding)

Instead of adding an absolute positional embedding table, RoPE rotates adjacent pairs of channels on the complex plane for Query and Key vectors, allowing the dot product to naturally encode relative positional distance.

### 2.1 Forward Pass
For vector channels $(2i, 2i+1)$ at sequence position $m \in [0, T-1]$:

$$\theta_i = b^{-2i/d} \quad (b = 10000 \text{ or } 500000)$$

$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

### 2.2 Analytical Derivative Derivation (Backward Pass)
The rotation matrix $R_m = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$ is an **orthogonal matrix**.

Because the inverse of an orthogonal matrix equals its transpose ($R_m^T = R_m^{-1} = R_{-m}$), the input gradient with respect to upstream gradient $\frac{\partial L}{\partial q'}$ is simply computed by **applying the reverse rotation (inverting the sign of the angle)**:

$$\begin{pmatrix} \frac{\partial L}{\partial q_{2i}} \\ \frac{\partial L}{\partial q_{2i+1}} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & \sin(m\theta_i) \\ -\sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} \frac{\partial L}{\partial q'_{2i}} \\ \frac{\partial L}{\partial q'_{2i+1}} \end{pmatrix}$$

> [!NOTE]
> RoPE introduces **zero learnable parameters**.
> It requires no parameter gradient tracking or optimizer states, and its backward pass executes in-place using nearly identical logic to the forward pass by simply flipping the sine term.

---

## 3. SwiGLU (Swish Gated Linear Unit)

SwiGLU replaces the conventional `GELU(x W_1) W_2` MLP with a gated element-wise product of two linear projections, substantially enhancing representation fidelity.

### 3.1 Forward Pass
For input $x \in \mathbb{R}^d$ and hidden dimension $d_{\text{ffn}}$, using weight matrices $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ffn}}}$, $W_{\text{down}} \in \mathbb{R}^{d_{\text{ffn}} \times d}$:

1. $u = x W_{\text{gate}}$ (gate branch)
2. $v = x W_{\text{up}}$ (up branch)
3. $h = \text{Swish}(u) \odot v = \big( u \cdot \sigma(u) \big) \odot v$ ($\sigma$ is the sigmoid function)
4. $\text{out} = h W_{\text{down}}$

### 3.2 Analytical Derivative Derivation (Backward Pass)
Let the incoming gradient be $\frac{\partial L}{\partial \text{out}}$.

1. **Down-Projection Backward**:
   $$\frac{\partial L}{\partial h} = \frac{\partial L}{\partial \text{out}} W_{\text{down}}^T, \quad \frac{\partial L}{\partial W_{\text{down}}} = h^T \frac{\partial L}{\partial \text{out}}$$

2. **SwiGLU Element-wise Product Backward**:
   By the product rule:
   $$\frac{\partial L}{\partial v} = \frac{\partial L}{\partial h} \odot \text{Swish}(u)$$

   $$\frac{\partial L}{\partial u} = \frac{\partial L}{\partial h} \odot v \odot \text{Swish}'(u)$$

   Where the derivative of $\text{Swish}(u) = u \sigma(u)$ is:
   $$\text{Swish}'(u) = \sigma(u) + u \sigma(u)(1 - \sigma(u)) = \sigma(u) \big[ 1 + u (1 - \sigma(u)) \big]$$

3. **Gate / Up Projections Backward**:
   $$\frac{\partial L}{\partial W_{\text{gate}}} = x^T \frac{\partial L}{\partial u}, \quad \frac{\partial L}{\partial W_{\text{up}}} = x^T \frac{\partial L}{\partial v}$$
   $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial u} W_{\text{gate}}^T + \frac{\partial L}{\partial v} W_{\text{up}}^T$$

---

## 4. GQA (Grouped-Query Attention)

While Multi-Head Attention (MHA) pairs each Query head with its own Key/Value heads, GQA groups Query heads to share fewer Key/Value heads.

```text
Query Heads:    [Q0] [Q1]   [Q2] [Q3]   ... (HQ total)
                  \   /       \   /
KV Heads:         [KV0]       [KV1]     ... (HKV total, HQ / HKV = G)
```

### 4.1 Forward & Backward Pass Differences
* **Forward**:
  With group ratio $G = H_Q / H_{KV}$, Query head $q$ references shared KV head $k = \lfloor q / G \rfloor$.
* **Backward**:
  The gradients $\frac{\partial L}{\partial K_q}$ and $\frac{\partial L}{\partial V_q}$ calculated across Query heads reduce onto the shared KV heads via sum reduction across the group:

$$\frac{\partial L}{\partial K_k} = \sum_{g=0}^{G-1} \frac{\partial L}{\partial K_{k \cdot G + g}}, \quad \frac{\partial L}{\partial V_k} = \sum_{g=0}^{G-1} \frac{\partial L}{\partial V_{k \cdot G + g}}$$

---

## 5. Summary: Evolution from GPT-2 to oniwa-lm

| Primitive | GPT-2 (`llm.c`) | Modern (`oniwa-lm`) | Implementation Highlights in Pure Rust |
| :--- | :--- | :--- | :--- |
| **Normalizer** | LayerNorm | **RMSNorm** | Simplified backward compute path; enhanced cache locality |
| **Positional Encoding** | Absolute (WPE) | **RoPE** | Zero parameter embedding table; in-place reverse rotation |
| **Feed-Forward** | 2-layer MLP (GELU) | **SwiGLU (Gate+Up+Down)** | Superior expressivity; vectorized sigmoid activations |
| **Attention** | MHA | **GQA** | Drastically reduces KV memory traffic on edge CPUs |
| **Biases** | In all linears/norms | **No Bias** | Eliminates bias parameters and backward accumulation loops |
