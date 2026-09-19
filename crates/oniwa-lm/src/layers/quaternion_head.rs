//! Quaternion LM Head with Strict Weight Tying
//!
//! Dual structure connecting token embedding and language model vocabulary projection
//! in the quaternion domain $\mathbb{H}^{D/4}$.
//!
//! Mathematical formulation:
//! - Vocabulary embedding weight: $E \in \mathbb{H}^{V \times (D/4)}$ (real dimension $V \times D$)
//! - Input hidden state: $h \in \mathbb{H}^{N \times (D/4)}$ (real dimension $N \times D$, where $N = B \times T$)
//! - Forward Logits:
//!   $$\text{logits}_{i, v} = \sum_{k=1}^{D/4} \text{Re}(E[v, k]^* \otimes h_i[k]) = \sum_{k=1}^{D/4} \text{dot\_product\_4d}(E[v, k], h_i[k])$$
//! - Backward:
//!   - Upstream gradient w.r.t logits: $d\text{logits} \in \mathbb{R}^{N \times V}$
//!   - Gradient w.r.t hidden state $h_i$:
//!     $$dh_i = \sum_{v=1}^V d\text{logits}_{i, v} \cdot E[v]$$
//!   - Gradient w.r.t vocabulary embedding $E[v]$:
//!     $$dE[v] \mathrel{+}= \sum_{i=1}^N d\text{logits}_{i, v} \cdot h_i$$

use crate::simd::dot_product_4d_simd;

pub struct QuaternionLMHead;

impl QuaternionLMHead {
    /// Forward pass of Quaternion LM Head
    ///
    /// - `logits`: Output logits `[n, v]`
    /// - `h`: Final normalized hidden states `[n, d]` (interpreted as `[n, d/4]` quaternions)
    /// - `weight`: Vocabulary embedding matrix $E$ `[v, d]` (interpreted as `[v, d/4]` quaternions)
    /// - `n`: Number of tokens ($B \times T$)
    /// - `d`: Hidden dimension (must be divisible by 4)
    /// - `v`: Vocabulary size
    pub fn forward(logits: &mut [f32], h: &[f32], weight: &[f32], n: usize, d: usize, v: usize) {
        assert_eq!(d % 4, 0, "Hidden dimension d must be a multiple of 4");
        let n_quat = d / 4;

        for i in 0..n {
            let h_row = &h[i * d..(i + 1) * d];
            let logits_row = &mut logits[i * v..(i + 1) * v];

            for j in 0..v {
                let w_row = &weight[j * d..(j + 1) * d];
                let mut logit_val = 0.0f32;

                for k in 0..n_quat {
                    let w_q: &[f32; 4] = w_row[k * 4..(k + 1) * 4].try_into().unwrap();
                    let h_q: &[f32; 4] = h_row[k * 4..(k + 1) * 4].try_into().unwrap();
                    logit_val += dot_product_4d_simd(w_q, h_q);
                }

                logits_row[j] = logit_val;
            }
        }
    }

    /// Backward pass of Quaternion LM Head
    ///
    /// - `dh`: Gradient w.r.t input hidden state `[n, d]` (accumulated)
    /// - `dw`: Gradient w.r.t vocabulary embedding matrix `[v, d]` (accumulated)
    /// - `dlogits`: Upstream gradient w.r.t logits `[n, v]`
    /// - `h`: Forward input hidden states `[n, d]`
    /// - `weight`: Forward vocabulary embedding matrix `[v, d]`
    /// - `n`: Number of tokens ($B \times T$)
    /// - `d`: Hidden dimension (must be divisible by 4)
    /// - `v`: Vocabulary size
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        dh: &mut [f32],
        dw: &mut [f32],
        dlogits: &[f32],
        h: &[f32],
        weight: &[f32],
        n: usize,
        d: usize,
        v: usize,
    ) {
        assert_eq!(d % 4, 0, "Hidden dimension d must be a multiple of 4");

        for i in 0..n {
            let dlogits_row = &dlogits[i * v..(i + 1) * v];
            let h_row = &h[i * d..(i + 1) * d];
            let dh_row = &mut dh[i * d..(i + 1) * d];

            for j in 0..v {
                let dl = dlogits_row[j];
                if dl == 0.0 {
                    continue;
                }
                let w_row = &weight[j * d..(j + 1) * d];
                let dw_row = &mut dw[j * d..(j + 1) * d];

                for k in 0..d {
                    dh_row[k] += dl * w_row[k];
                    dw_row[k] += dl * h_row[k];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reproducibility::DeterministicRng;

    #[test]
    fn test_quaternion_head_finite_difference_gradcheck() {
        let mut rng = DeterministicRng::new(42);
        let n = 2;
        let d = 16; // 4 quaternions
        let v = 7;

        let mut h = vec![0.0f32; n * d];
        let mut weight = vec![0.0f32; v * d];
        let mut dlogits = vec![0.0f32; n * v];

        rng.fill_gaussian(&mut h, 0.0, 1.0);
        rng.fill_gaussian(&mut weight, 0.0, 1.0 / (d as f32).sqrt());
        rng.fill_gaussian(&mut dlogits, 0.0, 1.0);

        // Analytical backward
        let mut dh_analytic = vec![0.0f32; n * d];
        let mut dw_analytic = vec![0.0f32; v * d];
        QuaternionLMHead::backward(
            &mut dh_analytic,
            &mut dw_analytic,
            &dlogits,
            &h,
            &weight,
            n,
            d,
            v,
        );

        let eps = 1e-3f32;

        // Numerical grad check for h
        for idx in 0..(n * d) {
            let orig = h[idx];

            h[idx] = orig + eps;
            let mut logits_plus = vec![0.0f32; n * v];
            QuaternionLMHead::forward(&mut logits_plus, &h, &weight, n, d, v);

            h[idx] = orig - eps;
            let mut logits_minus = vec![0.0f32; n * v];
            QuaternionLMHead::forward(&mut logits_minus, &h, &weight, n, d, v);

            h[idx] = orig;

            let mut num_grad = 0.0f32;
            for j in 0..(n * v) {
                num_grad += dlogits[j] * (logits_plus[j] - logits_minus[j]) / (2.0 * eps);
            }

            let diff = (dh_analytic[idx] - num_grad).abs();
            assert!(
                diff < 5e-3,
                "dh mismatch at {}: analytic {} vs numeric {}",
                idx,
                dh_analytic[idx],
                num_grad
            );
        }

        // Numerical grad check for weight
        for idx in 0..(v * d) {
            let orig = weight[idx];

            weight[idx] = orig + eps;
            let mut logits_plus = vec![0.0f32; n * v];
            QuaternionLMHead::forward(&mut logits_plus, &h, &weight, n, d, v);

            weight[idx] = orig - eps;
            let mut logits_minus = vec![0.0f32; n * v];
            QuaternionLMHead::forward(&mut logits_minus, &h, &weight, n, d, v);

            weight[idx] = orig;

            let mut num_grad = 0.0f32;
            for j in 0..(n * v) {
                num_grad += dlogits[j] * (logits_plus[j] - logits_minus[j]) / (2.0 * eps);
            }

            let diff = (dw_analytic[idx] - num_grad).abs();
            assert!(
                diff < 5e-3,
                "dw mismatch at {}: analytic {} vs numeric {}",
                idx,
                dw_analytic[idx],
                num_grad
            );
        }
    }
}
