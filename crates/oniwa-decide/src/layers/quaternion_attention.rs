//! Bidirectional Quaternion Multi-Head Self-Attention with RoPE and GHR Calculus Gradients
//!
//! Features:
//! - Interprets an input sequence $h \in \mathbb{R}^{B \times T \times D}$ as $\mathbb{H}^{B \times T \times D/4}$
//! - Q, K, V projections via QuaternionLinear ($D/4$ quaternions $\to D/4$ quaternions)
//! - Attention score calculation via Quaternion Inner Product (extracting the real part / Shared-Score approach):
//!   $$\text{score}(q_i, k_j) = \frac{\text{Re}(q_i \otimes k_j^*)}{\sqrt{d_h}}$$
//! - Output projection via QuaternionLinear ($D/4$ quaternions $\to D/4$ quaternions)
//! - Rigorous analytical gradients derived from GHR calculus.

use crate::layers::Attention;
use crate::layers::QuaternionLinear;
use crate::simd::dot_product_simd;

pub struct QuaternionAttention;

impl QuaternionAttention {
    /// Forward pass of Quaternion Self-Attention
    ///
    /// - `dim`: Total real dimension (must be divisible by 4, and dim/num_heads must be divisible by 4)
    /// - `num_heads`: Number of attention heads
    /// - `w_qkv`: Quaternion weights for Q, K, V `[3 * c_quat, c_quat * 4]` where `c_quat = dim / 4`
    /// - `w_proj`: Quaternion weights for output projection `[c_quat, c_quat * 4]`
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        out: &mut [f32],
        act_q: &mut [f32],
        act_k: &mut [f32],
        act_v: &mut [f32],
        act_att: &mut [f32],     // [B, NH, T, T] weights after Softmax
        act_att_out: &mut [f32], // [B, T, C]
        inp: &[f32],
        w_qkv: &[f32],
        w_proj: &[f32],
        b: usize,
        t: usize,
        dim: usize,
        num_heads: usize,
    ) {
        assert_eq!(dim % 4, 0, "Embedding dim must be divisible by 4");
        let n = b * t;
        let c_quat = dim / 4;
        let head_dim = dim / num_heads;
        assert_eq!(head_dim % 4, 0, "Head dim must be divisible by 4");
        let _d_h_quat = head_dim / 4;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // 1. QKV Projections using QuaternionLinear
        // w_qkv contains 3 sets of quaternion projection weights: [3 * c_quat, c_quat * 4]
        let w_q_slice = &w_qkv[0..c_quat * c_quat * 4];
        let w_k_slice = &w_qkv[c_quat * c_quat * 4..2 * c_quat * c_quat * 4];
        let w_v_slice = &w_qkv[2 * c_quat * c_quat * 4..3 * c_quat * c_quat * 4];

        QuaternionLinear::forward(act_q, inp, w_q_slice, None, n, c_quat, c_quat);
        QuaternionLinear::forward(act_k, inp, w_k_slice, None, n, c_quat, c_quat);
        QuaternionLinear::forward(act_v, inp, w_v_slice, None, n, c_quat, c_quat);

        // 2. Apply RoPE to Q and K
        Attention::apply_rope(act_q, b, t, num_heads, head_dim, false);
        Attention::apply_rope(act_k, b, t, num_heads, head_dim, false);

        // 3. Bidirectional Attention Matrix: S = Re(Q ⊗ K*) / sqrt(D_h)
        for bi in 0..b {
            for hi in 0..num_heads {
                let att_offset = (bi * num_heads + hi) * (t * t);
                for i in 0..t {
                    let q_offset = ((bi * t + i) * num_heads + hi) * head_dim;
                    let row_offset = att_offset + i * t;

                    let q_vec = &act_q[q_offset..q_offset + head_dim];

                    let mut max_val = f32::NEG_INFINITY;
                    for j in 0..t {
                        let k_offset = ((bi * t + j) * num_heads + hi) * head_dim;
                        let k_vec = &act_k[k_offset..k_offset + head_dim];

                        // Quaternion inner product: sum_k Re(q_k ⊗ k_k*) = sum_k dot(q_k, k_k) = dot_product_simd(q_vec, k_vec)
                        let dot = dot_product_simd(q_vec, k_vec);
                        let score = dot * scale;
                        act_att[row_offset + j] = score;
                        if score > max_val {
                            max_val = score;
                        }
                    }

                    // Softmax normalization across sequence
                    let mut sum_exp = 0.0f32;
                    for j in 0..t {
                        let exp_v = (act_att[row_offset + j] - max_val).exp();
                        act_att[row_offset + j] = exp_v;
                        sum_exp += exp_v;
                    }
                    let inv_sum = 1.0f32 / sum_exp.max(1e-12);
                    for j in 0..t {
                        act_att[row_offset + j] *= inv_sum;
                    }

                    // 4. Output = Att * V
                    let out_offset = ((bi * t + i) * num_heads + hi) * head_dim;
                    let out_slice = &mut act_att_out[out_offset..out_offset + head_dim];
                    out_slice.fill(0.0f32);
                    for j in 0..t {
                        let a = act_att[row_offset + j];
                        let v_offset = ((bi * t + j) * num_heads + hi) * head_dim;
                        let v_vec = &act_v[v_offset..v_offset + head_dim];
                        for d in 0..head_dim {
                            out_slice[d] += a * v_vec[d];
                        }
                    }
                }
            }
        }

        // 5. Output Projection using QuaternionLinear: act_att_out -> out
        QuaternionLinear::forward(out, act_att_out, w_proj, None, n, c_quat, c_quat);
    }

    /// Backward pass of Quaternion Self-Attention using GHR calculus
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    pub fn backward(
        dinp: &mut [f32],
        dw_qkv: &mut [f32],
        dw_proj: &mut [f32],
        dout: &[f32],
        inp: &[f32],
        act_q: &[f32],
        act_k: &[f32],
        act_v: &[f32],
        act_att: &[f32],
        act_att_out: &[f32],
        w_qkv: &[f32],
        w_proj: &[f32],
        b: usize,
        t: usize,
        dim: usize,
        num_heads: usize,
    ) {
        let n = b * t;
        let c_quat = dim / 4;
        let head_dim = dim / num_heads;
        let _d_h_quat = head_dim / 4;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // 1. Output projection backward via QuaternionLinear
        let mut d_att_out = vec![0.0f32; n * dim];
        QuaternionLinear::backward(
            &mut d_att_out,
            dw_proj,
            None,
            dout,
            act_att_out,
            w_proj,
            n,
            c_quat,
            c_quat,
        );

        // 2. Attention Matrix Backward
        let mut dq = vec![0.0f32; n * dim];
        let mut dk = vec![0.0f32; n * dim];
        let mut dv = vec![0.0f32; n * dim];

        for bi in 0..b {
            for hi in 0..num_heads {
                let att_offset = (bi * num_heads + hi) * (t * t);
                for i in 0..t {
                    let d_out_offset = ((bi * t + i) * num_heads + hi) * head_dim;
                    let d_out_vec = &d_att_out[d_out_offset..d_out_offset + head_dim];
                    let row_offset = att_offset + i * t;

                    // dV: sum_i (A_ij * d_out_i)
                    for j in 0..t {
                        let a = act_att[row_offset + j];
                        let v_offset = ((bi * t + j) * num_heads + hi) * head_dim;
                        for d in 0..head_dim {
                            dv[v_offset + d] += a * d_out_vec[d];
                        }
                    }

                    // dA_ij = d_out_i * V_j
                    let mut da = vec![0.0f32; t];
                    for j in 0..t {
                        let v_offset = ((bi * t + j) * num_heads + hi) * head_dim;
                        let v_vec = &act_v[v_offset..v_offset + head_dim];
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += d_out_vec[d] * v_vec[d];
                        }
                        da[j] = dot;
                    }

                    // Softmax backward: dS = A * (dA - sum(dA * A))
                    let mut sum_da_a = 0.0f32;
                    for j in 0..t {
                        sum_da_a += da[j] * act_att[row_offset + j];
                    }

                    let q_offset = ((bi * t + i) * num_heads + hi) * head_dim;
                    for j in 0..t {
                        let a = act_att[row_offset + j];
                        let ds = a * (da[j] - sum_da_a) * scale;

                        let k_offset = ((bi * t + j) * num_heads + hi) * head_dim;
                        // Score is Re(q_k ⊗ k_k*) = dot(q_k, k_k)
                        // Derivative w.r.t q_k is ds * k_k
                        // Derivative w.r.t k_k is ds * q_k
                        for d in 0..head_dim {
                            dq[q_offset + d] += ds * act_k[k_offset + d];
                            dk[k_offset + d] += ds * act_q[q_offset + d];
                        }
                    }
                }
            }
        }

        // 3. Inverse RoPE rotation on dQ and dK
        Attention::apply_rope(&mut dq, b, t, num_heads, head_dim, true);
        Attention::apply_rope(&mut dk, b, t, num_heads, head_dim, true);

        // 4. QKV Projections backward using QuaternionLinear
        let qkv_block = c_quat * c_quat * 4;
        let w_q_slice = &w_qkv[0..qkv_block];
        let w_k_slice = &w_qkv[qkv_block..2 * qkv_block];
        let w_v_slice = &w_qkv[2 * qkv_block..3 * qkv_block];

        let (dw_q_slice, rest) = dw_qkv.split_at_mut(qkv_block);
        let (dw_k_slice, dw_v_slice) = rest.split_at_mut(qkv_block);

        let mut dinp_q = vec![0.0f32; n * dim];
        let mut dinp_k = vec![0.0f32; n * dim];
        let mut dinp_v = vec![0.0f32; n * dim];

        QuaternionLinear::backward(
            &mut dinp_q,
            dw_q_slice,
            None,
            &dq,
            inp,
            w_q_slice,
            n,
            c_quat,
            c_quat,
        );
        QuaternionLinear::backward(
            &mut dinp_k,
            dw_k_slice,
            None,
            &dk,
            inp,
            w_k_slice,
            n,
            c_quat,
            c_quat,
        );
        QuaternionLinear::backward(
            &mut dinp_v,
            dw_v_slice,
            None,
            &dv,
            inp,
            w_v_slice,
            n,
            c_quat,
            c_quat,
        );

        for idx in 0..n * dim {
            dinp[idx] += dinp_q[idx] + dinp_k[idx] + dinp_v[idx];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oniwa_lm::reproducibility::DeterministicRng;

    #[test]
    fn test_quaternion_attention_forward_and_backward_gradcheck() {
        let b = 1;
        let t = 4;
        let c = 8; // 2 quaternions total
        let nh = 2; // head_dim = 4 (1 quaternion per head)
        let c_quat = c / 4;

        let mut rng = DeterministicRng::new(101);
        let n = b * t;

        let mut inp = vec![0.0f32; n * c];
        for val in inp.iter_mut() {
            *val = rng.next_f32() * 0.4 - 0.2;
        }

        let mut w_qkv = vec![0.0f32; 3 * c_quat * c_quat * 4];
        for val in w_qkv.iter_mut() {
            *val = rng.next_f32() * 0.4 - 0.2;
        }

        let mut w_proj = vec![0.0f32; c_quat * c_quat * 4];
        for val in w_proj.iter_mut() {
            *val = rng.next_f32() * 0.4 - 0.2;
        }

        let mut out = vec![0.0f32; n * c];
        let mut act_q = vec![0.0f32; n * c];
        let mut act_k = vec![0.0f32; n * c];
        let mut act_v = vec![0.0f32; n * c];
        let mut act_att = vec![0.0f32; b * nh * t * t];
        let mut act_att_out = vec![0.0f32; n * c];

        QuaternionAttention::forward(
            &mut out,
            &mut act_q,
            &mut act_k,
            &mut act_v,
            &mut act_att,
            &mut act_att_out,
            &inp,
            &w_qkv,
            &w_proj,
            b,
            t,
            c,
            nh,
        );

        // Define a scalar loss: L = 0.5 * sum(out^2)
        let dout = out.clone();
        let mut dinp = vec![0.0f32; n * c];
        let mut dw_qkv = vec![0.0f32; w_qkv.len()];
        let mut dw_proj = vec![0.0f32; w_proj.len()];

        QuaternionAttention::backward(
            &mut dinp,
            &mut dw_qkv,
            &mut dw_proj,
            &dout,
            &inp,
            &act_q,
            &act_k,
            &act_v,
            &act_att,
            &act_att_out,
            &w_qkv,
            &w_proj,
            b,
            t,
            c,
            nh,
        );

        // 1. Gradcheck for input
        let eps = 1e-3f32;
        let test_idx = 3;
        let orig_x = inp[test_idx];

        inp[test_idx] = orig_x + eps;
        let mut out_p = vec![0.0f32; n * c];
        QuaternionAttention::forward(
            &mut out_p,
            &mut act_q,
            &mut act_k,
            &mut act_v,
            &mut act_att,
            &mut act_att_out,
            &inp,
            &w_qkv,
            &w_proj,
            b,
            t,
            c,
            nh,
        );
        let loss_p: f32 = 0.5 * out_p.iter().map(|&v| v * v).sum::<f32>();

        inp[test_idx] = orig_x - eps;
        let mut out_m = vec![0.0f32; n * c];
        QuaternionAttention::forward(
            &mut out_m,
            &mut act_q,
            &mut act_k,
            &mut act_v,
            &mut act_att,
            &mut act_att_out,
            &inp,
            &w_qkv,
            &w_proj,
            b,
            t,
            c,
            nh,
        );
        let loss_m: f32 = 0.5 * out_m.iter().map(|&v| v * v).sum::<f32>();
        inp[test_idx] = orig_x;

        let num_grad = (loss_p - loss_m) / (2.0 * eps);
        let ana_grad = dinp[test_idx];
        let diff = (num_grad - ana_grad).abs();
        assert!(
            diff < 5e-3 || diff / (ana_grad.abs() + num_grad.abs()).max(1e-5) < 0.05,
            "Input gradcheck failed! Analytic: {}, Numerical: {}, Diff: {}",
            ana_grad,
            num_grad,
            diff
        );

        // 2. Gradcheck for projection weight
        let w_idx = 2;
        let orig_w = w_proj[w_idx];
        w_proj[w_idx] = orig_w + eps;
        QuaternionAttention::forward(
            &mut out_p,
            &mut act_q,
            &mut act_k,
            &mut act_v,
            &mut act_att,
            &mut act_att_out,
            &inp,
            &w_qkv,
            &w_proj,
            b,
            t,
            c,
            nh,
        );
        let loss_w_p: f32 = 0.5 * out_p.iter().map(|&v| v * v).sum::<f32>();

        w_proj[w_idx] = orig_w - eps;
        QuaternionAttention::forward(
            &mut out_m,
            &mut act_q,
            &mut act_k,
            &mut act_v,
            &mut act_att,
            &mut act_att_out,
            &inp,
            &w_qkv,
            &w_proj,
            b,
            t,
            c,
            nh,
        );
        let loss_w_m: f32 = 0.5 * out_m.iter().map(|&v| v * v).sum::<f32>();
        w_proj[w_idx] = orig_w;

        let num_w_grad = (loss_w_p - loss_w_m) / (2.0 * eps);
        let ana_w_grad = dw_proj[w_idx];
        let diff_w = (num_w_grad - ana_w_grad).abs();
        assert!(
            diff_w < 5e-3 || diff_w / (ana_w_grad.abs() + num_w_grad.abs()).max(1e-5) < 0.05,
            "Weight gradcheck failed! Analytic: {}, Numerical: {}, Diff: {}",
            ana_w_grad,
            num_w_grad,
            diff_w
        );

        // 3. Gradcheck for QKV weight
        let qkv_idx = 5;
        let orig_qkv_w = w_qkv[qkv_idx];
        w_qkv[qkv_idx] = orig_qkv_w + eps;
        QuaternionAttention::forward(
            &mut out_p,
            &mut act_q,
            &mut act_k,
            &mut act_v,
            &mut act_att,
            &mut act_att_out,
            &inp,
            &w_qkv,
            &w_proj,
            b,
            t,
            c,
            nh,
        );
        let loss_qkv_p: f32 = 0.5 * out_p.iter().map(|&v| v * v).sum::<f32>();

        w_qkv[qkv_idx] = orig_qkv_w - eps;
        QuaternionAttention::forward(
            &mut out_m,
            &mut act_q,
            &mut act_k,
            &mut act_v,
            &mut act_att,
            &mut act_att_out,
            &inp,
            &w_qkv,
            &w_proj,
            b,
            t,
            c,
            nh,
        );
        let loss_qkv_m: f32 = 0.5 * out_m.iter().map(|&v| v * v).sum::<f32>();
        w_qkv[qkv_idx] = orig_qkv_w;

        let num_qkv_grad = (loss_qkv_p - loss_qkv_m) / (2.0 * eps);
        let ana_qkv_grad = dw_qkv[qkv_idx];
        let diff_qkv = (num_qkv_grad - ana_qkv_grad).abs();
        assert!(
            diff_qkv < 5e-3
                || diff_qkv / (ana_qkv_grad.abs() + num_qkv_grad.abs()).max(1e-5) < 0.05,
            "QKV weight gradcheck failed! Analytic: {}, Numerical: {}, Diff: {}",
            ana_qkv_grad,
            num_qkv_grad,
            diff_qkv
        );
    }
}
