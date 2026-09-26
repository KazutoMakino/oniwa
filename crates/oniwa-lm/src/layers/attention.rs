//! Causal Multi-Head Self-Attention with RoPE
//!
//! - Input: X [B, T, C]
//! - Projection: QKV = X * W_qkv [C, 3*C]
//! - RoPE: Rotary Position Embedding based on token position
//! - Causal Attention: Attn = Softmax(Mask(Q * K^T / sqrt(D_h))) * V
//! - Output Projection: Y = Attn * W_proj [C, C]

pub struct CausalSelfAttention;

impl CausalSelfAttention {
    /// RoPE (Rotary Position Embedding) forward & backward pass
    ///
    /// In the backward pass, simply negating angle reverses the rotation.
    pub fn apply_rope(
        vec: &mut [f32],
        b: usize,
        t: usize,
        num_heads: usize,
        head_dim: usize,
        inverse: bool,
    ) {
        let sign = if inverse { -1.0f32 } else { 1.0f32 };
        let half = head_dim / 2;

        for bi in 0..b {
            for ti in 0..t {
                let pos = ti as f32;
                for hi in 0..num_heads {
                    let offset = ((bi * t + ti) * num_heads + hi) * head_dim;
                    for i in 0..half {
                        let theta = 10000.0f32.powf(-2.0 * (i as f32) / (head_dim as f32));
                        let angle = sign * pos * theta;
                        let cos = angle.cos();
                        let sin = angle.sin();

                        let v0 = vec[offset + i];
                        let v1 = vec[offset + half + i];

                        vec[offset + i] = v0 * cos - v1 * sin;
                        vec[offset + half + i] = v0 * sin + v1 * cos;
                    }
                }
            }
        }
    }

    /// Forward pass
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        out: &mut [f32],
        act_q: &mut [f32],
        act_k: &mut [f32],
        act_v: &mut [f32],
        act_att: &mut [f32],     // [B, NH, T, T] Softmax weights
        act_att_out: &mut [f32], // [B, T, C]
        inp: &[f32],
        w_qkv: &[f32],
        w_proj: &[f32],
        b: usize,
        t: usize,
        dim: usize,
        num_heads: usize,
    ) {
        let n = b * t;
        let head_dim = dim / num_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // 1. QKV projection: inp [N, C] * w_qkv [C, 3*C]
        for i in 0..n {
            let x_row = &inp[i * dim..(i + 1) * dim];
            for j in 0..dim {
                let mut dot_q = 0.0f32;
                let mut dot_k = 0.0f32;
                let mut dot_v = 0.0f32;
                for (k, &x_k) in x_row.iter().enumerate() {
                    let w_offset = k * (3 * dim);
                    dot_q += x_k * w_qkv[w_offset + j];
                    dot_k += x_k * w_qkv[w_offset + dim + j];
                    dot_v += x_k * w_qkv[w_offset + 2 * dim + j];
                }
                act_q[i * dim + j] = dot_q;
                act_k[i * dim + j] = dot_k;
                act_v[i * dim + j] = dot_v;
            }
        }

        // 2. Apply RoPE (Q and K)
        Self::apply_rope(act_q, b, t, num_heads, head_dim, false);
        Self::apply_rope(act_k, b, t, num_heads, head_dim, false);

        // 3. Causal Attention Matrix: S = Q * K^T / sqrt(D_h) + Mask
        // act_att: [B, NH, T, T]
        for bi in 0..b {
            for hi in 0..num_heads {
                let att_offset = (bi * num_heads + hi) * (t * t);
                for i in 0..t {
                    let q_offset = ((bi * t + i) * num_heads + hi) * head_dim;
                    let q_vec = &act_q[q_offset..q_offset + head_dim];

                    let row_offset = att_offset + i * t;

                    // Inner product calculation (j <= i only for causal masking)
                    let mut max_val = f32::NEG_INFINITY;
                    for j in 0..=i {
                        let k_offset = ((bi * t + j) * num_heads + hi) * head_dim;
                        let k_vec = &act_k[k_offset..k_offset + head_dim];

                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_vec[d] * k_vec[d];
                        }
                        let score = dot * scale;
                        act_att[row_offset + j] = score;
                        if score > max_val {
                            max_val = score;
                        }
                    }

                    // Softmax
                    let mut sum_exp = 0.0f32;
                    for j in 0..=i {
                        let exp_v = (act_att[row_offset + j] - max_val).exp();
                        act_att[row_offset + j] = exp_v;
                        sum_exp += exp_v;
                    }
                    for j in 0..=i {
                        act_att[row_offset + j] /= sum_exp;
                    }
                    for j in i + 1..t {
                        act_att[row_offset + j] = 0.0;
                    }

                    // 4. Output = Att * V
                    let out_offset = ((bi * t + i) * num_heads + hi) * head_dim;
                    for d in 0..head_dim {
                        let mut sum_v = 0.0f32;
                        for j in 0..=i {
                            let a = act_att[row_offset + j];
                            let v_offset = ((bi * t + j) * num_heads + hi) * head_dim;
                            sum_v += a * act_v[v_offset + d];
                        }
                        act_att_out[out_offset + d] = sum_v;
                    }
                }
            }
        }

        // 5. Output Projection: act_att_out [N, C] * w_proj [C, C]
        for i in 0..n {
            let att_row = &act_att_out[i * dim..(i + 1) * dim];
            for j in 0..dim {
                let mut dot = 0.0f32;
                for k in 0..dim {
                    dot += att_row[k] * w_proj[k * dim + j];
                }
                out[i * dim + j] = dot;
            }
        }
    }

    /// Backward pass
    #[allow(clippy::too_many_arguments)]
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
        let head_dim = dim / num_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // 1. d_att_out = dout * w_proj^T,  dw_proj += act_att_out^T * dout
        let mut d_att_out = vec![0.0f32; n * dim];
        for i in 0..n {
            let dout_row = &dout[i * dim..(i + 1) * dim];
            let att_row = &act_att_out[i * dim..(i + 1) * dim];
            for k in 0..dim {
                let mut dot = 0.0f32;
                for j in 0..dim {
                    dot += dout_row[j] * w_proj[k * dim + j];
                    dw_proj[k * dim + j] += att_row[k] * dout_row[j];
                }
                d_att_out[i * dim + k] = dot;
            }
        }

        // 2. Attention Backward (d_att_out -> dV, dAtt -> dQ, dK)
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
                    for j in 0..=i {
                        let a = act_att[row_offset + j];
                        let v_offset = ((bi * t + j) * num_heads + hi) * head_dim;
                        for d in 0..head_dim {
                            dv[v_offset + d] += a * d_out_vec[d];
                        }
                    }

                    // dA_ij = d_out_i * V_j
                    let mut da = vec![0.0f32; i + 1];
                    for (j, da_j) in da.iter_mut().enumerate().take(i + 1) {
                        let v_offset = ((bi * t + j) * num_heads + hi) * head_dim;
                        let v_vec = &act_v[v_offset..v_offset + head_dim];
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += d_out_vec[d] * v_vec[d];
                        }
                        *da_j = dot;
                    }

                    // Softmax backward: dS = A * (dA - sum(dA * A))
                    let mut sum_da_a = 0.0f32;
                    for j in 0..=i {
                        sum_da_a += da[j] * act_att[row_offset + j];
                    }

                    let q_offset = ((bi * t + i) * num_heads + hi) * head_dim;
                    for j in 0..=i {
                        let a = act_att[row_offset + j];
                        let ds = a * (da[j] - sum_da_a) * scale;

                        let k_offset = ((bi * t + j) * num_heads + hi) * head_dim;
                        for d in 0..head_dim {
                            dq[q_offset + d] += ds * act_k[k_offset + d];
                            dk[k_offset + d] += ds * act_q[q_offset + d];
                        }
                    }
                }
            }
        }

        // 3. RoPE inverse rotation (dQ and dK)
        Self::apply_rope(&mut dq, b, t, num_heads, head_dim, true);
        Self::apply_rope(&mut dk, b, t, num_heads, head_dim, true);

        // 4. QKV Projection Backward: dinp += dQKV * w_qkv^T, dw_qkv += inp^T * dQKV
        for i in 0..n {
            let x_row = &inp[i * dim..(i + 1) * dim];
            let dinp_row = &mut dinp[i * dim..(i + 1) * dim];
            let dq_row = &dq[i * dim..(i + 1) * dim];
            let dk_row = &dk[i * dim..(i + 1) * dim];
            let dv_row = &dv[i * dim..(i + 1) * dim];

            for k in 0..dim {
                let mut dx_k = 0.0f32;
                let w_offset = k * (3 * dim);
                for j in 0..dim {
                    dx_k += dq_row[j] * w_qkv[w_offset + j];
                    dx_k += dk_row[j] * w_qkv[w_offset + dim + j];
                    dx_k += dv_row[j] * w_qkv[w_offset + 2 * dim + j];

                    dw_qkv[w_offset + j] += x_row[k] * dq_row[j];
                    dw_qkv[w_offset + dim + j] += x_row[k] * dk_row[j];
                    dw_qkv[w_offset + 2 * dim + j] += x_row[k] * dv_row[j];
                }
                dinp_row[k] += dx_k;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_gradcheck() {
        let b = 1;
        let t = 3;
        let c = 4;
        let nh = 2;

        let inp: Vec<f32> = (0..b * t * c).map(|i| (i as f32 * 0.1).sin()).collect();
        let w_qkv: Vec<f32> = (0..c * 3 * c)
            .map(|i| (i as f32 * 0.2).cos() * 0.1)
            .collect();
        let w_proj: Vec<f32> = (0..c * c).map(|i| (i as f32 * 0.3).sin() * 0.1).collect();

        let mut out = vec![0.0f32; b * t * c];
        let mut act_q = vec![0.0f32; b * t * c];
        let mut act_k = vec![0.0f32; b * t * c];
        let mut act_v = vec![0.0f32; b * t * c];
        let mut act_att = vec![0.0f32; b * nh * t * t];
        let mut act_att_out = vec![0.0f32; b * t * c];

        CausalSelfAttention::forward(
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

        let dout: Vec<f32> = (0..b * t * c).map(|i| (i as f32 * 0.5).cos()).collect();

        let mut dinp = vec![0.0f32; b * t * c];
        let mut dw_qkv = vec![0.0f32; c * 3 * c];
        let mut dw_proj = vec![0.0f32; c * c];

        CausalSelfAttention::backward(
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

        // Numerical gradient check for inp
        let eps = 1e-3f32;
        for i in 0..inp.len() {
            let mut inp_pos = inp.clone();
            let mut inp_neg = inp.clone();
            inp_pos[i] += eps;
            inp_neg[i] -= eps;

            let mut out_pos = vec![0.0f32; b * t * c];
            let mut out_neg = vec![0.0f32; b * t * c];
            let mut q1 = vec![0.0f32; b * t * c];
            let mut k1 = vec![0.0f32; b * t * c];
            let mut v1 = vec![0.0f32; b * t * c];
            let mut a1 = vec![0.0f32; b * nh * t * t];
            let mut ao1 = vec![0.0f32; b * t * c];

            CausalSelfAttention::forward(
                &mut out_pos,
                &mut q1,
                &mut k1,
                &mut v1,
                &mut a1,
                &mut ao1,
                &inp_pos,
                &w_qkv,
                &w_proj,
                b,
                t,
                c,
                nh,
            );
            CausalSelfAttention::forward(
                &mut out_neg,
                &mut q1,
                &mut k1,
                &mut v1,
                &mut a1,
                &mut ao1,
                &inp_neg,
                &w_qkv,
                &w_proj,
                b,
                t,
                c,
                nh,
            );

            let mut loss_pos = 0.0f32;
            let mut loss_neg = 0.0f32;
            for j in 0..out.len() {
                loss_pos += out_pos[j] * dout[j];
                loss_neg += out_neg[j] * dout[j];
            }
            let num_grad = (loss_pos - loss_neg) / (2.0 * eps);
            let ana_grad = dinp[i];
            assert!(
                (num_grad - ana_grad).abs() < 1e-2,
                "Attention GradCheck failed at {}: num={}, ana={}",
                i,
                num_grad,
                ana_grad
            );
        }
    }
}
