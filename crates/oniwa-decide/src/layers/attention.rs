//! Bidirectional Multi-Head Self-Attention with RoPE
//!
//! Omits the causal lower-triangular mask to allow bidirectional context aggregation,
//! similar to BERT / ModernBERT.
//! Used in the decision model (System One) to capture whole-sequence context in a single pass.

pub struct BidirectionalSelfAttention;

impl BidirectionalSelfAttention {
    /// Rotary Position Embedding (RoPE) forward & inverse rotation
    pub fn apply_rope(vec: &mut [f32], b: usize, t: usize, nh: usize, d_h: usize, inverse: bool) {
        let sign = if inverse { -1.0f32 } else { 1.0f32 };
        let half = d_h / 2;

        for bi in 0..b {
            for ti in 0..t {
                let pos = ti as f32;
                for hi in 0..nh {
                    let offset = ((bi * t + ti) * nh + hi) * d_h;
                    for i in 0..half {
                        let theta = 10000.0f32.powf(-2.0 * (i as f32) / (d_h as f32));
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
        act_att: &mut [f32],     // [B, NH, T, T] weights after Softmax
        act_att_out: &mut [f32], // [B, T, C]
        inp: &[f32],
        w_qkv: &[f32],
        w_proj: &[f32],
        b: usize,
        t: usize,
        c: usize,
        nh: usize,
    ) {
        let n = b * t;
        let d_h = c / nh;
        let scale = 1.0f32 / (d_h as f32).sqrt();

        // 1. QKV projection: inp [N, C] * w_qkv [C, 3*C]
        for i in 0..n {
            let x_row = &inp[i * c..(i + 1) * c];
            for j in 0..c {
                let mut dot_q = 0.0f32;
                let mut dot_k = 0.0f32;
                let mut dot_v = 0.0f32;
                for (k, &x_k) in x_row.iter().enumerate() {
                    let w_offset = k * (3 * c);
                    dot_q += x_k * w_qkv[w_offset + j];
                    dot_k += x_k * w_qkv[w_offset + c + j];
                    dot_v += x_k * w_qkv[w_offset + 2 * c + j];
                }
                act_q[i * c + j] = dot_q;
                act_k[i * c + j] = dot_k;
                act_v[i * c + j] = dot_v;
            }
        }

        // 2. Apply RoPE (Q and K)
        Self::apply_rope(act_q, b, t, nh, d_h, false);
        Self::apply_rope(act_k, b, t, nh, d_h, false);

        // 3. Bidirectional Attention Matrix: S = Q * K^T / sqrt(D_h) (fully dense, unmasked)
        for bi in 0..b {
            for hi in 0..nh {
                let att_offset = (bi * nh + hi) * (t * t);
                for i in 0..t {
                    let q_offset = ((bi * t + i) * nh + hi) * d_h;
                    let q_vec = &act_q[q_offset..q_offset + d_h];
                    let row_offset = att_offset + i * t;

                    // Compute inner product across all tokens j
                    let mut max_val = f32::NEG_INFINITY;
                    for j in 0..t {
                        let k_offset = ((bi * t + j) * nh + hi) * d_h;
                        let k_vec = &act_k[k_offset..k_offset + d_h];

                        let mut dot = 0.0f32;
                        for d in 0..d_h {
                            dot += q_vec[d] * k_vec[d];
                        }
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
                    let out_offset = ((bi * t + i) * nh + hi) * d_h;
                    for d in 0..d_h {
                        let mut sum_v = 0.0f32;
                        for j in 0..t {
                            let a = act_att[row_offset + j];
                            let v_offset = ((bi * t + j) * nh + hi) * d_h;
                            sum_v += a * act_v[v_offset + d];
                        }
                        act_att_out[out_offset + d] = sum_v;
                    }
                }
            }
        }

        // 5. Output Projection: act_att_out [N, C] * w_proj [C, C]
        for i in 0..n {
            let att_row = &act_att_out[i * c..(i + 1) * c];
            for j in 0..c {
                let mut dot = 0.0f32;
                for k in 0..c {
                    dot += att_row[k] * w_proj[k * c + j];
                }
                out[i * c + j] = dot;
            }
        }
    }

    /// Backward pass
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
        c: usize,
        nh: usize,
    ) {
        let n = b * t;
        let d_h = c / nh;
        let scale = 1.0f32 / (d_h as f32).sqrt();

        // 1. d_att_out = dout * w_proj^T,  dw_proj += act_att_out^T * dout
        let mut d_att_out = vec![0.0f32; n * c];
        for i in 0..n {
            let dout_row = &dout[i * c..(i + 1) * c];
            let att_row = &act_att_out[i * c..(i + 1) * c];
            for k in 0..c {
                let mut dot = 0.0f32;
                for j in 0..c {
                    dot += dout_row[j] * w_proj[k * c + j];
                    dw_proj[k * c + j] += att_row[k] * dout_row[j];
                }
                d_att_out[i * c + k] = dot;
            }
        }

        // 2. Attention Backward (dense bidirectional)
        let mut dq = vec![0.0f32; n * c];
        let mut dk = vec![0.0f32; n * c];
        let mut dv = vec![0.0f32; n * c];

        for bi in 0..b {
            for hi in 0..nh {
                let att_offset = (bi * nh + hi) * (t * t);
                for i in 0..t {
                    let d_out_offset = ((bi * t + i) * nh + hi) * d_h;
                    let d_out_vec = &d_att_out[d_out_offset..d_out_offset + d_h];
                    let row_offset = att_offset + i * t;

                    // dV: sum_i (A_ij * d_out_i)
                    for j in 0..t {
                        let a = act_att[row_offset + j];
                        let v_offset = ((bi * t + j) * nh + hi) * d_h;
                        for d in 0..d_h {
                            dv[v_offset + d] += a * d_out_vec[d];
                        }
                    }

                    // dA_ij = d_out_i * V_j
                    let mut da = vec![0.0f32; t];
                    for j in 0..t {
                        let v_offset = ((bi * t + j) * nh + hi) * d_h;
                        let v_vec = &act_v[v_offset..v_offset + d_h];
                        let mut dot = 0.0f32;
                        for d in 0..d_h {
                            dot += d_out_vec[d] * v_vec[d];
                        }
                        da[j] = dot;
                    }

                    // Softmax backward: dS = A * (dA - sum(dA * A))
                    let mut sum_da_a = 0.0f32;
                    for j in 0..t {
                        sum_da_a += da[j] * act_att[row_offset + j];
                    }

                    let q_offset = ((bi * t + i) * nh + hi) * d_h;
                    for j in 0..t {
                        let a = act_att[row_offset + j];
                        let ds = a * (da[j] - sum_da_a) * scale;

                        let k_offset = ((bi * t + j) * nh + hi) * d_h;
                        for d in 0..d_h {
                            dq[q_offset + d] += ds * act_k[k_offset + d];
                            dk[k_offset + d] += ds * act_q[q_offset + d];
                        }
                    }
                }
            }
        }

        // 3. Inverse RoPE rotation (dQ and dK)
        Self::apply_rope(&mut dq, b, t, nh, d_h, true);
        Self::apply_rope(&mut dk, b, t, nh, d_h, true);

        // 4. d_qkv = [dQ, dK, dV],  dinp = d_qkv * w_qkv^T,  dw_qkv += inp^T * d_qkv
        for i in 0..n {
            let x_row = &inp[i * c..(i + 1) * c];
            let dinp_row = &mut dinp[i * c..(i + 1) * c];

            let q_row = &dq[i * c..(i + 1) * c];
            let k_row = &dk[i * c..(i + 1) * c];
            let v_row = &dv[i * c..(i + 1) * c];

            for k in 0..c {
                let mut dot_x = 0.0f32;
                let w_offset = k * (3 * c);
                for j in 0..c {
                    let g_q = q_row[j];
                    let g_k = k_row[j];
                    let g_v = v_row[j];

                    dot_x += g_q * w_qkv[w_offset + j];
                    dot_x += g_k * w_qkv[w_offset + c + j];
                    dot_x += g_v * w_qkv[w_offset + 2 * c + j];

                    dw_qkv[w_offset + j] += x_row[k] * g_q;
                    dw_qkv[w_offset + c + j] += x_row[k] * g_k;
                    dw_qkv[w_offset + 2 * c + j] += x_row[k] * g_v;
                }
                dinp_row[k] += dot_x;
            }
        }
    }
}
