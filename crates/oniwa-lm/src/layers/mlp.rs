//! SwiGLU (Swish-Gated Linear Unit) MLP 層
//!
//! 順伝播:
//!   G = X * W_gate       (N, FFN)
//!   U = X * W_up         (N, FFN)
//!   H = Swish(G) * U     (N, FFN)
//!   Y = H * W_down       (N, C)
//!
//! ここで Swish(g) = g * sigmoid(g) = g / (1 + exp(-g))

pub struct SwiGLU;

impl SwiGLU {
    #[inline(always)]
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    #[inline(always)]
    fn silu_grad(x: f32) -> f32 {
        let sig = 1.0 / (1.0 + (-x).exp());
        sig * (1.0 + x * (1.0 - sig))
    }

    /// 順伝播
    ///
    /// - `out`: [N, C]
    /// - `act_g`: 逆伝播用キャッシュ [N, FFN]
    /// - `act_u`: 逆伝播用キャッシュ [N, FFN]
    /// - `act_h`: 逆伝播用キャッシュ [N, FFN]
    /// - `inp`: [N, C]
    /// - `w_gate_up`: [C, 2 * FFN] (前半がgate, 後半がup)
    /// - `w_down`: [FFN, C]
    pub fn forward(
        out: &mut [f32],
        act_g: &mut [f32],
        act_u: &mut [f32],
        act_h: &mut [f32],
        inp: &[f32],
        w_gate_up: &[f32],
        w_down: &[f32],
        n: usize,
        c: usize,
        ffn: usize,
    ) {
        // 1. G, U の計算: inp [N, C] * w_gate_up [C, 2*FFN]
        for i in 0..n {
            let x_row = &inp[i * c..(i + 1) * c];
            for j in 0..ffn {
                let mut dot_g = 0.0f32;
                let mut dot_u = 0.0f32;
                for k in 0..c {
                    let w_offset = k * (2 * ffn);
                    dot_g += x_row[k] * w_gate_up[w_offset + j];
                    dot_u += x_row[k] * w_gate_up[w_offset + ffn + j];
                }
                act_g[i * ffn + j] = dot_g;
                act_u[i * ffn + j] = dot_u;
                let h = Self::silu(dot_g) * dot_u;
                act_h[i * ffn + j] = h;
            }
        }

        // 2. Y の計算: H [N, FFN] * w_down [FFN, C]
        for i in 0..n {
            let h_row = &act_h[i * ffn..(i + 1) * ffn];
            for j in 0..c {
                let mut dot_y = 0.0f32;
                for k in 0..ffn {
                    dot_y += h_row[k] * w_down[k * c + j];
                }
                out[i * c + j] = dot_y;
            }
        }
    }

    /// 逆伝播
    pub fn backward(
        dinp: &mut [f32],
        dw_gate_up: &mut [f32],
        dw_down: &mut [f32],
        dout: &[f32],
        inp: &[f32],
        act_g: &[f32],
        act_u: &[f32],
        act_h: &[f32],
        w_gate_up: &[f32],
        w_down: &[f32],
        n: usize,
        c: usize,
        ffn: usize,
    ) {
        let mut dh = vec![0.0f32; n * ffn];
        let mut dg = vec![0.0f32; n * ffn];
        let mut du = vec![0.0f32; n * ffn];

        // 1. dH = dout * w_down^T,  dw_down += act_h^T * dout
        for i in 0..n {
            let dout_row = &dout[i * c..(i + 1) * c];
            let h_row = &act_h[i * ffn..(i + 1) * ffn];
            for k in 0..ffn {
                let mut dot_dh = 0.0f32;
                for j in 0..c {
                    dot_dh += dout_row[j] * w_down[k * c + j];
                    dw_down[k * c + j] += h_row[k] * dout_row[j];
                }
                dh[i * ffn + k] = dot_dh;
            }
        }

        // 2. dU = dH * Swish(G),  dG = dH * U * Swish'(G)
        for idx in 0..n * ffn {
            let g = act_g[idx];
            let u = act_u[idx];
            let dh_val = dh[idx];

            let swish_g = Self::silu(g);
            du[idx] = dh_val * swish_g;

            let dswish = dh_val * u;
            dg[idx] = dswish * Self::silu_grad(g);
        }

        // 3. dinp += dG * W_gate^T + dU * W_up^T
        //    dw_gate_up += inp^T * [dG, dU]
        for i in 0..n {
            let x_row = &inp[i * c..(i + 1) * c];
            let dinp_row = &mut dinp[i * c..(i + 1) * c];
            let dg_row = &dg[i * ffn..(i + 1) * ffn];
            let du_row = &du[i * ffn..(i + 1) * ffn];

            for k in 0..c {
                let mut dx_k = 0.0f32;
                let w_offset = k * (2 * ffn);
                for j in 0..ffn {
                    dx_k += dg_row[j] * w_gate_up[w_offset + j];
                    dx_k += du_row[j] * w_gate_up[w_offset + ffn + j];

                    dw_gate_up[w_offset + j] += x_row[k] * dg_row[j];
                    dw_gate_up[w_offset + ffn + j] += x_row[k] * du_row[j];
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
    fn test_swiglu_gradcheck() {
        let n = 2;
        let c = 4;
        let ffn = 8;

        let inp: Vec<f32> = (0..n * c).map(|i| (i as f32 * 0.1).sin()).collect();
        let w_gate_up: Vec<f32> = (0..c * 2 * ffn)
            .map(|i| (i as f32 * 0.2).cos() * 0.1)
            .collect();
        let w_down: Vec<f32> = (0..ffn * c).map(|i| (i as f32 * 0.3).sin() * 0.1).collect();

        let mut out = vec![0.0f32; n * c];
        let mut act_g = vec![0.0f32; n * ffn];
        let mut act_u = vec![0.0f32; n * ffn];
        let mut act_h = vec![0.0f32; n * ffn];

        SwiGLU::forward(
            &mut out, &mut act_g, &mut act_u, &mut act_h, &inp, &w_gate_up, &w_down, n, c, ffn,
        );

        let dout: Vec<f32> = (0..n * c).map(|i| (i as f32 * 0.5).cos()).collect();

        let mut dinp = vec![0.0f32; n * c];
        let mut dw_gate_up = vec![0.0f32; c * 2 * ffn];
        let mut dw_down = vec![0.0f32; ffn * c];

        SwiGLU::backward(
            &mut dinp,
            &mut dw_gate_up,
            &mut dw_down,
            &dout,
            &inp,
            &act_g,
            &act_u,
            &act_h,
            &w_gate_up,
            &w_down,
            n,
            c,
            ffn,
        );

        // 数値微分チェック for inp
        let eps = 1e-3f32;
        for i in 0..inp.len() {
            let mut inp_pos = inp.clone();
            let mut inp_neg = inp.clone();
            inp_pos[i] += eps;
            inp_neg[i] -= eps;

            let mut out_pos = vec![0.0f32; n * c];
            let mut out_neg = vec![0.0f32; n * c];
            let mut dummy_g = vec![0.0f32; n * ffn];
            let mut dummy_u = vec![0.0f32; n * ffn];
            let mut dummy_h = vec![0.0f32; n * ffn];

            SwiGLU::forward(
                &mut out_pos,
                &mut dummy_g,
                &mut dummy_u,
                &mut dummy_h,
                &inp_pos,
                &w_gate_up,
                &w_down,
                n,
                c,
                ffn,
            );
            SwiGLU::forward(
                &mut out_neg,
                &mut dummy_g,
                &mut dummy_u,
                &mut dummy_h,
                &inp_neg,
                &w_gate_up,
                &w_down,
                n,
                c,
                ffn,
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
                "Inp GradCheck failed at {}: num={}, ana={}",
                i,
                num_grad,
                ana_grad
            );
        }
    }
}
