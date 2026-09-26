//! SwiGlu (Swish-Gated Linear Unit) MLP layer
//!
//! Forward pass:
//!   G = X * W_gate       (N, FFN)
//!   U = X * W_up         (N, FFN)
//!   H = Swish(G) * U     (N, FFN)
//!   Y = H * W_down       (N, C)
//!
//! where Swish(g) = g * sigmoid(g) = g / (1 + exp(-g))

use crate::simd::mul_slices_assign_simd;

pub struct SwiGlu;

impl SwiGlu {
    #[inline(always)]
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    #[inline(always)]
    fn silu_grad(x: f32) -> f32 {
        let sig = 1.0 / (1.0 + (-x).exp());
        sig * (1.0 + x * (1.0 - sig))
    }

    /// Forward pass
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        out: &mut [f32],
        act_g: &mut [f32],
        act_u: &mut [f32],
        act_h: &mut [f32],
        inp: &[f32],
        w_gate_up: &[f32],
        w_down: &[f32],
        n: usize,
        dim: usize,
        ffn_dim: usize,
    ) {
        for i in 0..n {
            let x_row = &inp[i * dim..(i + 1) * dim];
            for j in 0..ffn_dim {
                let mut dot_g = 0.0f32;
                let mut dot_u = 0.0f32;
                for (k, &x_k) in x_row.iter().enumerate() {
                    let w_offset = k * (2 * ffn_dim);
                    dot_g += x_k * w_gate_up[w_offset + j];
                    dot_u += x_k * w_gate_up[w_offset + ffn_dim + j];
                }
                act_g[i * ffn_dim + j] = dot_g;
                act_u[i * ffn_dim + j] = dot_u;
                act_h[i * ffn_dim + j] = Self::silu(dot_g);
            }
        }

        let total_ffn = n * ffn_dim;
        mul_slices_assign_simd(&mut act_h[..total_ffn], &act_u[..total_ffn]);

        for i in 0..n {
            let h_row = &act_h[i * ffn_dim..(i + 1) * ffn_dim];
            for j in 0..dim {
                let mut dot_y = 0.0f32;
                for k in 0..ffn_dim {
                    dot_y += h_row[k] * w_down[k * dim + j];
                }
                out[i * dim + j] = dot_y;
            }
        }
    }

    /// Backward pass
    #[allow(clippy::too_many_arguments)]
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
        dim: usize,
        ffn_dim: usize,
    ) {
        let mut dh = vec![0.0f32; n * ffn_dim];
        let mut dg = vec![0.0f32; n * ffn_dim];
        let mut du = vec![0.0f32; n * ffn_dim];

        // 1. dH = dout * w_down^T,  dw_down += act_h^T * dout
        for i in 0..n {
            let dout_row = &dout[i * dim..(i + 1) * dim];
            let h_row = &act_h[i * ffn_dim..(i + 1) * ffn_dim];
            for k in 0..ffn_dim {
                let mut dot_dh = 0.0f32;
                for j in 0..dim {
                    dot_dh += dout_row[j] * w_down[k * dim + j];
                    dw_down[k * dim + j] += h_row[k] * dout_row[j];
                }
                dh[i * ffn_dim + k] = dot_dh;
            }
        }

        // 2. dU = dH * Swish(G),  dG = dH * U * Swish'(G)
        for idx in 0..n * ffn_dim {
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
            let x_row = &inp[i * dim..(i + 1) * dim];
            let dinp_row = &mut dinp[i * dim..(i + 1) * dim];
            let dg_row = &dg[i * ffn_dim..(i + 1) * ffn_dim];
            let du_row = &du[i * ffn_dim..(i + 1) * ffn_dim];

            for k in 0..dim {
                let mut dx_k = 0.0f32;
                let w_offset = k * (2 * ffn_dim);
                for j in 0..ffn_dim {
                    dx_k += dg_row[j] * w_gate_up[w_offset + j];
                    dx_k += du_row[j] * w_gate_up[w_offset + ffn_dim + j];

                    dw_gate_up[w_offset + j] += x_row[k] * dg_row[j];
                    dw_gate_up[w_offset + ffn_dim + j] += x_row[k] * du_row[j];
                }
                dinp_row[k] += dx_k;
            }
        }
    }
}
