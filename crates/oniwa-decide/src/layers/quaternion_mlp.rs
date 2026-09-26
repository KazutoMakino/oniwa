//! Quaternion SwiGLU MLP Layer with Hamilton Product Projections
//!
//! Features:
//! - Inputs $X \in \mathbb{R}^{N \times D}$ interpreted as $\mathbb{H}^{N \times D/4}$
//! - Gate & Up projections via QuaternionLinear: $\mathbb{H}^{D/4} \to \mathbb{H}^{D_{ffn}/4}$
//!   $$G = W_{gate} \otimes X, \quad U = W_{up} \otimes X$$
//! - SwiGLU activation applied component-wise to all 4 quaternion components:
//!   $$H = \text{SiLU}(G) \odot U$$
//!   where $\text{SiLU}(g) = \frac{g}{1 + e^{-g}}$
//! - Down projection via QuaternionLinear: $\mathbb{H}^{D_{ffn}/4} \to \mathbb{H}^{D/4}$
//!   $$Y = W_{down} \otimes H$$
//! - Analytical backward pass derived with GHR calculus.

use crate::layers::QuaternionLinear;
use crate::simd::mul_slices_assign_simd;

pub struct QuaternionSwiGlu;

impl QuaternionSwiGlu {
    #[inline(always)]
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    #[inline(always)]
    fn silu_grad(x: f32) -> f32 {
        let sig = 1.0 / (1.0 + (-x).exp());
        sig * (1.0 + x * (1.0 - sig))
    }

    /// Forward pass of Quaternion SwiGLU MLP
    ///
    /// - `w_gate_up`: Quaternion weights for Gate and Up projections:
    ///   `[2 * ffn_quat, in_quat * 4]` where `in_quat = dim / 4`, `ffn_quat = ffn_dim / 4`
    /// - `w_down`: Quaternion weights for Down projection:
    ///   `[in_quat, ffn_quat * 4]`
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
        assert_eq!(dim % 4, 0, "Embedding dim must be divisible by 4");
        assert_eq!(ffn_dim % 4, 0, "FFN dim must be divisible by 4");
        let in_quat = dim / 4;
        let ffn_quat = ffn_dim / 4;
        let gate_up_block = ffn_quat * in_quat * 4;

        let w_gate = &w_gate_up[0..gate_up_block];
        let w_up = &w_gate_up[gate_up_block..2 * gate_up_block];

        // 1. Gate & Up projections via QuaternionLinear
        QuaternionLinear::forward(act_g, inp, w_gate, None, n, in_quat, ffn_quat);
        QuaternionLinear::forward(act_u, inp, w_up, None, n, in_quat, ffn_quat);
        let total_ffn = n * ffn_dim;

        // 2. Component-wise SwiGLU: H = SiLU(G) * U across all N * FFN real components
        // Compute standard SiLU scalar into act_h temporarily, then SIMD multiply with act_u
        for idx in 0..total_ffn {
            act_h[idx] = Self::silu(act_g[idx]);
        }
        mul_slices_assign_simd(&mut act_h[..total_ffn], &act_u[..total_ffn]);

        // 3. Down projection via QuaternionLinear
        QuaternionLinear::forward(out, act_h, w_down, None, n, ffn_quat, in_quat);
    }

    /// Backward pass of Quaternion SwiGLU MLP using GHR calculus
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
        let in_quat = dim / 4;
        let ffn_quat = ffn_dim / 4;
        let gate_up_block = ffn_quat * in_quat * 4;

        let w_gate = &w_gate_up[0..gate_up_block];
        let w_up = &w_gate_up[gate_up_block..2 * gate_up_block];

        // 1. Backward through Down projection: dH and dw_down
        let mut dh = vec![0.0f32; n * ffn_dim];
        QuaternionLinear::backward(
            &mut dh, dw_down, None, dout, act_h, w_down, n, ffn_quat, in_quat,
        );

        // 2. Backward through SwiGLU component-wise activation
        let mut dg = vec![0.0f32; n * ffn_dim];
        let mut du = vec![0.0f32; n * ffn_dim];
        for idx in 0..n * ffn_dim {
            let g = act_g[idx];
            let u = act_u[idx];
            let dh_val = dh[idx];

            let swish_g = Self::silu(g);
            du[idx] = dh_val * swish_g;

            let dswish = dh_val * u;
            dg[idx] = dswish * Self::silu_grad(g);
        }

        // 3. Backward through Gate and Up projections: dinp and dw_gate_up
        let (dw_gate, dw_up) = dw_gate_up.split_at_mut(gate_up_block);
        let mut dinp_g = vec![0.0f32; n * dim];
        let mut dinp_u = vec![0.0f32; n * dim];

        QuaternionLinear::backward(
            &mut dinp_g,
            dw_gate,
            None,
            &dg,
            inp,
            w_gate,
            n,
            in_quat,
            ffn_quat,
        );
        QuaternionLinear::backward(
            &mut dinp_u,
            dw_up,
            None,
            &du,
            inp,
            w_up,
            n,
            in_quat,
            ffn_quat,
        );

        for idx in 0..n * dim {
            dinp[idx] += dinp_g[idx] + dinp_u[idx];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oniwa_lm::reproducibility::DeterministicRng;

    #[test]
    fn test_quaternion_mlp_forward_and_backward_gradcheck() {
        let n = 2;
        let c = 8; // 2 quaternions
        let ffn = 16; // 4 quaternions
        let in_quat = c / 4;
        let ffn_quat = ffn / 4;

        let mut rng = DeterministicRng::new(202);

        let mut inp = vec![0.0f32; n * c];
        for val in inp.iter_mut() {
            *val = rng.next_f32() * 0.4 - 0.2;
        }

        let mut w_gate_up = vec![0.0f32; 2 * ffn_quat * in_quat * 4];
        for val in w_gate_up.iter_mut() {
            *val = rng.next_f32() * 0.4 - 0.2;
        }

        let mut w_down = vec![0.0f32; in_quat * ffn_quat * 4];
        for val in w_down.iter_mut() {
            *val = rng.next_f32() * 0.4 - 0.2;
        }

        let mut out = vec![0.0f32; n * c];
        let mut act_g = vec![0.0f32; n * ffn];
        let mut act_u = vec![0.0f32; n * ffn];
        let mut act_h = vec![0.0f32; n * ffn];

        QuaternionSwiGlu::forward(
            &mut out, &mut act_g, &mut act_u, &mut act_h, &inp, &w_gate_up, &w_down, n, c, ffn,
        );

        let dout = out.clone();
        let mut dinp = vec![0.0f32; n * c];
        let mut dw_gate_up = vec![0.0f32; w_gate_up.len()];
        let mut dw_down = vec![0.0f32; w_down.len()];

        QuaternionSwiGlu::backward(
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

        // 1. Gradcheck for input
        let eps = 1e-3f32;
        let test_idx = 3;
        let orig_x = inp[test_idx];

        inp[test_idx] = orig_x + eps;
        let mut out_p = vec![0.0f32; n * c];
        QuaternionSwiGlu::forward(
            &mut out_p, &mut act_g, &mut act_u, &mut act_h, &inp, &w_gate_up, &w_down, n, c, ffn,
        );
        let loss_p: f32 = 0.5 * out_p.iter().map(|&v| v * v).sum::<f32>();

        inp[test_idx] = orig_x - eps;
        let mut out_m = vec![0.0f32; n * c];
        QuaternionSwiGlu::forward(
            &mut out_m, &mut act_g, &mut act_u, &mut act_h, &inp, &w_gate_up, &w_down, n, c, ffn,
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

        // 2. Gradcheck for down weight
        let down_idx = 4;
        let orig_w_down = w_down[down_idx];
        w_down[down_idx] = orig_w_down + eps;
        QuaternionSwiGlu::forward(
            &mut out_p, &mut act_g, &mut act_u, &mut act_h, &inp, &w_gate_up, &w_down, n, c, ffn,
        );
        let loss_down_p: f32 = 0.5 * out_p.iter().map(|&v| v * v).sum::<f32>();

        w_down[down_idx] = orig_w_down - eps;
        QuaternionSwiGlu::forward(
            &mut out_m, &mut act_g, &mut act_u, &mut act_h, &inp, &w_gate_up, &w_down, n, c, ffn,
        );
        let loss_down_m: f32 = 0.5 * out_m.iter().map(|&v| v * v).sum::<f32>();
        w_down[down_idx] = orig_w_down;

        let num_down_grad = (loss_down_p - loss_down_m) / (2.0 * eps);
        let ana_down_grad = dw_down[down_idx];
        let diff_down = (num_down_grad - ana_down_grad).abs();
        assert!(
            diff_down < 5e-3
                || diff_down / (ana_down_grad.abs() + num_down_grad.abs()).max(1e-5) < 0.05,
            "Down weight gradcheck failed! Analytic: {}, Numerical: {}, Diff: {}",
            ana_down_grad,
            num_down_grad,
            diff_down
        );

        // 3. Gradcheck for gate_up weight
        let gu_idx = 7;
        let orig_gu = w_gate_up[gu_idx];
        w_gate_up[gu_idx] = orig_gu + eps;
        QuaternionSwiGlu::forward(
            &mut out_p, &mut act_g, &mut act_u, &mut act_h, &inp, &w_gate_up, &w_down, n, c, ffn,
        );
        let loss_gu_p: f32 = 0.5 * out_p.iter().map(|&v| v * v).sum::<f32>();

        w_gate_up[gu_idx] = orig_gu - eps;
        QuaternionSwiGlu::forward(
            &mut out_m, &mut act_g, &mut act_u, &mut act_h, &inp, &w_gate_up, &w_down, n, c, ffn,
        );
        let loss_gu_m: f32 = 0.5 * out_m.iter().map(|&v| v * v).sum::<f32>();
        w_gate_up[gu_idx] = orig_gu;

        let num_gu_grad = (loss_gu_p - loss_gu_m) / (2.0 * eps);
        let ana_gu_grad = dw_gate_up[gu_idx];
        let diff_gu = (num_gu_grad - ana_gu_grad).abs();
        assert!(
            diff_gu < 5e-3 || diff_gu / (ana_gu_grad.abs() + num_gu_grad.abs()).max(1e-5) < 0.05,
            "Gate/Up weight gradcheck failed! Analytic: {}, Numerical: {}, Diff: {}",
            ana_gu_grad,
            num_gu_grad,
            diff_gu
        );
    }
}
