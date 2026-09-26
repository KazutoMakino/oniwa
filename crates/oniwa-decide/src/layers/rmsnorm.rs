//! RMSNorm (Root Mean Square Normalization) layer
//!
//! Fast and numerically stable normalization adopted in modern architectures like LLaMA / Gemma.
//! y = x / RMS(x) * weight

use crate::simd::{mul_slices_assign_simd, scale_slice_simd, sum_squares_simd};

pub struct RmsNorm;

impl RmsNorm {
    /// Forward pass
    pub fn forward(
        out: &mut [f32],
        rstd: &mut [f32],
        inp: &[f32],
        weight: &[f32],
        n: usize,
        dim: usize,
        eps: f32,
    ) {
        for i in 0..n {
            let x = &inp[i * dim..(i + 1) * dim];
            let y = &mut out[i * dim..(i + 1) * dim];

            let sum_sq = sum_squares_simd(x);
            let mean_sq = sum_sq / (dim as f32);
            let inv_std = 1.0f32 / (mean_sq + eps).sqrt();
            rstd[i] = inv_std;

            // y = (x * inv_std) * weight
            scale_slice_simd(y, x, inv_std);
            mul_slices_assign_simd(y, weight);
        }
    }

    /// Backward pass
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        dinp: &mut [f32],
        dweight: &mut [f32],
        dout: &[f32],
        inp: &[f32],
        weight: &[f32],
        rstd: &[f32],
        n: usize,
        dim: usize,
    ) {
        let inv_c = 1.0f32 / (dim as f32);
        for i in 0..n {
            let dy = &dout[i * dim..(i + 1) * dim];
            let x = &inp[i * dim..(i + 1) * dim];
            let dx = &mut dinp[i * dim..(i + 1) * dim];
            let inv_std = rstd[i];

            let mut sum_dy_x_w = 0.0f32;
            for j in 0..dim {
                sum_dy_x_w += dy[j] * x[j] * weight[j];
                dweight[j] += dy[j] * x[j] * inv_std;
            }

            for j in 0..dim {
                let term1 = dy[j] * weight[j] * inv_std;
                let term2 = x[j] * (inv_std * inv_std * inv_std) * inv_c * sum_dy_x_w;
                dx[j] += term1 - term2;
            }
        }
    }
}
