//! RMSNorm (Root Mean Square Normalization) 層
//!
//! LLaMA / Gemma などの現代的アーキテクチャで採用されている高速かつ安定した正規化。
//! y = x / RMS(x) * weight

pub struct RMSNorm;

impl RMSNorm {
    /// 順伝播
    pub fn forward(
        out: &mut [f32],
        rstd: &mut [f32],
        inp: &[f32],
        weight: &[f32],
        n: usize,
        c: usize,
        eps: f32,
    ) {
        for i in 0..n {
            let x = &inp[i * c..(i + 1) * c];
            let y = &mut out[i * c..(i + 1) * c];

            let mut sum_sq = 0.0f32;
            for &val in x {
                sum_sq += val * val;
            }
            let mean_sq = sum_sq / (c as f32);
            let inv_std = 1.0f32 / (mean_sq + eps).sqrt();
            rstd[i] = inv_std;

            for j in 0..c {
                y[j] = x[j] * inv_std * weight[j];
            }
        }
    }

    /// 逆伝播
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        dinp: &mut [f32],
        dweight: &mut [f32],
        dout: &[f32],
        inp: &[f32],
        weight: &[f32],
        rstd: &[f32],
        n: usize,
        c: usize,
    ) {
        let inv_c = 1.0f32 / (c as f32);
        for i in 0..n {
            let dy = &dout[i * c..(i + 1) * c];
            let x = &inp[i * c..(i + 1) * c];
            let dx = &mut dinp[i * c..(i + 1) * c];
            let inv_std = rstd[i];

            let mut sum_dy_x_w = 0.0f32;
            for j in 0..c {
                sum_dy_x_w += dy[j] * x[j] * weight[j];
                dweight[j] += dy[j] * x[j] * inv_std;
            }

            for j in 0..c {
                let term1 = dy[j] * weight[j] * inv_std;
                let term2 = x[j] * (inv_std * inv_std * inv_std) * inv_c * sum_dy_x_w;
                dx[j] += term1 - term2;
            }
        }
    }
}
