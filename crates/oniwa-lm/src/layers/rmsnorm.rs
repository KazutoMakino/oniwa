//! RMSNorm (Root Mean Square Normalization)
//!
//! Forward pass formula:
//!   RMS(x) = sqrt( mean(x^2) + eps )
//!   x_hat = x / RMS(x)
//!   y = x_hat * weight
//!
//! Backward pass formula:
//!   dweight = sum( dout * x_hat )
//!   dx = (1 / RMS(x)) * [ weight * dout - (x_hat / d) * sum(dout * weight * x_hat) ]

pub struct RmsNorm;

impl RmsNorm {
    /// Forward pass
    ///
    /// - `out`: Output buffer [N, D]
    /// - `rstd`: Cache for backward pass (1 / RMS(x)) [N]
    /// - `inp`: Input tensor [N, D]
    /// - `weight`: Weight parameter [D]
    /// - `eps`: Epsilon for numerical stability
    pub fn forward(
        out: &mut [f32],
        rstd: &mut [f32],
        inp: &[f32],
        weight: &[f32],
        eps: f32,
        dim: usize,
    ) {
        let n = inp.len() / dim;
        assert_eq!(out.len(), inp.len());
        assert_eq!(rstd.len(), n);
        assert_eq!(weight.len(), dim);

        for (row, r_out) in rstd.iter_mut().enumerate().take(n) {
            let offset = row * dim;
            let inp_row = &inp[offset..offset + dim];
            let out_row = &mut out[offset..offset + dim];

            // 1. Sum of squares
            let mut sum_sq = 0.0f32;
            for &x in inp_row {
                sum_sq += x * x;
            }

            // 2. Reciprocal of RMS (rstd = 1 / sqrt(mean + eps))
            let mean_sq = sum_sq / (dim as f32);
            let r = 1.0f32 / (mean_sq + eps).sqrt();
            *r_out = r;

            // 3. Normalize and scale: out = (x * r) * weight
            for i in 0..dim {
                out_row[i] = inp_row[i] * r * weight[i];
            }
        }
    }

    /// Backward pass
    ///
    /// - `dinp`: Input gradient buffer [N, D] (accumulated)
    /// - `dweight`: Weight gradient buffer [D] (accumulated)
    /// - `dout`: Upstream output gradient [N, D]
    /// - `inp`: Input from forward pass [N, D]
    /// - `rstd`: Saved cache from forward pass [N]
    /// - `weight`: Weight parameter [D]
    pub fn backward(
        dinp: &mut [f32],
        dweight: &mut [f32],
        dout: &[f32],
        inp: &[f32],
        rstd: &[f32],
        weight: &[f32],
        dim: usize,
    ) {
        let n = inp.len() / dim;
        assert_eq!(dinp.len(), inp.len());
        assert_eq!(dout.len(), inp.len());
        assert_eq!(rstd.len(), n);
        assert_eq!(dweight.len(), dim);
        assert_eq!(weight.len(), dim);

        for (row, &r) in rstd.iter().enumerate().take(n) {
            let offset = row * dim;
            let inp_row = &inp[offset..offset + dim];
            let dout_row = &dout[offset..offset + dim];
            let dinp_row = &mut dinp[offset..offset + dim];

            // Inner product S = sum( dout * weight * x_hat )
            let mut s = 0.0f32;
            for i in 0..dim {
                let x_hat = inp_row[i] * r;
                s += dout_row[i] * weight[i] * x_hat;
                // Accumulate weight gradient
                dweight[i] += dout_row[i] * x_hat;
            }

            // Input gradient dx = r * [ weight * dout - (x_hat / d) * S ]
            let factor = s / (dim as f32);
            for i in 0..dim {
                let x_hat = inp_row[i] * r;
                dinp_row[i] += r * (weight[i] * dout_row[i] - x_hat * factor);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_gradcheck() {
        // Finite Differences vs Analytic Gradient comparison test
        let d = 4;
        let n = 2;
        let eps = 1e-5f32;

        let inp = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0];
        let weight = vec![0.5, 1.2, 0.8, 1.0];
        let dout = vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.2, 0.1, -0.3];

        let mut out = vec![0.0; n * d];
        let mut rstd = vec![0.0; n];
        RmsNorm::forward(&mut out, &mut rstd, &inp, &weight, eps, d);

        let mut dinp = vec![0.0; n * d];
        let mut dweight = vec![0.0; d];
        RmsNorm::backward(&mut dinp, &mut dweight, &dout, &inp, &rstd, &weight, d);

        // Objective: L = sum(out * dout)
        let delta = 1e-3f32;

        // 1. Numerical check for input gradient dinp
        for i in 0..(n * d) {
            let mut inp_plus = inp.clone();
            let mut inp_minus = inp.clone();
            inp_plus[i] += delta;
            inp_minus[i] -= delta;

            let mut out_plus = vec![0.0; n * d];
            let mut rstd_dummy = vec![0.0; n];
            RmsNorm::forward(&mut out_plus, &mut rstd_dummy, &inp_plus, &weight, eps, d);

            let mut out_minus = vec![0.0; n * d];
            RmsNorm::forward(&mut out_minus, &mut rstd_dummy, &inp_minus, &weight, eps, d);

            let l_plus: f32 = out_plus.iter().zip(&dout).map(|(a, b)| a * b).sum();
            let l_minus: f32 = out_minus.iter().zip(&dout).map(|(a, b)| a * b).sum();
            let num_grad = (l_plus - l_minus) / (2.0 * delta);

            let abs_err = (dinp[i] - num_grad).abs();
            assert!(
                abs_err < 1e-3,
                "dinp[{}] diff too large: analytic={}, numeric={}",
                i,
                dinp[i],
                num_grad
            );
        }

        // 2. Numerical check for weight gradient dweight
        for i in 0..d {
            let mut w_plus = weight.clone();
            let mut w_minus = weight.clone();
            w_plus[i] += delta;
            w_minus[i] -= delta;

            let mut out_plus = vec![0.0; n * d];
            let mut rstd_dummy = vec![0.0; n];
            RmsNorm::forward(&mut out_plus, &mut rstd_dummy, &inp, &w_plus, eps, d);

            let mut out_minus = vec![0.0; n * d];
            RmsNorm::forward(&mut out_minus, &mut rstd_dummy, &inp, &w_minus, eps, d);

            let l_plus: f32 = out_plus.iter().zip(&dout).map(|(a, b)| a * b).sum();
            let l_minus: f32 = out_minus.iter().zip(&dout).map(|(a, b)| a * b).sum();
            let num_grad = (l_plus - l_minus) / (2.0 * delta);

            let abs_err = (dweight[i] - num_grad).abs();
            assert!(
                abs_err < 1e-3,
                "dweight[{}] diff too large: analytic={}, numeric={}",
                i,
                dweight[i],
                num_grad
            );
        }
    }
}
