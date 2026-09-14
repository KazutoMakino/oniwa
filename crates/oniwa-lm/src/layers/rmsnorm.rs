//! RMSNorm (Root Mean Square Normalization)
//!
//! 順伝播数式:
//!   RMS(x) = sqrt( mean(x^2) + eps )
//!   x_hat = x / RMS(x)
//!   y = x_hat * weight
//!
//! 逆伝播数式:
//!   dweight = sum( dout * x_hat )
//!   dx = (1 / RMS(x)) * [ weight * dout - (x_hat / d) * sum(dout * weight * x_hat) ]

pub struct RMSNorm;

impl RMSNorm {
    /// 順伝播
    ///
    /// - `out`: 出力バッファ [N, D]
    /// - `rstd`: 逆伝播用キャッシュ (1 / RMS(x)) [N]
    /// - `inp`: 入力テンソル [N, D]
    /// - `weight`: 重みパラメータ [D]
    /// - `eps`: ゼロ除算防止用イプシロン
    pub fn forward(
        out: &mut [f32],
        rstd: &mut [f32],
        inp: &[f32],
        weight: &[f32],
        eps: f32,
        d: usize,
    ) {
        let n = inp.len() / d;
        assert_eq!(out.len(), inp.len());
        assert_eq!(rstd.len(), n);
        assert_eq!(weight.len(), d);

        for row in 0..n {
            let offset = row * d;
            let inp_row = &inp[offset..offset + d];
            let out_row = &mut out[offset..offset + d];

            // 1. 二乗和の計算
            let mut sum_sq = 0.0f32;
            for &x in inp_row {
                sum_sq += x * x;
            }

            // 2. RMSの逆数 (rstd = 1 / sqrt(mean + eps))
            let mean_sq = sum_sq / (d as f32);
            let r = 1.0f32 / (mean_sq + eps).sqrt();
            rstd[row] = r;

            // 3. 正規化とスケーリング: out = (x * r) * weight
            for i in 0..d {
                out_row[i] = inp_row[i] * r * weight[i];
            }
        }
    }

    /// 逆伝播
    ///
    /// - `dinp`: 入力勾配バッファ [N, D] (加算・累積)
    /// - `dweight`: 重み勾配バッファ [D] (加算・累積)
    /// - `dout`: 上流からの出力勾配 [N, D]
    /// - `inp`: 順伝播時の入力 [N, D]
    /// - `rstd`: 順伝播時に保存したキャッシュ [N]
    /// - `weight`: 重みパラメータ [D]
    pub fn backward(
        dinp: &mut [f32],
        dweight: &mut [f32],
        dout: &[f32],
        inp: &[f32],
        rstd: &[f32],
        weight: &[f32],
        d: usize,
    ) {
        let n = inp.len() / d;
        assert_eq!(dinp.len(), inp.len());
        assert_eq!(dout.len(), inp.len());
        assert_eq!(rstd.len(), n);
        assert_eq!(dweight.len(), d);
        assert_eq!(weight.len(), d);

        for row in 0..n {
            let offset = row * d;
            let inp_row = &inp[offset..offset + d];
            let dout_row = &dout[offset..offset + d];
            let dinp_row = &mut dinp[offset..offset + d];
            let r = rstd[row];

            // 内積 S = sum( dout * weight * x_hat )
            let mut s = 0.0f32;
            for i in 0..d {
                let x_hat = inp_row[i] * r;
                s += dout_row[i] * weight[i] * x_hat;
                // 重み勾配の蓄積
                dweight[i] += dout_row[i] * x_hat;
            }

            // 入力勾配 dx = r * [ weight * dout - (x_hat / d) * S ]
            let factor = s / (d as f32);
            for i in 0..d {
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
        // 数値微分 (Finite Differences) と解析的微分の比較テスト
        let d = 4;
        let n = 2;
        let eps = 1e-5f32;

        let inp = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0];
        let weight = vec![0.5, 1.2, 0.8, 1.0];
        let dout = vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.2, 0.1, -0.3];

        let mut out = vec![0.0; n * d];
        let mut rstd = vec![0.0; n];
        RMSNorm::forward(&mut out, &mut rstd, &inp, &weight, eps, d);

        let mut dinp = vec![0.0; n * d];
        let mut dweight = vec![0.0; d];
        RMSNorm::backward(&mut dinp, &mut dweight, &dout, &inp, &rstd, &weight, d);

        // 目的関数: L = sum(out * dout)
        let delta = 1e-3f32;

        // 1. 入力勾配 dinp の数値チェック
        for i in 0..(n * d) {
            let mut inp_plus = inp.clone();
            let mut inp_minus = inp.clone();
            inp_plus[i] += delta;
            inp_minus[i] -= delta;

            let mut out_plus = vec![0.0; n * d];
            let mut rstd_dummy = vec![0.0; n];
            RMSNorm::forward(&mut out_plus, &mut rstd_dummy, &inp_plus, &weight, eps, d);

            let mut out_minus = vec![0.0; n * d];
            RMSNorm::forward(&mut out_minus, &mut rstd_dummy, &inp_minus, &weight, eps, d);

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

        // 2. 重み勾配 dweight の数値チェック
        for i in 0..d {
            let mut w_plus = weight.clone();
            let mut w_minus = weight.clone();
            w_plus[i] += delta;
            w_minus[i] -= delta;

            let mut out_plus = vec![0.0; n * d];
            let mut rstd_dummy = vec![0.0; n];
            RMSNorm::forward(&mut out_plus, &mut rstd_dummy, &inp, &w_plus, eps, d);

            let mut out_minus = vec![0.0; n * d];
            RMSNorm::forward(&mut out_minus, &mut rstd_dummy, &inp, &w_minus, eps, d);

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
