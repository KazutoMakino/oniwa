//! Quaternion Linear Transformation Layer
//!
//! Maps an input vector of $N_{in}$ quaternions to $N_{out}$ quaternions
//! via Hamilton product matrix multiplication:
//!
//! $$\mathbf{y}_j = \sum_{i=1}^{N_{in}} \mathbf{W}_{ji} \otimes \mathbf{x}_i + \mathbf{b}_j$$
//!
//! Parameter efficiency:
//! - An equivalent real linear layer has $(4 N_{in}) \times (4 N_{out}) = 16 N_{in} N_{out}$ weights.
//! - A QuaternionLinear layer has $4 N_{in} N_{out}$ real weights, yielding a 4x parameter reduction
//!   while retaining rich cross-component rotational coupling.

use crate::simd::{
    accumulate_backward_din_simd, accumulate_backward_dw_simd, accumulate_hamilton_simd,
};

pub struct QuaternionLinear;

impl QuaternionLinear {
    /// Forward pass of QuaternionLinear layer
    ///
    /// - `out`: Output array `[B, out_quat * 4]`
    /// - `input`: Input array `[B, in_quat * 4]`
    /// - `weight`: Quaternion weight array `[out_quat, in_quat * 4]`
    /// - `bias`: Optional quaternion bias array `[out_quat * 4]`
    /// - `batch_size`: Batch size B
    /// - `in_quat`: Number of input quaternions (real dim = in_quat * 4)
    /// - `out_quat`: Number of output quaternions (real dim = out_quat * 4)
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        out: &mut [f32],
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        batch_size: usize,
        in_quat: usize,
        out_quat: usize,
    ) {
        let in_dim = in_quat * 4;
        let out_dim = out_quat * 4;

        for b in 0..batch_size {
            let x_b = &input[b * in_dim..(b + 1) * in_dim];
            let out_b = &mut out[b * out_dim..(b + 1) * out_dim];

            for j in 0..out_quat {
                let mut acc = if let Some(b_vec) = bias {
                    let b_slice: &[f32; 4] = b_vec[j * 4..(j + 1) * 4].try_into().unwrap();
                    *b_slice
                } else {
                    [0.0f32; 4]
                };

                let w_row = &weight[j * in_quat * 4..(j + 1) * in_quat * 4];
                for i in 0..in_quat {
                    let w_q: &[f32; 4] = w_row[i * 4..(i + 1) * 4].try_into().unwrap();
                    let x_q: &[f32; 4] = x_b[i * 4..(i + 1) * 4].try_into().unwrap();
                    // y_j += W_ji ⊗ x_i via SIMD
                    accumulate_hamilton_simd(&mut acc, w_q, x_q);
                }

                out_b[j * 4..(j + 1) * 4].copy_from_slice(&acc);
            }
        }
    }

    /// Backward pass of QuaternionLinear layer using GHR calculus
    ///
    /// - `d_in`: Gradient w.r.t input `[B, in_quat * 4]`
    /// - `d_weight`: Gradient w.r.t weight accumulator `[out_quat, in_quat * 4]`
    /// - `d_bias`: Optional gradient w.r.t bias accumulator `[out_quat * 4]`
    /// - `d_out`: Gradient w.r.t output `[B, out_quat * 4]`
    /// - `input`: Forward input `[B, in_quat * 4]`
    /// - `weight`: Forward weight `[out_quat, in_quat * 4]`
    /// - `batch_size`: Batch size B
    /// - `in_quat`: Number of input quaternions
    /// - `out_quat`: Number of output quaternions
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        d_in: &mut [f32],
        d_weight: &mut [f32],
        mut d_bias: Option<&mut [f32]>,
        d_out: &[f32],
        input: &[f32],
        weight: &[f32],
        batch_size: usize,
        in_quat: usize,
        out_quat: usize,
    ) {
        let in_dim = in_quat * 4;
        let out_dim = out_quat * 4;

        // Zero out d_in before accumulating
        d_in.fill(0.0);

        for b in 0..batch_size {
            let x_b = &input[b * in_dim..(b + 1) * in_dim];
            let d_out_b = &d_out[b * out_dim..(b + 1) * out_dim];
            let d_in_b = &mut d_in[b * in_dim..(b + 1) * in_dim];

            for j in 0..out_quat {
                let d_y_j: &[f32; 4] = d_out_b[j * 4..(j + 1) * 4].try_into().unwrap();

                // Gradient w.r.t bias: d_bias += d_y_j
                if let Some(ref mut db) = d_bias {
                    db[j * 4] += d_y_j[0];
                    db[j * 4 + 1] += d_y_j[1];
                    db[j * 4 + 2] += d_y_j[2];
                    db[j * 4 + 3] += d_y_j[3];
                }

                let w_row = &weight[j * in_quat * 4..(j + 1) * in_quat * 4];
                let dw_row = &mut d_weight[j * in_quat * 4..(j + 1) * in_quat * 4];

                for i in 0..in_quat {
                    let w_ji: &[f32; 4] = w_row[i * 4..(i + 1) * 4].try_into().unwrap();
                    let x_i: &[f32; 4] = x_b[i * 4..(i + 1) * 4].try_into().unwrap();

                    // d_loss/d_x = W* ⊗ d_loss/d_y via SIMD
                    let d_in_slot: &mut [f32; 4] =
                        (&mut d_in_b[i * 4..(i + 1) * 4]).try_into().unwrap();
                    accumulate_backward_din_simd(d_in_slot, w_ji, d_y_j);

                    // d_loss/d_W = d_loss/d_y ⊗ x* via SIMD
                    let dw_slot: &mut [f32; 4] =
                        (&mut dw_row[i * 4..(i + 1) * 4]).try_into().unwrap();
                    accumulate_backward_dw_simd(dw_slot, d_y_j, x_i);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quaternion_linear_forward_shape() {
        let b = 2;
        let in_quat = 3;
        let out_quat = 4;
        let input = vec![0.5f32; b * in_quat * 4];
        let weight = vec![0.1f32; out_quat * in_quat * 4];
        let bias = vec![0.05f32; out_quat * 4];
        let mut out = vec![0.0f32; b * out_quat * 4];

        QuaternionLinear::forward(&mut out, &input, &weight, Some(&bias), b, in_quat, out_quat);
        assert_eq!(out.len(), b * out_quat * 4);
        for &val in &out {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_quaternion_linear_finite_difference_gradcheck() {
        let b = 2;
        let in_quat = 2;
        let out_quat = 2;

        let in_dim = in_quat * 4;
        let out_dim = out_quat * 4;
        let w_dim = out_quat * in_quat * 4;

        // Pseudo-random deterministic inputs
        let mut input = vec![0.0f32; b * in_dim];
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i as f32 * 1.3).sin() + 0.1) * 0.5;
        }

        let mut weight = vec![0.0f32; w_dim];
        for (i, v) in weight.iter_mut().enumerate() {
            *v = ((i as f32 * 2.1).cos() - 0.2) * 0.5;
        }

        let mut bias = vec![0.0f32; out_dim];
        for (i, v) in bias.iter_mut().enumerate() {
            *v = (i as f32 * 0.7).sin() * 0.1;
        }

        let mut d_out = vec![0.0f32; b * out_dim];
        for (i, v) in d_out.iter_mut().enumerate() {
            *v = ((i as f32 * 0.9).cos() + 0.3) * 0.4;
        }

        // Analytical backward
        let mut d_in = vec![0.0f32; b * in_dim];
        let mut d_weight = vec![0.0f32; w_dim];
        let mut d_bias = vec![0.0f32; out_dim];

        QuaternionLinear::backward(
            &mut d_in,
            &mut d_weight,
            Some(&mut d_bias),
            &d_out,
            &input,
            &weight,
            b,
            in_quat,
            out_quat,
        );

        // Finite difference check for weight
        let eps = 1e-3f32;
        for i in 0..w_dim {
            let orig = weight[i];

            weight[i] = orig + eps;
            let mut out_pos = vec![0.0f32; b * out_dim];
            QuaternionLinear::forward(
                &mut out_pos,
                &input,
                &weight,
                Some(&bias),
                b,
                in_quat,
                out_quat,
            );

            weight[i] = orig - eps;
            let mut out_neg = vec![0.0f32; b * out_dim];
            QuaternionLinear::forward(
                &mut out_neg,
                &input,
                &weight,
                Some(&bias),
                b,
                in_quat,
                out_quat,
            );

            weight[i] = orig;

            // Loss = sum(out * d_out)
            let mut loss_pos = 0.0f32;
            let mut loss_neg = 0.0f32;
            for k in 0..b * out_dim {
                loss_pos += out_pos[k] * d_out[k];
                loss_neg += out_neg[k] * d_out[k];
            }

            let num_grad = (loss_pos - loss_neg) / (2.0 * eps);
            let ana_grad = d_weight[i];
            let diff = (num_grad - ana_grad).abs();
            assert!(
                diff < 5e-3,
                "Weight gradcheck failed at index {}: num={}, ana={}, diff={}",
                i,
                num_grad,
                ana_grad,
                diff
            );
        }

        // Finite difference check for input
        for i in 0..b * in_dim {
            let orig = input[i];

            input[i] = orig + eps;
            let mut out_pos = vec![0.0f32; b * out_dim];
            QuaternionLinear::forward(
                &mut out_pos,
                &input,
                &weight,
                Some(&bias),
                b,
                in_quat,
                out_quat,
            );

            input[i] = orig - eps;
            let mut out_neg = vec![0.0f32; b * out_dim];
            QuaternionLinear::forward(
                &mut out_neg,
                &input,
                &weight,
                Some(&bias),
                b,
                in_quat,
                out_quat,
            );

            input[i] = orig;

            let mut loss_pos = 0.0f32;
            let mut loss_neg = 0.0f32;
            for k in 0..b * out_dim {
                loss_pos += out_pos[k] * d_out[k];
                loss_neg += out_neg[k] * d_out[k];
            }

            let num_grad = (loss_pos - loss_neg) / (2.0 * eps);
            let ana_grad = d_in[i];
            let diff = (num_grad - ana_grad).abs();
            assert!(
                diff < 5e-3,
                "Input gradcheck failed at index {}: num={}, ana={}, diff={}",
                i,
                num_grad,
                ana_grad,
                diff
            );
        }

        // Finite difference check for bias
        for i in 0..out_dim {
            let orig = bias[i];

            bias[i] = orig + eps;
            let mut out_pos = vec![0.0f32; b * out_dim];
            QuaternionLinear::forward(
                &mut out_pos,
                &input,
                &weight,
                Some(&bias),
                b,
                in_quat,
                out_quat,
            );

            bias[i] = orig - eps;
            let mut out_neg = vec![0.0f32; b * out_dim];
            QuaternionLinear::forward(
                &mut out_neg,
                &input,
                &weight,
                Some(&bias),
                b,
                in_quat,
                out_quat,
            );

            bias[i] = orig;

            let mut loss_pos = 0.0f32;
            let mut loss_neg = 0.0f32;
            for k in 0..b * out_dim {
                loss_pos += out_pos[k] * d_out[k];
                loss_neg += out_neg[k] * d_out[k];
            }

            let num_grad = (loss_pos - loss_neg) / (2.0 * eps);
            let ana_grad = d_bias[i];
            let diff = (num_grad - ana_grad).abs();
            assert!(
                diff < 5e-3,
                "Bias gradcheck failed at index {}: num={}, ana={}, diff={}",
                i,
                num_grad,
                ana_grad,
                diff
            );
        }
    }
}
