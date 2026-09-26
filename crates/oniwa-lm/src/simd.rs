//! ARM NEON SIMD Acceleration for Quaternion Operations
//!
//! Provides vector intrinsics (`float32x4_t`) for:
//! - Hamilton product: $q_1 \otimes q_2$
//! - Accumulating Hamilton product: $y \mathrel{+}= w \otimes x$
//! - Quaternion conjugate Hamilton product: $w^* \otimes d$ and $d \otimes x^*$
//! - 4D Vector Dot Product
//!
//! When compiling on `aarch64` targets, NEON intrinsics are utilized directly.
//! On non-aarch64 platforms, optimized scalar fallback functions are transparently executed.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Computes Hamilton product `out = a ⊗ b`
///
/// a = [w_a, x_a, y_a, z_a]
/// b = [w_b, x_b, y_b, z_b]
///
/// Formula:
/// w =  w_a*w_b - x_a*x_b - y_a*y_b - z_a*z_b
/// x =  x_a*w_b + w_a*x_b - z_a*y_b + y_a*z_b
/// y =  y_a*w_b + z_a*x_b + w_a*y_b - x_a*z_b
/// z =  z_a*w_b - y_a*x_b + x_a*y_b + w_a*z_b
///
/// Returns [w, x, y, z]
#[inline(always)]
pub fn hamilton_product_simd(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let wb = vdupq_n_f32(b[0]);
        let xb = vdupq_n_f32(b[1]);
        let yb = vdupq_n_f32(b[2]);
        let zb = vdupq_n_f32(b[3]);

        // Four vector columns of a:
        // col_w: [ w_a,  x_a,  y_a,  z_a]
        // col_x: [-x_a,  w_a,  z_a, -y_a]
        // col_y: [-y_a, -z_a,  w_a,  x_a]
        // col_z: [-z_a,  y_a, -x_a,  w_a]
        let col_w = vld1q_f32(a.as_ptr());
        let col_x_arr = [-a[1], a[0], a[3], -a[2]];
        let col_y_arr = [-a[2], -a[3], a[0], a[1]];
        let col_z_arr = [-a[3], a[2], -a[1], a[0]];

        let col_x = vld1q_f32(col_x_arr.as_ptr());
        let col_y = vld1q_f32(col_y_arr.as_ptr());
        let col_z = vld1q_f32(col_z_arr.as_ptr());

        // res = col_w * wb + col_x * xb + col_y * yb + col_z * zb
        let mut res = vmulq_f32(col_w, wb);
        res = vfmaq_f32(res, col_x, xb);
        res = vfmaq_f32(res, col_y, yb);
        res = vfmaq_f32(res, col_z, zb);

        let mut out = [0.0f32; 4];
        vst1q_f32(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ]
    }
}

/// Accumulates `acc += w ⊗ x` using SIMD FMA operations
#[inline(always)]
pub fn accumulate_hamilton_simd(acc: &mut [f32; 4], w: &[f32; 4], x: &[f32; 4]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let xx = vdupq_n_f32(x[0]);
        let xy = vdupq_n_f32(x[1]);
        let xz = vdupq_n_f32(x[2]);
        let xw = vdupq_n_f32(x[3]);

        let col_w = vld1q_f32(w.as_ptr());
        let col_x_arr = [-w[1], w[0], w[3], -w[2]];
        let col_y_arr = [-w[2], -w[3], w[0], w[1]];
        let col_z_arr = [-w[3], w[2], -w[1], w[0]];

        let col_x = vld1q_f32(col_x_arr.as_ptr());
        let col_y = vld1q_f32(col_y_arr.as_ptr());
        let col_z = vld1q_f32(col_z_arr.as_ptr());

        let mut vacc = vld1q_f32(acc.as_ptr());
        vacc = vfmaq_f32(vacc, col_w, xx);
        vacc = vfmaq_f32(vacc, col_x, xy);
        vacc = vfmaq_f32(vacc, col_y, xz);
        vacc = vfmaq_f32(vacc, col_z, xw);

        vst1q_f32(acc.as_mut_ptr(), vacc);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        acc[0] += w[0] * x[0] - w[1] * x[1] - w[2] * x[2] - w[3] * x[3];
        acc[1] += w[0] * x[1] + w[1] * x[0] + w[2] * x[3] - w[3] * x[2];
        acc[2] += w[0] * x[2] - w[1] * x[3] + w[2] * x[0] + w[3] * x[1];
        acc[3] += w[0] * x[3] + w[1] * x[2] - w[2] * x[1] + w[3] * x[0];
    }
}

/// Accumulates backward gradient for input: `din += w* ⊗ dout`
#[inline(always)]
pub fn accumulate_backward_din_simd(din: &mut [f32; 4], w: &[f32; 4], dout: &[f32; 4]) {
    let w_conj = [w[0], -w[1], -w[2], -w[3]];
    accumulate_hamilton_simd(din, &w_conj, dout);
}

/// Accumulates backward gradient for weight: `dw += dout ⊗ x*`
#[inline(always)]
pub fn accumulate_backward_dw_simd(dw: &mut [f32; 4], dout: &[f32; 4], x: &[f32; 4]) {
    let x_conj = [x[0], -x[1], -x[2], -x[3]];
    accumulate_hamilton_simd(dw, dout, &x_conj);
}

/// Vector dot product in 4D: `a . b = a0*b0 + a1*b1 + a2*b2 + a3*b3`
#[inline(always)]
pub fn dot_product_4d_simd(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let va = vld1q_f32(a.as_ptr());
        let vb = vld1q_f32(b.as_ptr());
        let prod = vmulq_f32(va, vb);
        vaddvq_f32(prod)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    }
}

/// Computes the dot product of two slices: `\sum a[i] * b[i]`
#[inline(always)]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "Slice length mismatch: a len {} vs b len {}",
        a.len(),
        b.len()
    );

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let (a_chunks, a_rem) = a.as_chunks::<4>();
        let (b_chunks, b_rem) = b.as_chunks::<4>();

        let mut vacc = vdupq_n_f32(0.0);
        for (ca, cb) in a_chunks.iter().zip(b_chunks.iter()) {
            let va = vld1q_f32(ca.as_ptr());
            let vb = vld1q_f32(cb.as_ptr());
            vacc = vfmaq_f32(vacc, va, vb);
        }

        let mut sum = vaddvq_f32(vacc);
        for i in 0..a_rem.len() {
            sum += a_rem[i] * b_rem[i];
        }
        sum
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f32>()
    }
}

/// Computes the sum of squared elements in a slice: `\sum x[i]^2`
#[inline(always)]
pub fn sum_squares_simd(x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let (chunks, rem) = x.as_chunks::<4>();
        let mut vacc = vdupq_n_f32(0.0);
        for chunk in chunks {
            let vx = vld1q_f32(chunk.as_ptr());
            vacc = vfmaq_f32(vacc, vx, vx);
        }
        let mut sum = vaddvq_f32(vacc);
        for &val in rem {
            sum += val * val;
        }
        sum
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        x.iter().map(|&v| v * v).sum::<f32>()
    }
}

/// Element-wise slice multiplication: `out[i] = a[i] * b[i]`
#[inline(always)]
pub fn mul_slices_simd(out: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(
        out.len(),
        a.len(),
        "Slice length mismatch: out len {} vs a len {}",
        out.len(),
        a.len()
    );
    assert_eq!(
        a.len(),
        b.len(),
        "Slice length mismatch: a len {} vs b len {}",
        a.len(),
        b.len()
    );

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let (out_chunks, out_rem) = out.as_chunks_mut::<4>();
        let (a_chunks, a_rem) = a.as_chunks::<4>();
        let (b_chunks, b_rem) = b.as_chunks::<4>();

        for (o, (ca, cb)) in out_chunks
            .iter_mut()
            .zip(a_chunks.iter().zip(b_chunks.iter()))
        {
            let va = vld1q_f32(ca.as_ptr());
            let vb = vld1q_f32(cb.as_ptr());
            let vprod = vmulq_f32(va, vb);
            vst1q_f32(o.as_mut_ptr(), vprod);
        }

        for i in 0..out_rem.len() {
            out_rem[i] = a_rem[i] * b_rem[i];
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..out.len() {
            out[i] = a[i] * b[i];
        }
    }
}

/// In-place element-wise slice multiplication: `a[i] *= b[i]`
#[inline(always)]
pub fn mul_slices_assign_simd(a: &mut [f32], b: &[f32]) {
    assert_eq!(
        a.len(),
        b.len(),
        "Slice length mismatch: a len {} vs b len {}",
        a.len(),
        b.len()
    );

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let (a_chunks, a_rem) = a.as_chunks_mut::<4>();
        let (b_chunks, b_rem) = b.as_chunks::<4>();

        for (ca, cb) in a_chunks.iter_mut().zip(b_chunks.iter()) {
            let va = vld1q_f32(ca.as_ptr());
            let vb = vld1q_f32(cb.as_ptr());
            let vprod = vmulq_f32(va, vb);
            vst1q_f32(ca.as_mut_ptr(), vprod);
        }

        for i in 0..a_rem.len() {
            a_rem[i] *= b_rem[i];
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..a.len() {
            a[i] *= b[i];
        }
    }
}

/// Multiplies slice elements by a scalar: `out[i] = a[i] * scalar`
#[inline(always)]
pub fn scale_slice_simd(out: &mut [f32], a: &[f32], scalar: f32) {
    assert_eq!(
        out.len(),
        a.len(),
        "Slice length mismatch: out len {} vs a len {}",
        out.len(),
        a.len()
    );

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let vscalar = vdupq_n_f32(scalar);
        let (out_chunks, out_rem) = out.as_chunks_mut::<4>();
        let (a_chunks, a_rem) = a.as_chunks::<4>();

        for (o, ca) in out_chunks.iter_mut().zip(a_chunks.iter()) {
            let va = vld1q_f32(ca.as_ptr());
            let vprod = vmulq_f32(va, vscalar);
            vst1q_f32(o.as_mut_ptr(), vprod);
        }

        for i in 0..out_rem.len() {
            out_rem[i] = a_rem[i] * scalar;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..out.len() {
            out[i] = a[i] * scalar;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quaternion::Quaternion;

    #[test]
    fn test_hamilton_product_simd_equivalence() {
        let a = [1.5f32, -2.0, 3.2, -0.7];
        let b = [-0.8f32, 1.1, -2.4, 3.0];

        let q_a = Quaternion::from_slice(&a);
        let q_b = Quaternion::from_slice(&b);
        let q_ref = q_a.hamilton_product(&q_b);

        let simd_out = hamilton_product_simd(&a, &b);

        for (k, &simd_val) in simd_out.iter().enumerate() {
            let ref_val = match k {
                0 => q_ref.w,
                1 => q_ref.x,
                2 => q_ref.y,
                _ => q_ref.z,
            };
            let diff = (simd_val - ref_val).abs();
            assert!(
                diff < 1e-6,
                "Mismatch at lane {}: simd {} vs ref {}",
                k,
                simd_val,
                ref_val
            );
        }
    }

    #[test]
    fn test_accumulate_hamilton_simd_equivalence() {
        let mut acc = [0.5f32, -0.2, 1.0, -1.5];
        let w = [2.0f32, -1.0, 0.5, 3.0];
        let x = [1.2f32, 0.4, -1.1, 0.8];

        let q_acc = Quaternion::from_slice(&acc);
        let q_w = Quaternion::from_slice(&w);
        let q_x = Quaternion::from_slice(&x);
        let q_ref = q_acc + q_w.hamilton_product(&q_x);

        accumulate_hamilton_simd(&mut acc, &w, &x);

        assert!((acc[0] - q_ref.w).abs() < 1e-6);
        assert!((acc[1] - q_ref.x).abs() < 1e-6);
        assert!((acc[2] - q_ref.y).abs() < 1e-6);
        assert!((acc[3] - q_ref.z).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_4d_simd_equivalence() {
        let a = [1.2f32, -3.4, 5.6, -7.8];
        let b = [-2.1f32, 4.3, -6.5, 8.7];

        let q_a = Quaternion::from_slice(&a);
        let q_b = Quaternion::from_slice(&b);
        let ref_dot = q_a.dot(&q_b);

        let simd_dot = dot_product_4d_simd(&a, &b);
        assert!((simd_dot - ref_dot).abs() < 1e-5);
    }

    #[test]
    fn test_sum_squares_simd_equivalence() {
        // Test multiple lengths including multiples and non-multiples of 4
        for len in [0, 1, 3, 4, 7, 16, 35, 128] {
            let input: Vec<f32> = (0..len).map(|i| (i as f32) * 0.15 - 1.2).collect();
            let ref_sum = input.iter().map(|&v| v * v).sum::<f32>();
            let simd_sum = sum_squares_simd(&input);
            let diff = (simd_sum - ref_sum).abs();
            assert!(
                diff < 1e-5 || diff / ref_sum.abs().max(1.0) < 1e-5,
                "Failed for length {}: simd {} vs ref {}",
                len,
                simd_sum,
                ref_sum
            );
        }
    }

    #[test]
    fn test_mul_slices_simd_equivalence() {
        for len in [0, 1, 3, 4, 7, 16, 29, 64] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.25 - 0.5).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32) * -0.3 + 1.2).collect();
            let mut out = vec![0.0f32; len];

            mul_slices_simd(&mut out, &a, &b);

            for i in 0..len {
                let ref_val = a[i] * b[i];
                let diff = (out[i] - ref_val).abs();
                assert!(
                    diff < 1e-6,
                    "Failed at index {} for length {}: out {} vs ref {}",
                    i,
                    len,
                    out[i],
                    ref_val
                );
            }
        }
    }

    #[test]
    fn test_scale_slice_simd_equivalence() {
        for len in [0, 1, 3, 4, 7, 16, 33, 64] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1 - 2.0).collect();
            let scalar = 1.75f32;
            let mut out = vec![0.0f32; len];

            scale_slice_simd(&mut out, &a, scalar);

            for i in 0..len {
                let ref_val = a[i] * scalar;
                let diff = (out[i] - ref_val).abs();
                assert!(
                    diff < 1e-6,
                    "Failed at index {} for length {}: out {} vs ref {}",
                    i,
                    len,
                    out[i],
                    ref_val
                );
            }
        }
    }

    #[test]
    fn test_dot_product_simd_equivalence() {
        for len in [0, 1, 3, 4, 7, 16, 31, 64] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.15 - 1.0).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32) * -0.2 + 0.8).collect();
            let ref_dot = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f32>();
            let simd_dot = dot_product_simd(&a, &b);
            let diff = (simd_dot - ref_dot).abs();
            assert!(
                diff < 1e-5 || diff / ref_dot.abs().max(1.0) < 1e-5,
                "Failed for length {}: simd {} vs ref {}",
                len,
                simd_dot,
                ref_dot
            );
        }
    }

    #[test]
    fn test_mul_slices_assign_simd_equivalence() {
        for len in [0, 1, 3, 4, 7, 16, 29, 64] {
            let a_orig: Vec<f32> = (0..len).map(|i| (i as f32) * 0.25 - 0.5).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32) * -0.3 + 1.2).collect();
            let mut a = a_orig.clone();

            mul_slices_assign_simd(&mut a, &b);

            for i in 0..len {
                let ref_val = a_orig[i] * b[i];
                let diff = (a[i] - ref_val).abs();
                assert!(
                    diff < 1e-6,
                    "Failed at index {} for length {}: out {} vs ref {}",
                    i,
                    len,
                    a[i],
                    ref_val
                );
            }
        }
    }
}
