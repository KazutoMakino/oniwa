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
}
