//! ARM NEON SIMD Acceleration for Quaternion Operations
//!
//! Re-exported from `oniwa_lm::simd`.

pub use oniwa_lm::simd::*;

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
