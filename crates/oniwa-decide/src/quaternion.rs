//! Pure-Rust Quaternion Algebra for System One Decision Models
//!
//! Re-exported from `oniwa_lm::quaternion`.

pub use oniwa_lm::quaternion::Quaternion;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quaternion_hamilton_fundamental_rules() {
        let i = Quaternion::new(0.0, 1.0, 0.0, 0.0);
        let j = Quaternion::new(0.0, 0.0, 1.0, 0.0);
        let k = Quaternion::new(0.0, 0.0, 0.0, 1.0);

        let i2 = i.hamilton_product(&i);
        assert_eq!(i2, Quaternion::new(-1.0, 0.0, 0.0, 0.0));

        let j2 = j.hamilton_product(&j);
        assert_eq!(j2, Quaternion::new(-1.0, 0.0, 0.0, 0.0));

        let k2 = k.hamilton_product(&k);
        assert_eq!(k2, Quaternion::new(-1.0, 0.0, 0.0, 0.0));

        let ij = i.hamilton_product(&j);
        assert_eq!(ij, k);

        let ji = j.hamilton_product(&i);
        assert_eq!(ji, Quaternion::new(0.0, 0.0, 0.0, -1.0)); // -k

        assert_ne!(ij, ji);

        let jk = j.hamilton_product(&k);
        assert_eq!(jk, i);

        let ki = k.hamilton_product(&i);
        assert_eq!(ki, j);

        let ijk = ij.hamilton_product(&k);
        assert_eq!(ijk, Quaternion::new(-1.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_quaternion_conjugate_and_norm() {
        let q = Quaternion::new(1.0, -2.0, 3.0, -4.0);
        let q_conj = q.conjugate();
        assert_eq!(q_conj, Quaternion::new(1.0, 2.0, -3.0, 4.0));

        let prod = q.hamilton_product(&q_conj);
        let expected_norm_sq = 1.0 + 4.0 + 9.0 + 16.0;
        assert!((prod.w - expected_norm_sq).abs() < 1e-6);
        assert!(prod.x.abs() < 1e-6);
        assert!(prod.y.abs() < 1e-6);
        assert!(prod.z.abs() < 1e-6);

        assert!((q.norm() - 30.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_quaternion_inverse() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let q_inv = q.inverse().expect("Inverse exists");

        let identity1 = q.hamilton_product(&q_inv);
        let identity2 = q_inv.hamilton_product(&q);

        assert!((identity1.w - 1.0).abs() < 1e-6);
        assert!(identity1.x.abs() < 1e-6);
        assert!(identity1.y.abs() < 1e-6);
        assert!(identity1.z.abs() < 1e-6);

        assert!((identity2.w - 1.0).abs() < 1e-6);
        assert!(identity2.x.abs() < 1e-6);
        assert!(identity2.y.abs() < 1e-6);
        assert!(identity2.z.abs() < 1e-6);
    }
}
