//! Pure-Rust Quaternion Algebra for System One Decision Models
//!
//! Represents 4D hypercomplex numbers $q = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}$
//! with non-commutative Hamilton product, conjugation, norm, and GHR-calculus operations.

use serde::{Deserialize, Serialize};

/// 4D Quaternion primitive $q = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}$
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Quaternion {
    pub w: f32, // Real / scalar part
    pub x: f32, // Imaginary unit i
    pub y: f32, // Imaginary unit j
    pub z: f32, // Imaginary unit k
}

impl Quaternion {
    /// Create a new quaternion
    pub const fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }

    /// Additive identity (0)
    pub const fn zero() -> Self {
        Self {
            w: 0.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Multiplicative identity (1)
    pub const fn one() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Hamilton product: `self ⊗ other`
    ///
    /// Note: Non-commutative! `a.hamilton_product(b) != b.hamilton_product(a)` in general.
    #[inline(always)]
    pub fn hamilton_product(&self, other: &Self) -> Self {
        Self {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    /// Quaternion conjugate $q^* = w - x\mathbf{i} - y\mathbf{j} - z\mathbf{k}$
    #[inline(always)]
    pub fn conjugate(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Squared Euclidean norm $\|q\|^2 = w^2 + x^2 + y^2 + z^2$
    #[inline(always)]
    pub fn norm_squared(&self) -> f32 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Euclidean norm $\|q\| = \sqrt{w^2 + x^2 + y^2 + z^2}$
    #[inline(always)]
    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }

    /// Normalize to unit quaternion (versor)
    #[inline(always)]
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n > 1e-12 {
            let inv = 1.0 / n;
            Self {
                w: self.w * inv,
                x: self.x * inv,
                y: self.y * inv,
                z: self.z * inv,
            }
        } else {
            Self::zero()
        }
    }

    /// Multiplicative inverse $q^{-1} = q^* / \|q\|^2$
    #[inline(always)]
    pub fn inverse(&self) -> Option<Self> {
        let n_sq = self.norm_squared();
        if n_sq > 1e-12 {
            let inv = 1.0 / n_sq;
            Some(Self {
                w: self.w * inv,
                x: -self.x * inv,
                y: -self.y * inv,
                z: -self.z * inv,
            })
        } else {
            None
        }
    }

    /// Vector dot product in $\mathbb{R}^4$
    #[inline(always)]
    pub fn dot(&self, other: &Self) -> f32 {
        self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Convert from a flat 4-slice `[w, x, y, z]`
    #[inline(always)]
    pub fn from_slice(slice: &[f32]) -> Self {
        Self {
            w: slice[0],
            x: slice[1],
            y: slice[2],
            z: slice[3],
        }
    }

    /// Store to a flat 4-slice `[w, x, y, z]`
    #[inline(always)]
    pub fn to_slice(&self, out: &mut [f32]) {
        out[0] = self.w;
        out[1] = self.x;
        out[2] = self.y;
        out[3] = self.z;
    }
}

impl std::ops::Add for Quaternion {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            w: self.w + rhs.w,
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Sub for Quaternion {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            w: self.w - rhs.w,
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Mul<f32> for Quaternion {
    type Output = Self;
    #[inline(always)]
    fn mul(self, scalar: f32) -> Self {
        Self {
            w: self.w * scalar,
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quaternion_hamilton_fundamental_rules() {
        // Fundamental quaternion identities:
        // i^2 = j^2 = k^2 = ijk = -1
        // ij = k, jk = i, ki = j
        // ji = -k, kj = -i, ik = -j
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

        // Non-commutative check
        assert_ne!(ij, ji);

        let jk = j.hamilton_product(&k);
        assert_eq!(jk, i);

        let ki = k.hamilton_product(&i);
        assert_eq!(ki, j);

        // i*j*k = -1
        let ijk = ij.hamilton_product(&k);
        assert_eq!(ijk, Quaternion::new(-1.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_quaternion_conjugate_and_norm() {
        let q = Quaternion::new(1.0, -2.0, 3.0, -4.0);
        let q_conj = q.conjugate();
        assert_eq!(q_conj, Quaternion::new(1.0, 2.0, -3.0, 4.0));

        // q * q* = |q|^2
        let prod = q.hamilton_product(&q_conj);
        let expected_norm_sq = 1.0 + 4.0 + 9.0 + 16.0; // 30.0
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
