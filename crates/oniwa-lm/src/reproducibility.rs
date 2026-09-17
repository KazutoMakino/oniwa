//! Deterministic random number generation and hash checksums for perfect reproducibility.

use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};

/// Deterministic random number generator.
///
/// Guarantees bit-for-bit identical pseudo-random sequences from the same seed
/// across x86_64, ARM64 (Raspberry Pi), and any other supported architectures.
pub struct DeterministicRng {
    rng: ChaCha8Rng,
    seed: u64,
}

impl DeterministicRng {
    /// Initialize deterministic RNG from a 64-bit seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            seed,
        }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Generate uniform random f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        let val = self.rng.next_u32() >> 8; // 24 bits
        (val as f32) / (16777216.0f32)
    }

    /// Generate standard normal random variable with mean and std (Box-Muller transform).
    pub fn next_gaussian(&mut self, mean: f32, std: f32) -> f32 {
        let mut u1 = self.next_f32();
        while u1 <= 1e-7f32 {
            u1 = self.next_f32();
        }
        let u2 = self.next_f32();

        let z0 = (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos();
        mean + z0 * std
    }

    /// Fill an entire slice with normally distributed values of specified standard deviation.
    pub fn fill_gaussian(&mut self, dest: &mut [f32], mean: f32, std: f32) {
        for x in dest.iter_mut() {
            *x = self.next_gaussian(mean, std);
        }
    }
}

/// Compute SHA-256 checksum of a floating point slice (weights or activations).
///
/// Always hashed as little-endian bytes to ensure cross-platform endian consistency.
pub fn compute_checksum_f32(data: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for &val in data {
        hasher.update(val.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Compute SHA-256 checksum of raw bytes (dataset files, etc.).
pub fn compute_checksum_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_reproducibility() {
        // Verify random sequences and weight checksums match identically from same seed
        let seed = 42u64;

        let mut rng1 = DeterministicRng::new(seed);
        let mut weights1 = vec![0.0f32; 1000];
        rng1.fill_gaussian(&mut weights1, 0.0, 0.02);
        let checksum1 = compute_checksum_f32(&weights1);

        let mut rng2 = DeterministicRng::new(seed);
        let mut weights2 = vec![0.0f32; 1000];
        rng2.fill_gaussian(&mut weights2, 0.0, 0.02);
        let checksum2 = compute_checksum_f32(&weights2);

        // Bit-for-bit identical match
        assert_eq!(weights1, weights2);
        assert_eq!(checksum1, checksum2);

        // Verify different seeds produce different checksums
        let mut rng3 = DeterministicRng::new(seed + 1);
        let mut weights3 = vec![0.0f32; 1000];
        rng3.fill_gaussian(&mut weights3, 0.0, 0.02);
        let checksum3 = compute_checksum_f32(&weights3);

        assert_ne!(checksum1, checksum3);
    }
}
