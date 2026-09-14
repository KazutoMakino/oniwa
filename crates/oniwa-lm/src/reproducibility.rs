//! 完璧な再現性を担保するための決定論的乱数生成とハッシュチェックサム

use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};

/// 確定論的乱数ジェネレータ
///
/// x86_64、ARM64 (Raspberry Pi)、その他どのアーキテクチャで実行しても
/// 同じシード値からはビット単位で全く同一の乱数列を生成することを保証します。
pub struct DeterministicRng {
    rng: ChaCha8Rng,
    seed: u64,
}

impl DeterministicRng {
    /// 64-bit シードから決定論的 RNG を初期化
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            seed,
        }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// [0, 1) の一様分布 f32 を生成
    pub fn next_f32(&mut self) -> f32 {
        let val = self.rng.next_u32() >> 8; // 24 bits
        (val as f32) / (16777216.0f32)
    }

    /// 平均 0, 標準偏差 std の正規分布乱数を生成 (Box-Muller変換)
    pub fn next_gaussian(&mut self, mean: f32, std: f32) -> f32 {
        let mut u1 = self.next_f32();
        while u1 <= 1e-7f32 {
            u1 = self.next_f32();
        }
        let u2 = self.next_f32();

        let z0 = (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos();
        mean + z0 * std
    }

    /// スライス全体を指定した標準偏差の正規分布で初期化
    pub fn fill_gaussian(&mut self, dest: &mut [f32], mean: f32, std: f32) {
        for x in dest.iter_mut() {
            *x = self.next_gaussian(mean, std);
        }
    }
}

/// 浮動小数点スライス（重みや活性化値）の SHA-256 チェックサムを算出
///
/// プラットフォーム間のエンディアン差異を防ぐため、常にリトルエンディアンバイト列としてハッシュ化します。
pub fn compute_checksum_f32(data: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for &val in data {
        hasher.update(val.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// 生バイト列（データセットファイルなど）の SHA-256 チェックサムを算出
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
        // 同一シードから生成した乱数列と重みチェックサムが完全一致することを検証
        let seed = 42u64;

        let mut rng1 = DeterministicRng::new(seed);
        let mut weights1 = vec![0.0f32; 1000];
        rng1.fill_gaussian(&mut weights1, 0.0, 0.02);
        let checksum1 = compute_checksum_f32(&weights1);

        let mut rng2 = DeterministicRng::new(seed);
        let mut weights2 = vec![0.0f32; 1000];
        rng2.fill_gaussian(&mut weights2, 0.0, 0.02);
        let checksum2 = compute_checksum_f32(&weights2);

        // ビット単位で完全一致
        assert_eq!(weights1, weights2);
        assert_eq!(checksum1, checksum2);

        // 異なるシードでは一致しないことを検証
        let mut rng3 = DeterministicRng::new(seed + 1);
        let mut weights3 = vec![0.0f32; 1000];
        rng3.fill_gaussian(&mut weights3, 0.0, 0.02);
        let checksum3 = compute_checksum_f32(&weights3);

        assert_ne!(checksum1, checksum3);
    }
}
