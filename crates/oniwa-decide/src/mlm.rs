//! Deterministic BERT-style masking without a new dependency.

use oniwa_lm::reproducibility::DeterministicRng;

pub const IGNORE_LABEL: u16 = u16::MAX;

pub struct MaskedTokens {
    pub tokens: Vec<u16>,
    pub labels: Vec<u16>,
}

pub fn apply_mlm_mask(
    tokens: &[u16],
    vocab_size: usize,
    mask_token: u16,
    rng: &mut DeterministicRng,
) -> MaskedTokens {
    let count = (tokens.len() * 15).div_ceil(100);
    let mut positions: Vec<usize> = (0..tokens.len()).collect();
    for i in (1..positions.len()).rev() {
        let j = (rng.next_f32() * (i + 1) as f32) as usize;
        positions.swap(i, j.min(i));
    }
    apply_mask_positions(tokens, vocab_size, mask_token, &positions[..count], rng)
}

/// Apply MLM corruption at dataset-provided evaluation positions.
pub fn apply_mlm_mask_at_indices(
    tokens: &[u16],
    vocab_size: usize,
    mask_token: u16,
    indices: &[usize],
    rng: &mut DeterministicRng,
) -> Result<MaskedTokens, &'static str> {
    if indices.iter().any(|&index| index >= tokens.len()) {
        return Err("MLM mask index is outside the input sequence");
    }
    if indices.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("MLM mask indices must be sorted and unique");
    }
    Ok(apply_mask_positions(
        tokens, vocab_size, mask_token, indices, rng,
    ))
}

fn apply_mask_positions(
    tokens: &[u16],
    vocab_size: usize,
    mask_token: u16,
    positions: &[usize],
    rng: &mut DeterministicRng,
) -> MaskedTokens {
    let mut result = MaskedTokens {
        tokens: tokens.to_vec(),
        labels: vec![IGNORE_LABEL; tokens.len()],
    };
    for &index in positions {
        result.labels[index] = tokens[index];
        let draw = rng.next_f32();
        if draw < 0.8 {
            result.tokens[index] = mask_token;
        } else if draw < 0.9 && vocab_size > 1 {
            result.tokens[index] = 1 + (rng.next_f32() * (vocab_size - 1) as f32) as u16;
        }
    }
    result
}
