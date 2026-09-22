//! Calibrated Multi-Task Loss Functions
//!
//! Pure-Rust implementation of confidence calibration and multi-task loss functions
//! inspired by TypeSafe AI's RLCD (Reinforcement Learning from Calibrated Decisions):
//! - Choice: Cross Entropy with Label Smoothing
//! - Noul: Binary Cross Entropy with Brier Score Regularization
//! - Score: Smooth L1 (Huber) Loss
//! - Temperature Scaling for Calibrated Confidence

#[derive(Clone, Debug)]
pub struct LossConfig {
    pub label_smoothing: f32,
    pub brier_weight: f32,
    pub huber_delta: f32,
    pub choice_weight: f32,
    pub noul_weight: f32,
    pub score_weight: f32,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            label_smoothing: 0.05,
            brier_weight: 0.1,
            huber_delta: 0.5,
            choice_weight: 1.0,
            noul_weight: 1.0,
            score_weight: 1.0,
        }
    }
}

pub struct LossCalculator;

impl LossCalculator {
    /// Cross-entropy over selected MLM positions. `u16::MAX` excludes a position.
    pub fn masked_language_model_loss(
        logits: &[f32],
        labels: &[u16],
        vocab_size: usize,
    ) -> (f32, Vec<f32>) {
        assert_eq!(logits.len(), labels.len() * vocab_size);
        let mut loss = 0.0;
        let mut gradients = vec![0.0; logits.len()];
        let selected = labels
            .iter()
            .filter(|&&label| label != u16::MAX)
            .count()
            .max(1) as f32;
        for (position, &label) in labels.iter().enumerate() {
            if label == u16::MAX {
                continue;
            }
            let offset = position * vocab_size;
            let probs = Self::softmax(&logits[offset..offset + vocab_size], 1.0);
            let target = label as usize;
            assert!(target < vocab_size, "MLM label is outside the vocabulary");
            loss -= probs[target].max(1e-12).ln();
            for token in 0..vocab_size {
                gradients[offset + token] =
                    (probs[token] - usize::from(token == target) as f32) / selected;
            }
        }
        (loss / selected, gradients)
    }

    /// Compute Softmax with temperature scaling
    pub fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
        let temp = temperature.max(1e-4);
        let max_val = logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, |a, b| a.max(b / temp));
        let mut exps = Vec::with_capacity(logits.len());
        let mut sum_exp = 0.0f32;
        for &v in logits {
            let e = ((v / temp) - max_val).exp();
            exps.push(e);
            sum_exp += e;
        }
        let inv_sum = 1.0f32 / sum_exp.max(1e-12);
        for e in exps.iter_mut() {
            *e *= inv_sum;
        }
        exps
    }

    /// Compute Sigmoid function
    #[inline(always)]
    pub fn sigmoid(x: f32) -> f32 {
        1.0f32 / (1.0f32 + (-x).exp())
    }

    /// Choice Loss and Gradient (Cross Entropy + Label Smoothing)
    ///
    /// Returns: (loss, dlogits)
    #[allow(clippy::needless_range_loop)]
    pub fn choice_loss(
        logits: &[f32],
        target_idx: usize,
        num_classes: usize,
        label_smoothing: f32,
    ) -> (f32, Vec<f32>) {
        let probs = Self::softmax(logits, 1.0);
        let eps = label_smoothing;
        let uniform = eps / (num_classes as f32);

        let mut loss = 0.0f32;
        let mut dlogits = Vec::with_capacity(num_classes);

        for k in 0..num_classes {
            let target_prob = if k == target_idx {
                (1.0 - eps) + uniform
            } else {
                uniform
            };
            loss -= target_prob * probs[k].max(1e-12).ln();
            dlogits.push(probs[k] - target_prob);
        }

        (loss, dlogits)
    }

    /// Noul Loss and Gradient (Binary Cross Entropy + Brier Score)
    ///
    /// Returns: (loss, dlogit)
    pub fn noul_loss(logit: f32, target: bool, brier_weight: f32) -> (f32, f32) {
        let p = Self::sigmoid(logit);
        let y = if target { 1.0f32 } else { 0.0f32 };

        // BCE: - [y * ln(p) + (1-y) * ln(1-p)]
        let bce = -(y * p.max(1e-12).ln() + (1.0 - y) * (1.0 - p).max(1e-12).ln());
        // Brier: (p - y)^2
        let brier = (p - y) * (p - y);
        let total_loss = bce + brier_weight * brier;

        // d(BCE)/dz = p - y
        // d(Brier)/dz = 2 * (p - y) * p * (1 - p)
        let grad = (p - y) * (1.0 + 2.0 * brier_weight * p * (1.0 - p));

        (total_loss, grad)
    }

    /// Score Loss and Gradient (Smooth L1 / Huber)
    ///
    /// Returns: (loss, dpred)
    pub fn score_loss(pred: f32, target: f32, delta: f32) -> (f32, f32) {
        let diff = pred - target;
        let abs_diff = diff.abs();

        if abs_diff <= delta {
            let loss = 0.5 * diff * diff / delta;
            let grad = diff / delta;
            (loss, grad)
        } else {
            let loss = abs_diff - 0.5 * delta;
            let grad = if diff > 0.0 { 1.0 } else { -1.0 };
            (loss, grad)
        }
    }
}
