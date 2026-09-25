//! Bidirectional Transformer Decision Model (System One Decision Model)
//!
//! - Input: Token sequence [B, T]
//! - Bidirectional Transformer Encoder (RoPE + RMSNorm + SwiGLU)
//! - Mean Pooling produces a fixed-length context vector h [B, C]
//! - Three decision heads (ChoiceHead, NoulHead, ScoreHead)
//! - Outputs type-safe decisions in a single forward pass

use crate::layers::{
    BidirectionalSelfAttention, QuaternionLinear, QuaternionSelfAttention, QuaternionSwiGLU,
    RMSNorm, SwiGLU,
};
use crate::loss::LossCalculator;
use oniwa_lm::reproducibility::DeterministicRng;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionConfig {
    pub vocab_size: usize,
    pub seq_len: usize,
    pub dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub num_choices: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_use_quaternion_head")]
    pub use_quaternion_head: bool,
    #[serde(default = "default_quaternion_backbone")]
    pub quaternion_backbone: bool,
    #[serde(default)]
    pub score_unit_interval: bool,
}

fn default_use_quaternion_head() -> bool {
    false
}

fn default_quaternion_backbone() -> bool {
    false
}

fn default_temperature() -> f32 {
    1.0
}

impl Default for DecisionConfig {
    fn default() -> Self {
        Self {
            vocab_size: 4721,
            seq_len: 128,
            dim: 128,
            num_layers: 4,
            num_heads: 4,
            head_dim: 32,
            ffn_dim: 256,
            num_choices: 4,
            temperature: 1.0,
            use_quaternion_head: false,
            quaternion_backbone: false,
            score_unit_interval: false,
        }
    }
}

impl DecisionConfig {
    /// System 1 target: V=4096, d_model=256, T=128, B=4 fits the 100 MiB budget.
    pub fn system1_mlm() -> Self {
        Self {
            vocab_size: 4_096,
            seq_len: 128,
            dim: 256,
            num_layers: 4,
            num_heads: 4,
            head_dim: 64,
            ffn_dim: 1_024,
            num_choices: 4,
            temperature: 1.0,
            use_quaternion_head: true,
            quaternion_backbone: true,
            score_unit_interval: true,
        }
    }
    /// Standard baseline configuration (~1.26M parameters)
    pub fn standard_baseline(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            use_quaternion_head: false,
            quaternion_backbone: false,
            score_unit_interval: false,
            ..Default::default()
        }
    }

    /// Quaternion head configuration (~1.26M parameters with 4x compressed head)
    pub fn quaternion_head(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            use_quaternion_head: true,
            quaternion_backbone: false,
            score_unit_interval: false,
            ..Default::default()
        }
    }

    /// Full Quaternion Transformer configuration (~315K parameters)
    /// Both backbone (Attention + SwiGLU) and decision head operate in quaternion space.
    pub fn full_quaternion_transformer(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            use_quaternion_head: true,
            quaternion_backbone: true,
            score_unit_interval: false,
            ..Default::default()
        }
    }

    /// Iso-parameter configuration (~315K parameters matching Full Q-Transformer scale)
    /// Shrinks hidden dimension from 128 to 64, head_dim from 32 to 16, and ffn_dim from 256 to 128
    pub fn iso_parameter(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            seq_len: 128,
            dim: 64,
            num_layers: 4,
            num_heads: 4,
            head_dim: 16,
            ffn_dim: 128,
            num_choices: 4,
            temperature: 1.0,
            use_quaternion_head: false,
            quaternion_backbone: false,
            score_unit_interval: false,
        }
    }
}

/// Decision output for a single sample
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawDecision {
    pub choice_probs: Vec<f32>,
    pub choice_idx: usize,
    pub choice_confidence: f32,
    pub noul_prob: f32,
    pub noul_value: bool,
    pub noul_confidence: f32,
    pub score_value: f32,
    pub score_confidence: f32,
}

pub struct DecisionModel {
    pub config: DecisionConfig,
    // Parameters
    pub params: Vec<f32>,
    pub grads: Vec<f32>,
    pub m: Vec<f32>, // AdamW 1st moment
    pub v: Vec<f32>, // AdamW 2nd moment

    // Parameter offsets
    pub offset_wte: usize,
    pub offset_layers: Vec<LayerOffsets>,
    pub offset_ln_f: usize,
    pub offset_head_choice: usize,
    pub offset_head_noul: usize,
    pub offset_head_score: usize,
    pub offset_head_quat: usize,
}

#[derive(Clone, Debug)]
pub struct LayerOffsets {
    pub ln1_gamma: usize,
    pub attn_w_qkv: usize,
    pub attn_w_proj: usize,
    pub ln2_gamma: usize,
    pub mlp_w_gate_up: usize,
    pub mlp_w_down: usize,
}

/// Forward pass activation cache per layer (used in backward pass)
pub struct LayerCache {
    pub x1: Vec<f32>,
    pub rstd1: Vec<f32>,
    pub act_q: Vec<f32>,
    pub act_k: Vec<f32>,
    pub act_v: Vec<f32>,
    pub act_att: Vec<f32>,
    pub act_att_out: Vec<f32>,
    pub x2: Vec<f32>,
    pub rstd2: Vec<f32>,
    pub act_g: Vec<f32>,
    pub act_u: Vec<f32>,
    pub act_h: Vec<f32>,
    pub out: Vec<f32>,
}

pub struct ForwardCache {
    pub embedded: Vec<f32>,
    pub layer_caches: Vec<LayerCache>,
    pub norm_f_out: Vec<f32>,
    pub rstd_f: Vec<f32>,
    pub pooled: Vec<f32>,        // [B, C] Mean pooling
    pub choice_logits: Vec<f32>, // [B, num_choices]
    pub noul_logits: Vec<f32>,   // [B]
    pub score_preds: Vec<f32>,   // [B]
    pub mlm_logits: Vec<f32>,    // [B, T, vocab_size], tied to input embeddings
    pub quat_head_out: Vec<f32>, // [B, 24] Optional cache for Quaternion Head backward pass
}

impl DecisionModel {
    pub fn new(config: DecisionConfig, rng: &mut DeterministicRng) -> Self {
        let c = config.dim;
        let v = config.vocab_size;
        let ffn = config.ffn_dim;
        let num_choices = config.num_choices;

        let mut offset = 0;
        let offset_wte = offset;
        offset += v * c;

        let mut offset_layers = Vec::with_capacity(config.num_layers);
        let in_quat = c / 4;
        let ffn_quat = ffn / 4;

        if config.quaternion_backbone {
            assert_eq!(
                c % 4,
                0,
                "Embedding dim must be divisible by 4 for quaternion backbone"
            );
            assert_eq!(
                ffn % 4,
                0,
                "FFN dim must be divisible by 4 for quaternion backbone"
            );
            let d_h = c / config.num_heads;
            assert_eq!(
                d_h % 4,
                0,
                "Head dim must be divisible by 4 for quaternion backbone"
            );
        }

        for _ in 0..config.num_layers {
            let ln1 = offset;
            offset += c;

            let (qkv, proj, ln2, gate_up, down) = if config.quaternion_backbone {
                let qkv = offset;
                offset += 3 * in_quat * in_quat * 4;
                let proj = offset;
                offset += in_quat * in_quat * 4;
                let ln2 = offset;
                offset += c;
                let gate_up = offset;
                offset += 2 * ffn_quat * in_quat * 4;
                let down = offset;
                offset += in_quat * ffn_quat * 4;
                (qkv, proj, ln2, gate_up, down)
            } else {
                let qkv = offset;
                offset += c * (3 * c);
                let proj = offset;
                offset += c * c;
                let ln2 = offset;
                offset += c;
                let gate_up = offset;
                offset += c * (2 * ffn);
                let down = offset;
                offset += ffn * c;
                (qkv, proj, ln2, gate_up, down)
            };

            offset_layers.push(LayerOffsets {
                ln1_gamma: ln1,
                attn_w_qkv: qkv,
                attn_w_proj: proj,
                ln2_gamma: ln2,
                mlp_w_gate_up: gate_up,
                mlp_w_down: down,
            });
        }

        let offset_ln_f = offset;
        offset += c;

        let mut offset_head_choice = 0;
        let mut offset_head_noul = 0;
        let mut offset_head_score = 0;
        let mut offset_head_quat = 0;

        if config.use_quaternion_head {
            assert_eq!(
                c % 4,
                0,
                "Embedding dim must be divisible by 4 for quaternion head"
            );
            assert_eq!(
                num_choices, 4,
                "Quaternion head currently supports exactly 4 choices"
            );
            offset_head_quat = offset;
            let in_quat = c / 4;
            let out_quat = 6;
            offset += out_quat * in_quat * 4;
        } else {
            offset_head_choice = offset;
            offset += c * num_choices;
            offset_head_noul = offset;
            offset += c;
            offset_head_score = offset;
            offset += c;
        }

        let total_params = offset;
        let mut params = vec![0.0f32; total_params];

        // Xavier / Glorot initialization
        let scale_wte = (2.0f32 / (v + c) as f32).sqrt();
        for i in 0..v * c {
            params[offset_wte + i] = (rng.next_f32() * 2.0 - 1.0) * scale_wte;
        }

        for l in &offset_layers {
            for i in 0..c {
                params[l.ln1_gamma + i] = 1.0;
                params[l.ln2_gamma + i] = 1.0;
            }
            if config.quaternion_backbone {
                let qkv_len = 3 * in_quat * in_quat * 4;
                let scale_qkv = (2.0f32 / (in_quat * 4 + 3 * in_quat * 4) as f32).sqrt();
                for i in 0..qkv_len {
                    params[l.attn_w_qkv + i] = (rng.next_f32() * 2.0 - 1.0) * scale_qkv;
                }
                let proj_len = in_quat * in_quat * 4;
                let scale_proj = (2.0f32 / (2 * in_quat * 4) as f32).sqrt();
                for i in 0..proj_len {
                    params[l.attn_w_proj + i] = (rng.next_f32() * 2.0 - 1.0) * scale_proj;
                }
                let gu_len = 2 * ffn_quat * in_quat * 4;
                let scale_gu = (2.0f32 / (in_quat * 4 + 2 * ffn_quat * 4) as f32).sqrt();
                for i in 0..gu_len {
                    params[l.mlp_w_gate_up + i] = (rng.next_f32() * 2.0 - 1.0) * scale_gu;
                }
                let dn_len = in_quat * ffn_quat * 4;
                let scale_dn = (2.0f32 / (ffn_quat * 4 + in_quat * 4) as f32).sqrt();
                for i in 0..dn_len {
                    params[l.mlp_w_down + i] = (rng.next_f32() * 2.0 - 1.0) * scale_dn;
                }
            } else {
                let scale_qkv = (2.0f32 / (c + 3 * c) as f32).sqrt();
                for i in 0..c * (3 * c) {
                    params[l.attn_w_qkv + i] = (rng.next_f32() * 2.0 - 1.0) * scale_qkv;
                }
                let scale_proj = (2.0f32 / (2 * c) as f32).sqrt();
                for i in 0..c * c {
                    params[l.attn_w_proj + i] = (rng.next_f32() * 2.0 - 1.0) * scale_proj;
                }
                let scale_gu = (2.0f32 / (c + 2 * ffn) as f32).sqrt();
                for i in 0..c * (2 * ffn) {
                    params[l.mlp_w_gate_up + i] = (rng.next_f32() * 2.0 - 1.0) * scale_gu;
                }
                let scale_dn = (2.0f32 / (ffn + c) as f32).sqrt();
                for i in 0..ffn * c {
                    params[l.mlp_w_down + i] = (rng.next_f32() * 2.0 - 1.0) * scale_dn;
                }
            }
        }

        for i in 0..c {
            params[offset_ln_f + i] = 1.0;
        }

        if config.use_quaternion_head {
            let in_quat = c / 4;
            let out_quat = 6;
            let quat_w_len = out_quat * in_quat * 4;
            let scale_quat = (2.0f32 / (in_quat * 4 + out_quat * 4) as f32).sqrt();
            for i in 0..quat_w_len {
                params[offset_head_quat + i] = (rng.next_f32() * 2.0 - 1.0) * scale_quat;
            }
        } else {
            let scale_hc = (2.0f32 / (c + num_choices) as f32).sqrt();
            for i in 0..c * num_choices {
                params[offset_head_choice + i] = (rng.next_f32() * 2.0 - 1.0) * scale_hc;
            }
            let scale_single = (2.0f32 / (c + 1) as f32).sqrt();
            for i in 0..c {
                params[offset_head_noul + i] = (rng.next_f32() * 2.0 - 1.0) * scale_single;
                params[offset_head_score + i] = (rng.next_f32() * 2.0 - 1.0) * scale_single;
            }
        }

        let grads = vec![0.0f32; total_params];
        let m = vec![0.0f32; total_params];
        let v_arr = vec![0.0f32; total_params];

        Self {
            config,
            params,
            grads,
            m,
            v: v_arr,
            offset_wte,
            offset_layers,
            offset_ln_f,
            offset_head_choice,
            offset_head_noul,
            offset_head_score,
            offset_head_quat,
        }
    }

    pub fn zero_grad(&mut self) {
        self.grads.fill(0.0);
    }

    /// Forward pass
    pub fn forward(&self, tokens: &[u16], b: usize, t: usize) -> ForwardCache {
        let c = self.config.dim;
        let n = b * t;
        let nh = self.config.num_heads;
        let ffn = self.config.ffn_dim;
        let num_choices = self.config.num_choices;
        let vocab_size = self.config.vocab_size;

        // 1. Embedding
        let mut cur = vec![0.0f32; n * c];
        for i in 0..n {
            let tok = tokens[i] as usize % self.config.vocab_size;
            let wte_slice =
                &self.params[self.offset_wte + tok * c..self.offset_wte + (tok + 1) * c];
            cur[i * c..(i + 1) * c].copy_from_slice(wte_slice);
        }
        let embedded = cur.clone();

        // 2. Transformer layers
        let mut layer_caches = Vec::with_capacity(self.config.num_layers);
        for l in &self.offset_layers {
            let mut x1 = vec![0.0f32; n * c];
            let mut rstd1 = vec![0.0f32; n];
            let gamma1 = &self.params[l.ln1_gamma..l.ln1_gamma + c];
            RMSNorm::forward(&mut x1, &mut rstd1, &cur, gamma1, n, c, 1e-5);

            let mut act_q = vec![0.0f32; n * c];
            let mut act_k = vec![0.0f32; n * c];
            let mut act_v = vec![0.0f32; n * c];
            let mut act_att = vec![0.0f32; b * nh * t * t];
            let mut act_att_out = vec![0.0f32; n * c];
            let mut attn_out = vec![0.0f32; n * c];

            if self.config.quaternion_backbone {
                let in_quat = c / 4;
                let w_qkv = &self.params[l.attn_w_qkv..l.attn_w_qkv + 3 * in_quat * in_quat * 4];
                let w_proj = &self.params[l.attn_w_proj..l.attn_w_proj + in_quat * in_quat * 4];
                QuaternionSelfAttention::forward(
                    &mut attn_out,
                    &mut act_q,
                    &mut act_k,
                    &mut act_v,
                    &mut act_att,
                    &mut act_att_out,
                    &x1,
                    w_qkv,
                    w_proj,
                    b,
                    t,
                    c,
                    nh,
                );
            } else {
                let w_qkv = &self.params[l.attn_w_qkv..l.attn_w_qkv + c * (3 * c)];
                let w_proj = &self.params[l.attn_w_proj..l.attn_w_proj + c * c];
                BidirectionalSelfAttention::forward(
                    &mut attn_out,
                    &mut act_q,
                    &mut act_k,
                    &mut act_v,
                    &mut act_att,
                    &mut act_att_out,
                    &x1,
                    w_qkv,
                    w_proj,
                    b,
                    t,
                    c,
                    nh,
                );
            }

            // Residual connection 1
            for i in 0..n * c {
                cur[i] += attn_out[i];
            }

            // LN2 + MLP
            let mut x2 = vec![0.0f32; n * c];
            let mut rstd2 = vec![0.0f32; n];
            let gamma2 = &self.params[l.ln2_gamma..l.ln2_gamma + c];
            RMSNorm::forward(&mut x2, &mut rstd2, &cur, gamma2, n, c, 1e-5);

            let mut act_g = vec![0.0f32; n * ffn];
            let mut act_u = vec![0.0f32; n * ffn];
            let mut act_h = vec![0.0f32; n * ffn];
            let mut mlp_out = vec![0.0f32; n * c];

            if self.config.quaternion_backbone {
                let in_quat = c / 4;
                let ffn_quat = ffn / 4;
                let w_gu =
                    &self.params[l.mlp_w_gate_up..l.mlp_w_gate_up + 2 * ffn_quat * in_quat * 4];
                let w_dn = &self.params[l.mlp_w_down..l.mlp_w_down + in_quat * ffn_quat * 4];
                QuaternionSwiGLU::forward(
                    &mut mlp_out,
                    &mut act_g,
                    &mut act_u,
                    &mut act_h,
                    &x2,
                    w_gu,
                    w_dn,
                    n,
                    c,
                    ffn,
                );
            } else {
                let w_gu = &self.params[l.mlp_w_gate_up..l.mlp_w_gate_up + c * (2 * ffn)];
                let w_dn = &self.params[l.mlp_w_down..l.mlp_w_down + ffn * c];
                SwiGLU::forward(
                    &mut mlp_out,
                    &mut act_g,
                    &mut act_u,
                    &mut act_h,
                    &x2,
                    w_gu,
                    w_dn,
                    n,
                    c,
                    ffn,
                );
            }

            // Residual connection 2
            for i in 0..n * c {
                cur[i] += mlp_out[i];
            }

            layer_caches.push(LayerCache {
                x1,
                rstd1,
                act_q,
                act_k,
                act_v,
                act_att,
                act_att_out,
                x2,
                rstd2,
                act_g,
                act_u,
                act_h,
                out: cur.clone(),
            });
        }

        // 3. Final LN
        let mut norm_f_out = vec![0.0f32; n * c];
        let mut rstd_f = vec![0.0f32; n];
        let gamma_f = &self.params[self.offset_ln_f..self.offset_ln_f + c];
        RMSNorm::forward(&mut norm_f_out, &mut rstd_f, &cur, gamma_f, n, c, 1e-5);

        // Tied MLM projection: each token state is scored against the input embedding table.
        let mut mlm_logits = vec![0.0f32; n * vocab_size];
        for position in 0..n {
            let h = &norm_f_out[position * c..(position + 1) * c];
            for token in 0..vocab_size {
                let w =
                    &self.params[self.offset_wte + token * c..self.offset_wte + (token + 1) * c];
                mlm_logits[position * vocab_size + token] =
                    h.iter().zip(w).map(|(a, b)| a * b).sum();
            }
        }

        // 4. Mean Pooling: pooled [B, C]
        let mut pooled = vec![0.0f32; b * c];
        let inv_t = 1.0f32 / (t as f32);
        for bi in 0..b {
            for ti in 0..t {
                let tok_offset = (bi * t + ti) * c;
                let pool_offset = bi * c;
                for j in 0..c {
                    pooled[pool_offset + j] += norm_f_out[tok_offset + j] * inv_t;
                }
            }
        }

        // 5. Decision heads
        let mut choice_logits = vec![0.0f32; b * num_choices];
        let mut noul_logits = vec![0.0f32; b];
        let mut score_preds = vec![0.0f32; b];
        let mut quat_head_out = Vec::new();

        if self.config.use_quaternion_head {
            let in_quat = c / 4;
            let out_quat = 6;
            let w_quat =
                &self.params[self.offset_head_quat..self.offset_head_quat + out_quat * in_quat * 4];
            quat_head_out = vec![0.0f32; b * out_quat * 4];

            QuaternionLinear::forward(
                &mut quat_head_out,
                &pooled,
                w_quat,
                None,
                b,
                in_quat,
                out_quat,
            );

            for bi in 0..b {
                let q_out_bi = &quat_head_out[bi * out_quat * 4..(bi + 1) * out_quat * 4];

                // Choice logits from real parts w of output quaternions 0..4
                for k in 0..num_choices {
                    choice_logits[bi * num_choices + k] = q_out_bi[k * 4];
                }

                // Noul logit from real part w of output quaternion 4
                noul_logits[bi] = q_out_bi[4 * 4];

                // Score prediction from real part w of output quaternion 5
                let s_dot = q_out_bi[5 * 4];
                let tanh = (s_dot * 0.5).tanh();
                score_preds[bi] = if self.config.score_unit_interval {
                    0.5 + 0.5 * tanh
                } else {
                    3.0 + 2.0 * tanh
                };
            }
        } else {
            let w_hc =
                &self.params[self.offset_head_choice..self.offset_head_choice + c * num_choices];
            let w_hn = &self.params[self.offset_head_noul..self.offset_head_noul + c];
            let w_hs = &self.params[self.offset_head_score..self.offset_head_score + c];

            for bi in 0..b {
                let h = &pooled[bi * c..(bi + 1) * c];

                // Choice Head
                for k in 0..num_choices {
                    let mut dot = 0.0f32;
                    for j in 0..c {
                        dot += h[j] * w_hc[j * num_choices + k];
                    }
                    choice_logits[bi * num_choices + k] = dot;
                }

                // Noul Head
                let mut dot_n = 0.0f32;
                for j in 0..c {
                    dot_n += h[j] * w_hn[j];
                }
                noul_logits[bi] = dot_n;

                // Score Head: mapped to [1.0, 5.0] via 3.0 + 2.0 * tanh(dot / 2.0)
                let mut dot_s = 0.0f32;
                for j in 0..c {
                    dot_s += h[j] * w_hs[j];
                }
                let tanh = (dot_s * 0.5).tanh();
                score_preds[bi] = if self.config.score_unit_interval {
                    0.5 + 0.5 * tanh
                } else {
                    3.0 + 2.0 * tanh
                };
            }
        }

        ForwardCache {
            embedded,
            layer_caches,
            norm_f_out,
            rstd_f,
            pooled,
            choice_logits,
            noul_logits,
            score_preds,
            mlm_logits,
            quat_head_out,
        }
    }

    /// Backward pass
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    pub fn backward(
        &mut self,
        tokens: &[u16],
        cache: &ForwardCache,
        dchoice_logits: &[f32],
        dnoul_logits: &[f32],
        dscore_preds: &[f32],
        b: usize,
        t: usize,
    ) {
        self.backward_with_mlm(
            tokens,
            cache,
            dchoice_logits,
            dnoul_logits,
            dscore_preds,
            &[],
            b,
            t,
        );
    }

    /// Backward pass with optional tied-embedding MLM gradients.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    pub fn backward_with_mlm(
        &mut self,
        tokens: &[u16],
        cache: &ForwardCache,
        dchoice_logits: &[f32],
        dnoul_logits: &[f32],
        dscore_preds: &[f32],
        dmlm_logits: &[f32],
        b: usize,
        t: usize,
    ) {
        let c = self.config.dim;
        let n = b * t;
        let nh = self.config.num_heads;
        let ffn = self.config.ffn_dim;
        let num_choices = self.config.num_choices;

        // 1. Compute gradients from decision heads -> d_pooled [B, C]
        let mut d_pooled = vec![0.0f32; b * c];

        if self.config.use_quaternion_head {
            let in_quat = c / 4;
            let out_quat = 6;
            let mut d_quat_out = vec![0.0f32; b * out_quat * 4];

            for bi in 0..b {
                let d_q_bi = &mut d_quat_out[bi * out_quat * 4..(bi + 1) * out_quat * 4];

                // Choice head gradients -> d_loss / d(w_k) for k in 0..4
                let d_cl = &dchoice_logits[bi * num_choices..(bi + 1) * num_choices];
                for k in 0..num_choices {
                    d_q_bi[k * 4] = d_cl[k]; // w-component
                }

                // Noul head gradient -> d_loss / d(w_4)
                d_q_bi[4 * 4] = dnoul_logits[bi];

                // Score head gradient -> d_loss / d(w_5)
                // y = 3.0 + 2.0 * tanh(s_dot * 0.5)
                let tanh_u = if self.config.score_unit_interval {
                    2.0 * cache.score_preds[bi] - 1.0
                } else {
                    (cache.score_preds[bi] - 3.0) / 2.0
                };
                let output_scale = if self.config.score_unit_interval {
                    0.25
                } else {
                    1.0
                };
                let d_dots = dscore_preds[bi] * output_scale * (1.0 - tanh_u * tanh_u);
                d_q_bi[5 * 4] = d_dots;
            }

            let w_quat =
                &self.params[self.offset_head_quat..self.offset_head_quat + out_quat * in_quat * 4];
            let d_w_quat = &mut self.grads
                [self.offset_head_quat..self.offset_head_quat + out_quat * in_quat * 4];

            QuaternionLinear::backward(
                &mut d_pooled,
                d_w_quat,
                None,
                &d_quat_out,
                &cache.pooled,
                w_quat,
                b,
                in_quat,
                out_quat,
            );
        } else {
            let w_hc =
                &self.params[self.offset_head_choice..self.offset_head_choice + c * num_choices];
            let w_hn = &self.params[self.offset_head_noul..self.offset_head_noul + c];
            let w_hs = &self.params[self.offset_head_score..self.offset_head_score + c];

            for bi in 0..b {
                let h = &cache.pooled[bi * c..(bi + 1) * c];
                let dh = &mut d_pooled[bi * c..(bi + 1) * c];

                // Choice head backward pass
                let d_cl = &dchoice_logits[bi * num_choices..(bi + 1) * num_choices];
                for j in 0..c {
                    let mut sum_dh = 0.0f32;
                    for k in 0..num_choices {
                        sum_dh += d_cl[k] * w_hc[j * num_choices + k];
                        self.grads[self.offset_head_choice + j * num_choices + k] += h[j] * d_cl[k];
                    }
                    dh[j] += sum_dh;
                }

                // Noul head backward pass
                let d_nl = dnoul_logits[bi];
                for j in 0..c {
                    dh[j] += d_nl * w_hn[j];
                    self.grads[self.offset_head_noul + j] += h[j] * d_nl;
                }

                // Score head backward pass: y = 3.0 + 2.0 * tanh(u), u = 0.5 * (h . w_hs)
                let tanh_u = if self.config.score_unit_interval {
                    2.0 * cache.score_preds[bi] - 1.0
                } else {
                    (cache.score_preds[bi] - 3.0) / 2.0
                };
                let output_scale = if self.config.score_unit_interval {
                    0.25
                } else {
                    1.0
                };
                let d_dots = dscore_preds[bi] * output_scale * (1.0 - tanh_u * tanh_u);
                for j in 0..c {
                    dh[j] += d_dots * w_hs[j];
                    self.grads[self.offset_head_score + j] += h[j] * d_dots;
                }
            }
        }

        // 2. Mean Pooling backward pass -> d_norm_f_out [N, C]
        let mut d_norm_f_out = vec![0.0f32; n * c];
        let inv_t = 1.0f32 / (t as f32);
        for bi in 0..b {
            let dh = &d_pooled[bi * c..(bi + 1) * c];
            for ti in 0..t {
                let tok_offset = (bi * t + ti) * c;
                for j in 0..c {
                    d_norm_f_out[tok_offset + j] = dh[j] * inv_t;
                }
            }
        }

        // MLM gradients enter before mean pooling and also update the tied embedding rows.
        if !dmlm_logits.is_empty() {
            assert_eq!(dmlm_logits.len(), n * self.config.vocab_size);
            for position in 0..n {
                let h = &cache.norm_f_out[position * c..(position + 1) * c];
                for token in 0..self.config.vocab_size {
                    let gradient = dmlm_logits[position * self.config.vocab_size + token];
                    if gradient == 0.0 {
                        continue;
                    }
                    let weight_offset = self.offset_wte + token * c;
                    for j in 0..c {
                        d_norm_f_out[position * c + j] += gradient * self.params[weight_offset + j];
                        self.grads[weight_offset + j] += gradient * h[j];
                    }
                }
            }
        }

        // 3. Final LN backward pass -> dcur [N, C]
        let mut dcur = vec![0.0f32; n * c];
        let last_layer_out = if self.config.num_layers > 0 {
            &cache.layer_caches.last().unwrap().out
        } else {
            &cache.embedded
        };
        let gamma_f = &self.params[self.offset_ln_f..self.offset_ln_f + c];
        let dgamma_f = &mut self.grads[self.offset_ln_f..self.offset_ln_f + c];
        RMSNorm::backward(
            &mut dcur,
            dgamma_f,
            &d_norm_f_out,
            last_layer_out,
            gamma_f,
            &cache.rstd_f,
            n,
            c,
        );

        // 4. Layer backward passes in reverse order
        for l_idx in (0..self.config.num_layers).rev() {
            let l = &self.offset_layers[l_idx];
            let prev_layer_out = if l_idx > 0 {
                &cache.layer_caches[l_idx - 1].out
            } else {
                &cache.embedded
            };
            let lc = &cache.layer_caches[l_idx];

            // Residual connection 2 branch: d_mlp_out = dcur, dcur_residual = dcur
            // MLP Backward
            let mut dx2 = vec![0.0f32; n * c];
            let gamma2 = &self.params[l.ln2_gamma..l.ln2_gamma + c];

            if self.config.quaternion_backbone {
                let in_quat = c / 4;
                let ffn_quat = ffn / 4;
                let w_gu =
                    &self.params[l.mlp_w_gate_up..l.mlp_w_gate_up + 2 * ffn_quat * in_quat * 4];
                let w_dn = &self.params[l.mlp_w_down..l.mlp_w_down + in_quat * ffn_quat * 4];
                let mut dw_gu = vec![0.0f32; 2 * ffn_quat * in_quat * 4];
                let mut dw_dn = vec![0.0f32; in_quat * ffn_quat * 4];

                QuaternionSwiGLU::backward(
                    &mut dx2, &mut dw_gu, &mut dw_dn, &dcur, &lc.x2, &lc.act_g, &lc.act_u,
                    &lc.act_h, w_gu, w_dn, n, c, ffn,
                );

                for i in 0..2 * ffn_quat * in_quat * 4 {
                    self.grads[l.mlp_w_gate_up + i] += dw_gu[i];
                }
                for i in 0..in_quat * ffn_quat * 4 {
                    self.grads[l.mlp_w_down + i] += dw_dn[i];
                }
            } else {
                let w_gu = &self.params[l.mlp_w_gate_up..l.mlp_w_gate_up + c * (2 * ffn)];
                let w_dn = &self.params[l.mlp_w_down..l.mlp_w_down + ffn * c];
                let mut dw_gu = vec![0.0f32; c * (2 * ffn)];
                let mut dw_dn = vec![0.0f32; ffn * c];

                SwiGLU::backward(
                    &mut dx2, &mut dw_gu, &mut dw_dn, &dcur, &lc.x2, &lc.act_g, &lc.act_u,
                    &lc.act_h, w_gu, w_dn, n, c, ffn,
                );

                for i in 0..c * (2 * ffn) {
                    self.grads[l.mlp_w_gate_up + i] += dw_gu[i];
                }
                for i in 0..ffn * c {
                    self.grads[l.mlp_w_down + i] += dw_dn[i];
                }
            }

            // LN2 Backward
            let mut d_mid = vec![0.0f32; n * c];
            let dgamma2 = &mut self.grads[l.ln2_gamma..l.ln2_gamma + c];
            // LN2 Backward: approximate pre-normalization tensor via lc.out
            RMSNorm::backward(
                &mut d_mid, dgamma2, &dx2, &lc.out, // approximately used in RMSNorm backward
                gamma2, &lc.rstd2, n, c,
            );

            for i in 0..n * c {
                dcur[i] += d_mid[i];
            }

            // Attention Backward
            let mut dx1 = vec![0.0f32; n * c];

            if self.config.quaternion_backbone {
                let in_quat = c / 4;
                let w_qkv = &self.params[l.attn_w_qkv..l.attn_w_qkv + 3 * in_quat * in_quat * 4];
                let w_proj = &self.params[l.attn_w_proj..l.attn_w_proj + in_quat * in_quat * 4];
                let mut dw_qkv = vec![0.0f32; 3 * in_quat * in_quat * 4];
                let mut dw_proj = vec![0.0f32; in_quat * in_quat * 4];

                QuaternionSelfAttention::backward(
                    &mut dx1,
                    &mut dw_qkv,
                    &mut dw_proj,
                    &dcur,
                    &lc.x1,
                    &lc.act_q,
                    &lc.act_k,
                    &lc.act_v,
                    &lc.act_att,
                    &lc.act_att_out,
                    w_qkv,
                    w_proj,
                    b,
                    t,
                    c,
                    nh,
                );

                for i in 0..3 * in_quat * in_quat * 4 {
                    self.grads[l.attn_w_qkv + i] += dw_qkv[i];
                }
                for i in 0..in_quat * in_quat * 4 {
                    self.grads[l.attn_w_proj + i] += dw_proj[i];
                }
            } else {
                let w_qkv = &self.params[l.attn_w_qkv..l.attn_w_qkv + c * (3 * c)];
                let w_proj = &self.params[l.attn_w_proj..l.attn_w_proj + c * c];
                let mut dw_qkv = vec![0.0f32; c * (3 * c)];
                let mut dw_proj = vec![0.0f32; c * c];

                BidirectionalSelfAttention::backward(
                    &mut dx1,
                    &mut dw_qkv,
                    &mut dw_proj,
                    &dcur,
                    &lc.x1,
                    &lc.act_q,
                    &lc.act_k,
                    &lc.act_v,
                    &lc.act_att,
                    &lc.act_att_out,
                    w_qkv,
                    w_proj,
                    b,
                    t,
                    c,
                    nh,
                );

                for i in 0..c * (3 * c) {
                    self.grads[l.attn_w_qkv + i] += dw_qkv[i];
                }
                for i in 0..c * c {
                    self.grads[l.attn_w_proj + i] += dw_proj[i];
                }
            }

            // LN1 Backward
            let mut d_prev = vec![0.0f32; n * c];
            let gamma1 = &self.params[l.ln1_gamma..l.ln1_gamma + c];
            let dgamma1 = &mut self.grads[l.ln1_gamma..l.ln1_gamma + c];
            RMSNorm::backward(
                &mut d_prev,
                dgamma1,
                &dx1,
                prev_layer_out,
                gamma1,
                &lc.rstd1,
                n,
                c,
            );

            for i in 0..n * c {
                dcur[i] += d_prev[i];
            }
        }

        // 5. Embedding gradient accumulation
        for i in 0..n {
            let tok = tokens[i] as usize % self.config.vocab_size;
            let dcur_row = &dcur[i * c..(i + 1) * c];
            for j in 0..c {
                self.grads[self.offset_wte + tok * c + j] += dcur_row[j];
            }
        }
    }

    /// AdamW optimization step
    pub fn adamw_step(
        &mut self,
        lr: f32,
        weight_decay: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        step: usize,
    ) {
        let step_f = step as f32;
        let bias_corr1 = 1.0f32 - beta1.powf(step_f);
        let bias_corr2 = 1.0f32 - beta2.powf(step_f);
        let step_size = lr * (bias_corr2.sqrt() / bias_corr1);

        for i in 0..self.params.len() {
            let g = self.grads[i];
            // Weight decay
            self.params[i] -= lr * weight_decay * self.params[i];

            // Adam moment updates
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;

            let m_hat = self.m[i];
            let v_hat = self.v[i];
            self.params[i] -= step_size * m_hat / (v_hat.sqrt() + eps);
        }
    }

    /// Run inference on a single sequence (execute decision)
    pub fn decide(&self, tokens: &[u16]) -> RawDecision {
        let b = 1;
        let t = tokens.len().min(self.config.seq_len);
        let mut padded = vec![0u16; self.config.seq_len];
        padded[..t].copy_from_slice(&tokens[..t]);

        let cache = self.forward(&padded, b, self.config.seq_len);

        // Compute Choice (Softmax with temperature)
        let num_choices = self.config.num_choices;
        let choice_probs =
            LossCalculator::softmax(&cache.choice_logits[..num_choices], self.config.temperature);
        let mut best_choice = 0;
        let mut max_p = 0.0f32;
        for (k, &p) in choice_probs.iter().enumerate() {
            if p > max_p {
                max_p = p;
                best_choice = k;
            }
        }
        let choice_conf = max_p;

        // Compute Noul
        let noul_p = LossCalculator::sigmoid(cache.noul_logits[0]);
        let noul_val = noul_p >= 0.5;
        let noul_conf = (noul_p - 0.5).abs() * 2.0;

        // Compute Score
        let (score_val, score_conf) = if self.config.score_unit_interval {
            let val = cache.score_preds[0].clamp(0.0, 1.0);
            let conf = 1.0 - (val - 0.5).abs() * 0.4;
            (val, conf)
        } else {
            let val = cache.score_preds[0].clamp(1.0, 5.0);
            let conf = 1.0 - ((val - 3.0).abs() / 2.0) * 0.2; // Stability confidence
            (val, conf)
        };

        RawDecision {
            choice_probs,
            choice_idx: best_choice,
            choice_confidence: choice_conf,
            noul_prob: noul_p,
            noul_value: noul_val,
            noul_confidence: noul_conf,
            score_value: score_val,
            score_confidence: score_conf,
        }
    }

    /// Save model checkpoint
    pub fn save_checkpoint(
        &self,
        dir: &Path,
        step: usize,
        loss: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(dir)?;
        let meta = serde_json::json!({
            "step": step,
            "loss": loss,
            "config": self.config,
            "params_count": self.params.len(),
        });
        std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;

        // Save weights binary
        let mut bytes = Vec::with_capacity(self.params.len() * 4);
        for &p in &self.params {
            bytes.extend_from_slice(&p.to_le_bytes());
        }
        std::fs::write(dir.join("weights.bin"), bytes)?;
        Ok(())
    }

    /// Load model checkpoint
    #[allow(clippy::chunks_exact_to_as_chunks)]
    pub fn load_checkpoint(
        &mut self,
        dir: &Path,
    ) -> Result<(usize, f32), Box<dyn std::error::Error>> {
        let meta_str = std::fs::read_to_string(dir.join("meta.json"))?;
        let meta: serde_json::Value = serde_json::from_str(&meta_str)?;
        let step = meta["step"].as_u64().unwrap_or(0) as usize;
        let loss = meta["loss"].as_f64().unwrap_or(0.0) as f32;

        let bytes = std::fs::read(dir.join("weights.bin"))?;
        if bytes.len() != self.params.len() * 4 {
            return Err("Checkpoint weights size mismatch".into());
        }
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            self.params[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }
        Ok((step, loss))
    }
}
