//! oniwa-lm 超軽量モダンTransformerモデル（ONIWA）
//!
//! Gemma / Llama 系アーキテクチャに準拠:
//! - RMSNorm (各サブレイヤー前)
//! - RoPE (Rotary Position Embedding)
//! - SwiGLU 活性化層 (Gate-Up-Down FFN)
//! - Causal Self-Attention (因果マスク付き自己注意)
//! - 手動バックプロパゲーション (ゼロアロケーション・フラットバッファ)

use crate::layers::attention::CausalSelfAttention;
use crate::layers::mlp::SwiGLU;
use crate::layers::rmsnorm::RMSNorm;
use crate::reproducibility::DeterministicRng;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub seq_len: usize,
    pub dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    #[serde(default = "default_label_smoothing")]
    pub label_smoothing: f32,
    #[serde(default = "default_z_loss_weight")]
    pub z_loss_weight: f32,
}

pub fn default_label_smoothing() -> f32 {
    0.05
}

pub fn default_z_loss_weight() -> f32 {
    1e-4
}

impl Default for ModelConfig {
    fn default() -> Self {
        // 山月記の学習・検証用マイクロモデル
        Self {
            vocab_size: 800, // トークナイザー生成後に更新
            seq_len: 32,
            dim: 64,
            num_layers: 2,
            num_heads: 2,
            head_dim: 32, // dim / num_heads
            ffn_dim: 128,
            label_smoothing: default_label_smoothing(),
            z_loss_weight: default_z_loss_weight(),
        }
    }
}

/// 各レイヤーのパラメータオフセット
#[derive(Debug, Clone, Copy)]
pub struct LayerParamOffsets {
    pub rms_att: usize,
    pub qkv: usize,
    pub att_proj: usize,
    pub rms_ffn: usize,
    pub gate_up: usize,
    pub down: usize,
}

/// モデル全体のパラメータレイアウト（フラット配列内オフセット）
#[derive(Debug, Clone)]
pub struct ModelLayout {
    pub wte: usize,
    pub layers: Vec<LayerParamOffsets>,
    pub rms_final: usize,
    pub lm_head: usize,
}

impl ModelLayout {
    pub fn new(cfg: &ModelConfig) -> Self {
        let v = cfg.vocab_size;
        let c = cfg.dim;
        let l = cfg.num_layers;
        let ffn = cfg.ffn_dim;

        let mut offset = 0;
        let wte = offset;
        offset += v * c;

        let mut layers = Vec::with_capacity(l);
        for _ in 0..l {
            let rms_att = offset; offset += c;
            let qkv = offset; offset += c * 3 * c;
            let att_proj = offset; offset += c * c;
            let rms_ffn = offset; offset += c;
            let gate_up = offset; offset += c * 2 * ffn;
            let down = offset; offset += ffn * c;

            layers.push(LayerParamOffsets {
                rms_att,
                qkv,
                att_proj,
                rms_ffn,
                gate_up,
                down,
            });
        }

        let rms_final = offset; offset += c;
        let lm_head = offset;

        Self {
            wte,
            layers,
            rms_final,
            lm_head,
        }
    }
}

/// 単一のフラット配列で管理されるモデルパラメータ
pub struct ModelWeights {
    pub config: ModelConfig,
    /// 全重みバッファ
    pub params: Vec<f32>,
    /// 全勾配バッファ
    pub grads: Vec<f32>,
    /// AdamW 1次モーメンタム
    pub m: Vec<f32>,
    /// AdamW 2次モーメンタム
    pub v: Vec<f32>,
}

impl ModelWeights {
    pub fn new(config: ModelConfig, rng: &mut DeterministicRng) -> Self {
        let num_params = Self::calculate_num_params(&config);
        let mut params = vec![0.0f32; num_params];
        let grads = vec![0.0f32; num_params];
        let m = vec![0.0f32; num_params];
        let v = vec![0.0f32; num_params];

        // 正規分布 (std = 0.02) で初期化
        rng.fill_gaussian(&mut params, 0.0, 0.02);

        Self {
            config,
            params,
            grads,
            m,
            v,
        }
    }

    pub fn calculate_num_params(cfg: &ModelConfig) -> usize {
        let v = cfg.vocab_size;
        let c = cfg.dim;
        let l = cfg.num_layers;
        let ffn = cfg.ffn_dim;

        let mut total = 0;
        // 1. Embedding: [V, C]
        total += v * c;
        // 2. Per layer:
        for _ in 0..l {
            // rms_att weight: [C]
            total += c;
            // qkv matmul: [C, 3*C]
            total += c * 3 * c;
            // att_proj matmul: [C, C]
            total += c * c;
            // rms_ffn weight: [C]
            total += c;
            // gate_up matmul: [C, 2*FFN]
            total += c * 2 * ffn;
            // down matmul: [FFN, C]
            total += ffn * c;
        }
        // 3. Final RMSNorm: [C]
        total += c;
        // 4. LM Head (Logits Proj): [C, V]
        total += c * v;

        total
    }

    /// 勾配をゼロクリア
    pub fn zero_grad(&mut self) {
        self.grads.fill(0.0);
    }

    /// AdamW による1次元フラットパラメータ更新
    pub fn adamw_step(&mut self, lr: f32, wd: f32, beta1: f32, beta2: f32, eps: f32, step: usize) {
        let beta1_t = beta1.powi(step as i32);
        let beta2_t = beta2.powi(step as i32);

        for i in 0..self.params.len() {
            let p = self.params[i];
            let g = self.grads[i];

            // 重み減衰
            let mut p_updated = p - lr * wd * p;

            // モーメンタム更新
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;

            // バイアス補正
            let m_hat = self.m[i] / (1.0 - beta1_t);
            let v_hat = self.v[i] / (1.0 - beta2_t);

            // パラメータ更新
            p_updated -= lr * m_hat / (v_hat.sqrt() + eps);
            self.params[i] = p_updated;
        }
    }

    /// チェックポイントをディレクトリに保存 (メタ情報JSON + 重み/オプティマイザ生バイナリ)
    pub fn save_checkpoint<P: AsRef<std::path::Path>>(
        &self,
        dir: P,
        step: usize,
        loss: f32,
        seed: u64,
    ) -> std::io::Result<()> {
        let dir_p = dir.as_ref();
        std::fs::create_dir_all(dir_p)?;

        // 1. メタ情報 JSON
        let meta = serde_json::json!({
            "step": step,
            "loss": loss,
            "seed": seed,
            "vocab_size": self.config.vocab_size,
            "dim": self.config.dim,
            "num_layers": self.config.num_layers,
            "seq_len": self.config.seq_len,
            "label_smoothing": self.config.label_smoothing,
            "z_loss_weight": self.config.z_loss_weight,
            "params_checksum": crate::reproducibility::compute_checksum_f32(&self.params),
            "git_commit_hash": crate::logger::get_git_commit_hash(),
            "git_dirty": crate::logger::get_git_dirty(),
        });
        std::fs::write(dir_p.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;

        // 2. 生バイナリ (params, m, v を1つのファイルに連続書き出し)
        let mut f = std::io::BufWriter::new(std::fs::File::create(dir_p.join("weights.bin"))?);
        use std::io::Write;
        for &val in &self.params {
            f.write_all(&val.to_le_bytes())?;
        }
        for &val in &self.m {
            f.write_all(&val.to_le_bytes())?;
        }
        for &val in &self.v {
            f.write_all(&val.to_le_bytes())?;
        }
        f.flush()?;

        Ok(())
    }

    /// チェックポイントから重みとオプティマイザ状態を復元
    pub fn load_checkpoint<P: AsRef<std::path::Path>>(
        &mut self,
        dir: P,
    ) -> std::io::Result<(usize, f32, u64)> {
        let dir_p = dir.as_ref();

        // 1. メタ情報読み込み
        let meta_str = std::fs::read_to_string(dir_p.join("meta.json"))?;
        let meta: serde_json::Value = serde_json::from_str(&meta_str)?;
        let step = meta["step"].as_u64().unwrap_or(0) as usize;
        let loss = meta["loss"].as_f64().unwrap_or(0.0) as f32;
        let seed = meta["seed"].as_u64().unwrap_or(0);
        if let Some(ls) = meta["label_smoothing"].as_f64() {
            self.config.label_smoothing = ls as f32;
        }
        if let Some(zw) = meta["z_loss_weight"].as_f64() {
            self.config.z_loss_weight = zw as f32;
        }

        // 2. 生バイナリ読み込み
        let mut f = std::io::BufReader::new(std::fs::File::open(dir_p.join("weights.bin"))?);
        use std::io::Read;

        let num_params = self.params.len();
        let mut buf4 = [0u8; 4];

        for i in 0..num_params {
            f.read_exact(&mut buf4)?;
            self.params[i] = f32::from_le_bytes(buf4);
        }
        for i in 0..num_params {
            f.read_exact(&mut buf4)?;
            self.m[i] = f32::from_le_bytes(buf4);
        }
        for i in 0..num_params {
            f.read_exact(&mut buf4)?;
            self.v[i] = f32::from_le_bytes(buf4);
        }

        Ok((step, loss, seed))
    }

    /// フルTransformerの順伝播・逆伝播・勾配蓄積・損失計算
    pub fn forward_backward(
        &mut self,
        x: &[u16],
        y: &[u16],
        b: usize,
        t: usize,
    ) -> (f32, f32) {
        let n = b * t;
        let c = self.config.dim;
        let v = self.config.vocab_size;
        let l = self.config.num_layers;
        let nh = self.config.num_heads;
        let ffn = self.config.ffn_dim;

        let layout = ModelLayout::new(&self.config);

        // --- 1. 順伝播 (Forward) ---

        // (1) Token Embedding lookup: x0 [N, C]
        let mut x_curr = vec![0.0f32; n * c];
        let wte = &self.params[layout.wte..layout.wte + v * c];
        for i in 0..n {
            let tok = x[i] as usize;
            let offset = tok * c;
            x_curr[i * c..(i + 1) * c].copy_from_slice(&wte[offset..offset + c]);
        }

        // 各レイヤーの中間テンソルを保持するキャッシュ
        #[allow(dead_code)]
        struct LayerActivations {
            x_in: Vec<f32>,       // [N, C]
            norm1: Vec<f32>,      // [N, C]
            rstd1: Vec<f32>,      // [N]
            q: Vec<f32>,          // [N, C]
            k: Vec<f32>,          // [N, C]
            val: Vec<f32>,        // [N, C]
            att: Vec<f32>,        // [NH, N, T]
            att_out: Vec<f32>,    // [N, C]
            post_att: Vec<f32>,   // [N, C]
            x_mid: Vec<f32>,      // [N, C]
            norm2: Vec<f32>,      // [N, C]
            rstd2: Vec<f32>,      // [N]
            act_g: Vec<f32>,      // [N, FFN]
            act_u: Vec<f32>,      // [N, FFN]
            act_h: Vec<f32>,      // [N, FFN]
            post_mlp: Vec<f32>,   // [N, C]
        }

        let mut layer_acts = Vec::with_capacity(l);

        for li in 0..l {
            let lo = &layout.layers[li];
            let x_in = x_curr.clone();

            // 1. RMSNorm 1
            let mut norm1 = vec![0.0f32; n * c];
            let mut rstd1 = vec![0.0f32; n];
            let rms_att_w = &self.params[lo.rms_att..lo.rms_att + c];
            RMSNorm::forward(&mut norm1, &mut rstd1, &x_in, rms_att_w, 1e-5, c);

            // 2. Causal Self-Attention
            let mut q = vec![0.0f32; n * c];
            let mut k = vec![0.0f32; n * c];
            let mut val = vec![0.0f32; n * c];
            let mut att = vec![0.0f32; b * nh * t * t];
            let mut att_out = vec![0.0f32; n * c];
            let mut post_att = vec![0.0f32; n * c];

            let qkv_w = &self.params[lo.qkv..lo.qkv + c * 3 * c];
            let att_proj_w = &self.params[lo.att_proj..lo.att_proj + c * c];

            CausalSelfAttention::forward(
                &mut post_att,
                &mut q,
                &mut k,
                &mut val,
                &mut att,
                &mut att_out,
                &norm1,
                qkv_w,
                att_proj_w,
                b,
                t,
                c,
                nh,
            );

            // 3. 残差加算: x_mid = x_in + post_att
            let mut x_mid = vec![0.0f32; n * c];
            for i in 0..n * c {
                x_mid[i] = x_in[i] + post_att[i];
            }

            // 4. RMSNorm 2
            let mut norm2 = vec![0.0f32; n * c];
            let mut rstd2 = vec![0.0f32; n];
            let rms_ffn_w = &self.params[lo.rms_ffn..lo.rms_ffn + c];
            RMSNorm::forward(&mut norm2, &mut rstd2, &x_mid, rms_ffn_w, 1e-5, c);

            // 5. SwiGLU MLP
            let mut act_g = vec![0.0f32; n * ffn];
            let mut act_u = vec![0.0f32; n * ffn];
            let mut act_h = vec![0.0f32; n * ffn];
            let mut post_mlp = vec![0.0f32; n * c];

            let gate_up_w = &self.params[lo.gate_up..lo.gate_up + c * 2 * ffn];
            let down_w = &self.params[lo.down..lo.down + ffn * c];

            SwiGLU::forward(
                &mut post_mlp,
                &mut act_g,
                &mut act_u,
                &mut act_h,
                &norm2,
                gate_up_w,
                down_w,
                n,
                c,
                ffn,
            );

            // 6. 残差加算: x_out = x_mid + post_mlp
            let mut x_out = vec![0.0f32; n * c];
            for i in 0..n * c {
                x_out[i] = x_mid[i] + post_mlp[i];
            }

            layer_acts.push(LayerActivations {
                x_in,
                norm1,
                rstd1,
                q,
                k,
                val,
                att,
                att_out,
                post_att,
                x_mid,
                norm2,
                rstd2,
                act_g,
                act_u,
                act_h,
                post_mlp,
            });

            x_curr = x_out;
        }

        // Final RMSNorm
        let x_final_in = x_curr.clone();
        let mut final_norm = vec![0.0f32; n * c];
        let mut rstd_final = vec![0.0f32; n];
        let rms_final_w = &self.params[layout.rms_final..layout.rms_final + c];
        RMSNorm::forward(&mut final_norm, &mut rstd_final, &x_final_in, rms_final_w, 1e-5, c);

        // LM Head: final_norm [N, C] * lm_head [C, V] -> logits [N, V]
        let lm_head_w = &self.params[layout.lm_head..layout.lm_head + c * v];
        let mut total_loss = 0.0f32;
        let mut dlogits = vec![0.0f32; n * v];

        let eps = self.config.label_smoothing;
        let cz = self.config.z_loss_weight;
        let v_f32 = v as f32;
        let smooth_uniform = if eps > 0.0 { eps / v_f32 } else { 0.0 };
        let target_weight = 1.0 - eps + smooth_uniform;

        for i in 0..n {
            let target = y[i] as usize;
            let norm_row = &final_norm[i * c..(i + 1) * c];

            let mut max_logit = f32::NEG_INFINITY;
            let mut logits_row = vec![0.0f32; v];
            let mut sum_raw_logits = 0.0f32;
            for j in 0..v {
                let mut dot = 0.0f32;
                for k in 0..c {
                    dot += norm_row[k] * lm_head_w[k * v + j];
                }
                logits_row[j] = dot;
                sum_raw_logits += dot;
                if dot > max_logit {
                    max_logit = dot;
                }
            }

            // Softmax & Partition Function Z
            let mut sum_exp = 0.0f32;
            for val in logits_row.iter_mut() {
                *val = (*val - max_logit).exp();
                sum_exp += *val;
            }
            let log_z = max_logit + sum_exp.ln();

            for val in logits_row.iter_mut() {
                *val /= sum_exp;
            }

            // 1. Cross Entropy with Label Smoothing
            let prob_target = logits_row[target].max(1e-15);
            let ce_loss = -prob_target.ln();
            let token_loss = if eps > 0.0 {
                // E_{q}[-ln P(j)] = (1 - eps) * ce_loss + eps * (log_z - (sum_raw_logits / V))
                (1.0 - eps) * ce_loss + eps * (log_z - (sum_raw_logits / v_f32))
            } else {
                ce_loss
            };

            // 2. Z-loss Regularization: cz * (ln Z)^2
            let z_loss = if cz > 0.0 {
                cz * log_z * log_z
            } else {
                0.0
            };

            total_loss += token_loss + z_loss;

            // 3. Analytic Gradient:
            // dL/dz_j = P(j) * (1 + 2 * cz * log_z) - q(j)
            let z_grad_factor = if cz > 0.0 { 1.0 + 2.0 * cz * log_z } else { 1.0 };
            for j in 0..v {
                let p = logits_row[j];
                let q = if j == target { target_weight } else { smooth_uniform };
                let dl = p * z_grad_factor - q;
                dlogits[i * v + j] = dl / (n as f32);
            }
        }

        // --- 2. 逆伝播 (Backward) ---

        // (1) LM Head Backward:
        // dlm_head += final_norm^T * dlogits
        // d_final_norm = dlogits * lm_head^T
        let mut d_final_norm = vec![0.0f32; n * c];
        let dlm_head = &mut self.grads[layout.lm_head..layout.lm_head + c * v];
        for i in 0..n {
            let dlogits_row = &dlogits[i * v..(i + 1) * v];
            let norm_row = &final_norm[i * c..(i + 1) * c];
            let df_row = &mut d_final_norm[i * c..(i + 1) * c];

            for k in 0..c {
                let mut dot = 0.0f32;
                for j in 0..v {
                    let dlogit = dlogits_row[j];
                    dot += dlogit * lm_head_w[k * v + j];
                    dlm_head[k * v + j] += norm_row[k] * dlogit;
                }
                df_row[k] = dot;
            }
        }

        // (2) Final RMSNorm Backward
        let mut d_x_curr = vec![0.0f32; n * c];
        let d_rms_final = &mut self.grads[layout.rms_final..layout.rms_final + c];
        RMSNorm::backward(
            &mut d_x_curr,
            d_rms_final,
            &d_final_norm,
            &x_final_in,
            &rstd_final,
            rms_final_w,
            c,
        );

        // (3) レイヤーの逆伝播 (L-1 down to 0)
        for li in (0..l).rev() {
            let lo = &layout.layers[li];
            let acts = &layer_acts[li];

            // x_out = x_mid + post_mlp
            let dpost_mlp = d_x_curr.clone();
            let mut dx_mid = d_x_curr;

            // MLP Backward
            let mut dnorm2 = vec![0.0f32; n * c];
            let (grads_pre_down, grads_down) = self.grads.split_at_mut(lo.down);
            let dw_gate_up = &mut grads_pre_down[lo.gate_up..lo.gate_up + c * 2 * ffn];
            let dw_down = &mut grads_down[..ffn * c];
            let w_gate_up = &self.params[lo.gate_up..lo.gate_up + c * 2 * ffn];
            let w_down = &self.params[lo.down..lo.down + ffn * c];

            SwiGLU::backward(
                &mut dnorm2,
                dw_gate_up,
                dw_down,
                &dpost_mlp,
                &acts.norm2,
                &acts.act_g,
                &acts.act_u,
                &acts.act_h,
                w_gate_up,
                w_down,
                n,
                c,
                ffn,
            );

            // RMSNorm 2 Backward -> dx_mid に加算
            let d_rms_ffn = &mut self.grads[lo.rms_ffn..lo.rms_ffn + c];
            let rms_ffn_w = &self.params[lo.rms_ffn..lo.rms_ffn + c];
            RMSNorm::backward(
                &mut dx_mid,
                d_rms_ffn,
                &dnorm2,
                &acts.x_mid,
                &acts.rstd2,
                rms_ffn_w,
                c,
            );

            // x_mid = x_in + post_att
            let dpost_att = dx_mid.clone();
            let mut dx_in = dx_mid;

            // Attention Backward
            let mut dnorm1 = vec![0.0f32; n * c];
            let (grads_pre_proj, grads_proj) = self.grads.split_at_mut(lo.att_proj);
            let dw_qkv = &mut grads_pre_proj[lo.qkv..lo.qkv + c * 3 * c];
            let dw_proj = &mut grads_proj[..c * c];
            let w_qkv = &self.params[lo.qkv..lo.qkv + c * 3 * c];
            let w_proj = &self.params[lo.att_proj..lo.att_proj + c * c];

            CausalSelfAttention::backward(
                &mut dnorm1,
                dw_qkv,
                dw_proj,
                &dpost_att,
                &acts.norm1,
                &acts.q,
                &acts.k,
                &acts.val,
                &acts.att,
                &acts.att_out,
                w_qkv,
                w_proj,
                b,
                t,
                c,
                nh,
            );

            // RMSNorm 1 Backward -> dx_in に加算
            let d_rms_att = &mut self.grads[lo.rms_att..lo.rms_att + c];
            let rms_att_w = &self.params[lo.rms_att..lo.rms_att + c];
            RMSNorm::backward(
                &mut dx_in,
                d_rms_att,
                &dnorm1,
                &acts.x_in,
                &acts.rstd1,
                rms_att_w,
                c,
            );

            d_x_curr = dx_in;
        }

        // (4) Token Embedding Backward:
        let dwte = &mut self.grads[layout.wte..layout.wte + v * c];
        for i in 0..n {
            let tok = x[i] as usize;
            let offset = tok * c;
            let dx_row = &d_x_curr[i * c..(i + 1) * c];
            for k in 0..c {
                dwte[offset + k] += dx_row[k];
            }
        }

        // 勾配ノルム計算
        let mut grad_norm_sq = 0.0f32;
        for &g in self.grads.iter() {
            grad_norm_sq += g * g;
        }

        (total_loss / (n as f32), grad_norm_sq.sqrt())
    }

    /// 評価用: 順伝播のみでクロスエントロピー損失を計算 (逆伝播なし・勾配更新なし)
    pub fn evaluate_loss(
        &self,
        x: &[u16],
        y: &[u16],
        b: usize,
        t: usize,
    ) -> f32 {
        self.evaluate_loss_and_top_k(x, y, b, t, 1).0
    }

    /// 評価用: 順伝播のみでクロスエントロピー損失と Top-k 精度 (%) を同時に計算
    pub fn evaluate_loss_and_top_k(
        &self,
        x: &[u16],
        y: &[u16],
        b: usize,
        t: usize,
        k: usize,
    ) -> (f32, f32) {
        let n = b * t;
        let c = self.config.dim;
        let v = self.config.vocab_size;
        let l = self.config.num_layers;
        let nh = self.config.num_heads;
        let ffn = self.config.ffn_dim;

        let layout = ModelLayout::new(&self.config);

        // (1) Token Embedding lookup
        let mut x_curr = vec![0.0f32; n * c];
        let wte = &self.params[layout.wte..layout.wte + v * c];
        for i in 0..n {
            let tok = x[i] as usize;
            let offset = tok * c;
            x_curr[i * c..(i + 1) * c].copy_from_slice(&wte[offset..offset + c]);
        }

        // バッファを再利用してメモリ割り当てを最小化
        let mut norm1 = vec![0.0f32; n * c];
        let mut rstd1 = vec![0.0f32; n];
        let mut q = vec![0.0f32; n * c];
        let mut k_vec = vec![0.0f32; n * c];
        let mut val = vec![0.0f32; n * c];
        let mut att = vec![0.0f32; b * nh * t * t];
        let mut att_out = vec![0.0f32; n * c];
        let mut post_att = vec![0.0f32; n * c];
        let mut norm2 = vec![0.0f32; n * c];
        let mut rstd2 = vec![0.0f32; n];
        let mut act_g = vec![0.0f32; n * ffn];
        let mut act_u = vec![0.0f32; n * ffn];
        let mut act_h = vec![0.0f32; n * ffn];
        let mut post_mlp = vec![0.0f32; n * c];

        for li in 0..l {
            let lo = &layout.layers[li];

            // 1. RMSNorm 1
            let rms_att_w = &self.params[lo.rms_att..lo.rms_att + c];
            RMSNorm::forward(&mut norm1, &mut rstd1, &x_curr, rms_att_w, 1e-5, c);

            // 2. Attention
            let qkv_w = &self.params[lo.qkv..lo.qkv + c * 3 * c];
            let att_proj_w = &self.params[lo.att_proj..lo.att_proj + c * c];
            CausalSelfAttention::forward(
                &mut post_att,
                &mut q,
                &mut k_vec,
                &mut val,
                &mut att,
                &mut att_out,
                &norm1,
                qkv_w,
                att_proj_w,
                b,
                t,
                c,
                nh,
            );

            // 3. 残差加算: x_mid = x_curr + post_att
            for i in 0..n * c {
                x_curr[i] += post_att[i];
            }

            // 4. RMSNorm 2
            let rms_ffn_w = &self.params[lo.rms_ffn..lo.rms_ffn + c];
            RMSNorm::forward(&mut norm2, &mut rstd2, &x_curr, rms_ffn_w, 1e-5, c);

            // 5. SwiGLU
            let gate_up_w = &self.params[lo.gate_up..lo.gate_up + c * 2 * ffn];
            let down_w = &self.params[lo.down..lo.down + ffn * c];
            SwiGLU::forward(
                &mut post_mlp,
                &mut act_g,
                &mut act_u,
                &mut act_h,
                &norm2,
                gate_up_w,
                down_w,
                n,
                c,
                ffn,
            );

            // 6. 残差加算: x_curr += post_mlp
            for i in 0..n * c {
                x_curr[i] += post_mlp[i];
            }
        }

        // Final RMSNorm
        let mut final_norm = vec![0.0f32; n * c];
        let mut rstd_final = vec![0.0f32; n];
        let rms_final_w = &self.params[layout.rms_final..layout.rms_final + c];
        RMSNorm::forward(&mut final_norm, &mut rstd_final, &x_curr, rms_final_w, 1e-5, c);

        // LM Head & CrossEntropy & Top-k
        let lm_head_w = &self.params[layout.lm_head..layout.lm_head + c * v];
        let mut total_loss = 0.0f32;
        let mut top_k_hits = 0usize;
        let mut logits_row = vec![0.0f32; v];

        for i in 0..n {
            let target = y[i] as usize;
            let norm_row = &final_norm[i * c..(i + 1) * c];

            let mut max_logit = f32::NEG_INFINITY;
            for j in 0..v {
                let mut dot = 0.0f32;
                for k_idx in 0..c {
                    dot += norm_row[k_idx] * lm_head_w[k_idx * v + j];
                }
                logits_row[j] = dot;
                if dot > max_logit {
                    max_logit = dot;
                }
            }

            // Top-k check
            let target_logit = logits_row[target];
            let mut rank = 0;
            for &logit in &logits_row {
                if logit > target_logit {
                    rank += 1;
                    if rank >= k {
                        break;
                    }
                }
            }
            if rank < k {
                top_k_hits += 1;
            }

            // Softmax
            let mut sum_exp = 0.0f32;
            for val in logits_row.iter_mut() {
                *val = (*val - max_logit).exp();
                sum_exp += *val;
            }
            let prob_target = (logits_row[target] / sum_exp).max(1e-15);
            total_loss += -prob_target.ln();
        }

        let avg_loss = total_loss / (n as f32);
        let top_k_acc = (top_k_hits as f32 / n as f32) * 100.0;
        (avg_loss, top_k_acc)
    }

    /// 推論用順伝播: 与えられたトークン列（最大seq_len）から次のトークンのLogitsを計算
    pub fn forward_inference(&self, tokens: &[u16]) -> Vec<f32> {
        let t = tokens.len();
        let b = 1;
        let c = self.config.dim;
        let v = self.config.vocab_size;
        let l = self.config.num_layers;
        let nh = self.config.num_heads;
        let ffn = self.config.ffn_dim;

        let layout = ModelLayout::new(&self.config);

        // Embedding lookup
        let mut x_curr = vec![0.0f32; t * c];
        let wte = &self.params[layout.wte..layout.wte + v * c];
        for i in 0..t {
            let tok = tokens[i] as usize;
            let offset = tok * c;
            x_curr[i * c..(i + 1) * c].copy_from_slice(&wte[offset..offset + c]);
        }

        for li in 0..l {
            let lo = &layout.layers[li];
            let x_in = x_curr.clone();

            // 1. RMSNorm 1
            let mut norm1 = vec![0.0f32; t * c];
            let mut rstd1 = vec![0.0f32; t];
            let rms_att_w = &self.params[lo.rms_att..lo.rms_att + c];
            RMSNorm::forward(&mut norm1, &mut rstd1, &x_in, rms_att_w, 1e-5, c);

            // 2. Causal Self-Attention
            let mut q = vec![0.0f32; t * c];
            let mut k = vec![0.0f32; t * c];
            let mut val = vec![0.0f32; t * c];
            let mut att = vec![0.0f32; nh * t * t];
            let mut att_out = vec![0.0f32; t * c];
            let mut post_att = vec![0.0f32; t * c];

            let qkv_w = &self.params[lo.qkv..lo.qkv + c * 3 * c];
            let att_proj_w = &self.params[lo.att_proj..lo.att_proj + c * c];

            CausalSelfAttention::forward(
                &mut post_att,
                &mut q,
                &mut k,
                &mut val,
                &mut att,
                &mut att_out,
                &norm1,
                qkv_w,
                att_proj_w,
                b,
                t,
                c,
                nh,
            );

            // 3. 残差加算: x_mid = x_in + post_att
            let mut x_mid = vec![0.0f32; t * c];
            for i in 0..t * c {
                x_mid[i] = x_in[i] + post_att[i];
            }

            // 4. RMSNorm 2
            let mut norm2 = vec![0.0f32; t * c];
            let mut rstd2 = vec![0.0f32; t];
            let rms_ffn_w = &self.params[lo.rms_ffn..lo.rms_ffn + c];
            RMSNorm::forward(&mut norm2, &mut rstd2, &x_mid, rms_ffn_w, 1e-5, c);

            // 5. SwiGLU MLP
            let mut act_g = vec![0.0f32; t * ffn];
            let mut act_u = vec![0.0f32; t * ffn];
            let mut act_h = vec![0.0f32; t * ffn];
            let mut post_mlp = vec![0.0f32; t * c];

            let gate_up_w = &self.params[lo.gate_up..lo.gate_up + c * 2 * ffn];
            let down_w = &self.params[lo.down..lo.down + ffn * c];

            SwiGLU::forward(
                &mut post_mlp,
                &mut act_g,
                &mut act_u,
                &mut act_h,
                &norm2,
                gate_up_w,
                down_w,
                t,
                c,
                ffn,
            );

            // 6. 残差加算
            for i in 0..t * c {
                x_curr[i] = x_mid[i] + post_mlp[i];
            }
        }

        // Final RMSNorm
        let mut final_norm = vec![0.0f32; t * c];
        let mut rstd_final = vec![0.0f32; t];
        let rms_final_w = &self.params[layout.rms_final..layout.rms_final + c];
        RMSNorm::forward(&mut final_norm, &mut rstd_final, &x_curr, rms_final_w, 1e-5, c);

        // 最後のトークン (t - 1) に対する Logits
        let last_norm_row = &final_norm[(t - 1) * c..t * c];
        let lm_head_w = &self.params[layout.lm_head..layout.lm_head + c * v];
        let mut logits = vec![0.0f32; v];
        for j in 0..v {
            let mut dot = 0.0f32;
            for k in 0..c {
                dot += last_norm_row[k] * lm_head_w[k * v + j];
            }
            logits[j] = dot;
        }

        logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_loss_matches_forward_backward() {
        let mut rng = DeterministicRng::new(42);
        let config = ModelConfig {
            vocab_size: 20,
            seq_len: 4,
            dim: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            ffn_dim: 32,
            label_smoothing: 0.0,
            z_loss_weight: 0.0,
        };
        let mut model = ModelWeights::new(config, &mut rng);

        let b = 2;
        let t = 4;
        let x = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let y = vec![2, 3, 4, 5, 6, 7, 8, 9];

        // evaluate_loss による損失
        let eval_loss = model.evaluate_loss(&x, &y, b, t);

        // forward_backward による損失
        let (fb_loss, _) = model.forward_backward(&x, &y, b, t);

        assert!((eval_loss - fb_loss).abs() < 1e-5, "eval_loss: {}, fb_loss: {}", eval_loss, fb_loss);
        assert!(!eval_loss.is_nan());
        assert!(eval_loss > 0.0);
    }

    #[test]
    fn test_label_smoothing_and_z_loss_backward() {
        let mut rng = DeterministicRng::new(42);
        let config = ModelConfig {
            vocab_size: 15,
            seq_len: 4,
            dim: 16,
            num_layers: 1,
            num_heads: 1,
            head_dim: 16,
            ffn_dim: 32,
            label_smoothing: 0.05,
            z_loss_weight: 1e-4,
        };
        let mut model = ModelWeights::new(config, &mut rng);

        let b = 1;
        let t = 4;
        let x = vec![1, 2, 3, 4];
        let y = vec![2, 3, 4, 5];

        let (loss, grad_norm) = model.forward_backward(&x, &y, b, t);
        assert!(!loss.is_nan());
        assert!(loss > 0.0);
        assert!(!grad_norm.is_nan());
        assert!(grad_norm > 0.0);

        // 数値勾配チェック (LM Head の一部パラメータで解析的勾配と数値微分を比較)
        let param_idx = model.params.len() - 5;
        let orig_val = model.params[param_idx];
        let h = 1e-3f32;

        model.params[param_idx] = orig_val + h;
        let (loss_plus, _) = model.forward_backward(&x, &y, b, t);

        model.params[param_idx] = orig_val - h;
        let (loss_minus, _) = model.forward_backward(&x, &y, b, t);

        model.params[param_idx] = orig_val;
        model.forward_backward(&x, &y, b, t);
        let analytical_grad = model.grads[param_idx];
        let numerical_grad = (loss_plus - loss_minus) / (2.0 * h);

        let diff = (analytical_grad - numerical_grad).abs();
        assert!(
            diff < 5e-3,
            "analytical: {}, numerical: {}, diff: {}",
            analytical_grad,
            numerical_grad,
            diff
        );
    }
}
