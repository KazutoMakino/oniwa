//! oniwa-decide Training Binary
//!
//! Trains a bidirectional Transformer encoder and three decision heads (Choice, Noul, Score)
//! simultaneously using self-supervised automatically synthesized datasets.
//! Equipped with real-time hardware thermal throttling and power tracking (Pure Rust).

use oniwa_decide::{DatasetGenerator, DecisionConfig, DecisionModel, LossCalculator, LossConfig};
use oniwa_lm::power::PowerTracker;
use oniwa_lm::reproducibility::DeterministicRng;
use oniwa_lm::thermal::{ThermalConfig, ThermalController};
use oniwa_lm::tokenizer::CharTokenizer;
use sha2::Digest;
use std::env;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!(" 🧭 ONIWA-DECIDE: TypeSafe System One Training Engine");
    println!("    (Type-safe Decision Model Training Binary)");
    println!("============================================================\n");

    let args: Vec<String> = env::args().collect();
    let mut num_steps = 300usize;
    let mut batch_size = 8usize;
    let mut lr = 0.001f32;
    let mut seed = 42u64;

    let mut reset_mode = false;
    let mut add_steps_arg: Option<usize> = None;
    let mut quaternion_head_mode = false;
    let mut config_mode = "standard".to_string(); // "standard", "quaternion_head", "iso_parameter"
    let mut custom_checkpoint_dir: Option<std::path::PathBuf> = None;
    let mut metrics_path: Option<std::path::PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => {
                if let Some(v) = args.get(i + 1) {
                    num_steps = v.parse().unwrap_or(300);
                    i += 1;
                }
            }
            "--add-steps" => {
                if let Some(v) = args.get(i + 1) {
                    add_steps_arg = v.parse().ok();
                    i += 1;
                }
            }
            "--reset" => {
                reset_mode = true;
            }
            "--quaternion-head" => {
                quaternion_head_mode = true;
                config_mode = "quaternion_head".to_string();
            }
            "--config" => {
                if let Some(v) = args.get(i + 1) {
                    config_mode = v.clone();
                    if v == "quaternion_head" {
                        quaternion_head_mode = true;
                    }
                    i += 1;
                }
            }
            "--checkpoint-dir" => {
                if let Some(v) = args.get(i + 1) {
                    custom_checkpoint_dir = Some(std::path::PathBuf::from(v));
                    i += 1;
                }
            }
            "--output-metrics" => {
                if let Some(v) = args.get(i + 1) {
                    metrics_path = Some(std::path::PathBuf::from(v));
                    i += 1;
                }
            }
            "--batch-size" => {
                if let Some(v) = args.get(i + 1) {
                    batch_size = v.parse().unwrap_or(8);
                    i += 1;
                }
            }
            "--lr" => {
                if let Some(v) = args.get(i + 1) {
                    lr = v.parse().unwrap_or(0.001);
                    i += 1;
                }
            }
            "--seed" => {
                if let Some(v) = args.get(i + 1) {
                    seed = v.parse().unwrap_or(42);
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let workspace_root = oniwa_lm::find_workspace_root();
    let data_dir = workspace_root.join("data");
    let corpus_dir = data_dir.join("corpus");
    let base_dir = workspace_root.join("crates/oniwa-decide");

    // Checkpoint directories: use custom dir or fallback based on config_mode
    let (checkpoint_dir, best_checkpoint_dir) = if let Some(ref dir) = custom_checkpoint_dir {
        (dir.join("latest"), dir.join("best"))
    } else {
        match config_mode.as_str() {
            "quaternion_head" => (
                base_dir.join("checkpoints/quaternion_head/latest"),
                base_dir.join("checkpoints/quaternion_head/best"),
            ),
            "iso_parameter" => (
                base_dir.join("checkpoints/iso_parameter/latest"),
                base_dir.join("checkpoints/iso_parameter/best"),
            ),
            _ => (
                base_dir.join("checkpoints/standard_baseline/latest"),
                base_dir.join("checkpoints/standard_baseline/best"),
            ),
        }
    };

    // Initialize tokenizer and dataset generator
    let vocab_path = data_dir.join("vocab.json");
    let tokenizer = if vocab_path.exists() {
        CharTokenizer::load_vocab(&vocab_path)?
    } else {
        CharTokenizer::build_from_text("abcdefghijklmnopqrstuvwxyz 0123456789")
    };
    let dataset = DatasetGenerator::load_from_corpus_dir(&corpus_dir)?;

    let config = match config_mode.as_str() {
        "quaternion_head" => DecisionConfig::quaternion_head(tokenizer.vocab_size()),
        "iso_parameter" => DecisionConfig::iso_parameter(tokenizer.vocab_size()),
        _ => {
            if quaternion_head_mode {
                DecisionConfig::quaternion_head(tokenizer.vocab_size())
            } else {
                DecisionConfig::standard_baseline(tokenizer.vocab_size())
            }
        }
    };

    let mut rng = DeterministicRng::new(seed);
    let mut model = DecisionModel::new(config.clone(), &mut rng);
    let loss_config = LossConfig::default();

    // Checkpoint automatic resume handling
    let mut start_step = 1;
    let mut best_step = 1;
    let mut best_loss = if !reset_mode && best_checkpoint_dir.join("meta.json").exists() {
        let meta_str =
            std::fs::read_to_string(best_checkpoint_dir.join("meta.json")).unwrap_or_default();
        let meta: serde_json::Value = serde_json::from_str(&meta_str).unwrap_or_default();
        if let Some(s) = meta["step"].as_u64() {
            best_step = s as usize;
        }
        meta["loss"]
            .as_f64()
            .map(|v| v as f32)
            .unwrap_or(f32::INFINITY)
    } else {
        f32::INFINITY
    };

    if !reset_mode && checkpoint_dir.join("meta.json").exists() {
        match model.load_checkpoint(&checkpoint_dir) {
            Ok((resumed_step, resumed_loss)) => {
                start_step = resumed_step + 1;
                println!(
                    "  🔄 Existing checkpoint detected! Resuming from Step {} (Previous Loss: {:.4})",
                    start_step, resumed_loss
                );
                if !best_loss.is_infinite() {
                    println!("  🏆 Best Recorded Loss: {:.4}", best_loss);
                }
            }
            Err(e) => {
                println!("  ⚠️ Failed to resume checkpoint (starting fresh): {}", e);
            }
        }
    } else if reset_mode {
        println!(
            "  🔄 --reset specified: Starting fresh from Step 1 (ignoring existing checkpoints)"
        );
    }

    let target_steps = if let Some(add) = add_steps_arg {
        (start_step - 1) + add
    } else if start_step > 1 && num_steps <= (start_step - 1) {
        println!(
            "  💡 Specified --steps ({}) is <= current step ({}). Adding {} steps (Target: Step {})",
            num_steps,
            start_step - 1,
            num_steps,
            (start_step - 1) + num_steps
        );
        (start_step - 1) + num_steps
    } else {
        num_steps
    };

    let thermal = ThermalController::new(ThermalConfig::default());
    let mut power_tracker = PowerTracker::auto_detect();

    println!(
        "  - Model Scale: {} layers, {} heads, dim {} (parameters: {})",
        config.num_layers,
        config.num_heads,
        config.dim,
        model.params.len()
    );
    println!(
        "  - Training Range: Step {} ~ Step {} (Total: {} steps, batch size {}, initial lr {})",
        start_step,
        target_steps,
        target_steps.saturating_sub(start_step - 1),
        batch_size,
        lr
    );
    println!(
        "  - Decision Tasks: Choice (4 categories), Noul (syntax anomaly), Score (complexity)\n"
    );

    let start_time = Instant::now();
    let mut final_loss = if best_loss.is_finite() {
        best_loss
    } else {
        0.0
    };

    #[derive(serde::Serialize)]
    struct StepMetric {
        step: usize,
        loss: f32,
        choice_acc: f32,
        noul_acc: f32,
        score_mae: f32,
        net_watts: f32,
    }
    let mut step_metrics_log: Vec<StepMetric> = Vec::new();

    for step in start_step..=target_steps {
        let step_start = Instant::now();

        // 1. Batch generation
        let (tokens, target_choices, target_nouls, target_scores) =
            dataset.generate_batch(&tokenizer, batch_size, config.seq_len, &mut rng);

        // 2. Forward pass
        model.zero_grad();
        let cache = model.forward(&tokens, batch_size, config.seq_len);

        // 3. Loss computation
        let mut total_loss = 0.0f32;
        let mut dchoice_logits = vec![0.0f32; batch_size * config.num_choices];
        let mut dnoul_logits = vec![0.0f32; batch_size];
        let mut dscore_preds = vec![0.0f32; batch_size];

        let mut correct_choice = 0;
        let mut correct_noul = 0;
        let mut total_score_err = 0.0f32;

        for bi in 0..batch_size {
            // Choice Loss
            let logits =
                &cache.choice_logits[bi * config.num_choices..(bi + 1) * config.num_choices];
            let (l_c, d_c) = LossCalculator::choice_loss(
                logits,
                target_choices[bi],
                config.num_choices,
                loss_config.label_smoothing,
            );
            total_loss += loss_config.choice_weight * l_c;
            for k in 0..config.num_choices {
                dchoice_logits[bi * config.num_choices + k] =
                    loss_config.choice_weight * d_c[k] / (batch_size as f32);
            }
            let pred_choice = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(k, _)| k)
                .unwrap();
            if pred_choice == target_choices[bi] {
                correct_choice += 1;
            }

            // Noul Loss
            let noul_logit = cache.noul_logits[bi];
            let (l_n, d_n) =
                LossCalculator::noul_loss(noul_logit, target_nouls[bi], loss_config.brier_weight);
            total_loss += loss_config.noul_weight * l_n;
            dnoul_logits[bi] = loss_config.noul_weight * d_n / (batch_size as f32);
            let pred_noul = LossCalculator::sigmoid(noul_logit) >= 0.5;
            if pred_noul == target_nouls[bi] {
                correct_noul += 1;
            }

            // Score Loss
            let score_pred = cache.score_preds[bi];
            let (l_s, d_s) =
                LossCalculator::score_loss(score_pred, target_scores[bi], loss_config.huber_delta);
            total_loss += loss_config.score_weight * l_s;
            dscore_preds[bi] = loss_config.score_weight * d_s / (batch_size as f32);
            total_score_err += (score_pred - target_scores[bi]).abs();
        }

        let mean_loss = total_loss / (batch_size as f32);
        final_loss = mean_loss;

        // 4. Backward pass
        model.backward(
            &tokens,
            &cache,
            &dchoice_logits,
            &dnoul_logits,
            &dscore_preds,
            batch_size,
            config.seq_len,
        );

        // 5. AdamW optimization
        let cur_lr = lr * (1.0 - (step as f32 / target_steps as f32) * 0.8); // Gentle linear decay
        model.adamw_step(cur_lr, 0.01, 0.9, 0.999, 1e-8, step);

        // 6. Thermal control & power telemetry
        let calc_time = step_start.elapsed().as_millis();
        let (cpu_temp, throttle_ms) = thermal.step_throttle();
        let reading = power_tracker.tick(calc_time, throttle_ms);

        let choice_acc = (correct_choice as f32 / batch_size as f32) * 100.0;
        let noul_acc = (correct_noul as f32 / batch_size as f32) * 100.0;
        let score_mae = total_score_err / batch_size as f32;

        if metrics_path.is_some() {
            step_metrics_log.push(StepMetric {
                step,
                loss: mean_loss,
                choice_acc,
                noul_acc,
                score_mae,
                net_watts: reading.net_watts,
            });
        }

        // Periodic logging (every 25 steps, or first/last step)
        if step % 25 == 0 || step == start_step || step == target_steps {
            println!(
                "Step {:4}/{} | Loss: {:.4} | Choice Acc: {:5.1}% | Noul Acc: {:5.1}% | Score MAE: {:.3} | Temp: {} | Power: {:.1}W",
                step,
                target_steps,
                mean_loss,
                choice_acc,
                noul_acc,
                score_mae,
                cpu_temp.map(|t| format!("{:.1}C", t)).unwrap_or_else(|| "N/A".into()),
                reading.net_watts,
            );

            // Save best checkpoint
            if mean_loss < best_loss {
                best_loss = mean_loss;
                best_step = step;
                let _ = model.save_checkpoint(&best_checkpoint_dir, step, mean_loss);
            }
            let _ = model.save_checkpoint(&checkpoint_dir, step, mean_loss);
        }
    }

    if let Some(ref m_path) = metrics_path {
        if let Some(parent) = m_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(&step_metrics_log) {
            let _ = std::fs::write(m_path, json);
            println!("  📊 Exported execution metrics log to {:?}", m_path);
        }
    }

    let elapsed = start_time.elapsed();
    println!("\n============================================================");
    println!(" 🎉 oniwa-decide Training Complete!");
    println!("  - Elapsed Time: {:.2?}", elapsed);
    println!("  - Minimum Loss: {:.4} (Step {})", best_loss, best_step);
    println!("  - Best Checkpoint Path: {:?}", best_checkpoint_dir);
    println!(
        "  - ⚡ Cumulative Net Energy: {:.4} Wh",
        power_tracker.total_net_wh()
    );

    // 7. Provenance Ledger Recording
    let root_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let ledger_path = root_dir.join("logs").join("ledger_index.jsonl");

    let compute_file_sha256 = |path: &std::path::Path| -> String {
        if path.exists() {
            let bytes = std::fs::read(path).unwrap_or_default();
            let mut hasher = sha2::Sha256::new();
            hasher.update(&bytes);
            format!("{:x}", hasher.finalize())
        } else {
            "N/A".into()
        }
    };

    let best_sha256 = compute_file_sha256(&best_checkpoint_dir.join("weights.bin"));
    let latest_sha256 = compute_file_sha256(&checkpoint_dir.join("weights.bin"));

    if ledger_path.parent().map(|p| p.exists()).unwrap_or(false) {
        if let Ok(mut ledger) = oniwa_lm::logger::ProvenanceLedger::open(&ledger_path) {
            let git_commit = oniwa_lm::logger::get_git_commit_hash();
            let git_dirty = oniwa_lm::logger::get_git_dirty();
            let event =
                oniwa_lm::logger::ProvenanceEvent::TrainingRun(oniwa_lm::logger::TrainingRunLog {
                    timestamp_utc: oniwa_lm::logger::current_timestamp_utc(),
                    model_name: "oniwa-decide".into(),
                    git_commit_hash: git_commit,
                    git_dirty,
                    random_seed: seed,
                    total_steps: target_steps,
                    best_step,
                    best_loss,
                    final_loss,
                    best_weights_sha256: best_sha256.clone(),
                    latest_weights_sha256: latest_sha256,
                    elapsed_secs: elapsed.as_secs_f32(),
                    cumulative_energy_wh: power_tracker.total_net_wh(),
                    target_arch: std::env::consts::ARCH.into(),
                    target_os: std::env::consts::OS.into(),
                });
            if let Ok(()) = ledger.record(&event) {
                println!(
                    "  📜 Audit Ledger: Recorded TrainingRun event to {:?}",
                    ledger_path
                );
                println!("  🔒 Best Model SHA-256: {}", best_sha256);
            }
        }
    }
    println!("============================================================");

    Ok(())
}
