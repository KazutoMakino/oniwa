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

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => {
                if let Some(v) = args.get(i + 1) {
                    num_steps = v.parse().unwrap_or(300);
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
    let checkpoint_dir = base_dir.join("checkpoints").join("latest");
    let best_checkpoint_dir = base_dir.join("checkpoints").join("best");

    // Initialize tokenizer and dataset generator
    let vocab_path = data_dir.join("vocab.json");
    let tokenizer = if vocab_path.exists() {
        CharTokenizer::load_vocab(&vocab_path)?
    } else {
        CharTokenizer::build_from_text("abcdefghijklmnopqrstuvwxyz 0123456789")
    };
    let dataset = DatasetGenerator::load_from_corpus_dir(&corpus_dir)?;

    let config = DecisionConfig {
        vocab_size: tokenizer.vocab_size(),
        seq_len: 128,
        dim: 128,
        num_layers: 4,
        num_heads: 4,
        head_dim: 32,
        ffn_dim: 256,
        num_choices: 4,
        temperature: 1.0,
    };

    let mut rng = DeterministicRng::new(seed);
    let mut model = DecisionModel::new(config.clone(), &mut rng);
    let loss_config = LossConfig::default();

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
        "  - Training Configuration: {} steps, batch size {}, initial lr {}",
        num_steps, batch_size, lr
    );
    println!(
        "  - Decision Tasks: Choice (4 categories), Noul (syntax anomaly), Score (complexity)\n"
    );

    let mut best_loss = f32::INFINITY;
    let start_time = Instant::now();

    for step in 1..=num_steps {
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
        let cur_lr = lr * (1.0 - (step as f32 / num_steps as f32) * 0.8); // Gentle linear decay
        model.adamw_step(cur_lr, 0.01, 0.9, 0.999, 1e-8, step);

        // 6. Thermal control & power telemetry
        let calc_time = step_start.elapsed().as_millis();
        let (cpu_temp, throttle_ms) = thermal.step_throttle();
        let reading = power_tracker.tick(calc_time, throttle_ms);

        // Periodic logging (every 25 steps, or first/last step)
        if step % 25 == 0 || step == 1 || step == num_steps {
            let choice_acc = (correct_choice as f32 / batch_size as f32) * 100.0;
            let noul_acc = (correct_noul as f32 / batch_size as f32) * 100.0;
            let score_mae = total_score_err / batch_size as f32;

            println!(
                "Step {:4}/{} | Loss: {:.4} | Choice Acc: {:5.1}% | Noul Acc: {:5.1}% | Score MAE: {:.3} | Temp: {} | Power: {:.1}W",
                step,
                num_steps,
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
                let _ = model.save_checkpoint(&best_checkpoint_dir, step, mean_loss);
            }
            let _ = model.save_checkpoint(&checkpoint_dir, step, mean_loss);
        }
    }

    let elapsed = start_time.elapsed();
    println!("\n============================================================");
    println!(" 🎉 oniwa-decide Training Complete!");
    println!("  - Elapsed Time: {:.2?}", elapsed);
    println!("  - Minimum Loss: {:.4}", best_loss);
    println!("  - Best Checkpoint Path: {:?}", best_checkpoint_dir);
    println!(
        "  - ⚡ Cumulative Net Energy: {:.4} Wh",
        power_tracker.total_net_wh()
    );
    println!("============================================================");

    Ok(())
}
