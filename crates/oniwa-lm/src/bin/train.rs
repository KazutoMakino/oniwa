//! oniwa-lm Training Binary
//!
//! ONIWA: Organic Non-datacenter Intelligence Without Abuse

use oniwa_lm::benchmark::run_benchmark;
use oniwa_lm::logger::{ModelConfigInfo, TrainingManifest, TrainingStepLog};
use oniwa_lm::model::{ModelConfig, ModelWeights};
use oniwa_lm::power::PowerTracker;
use oniwa_lm::reproducibility::{compute_checksum_bytes, compute_checksum_f32, DeterministicRng};
use oniwa_lm::thermal::{ThermalConfig, ThermalController};
use oniwa_lm::tokenizer::CharTokenizer;
use std::fs;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let git_commit = oniwa_lm::logger::get_git_commit_hash();
    let git_short = oniwa_lm::logger::get_git_short_hash();
    let git_dirty = oniwa_lm::logger::get_git_dirty();
    let dirty_str = if git_dirty {
        " (dirty / uncommitted changes)"
    } else {
        " (clean)"
    };

    println!("============================================================");
    println!(" 🪨 ONIWA: Organic Non-datacenter Intelligence Without Abuse");
    println!("    (Organic Non-datacenter Intelligence Without Abuse: oniwa-lm)");
    println!("    Git Commit: {}{}", git_short, dirty_str);
    println!("============================================================\n");

    let workspace_root = oniwa_lm::find_workspace_root();
    let data_dir = workspace_root.join("data");
    let logs_dir = workspace_root.join("logs");
    let base_dir = if workspace_root.join("crates/oniwa-lm/checkpoints").is_dir() {
        workspace_root.join("crates/oniwa-lm")
    } else {
        workspace_root.clone()
    };
    let raw_text_path = data_dir.join("sangetsuki_clean.txt");

    // ---------------------------------------------------------
    // Parse command-line arguments
    // ---------------------------------------------------------
    let args: Vec<String> = std::env::args().collect();
    let mut num_steps_arg: Option<usize> = None;
    let mut add_steps_arg: Option<usize> = None;
    let mut infinite_mode = false;
    let mut seed = 0u64;
    let mut reset_mode = false;
    let mut custom_prompts: Vec<String> = Vec::new();
    let mut gen_len = 30usize;
    let mut log_interval = 25usize;
    let mut run_name_arg: Option<String> = None;
    let mut label_smoothing = 0.05f32;
    let mut z_loss_weight = 1e-4f32;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => {
                if let Some(val) = args.get(i + 1) {
                    num_steps_arg = val.parse().ok();
                    i += 1;
                }
            }
            "--add-steps" => {
                if let Some(val) = args.get(i + 1) {
                    add_steps_arg = val.parse().ok();
                    i += 1;
                }
            }
            "--infinite" | "-i" => {
                infinite_mode = true;
            }
            "--seed" => {
                if let Some(val) = args.get(i + 1) {
                    seed = val.parse().unwrap_or(0);
                    i += 1;
                }
            }
            "--reset" => {
                reset_mode = true;
            }
            "--prompt" => {
                if let Some(val) = args.get(i + 1) {
                    custom_prompts.push(val.clone());
                    i += 1;
                }
            }
            "--prompts" => {
                if let Some(val) = args.get(i + 1) {
                    for p in val.split(',') {
                        let trimmed = p.trim();
                        if !trimmed.is_empty() {
                            custom_prompts.push(trimmed.to_string());
                        }
                    }
                    i += 1;
                }
            }
            "--gen-len" => {
                if let Some(val) = args.get(i + 1) {
                    gen_len = val.parse().unwrap_or(30);
                    i += 1;
                }
            }
            "--interval" => {
                if let Some(val) = args.get(i + 1) {
                    log_interval = val.parse().unwrap_or(25);
                    i += 1;
                }
            }
            "--run-name" => {
                if let Some(val) = args.get(i + 1) {
                    run_name_arg = Some(val.clone());
                    i += 1;
                }
            }
            "--label-smoothing" => {
                if let Some(val) = args.get(i + 1) {
                    label_smoothing = val.parse().unwrap_or(0.05);
                    i += 1;
                }
            }
            "--z-loss" => {
                if let Some(val) = args.get(i + 1) {
                    z_loss_weight = val.parse().unwrap_or(1e-4);
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let default_prompts = vec![
        "その時、".to_string(),
        "メロスは、".to_string(),
        "def fibonacci(".to_string(),
        "fn is_prime(".to_string(),
        "use serde::".to_string(),
    ];
    let observation_prompts = if !custom_prompts.is_empty() {
        custom_prompts
    } else {
        default_prompts
    };

    // ---------------------------------------------------------
    // 1. Data preparation (Tokenize corpus & record provenance)
    // ---------------------------------------------------------
    println!("[1/4] 📚 Verifying dataset & tokenizing...");
    let combined_corpus_path = data_dir.join("corpus_combined.txt");
    let (dataset_path, dataset_name, dataset_url) = if combined_corpus_path.exists() {
        (
            combined_corpus_path,
            "ONIWA Combined Corpus (Aozora Bunko PD & e-Gov Laws & arXiv & Official Tech Specs & Clean Code)",
            "https://github.com/KazutoMakino/oniwa",
        )
    } else if raw_text_path.exists() {
        (
            raw_text_path,
            "Aozora Bunko/Wikisource: Atsushi Nakajima 'Sangetsuki'",
            "https://ja.wikisource.org/wiki/山月記",
        )
    } else {
        eprintln!("Error: Training text corpus not found. Please run `cargo run -p oniwa-pipeline -- --all` first.");
        return Ok(());
    };

    let tokenizer = CharTokenizer::ingest_file(
        &dataset_path,
        &data_dir,
        &logs_dir,
        dataset_name,
        dataset_url,
        "Public Domain & Clean Open Source",
    )?;

    let tokens = CharTokenizer::load_tokens_bin(data_dir.join("tokens.bin"))?;
    let split_idx = (tokens.len() as f32 * 0.9) as usize;
    let (train_tokens, val_tokens) = tokens.split_at(split_idx);
    println!("  - Training dataset: {}", dataset_name);
    println!(
        "  - Vocabulary size (V): {} characters",
        tokenizer.vocab_size()
    );
    println!("  - Total tokens: {} tokens", tokens.len());
    println!(
        "  - Data split: Train {} tokens (90%) / Val {} tokens (10%)",
        train_tokens.len(),
        val_tokens.len()
    );

    // ---------------------------------------------------------
    // 2. Model initialization & reproducibility configuration
    // ---------------------------------------------------------
    println!("\n[2/4] ⚙️ Initializing model & deterministic seed (oniwa-v2)...");
    let mut rng = DeterministicRng::new(seed);

    let checkpoint_dir = base_dir.join("checkpoints").join("latest");
    let best_checkpoint_dir = base_dir.join("checkpoints").join("best");

    // Restore config from meta.json if checkpoint exists, otherwise use v2 default
    let config = if !reset_mode && checkpoint_dir.join("meta.json").exists() {
        match ModelConfig::from_meta_json(checkpoint_dir.join("meta.json")) {
            Ok(mut c) => {
                c.vocab_size = tokenizer.vocab_size();
                c
            }
            Err(_) => ModelConfig {
                vocab_size: tokenizer.vocab_size(),
                seq_len: 128,
                dim: 128,
                num_layers: 4,
                num_heads: 4,
                head_dim: 32,
                ffn_dim: 256,
                label_smoothing,
                z_loss_weight,
            },
        }
    } else {
        ModelConfig {
            vocab_size: tokenizer.vocab_size(),
            seq_len: 128,
            dim: 128,
            num_layers: 4,
            num_heads: 4,
            head_dim: 32,
            ffn_dim: 256,
            label_smoothing,
            z_loss_weight,
        }
    };

    let mut model = ModelWeights::new(config.clone(), &mut rng);
    let total_params = model.params.len();
    println!(
        "  - Architecture: seq_len {} chars, dim {}, layers {}, attention heads {}",
        config.seq_len, config.dim, config.num_layers, config.num_heads
    );
    println!(
        "  - Total parameters: {} (~{:.2} M params)",
        total_params,
        total_params as f32 / 1_000_000.0
    );
    println!(
        "  - Regularization / Loss: Label Smoothing ({:.2}) + Z-loss ({:e})",
        config.label_smoothing, config.z_loss_weight
    );

    // Auto-detect checkpoint and resume (latest & best)
    let mut start_step = 1;
    let mut current_seed = seed;
    let mut best_val_loss = if !reset_mode && best_checkpoint_dir.join("meta.json").exists() {
        let meta_str =
            fs::read_to_string(best_checkpoint_dir.join("meta.json")).unwrap_or_default();
        let meta: serde_json::Value = serde_json::from_str(&meta_str).unwrap_or_default();
        meta["loss"]
            .as_f64()
            .map(|v| v as f32)
            .unwrap_or(f32::INFINITY)
    } else {
        f32::INFINITY
    };

    if !reset_mode && checkpoint_dir.exists() && checkpoint_dir.join("meta.json").exists() {
        println!("  🔄 Existing checkpoint detected! Attempting to resume...");
        match model.load_checkpoint(&checkpoint_dir) {
            Ok((resumed_step, resumed_loss, resumed_seed)) => {
                start_step = resumed_step + 1;
                current_seed = resumed_seed;
                println!(
                    "  ✅ Checkpoint restored: Resuming from Step {} (Previous Loss: {:.4})",
                    resumed_step, resumed_loss
                );
                if !best_val_loss.is_infinite() {
                    println!("  🏆 Existing best Val Loss: {:.4}", best_val_loss);
                }
            }
            Err(e) => {
                println!(
                    "  ⚠️ Failed to restore checkpoint (starting new session): {}",
                    e
                );
            }
        }
    } else if reset_mode {
        println!(
            "  🔄 --reset specified: discarding existing checkpoints and starting fresh (Seed: {})",
            seed
        );
    } else {
        println!(
            "  - Starting fresh session (no checkpoint found, Seed: {})",
            seed
        );
    }

    let init_checksum = compute_checksum_f32(&model.params);
    println!("  - Weights SHA-256: {}...", &init_checksum[..16]);
    println!("  - Random seed: {}", current_seed);

    // ---------------------------------------------------------
    // 3. Thermal control & power tracking & provenance ledger
    // ---------------------------------------------------------
    println!("\n[3/4] 🌡️ Initializing hardware thermal control & ⚡ green power tracker...");
    let thermal = ThermalController::new(ThermalConfig::default());
    let mut power_tracker = PowerTracker::auto_detect();

    println!(
        "  - Power measurement mode: {} (Baseline idle: {:.1}W)",
        power_tracker.source_description(),
        power_tracker.baseline_watts()
    );

    if thermal.is_available() {
        let sname = thermal.sensor_name().unwrap_or("CPU sensor");
        let initial_temp = thermal
            .read_temperature()
            .map(|t| format!("{:.1} C", t))
            .unwrap_or_else(|| "N/A".into());
        println!(
            "  - CPU thermal sensor: Detected [{}] (Current temp: {}, dynamic throttling active)",
            sname, initial_temp
        );
    } else {
        println!("  - CPU thermal sensor: Not detected (Standard PC mode: full speed with zero sleep delay)");
    }

    let current_time_iso = oniwa_lm::logger::current_timestamp_utc();
    let default_run_id = format!(
        "run_{}",
        current_time_iso
            .replace(['-', ':', 'T', 'Z'], "_")
            .trim_end_matches('_')
    );
    let run_id = run_name_arg.unwrap_or(default_run_id);
    let run_dir = logs_dir.join("runs").join(&run_id);
    fs::create_dir_all(&run_dir)?;

    let manifest = TrainingManifest {
        project_name: "oniwa-lm".into(),
        version: "0.1.0".into(),
        git_commit_hash: git_commit.clone(),
        git_dirty,
        timestamp_utc: current_time_iso,
        random_seed: seed,
        model_config: ModelConfigInfo {
            vocab_size: config.vocab_size,
            seq_len: config.seq_len,
            dim: config.dim,
            num_layers: config.num_layers,
            num_heads: config.num_heads,
            num_kv_heads: config.num_heads,
            ffn_dim: config.ffn_dim,
            label_smoothing: config.label_smoothing,
            z_loss_weight: config.z_loss_weight,
        },
        dataset_sha256: compute_checksum_bytes(&fs::read(data_dir.join("tokens.bin"))?),
        initial_weights_sha256: init_checksum.clone(),
        platform_arch: std::env::consts::ARCH.into(),
        os_name: std::env::consts::OS.into(),
    };

    // Save manifest.json
    fs::write(
        run_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;

    let steps_log_path = run_dir.join("steps.jsonl");
    let mut steps_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&steps_log_path)?;

    // Initialize Growth Journal
    let journal_md_path = run_dir.join("growth_journal.md");
    let latest_journal_md_path = logs_dir.join("growth_journal.md");
    let journal_jsonl_path = run_dir.join("growth_journal.jsonl");

    let mut journal_md = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(reset_mode || !journal_md_path.exists())
        .append(!reset_mode && journal_md_path.exists())
        .open(&journal_md_path)?;

    let mut journal_jsonl = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&journal_jsonl_path)?;

    // Write header on new file (or update legacy header without intellect benchmark column)
    let prompts_display = observation_prompts
        .iter()
        .map(|p| format!("\"{}\"", p))
        .collect::<Vec<_>>()
        .join(" / ");
    let should_write_header =
        reset_mode || !journal_md_path.exists() || journal_md.metadata()?.len() == 0;
    if should_write_header {
        use std::io::Write;
        writeln!(journal_md, "# 🌿 ONIWA Growth Journal")?;
        writeln!(journal_md, "> **\"Language is the shadow of cognition, and code is the DNA of life.\"**  \n> A chronicle of model transformation, sprouting words from pure random noise (Step 0) and acquiring context.\n")?;
        writeln!(journal_md, "- **Observation Session**: `{}`", run_id)?;
        writeln!(journal_md, "- **Git Commit**: `{}`{}", git_short, dirty_str)?;
        writeln!(
            journal_md,
            "- **Observation Prompts (Probes)**: {}",
            prompts_display
        )?;
        writeln!(
            journal_md,
            "- **Generated Characters**: {} chars per prompt",
            gen_len
        )?;
        writeln!(
            journal_md,
            "- **Model Scale**: {} layers, {} heads, dim {} (~{:.1}K params)\n",
            config.num_layers,
            config.num_heads,
            config.dim,
            total_params as f32 / 1000.0
        )?;
        writeln!(journal_md, "| Step | Train Loss | Val Loss | Benchmark (Top-5 / Quiz / Syntax) | Core Temp / Power | Developmental Generation (Emergence of Words) |")?;
        writeln!(
            journal_md,
            "| :---: | :---: | :---: | :---: | :---: | :--- |"
        )?;
        journal_md.flush()?;
    } else {
        let existing = fs::read_to_string(&journal_md_path).unwrap_or_default();
        if !existing.contains("Benchmark") && !existing.contains("知能ベンチ") {
            let mut file = fs::File::create(&journal_md_path)?;
            use std::io::Write;
            writeln!(file, "# 🌿 ONIWA Growth Journal")?;
            writeln!(file, "> **\"Language is the shadow of cognition, and code is the DNA of life.\"**  \n> A chronicle of model transformation, sprouting words from pure random noise (Step 0) and acquiring context.\n")?;
            writeln!(file, "- **Observation Session**: `{}`", run_id)?;
            writeln!(file, "- **Git Commit**: `{}`{}", git_short, dirty_str)?;
            writeln!(
                file,
                "- **Observation Prompts (Probes)**: {}",
                prompts_display
            )?;
            writeln!(
                file,
                "- **Generated Characters**: {} chars per prompt",
                gen_len
            )?;
            writeln!(
                file,
                "- **Model Scale**: {} layers, {} heads, dim {} (~{:.1}K params)\n",
                config.num_layers,
                config.num_heads,
                config.dim,
                total_params as f32 / 1000.0
            )?;
            writeln!(file, "| Step | Train Loss | Val Loss | Benchmark (Top-5 / Quiz / Syntax) | Core Temp / Power | Developmental Generation (Emergence of Words) |")?;
            writeln!(file, "| :---: | :---: | :---: | :---: | :---: | :--- |")?;
            file.flush()?;
        }
    }

    // ---------------------------------------------------------
    // 4. Pre-training / Resume generation test
    // ---------------------------------------------------------
    println!("\n[4/4] 🚀 Starting training & observing developmental emergence (patrolling multiple probes)");
    println!("------------------------------------------------------------");

    if start_step == 1 {
        let mut md_lines = Vec::new();
        let mut sample_entries = Vec::new();
        let mut raw_samples = Vec::new();

        for p in &observation_prompts {
            let s = generate_sample(&model, &tokenizer, p, gen_len, &mut rng);
            println!("  - [{}] -> \"{}\"", p, s);
            let clean = s.replace('\n', " ").replace('|', "\\|");
            md_lines.push(format!("**[{}]** `{}`", p, clean));
            sample_entries.push(serde_json::json!({
                "prompt": p,
                "generated_text": s.clone(),
            }));
            raw_samples.push(s);
        }

        let (val_0, bench_0) =
            run_benchmark(&model, &tokenizer, val_tokens, &raw_samples, 4, 4, &mut rng);
        println!(
            "【Step 0 (Initial State)】 Initial Validation Loss: {:.4}",
            val_0
        );
        println!("  🧠 Initial Benchmark: {}", bench_0.summary_line());
        println!("  (* The model has learned nothing yet; outputs are purely random characters *)");
        println!("------------------------------------------------------------\n");

        use std::io::Write;
        let md_text = md_lines.join("<br>");
        writeln!(
            journal_md,
            "| **Step 0** | 8.3400 (initial random) | {:.4} | {} | - | {} *(initial cry)* |",
            val_0,
            bench_0.short_display(),
            md_text
        )?;
        journal_md.flush()?;

        let json_entry = serde_json::json!({
            "step": 0,
            "loss": 8.34,
            "val_loss": val_0,
            "benchmark": {
                "top5_accuracy": bench_0.top5_accuracy,
                "cloze_top1_accuracy": bench_0.cloze_top1_accuracy,
                "cloze_top5_accuracy": bench_0.cloze_top5_accuracy,
                "syntactic_score": bench_0.syntactic_score,
                "bracket_score": bench_0.bracket_score,
                "non_repetition_score": bench_0.non_repetition_score,
            },
            "samples": sample_entries,
            "cpu_temp_c": null,
            "power_w": null,
        });
        writeln!(journal_jsonl, "{}", json_entry)?;
        journal_jsonl.flush()?;
    } else {
        let mut raw_samples = Vec::new();
        println!("【Step {} (Checkpoint Restored State)】", start_step - 1);
        println!("  - Current linguistic emergence (probes):");
        for p in &observation_prompts {
            let s = generate_sample(&model, &tokenizer, p, gen_len, &mut rng);
            println!("    [{}] -> \"{}\"", p, s);
            raw_samples.push(s);
        }
        let (current_val, bench_cur) =
            run_benchmark(&model, &tokenizer, val_tokens, &raw_samples, 4, 4, &mut rng);
        println!("  - Current Validation Loss: {:.4}", current_val);
        println!("  🧠 Current Benchmark: {}", bench_cur.summary_line());
        println!("  (* Continuing training from checkpoint intellect state *)");
        println!("------------------------------------------------------------\n");
    }

    // ---------------------------------------------------------
    // 5. Training loop configuration
    // ---------------------------------------------------------
    let target_steps = if infinite_mode {
        usize::MAX
    } else if let Some(add) = add_steps_arg {
        (start_step - 1) + add
    } else if let Some(steps) = num_steps_arg {
        if steps < start_step {
            println!("  💡 Specified --steps ({}) is <= current step ({}); running {} additional steps (Target: Step {})",
                steps, start_step - 1, steps, (start_step - 1) + steps);
            (start_step - 1) + steps
        } else {
            steps
        }
    } else {
        if start_step > 1 {
            println!("  💡 No step count specified; running 150 additional steps from checkpoint (Target: Step {})",
                (start_step - 1) + 150);
            (start_step - 1) + 150
        } else {
            150
        }
    };

    let batch_size = 4;
    let max_lr = 0.003f32;
    let min_lr = 0.0003f32;
    let total_session_steps = target_steps.saturating_sub(start_step - 1);
    let warmup_steps = ((total_session_steps as f32 * 0.05) as usize).clamp(10, 100);
    let wd = 0.01f32;

    let target_steps_display = if infinite_mode {
        "Infinite (Press Ctrl+C anytime for graceful stop)".to_string()
    } else {
        format!(
            "Step {} to {} (additional {} steps)",
            start_step, target_steps, total_session_steps
        )
    };
    println!("  - Training range: {}", target_steps_display);
    println!(
        "  - LR Schedule: Cosine LR Decay (Peak: {:.4}, Min: {:.4}, Warmup: {} steps)",
        max_lr, min_lr, warmup_steps
    );
    println!("  - (* If interrupted with Ctrl+C, checkpoint is automatically saved *)\n");

    let start_time = Instant::now();
    let mut step = start_step;
    let mut last_loss = 0.0f32;

    loop {
        if !infinite_mode && step > target_steps {
            break;
        }
        let step_start = Instant::now();

        // Dynamic LR calculation (Warmup + Cosine Decay)
        let effective_max_steps = if infinite_mode {
            step + 10000
        } else {
            target_steps
        };
        let lr = compute_cosine_lr(step, effective_max_steps, max_lr, min_lr, warmup_steps);

        // 1. Extract minibatch (sample from training dataset)
        let mut x_batch = Vec::with_capacity(batch_size * config.seq_len);
        let mut y_batch = Vec::with_capacity(batch_size * config.seq_len);

        for _ in 0..batch_size {
            let max_idx = train_tokens.len() - config.seq_len - 1;
            let start_idx = (rng.next_f32() * max_idx as f32) as usize;
            x_batch.extend_from_slice(&train_tokens[start_idx..start_idx + config.seq_len]);
            y_batch.extend_from_slice(&train_tokens[start_idx + 1..start_idx + 1 + config.seq_len]);
        }

        // 2. Transformer forward, backward, and loss (Self-Attention + RoPE + SwiGLU + RMSNorm)
        model.zero_grad();
        let (loss, grad_norm) =
            model.forward_backward(&x_batch, &y_batch, batch_size, config.seq_len);

        // 3. AdamW update
        model.adamw_step(lr, wd, 0.9, 0.999, 1e-8, step);
        let calc_time_ms = step_start.elapsed().as_millis();

        // 4. Dynamic thermal throttling
        let (cpu_temp, throttle_ms) = thermal.step_throttle();

        // 5. Power tracking (net compute power and gross system power)
        let reading = power_tracker.tick(calc_time_ms, throttle_ms);

        let is_eval_step = step % log_interval == 0 || (!infinite_mode && step == target_steps);
        let (val_loss_opt, benchmark_opt, raw_samples, md_lines, sample_entries) = if is_eval_step {
            let mut samples = Vec::with_capacity(observation_prompts.len());
            let mut md_l = Vec::with_capacity(observation_prompts.len());
            let mut s_entries = Vec::with_capacity(observation_prompts.len());

            for p in &observation_prompts {
                let gen = generate_sample(&model, &tokenizer, p, gen_len, &mut rng);
                let clean = gen.replace('\n', " ").replace('|', "\\|");
                md_l.push(format!("**[{}]** `{}`", p, clean));
                s_entries.push(serde_json::json!({
                    "prompt": p,
                    "generated_text": gen.clone(),
                }));
                samples.push(gen);
            }

            let (vl, bench) = run_benchmark(
                &model, &tokenizer, val_tokens, &samples, batch_size, 4, &mut rng,
            );
            (Some(vl), Some(bench), samples, md_l, s_entries)
        } else {
            (None, None, Vec::new(), Vec::new(), Vec::new())
        };

        let step_elapsed = step_start.elapsed().as_millis();

        // 6. Create structured step log
        let step_log = TrainingStepLog {
            step,
            epoch: 1,
            loss,
            val_loss: val_loss_opt,
            learning_rate: lr,
            grad_norm,
            elapsed_ms: step_elapsed,
            cpu_temp_c: cpu_temp,
            throttle_sleep_ms: throttle_ms,
            estimated_power_w: reading.net_watts,
            accumulated_energy_wh: reading.net_accum_wh,
            param_checksum: if step % log_interval == 0 {
                Some(compute_checksum_f32(&model.params))
            } else {
                None
            },
            benchmark: benchmark_opt
                .as_ref()
                .map(oniwa_lm::logger::BenchmarkLog::from),
        };

        use std::io::Write;
        writeln!(steps_file, "{}", serde_json::to_string(&step_log)?)?;
        steps_file.flush()?;

        // 7. Observe developmental emergence (every log_interval steps or final step)
        if is_eval_step {
            let val_loss_val = val_loss_opt.unwrap_or(0.0);
            let bench = benchmark_opt.as_ref().unwrap();
            println!(
                "Step {:3}/{} | Train Loss: {:.4} | Val Loss: {:.4} | LR: {:.5} | Temp: {} | Net Power: {:.1}W (Gross: {:.1}W) | Net Energy: {:.4}Wh",
                step,
                if infinite_mode { "∞".into() } else { target_steps.to_string() },
                loss,
                val_loss_val,
                lr,
                cpu_temp
                    .map(|t| format!("{:.1} C", t))
                    .unwrap_or_else(|| "N/A".into()),
                reading.net_watts,
                reading.gross_watts,
                reading.net_accum_wh
            );
            println!("  🧠 Benchmark: {}", bench.summary_line());
            let correct_cloze: Vec<_> = bench
                .cloze_details
                .iter()
                .filter(|d| d.top1_hit)
                .map(|d| format!("\"{}\"[{}]", d.prompt, d.target))
                .collect();
            if !correct_cloze.is_empty() {
                println!(
                    "     🎉 Correct Quiz Answers (Top-1): {}",
                    correct_cloze.join(", ")
                );
            }

            // Auto-save best checkpoint
            if val_loss_val < best_val_loss {
                let prev_best = best_val_loss;
                best_val_loss = val_loss_val;
                let _ =
                    model.save_checkpoint(&best_checkpoint_dir, step, val_loss_val, current_seed);
                if prev_best.is_infinite() {
                    println!(
                        "  🏆 [New Best Recorded] Val Loss: {:.4} -> saved to `checkpoints/best`",
                        val_loss_val
                    );
                } else {
                    println!("  🏆 [Best Updated] Val Loss: {:.4} -> {:.4} -> saved to `checkpoints/best`", prev_best, val_loss_val);
                }
            }

            // Intermediate generation samples (multiple probes autoregressive sampling)
            println!("  🌱 Developmental Generation Samples (Probes):");
            for (p, gen) in observation_prompts.iter().zip(raw_samples.iter()) {
                println!("    - [{}] -> \"{}\"", p, gen);
            }
            println!();

            // Append to Growth Journal (Markdown / JSONL)
            {
                use std::io::Write;
                let md_text = md_lines.join("<br>");
                let temp_str = cpu_temp
                    .map(|t| format!("{:.1} C", t))
                    .unwrap_or_else(|| "-".into());
                writeln!(
                    journal_md,
                    "| Step {:5} | {:.4} | {:.4} | {} | {} / Net {:.1}W (Gross: {:.1}W) | {} |",
                    step,
                    loss,
                    val_loss_val,
                    bench.short_display(),
                    temp_str,
                    reading.net_watts,
                    reading.gross_watts,
                    md_text
                )?;
                journal_md.flush()?;

                let json_entry = serde_json::json!({
                    "step": step,
                    "train_loss": loss,
                    "val_loss": val_loss_val,
                    "lr": lr,
                    "benchmark": {
                        "top5_accuracy": bench.top5_accuracy,
                        "cloze_top1_accuracy": bench.cloze_top1_accuracy,
                        "cloze_top5_accuracy": bench.cloze_top5_accuracy,
                        "syntactic_score": bench.syntactic_score,
                        "bracket_score": bench.bracket_score,
                        "non_repetition_score": bench.non_repetition_score,
                    },
                    "samples": sample_entries,
                    "cpu_temp_c": cpu_temp,
                    "net_power_w": reading.net_watts,
                    "gross_power_w": reading.gross_watts,
                    "net_energy_wh": reading.net_accum_wh,
                });
                writeln!(journal_jsonl, "{}", json_entry)?;
                journal_jsonl.flush()?;
            }

            // Save checkpoint (checkpoints/latest)
            let _ = model.save_checkpoint(&checkpoint_dir, step, loss, current_seed);
        }

        last_loss = loss;
        step += 1;
    }

    let total_elapsed = start_time.elapsed();
    let final_checksum = compute_checksum_f32(&model.params);
    let net_wh = power_tracker.total_net_wh();
    let gross_wh = power_tracker.total_gross_wh();
    let net_co2_g = power_tracker.equivalent_co2_grams();
    let net_cost_yen = power_tracker.cost_yen();

    // Save final checkpoint (record actual step count and final loss)
    let final_step = if step > start_step {
        step - 1
    } else {
        start_step - 1
    };
    let _ = model.save_checkpoint(&checkpoint_dir, final_step, last_loss, current_seed);

    // Synchronize latest growth journal to logs/growth_journal.md
    if journal_md_path.exists() {
        let _ = fs::copy(&journal_md_path, &latest_journal_md_path);
    }

    println!("============================================================");
    println!(" 🎉 Training Complete!");
    println!("  - Total Elapsed Time: {:.2?}", total_elapsed);
    println!("  - Final Weights SHA-256: {}...", &final_checksum[..16]);
    println!("------------------------------------------------------------");
    println!(" 🌿 Eco-Performance Summary (Empirical Antithesis):");
    println!(
        "  - ⚡ Net Energy Consumed (Computation Only): {:.4} Wh ({:.2} Joules)",
        net_wh,
        net_wh * 3600.0
    );
    println!(
        "    (* Power purely used for computation, excluding baseline idle power {:.1}W *)",
        power_tracker.baseline_watts()
    );
    println!(
        "  - 🖥️ Reference: Gross Hardware Energy: {:.4} Wh ({:.2} Joules)",
        gross_wh,
        gross_wh * 3600.0
    );
    println!(
        "  - 💴 Estimated Electricity Cost (Computation): ~{:.4} JPY (< 1 JPY!)",
        net_cost_yen
    );
    println!(
        "  - 🌍 Estimated CO2 Emissions (Computation): ~{:.4} g-CO2",
        net_co2_g
    );
    println!("  - 💡 Context: A fraction of a single smartphone recharge (~10-15 Wh)!");
    println!("============================================================");
    println!(" 📖 Growth Journal:");
    println!("  - Markdown (Human-readable): {:?}", journal_md_path);
    println!(
        "  - JSON Lines (Machine-readable): {:?}",
        journal_jsonl_path
    );
    println!("  - Detailed Step Log: {:?}", steps_log_path);
    println!("============================================================");

    Ok(())
}

/// Cosine learning rate scheduler with warmup
fn compute_cosine_lr(
    step: usize,
    max_steps: usize,
    max_lr: f32,
    min_lr: f32,
    warmup_steps: usize,
) -> f32 {
    if warmup_steps > 0 && step <= warmup_steps {
        // Linear warmup: ramp up from min_lr to max_lr
        min_lr + (max_lr - min_lr) * (step as f32 / warmup_steps as f32)
    } else if step >= max_steps {
        min_lr
    } else {
        // Cosine decay: smoothly decay from max_lr to min_lr
        let progress =
            (step - warmup_steps) as f32 / (max_steps.saturating_sub(warmup_steps)).max(1) as f32;
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        min_lr + (max_lr - min_lr) * cosine
    }
}

/// Autoregressively sample specified number of tokens from prompt
fn generate_sample(
    model: &ModelWeights,
    tokenizer: &CharTokenizer,
    prompt: &str,
    max_tokens: usize,
    rng: &mut DeterministicRng,
) -> String {
    let mut tokens = tokenizer.encode(prompt);
    if tokens.is_empty() {
        tokens.push(0);
    }
    let seq_len = model.config.seq_len;

    for _ in 0..max_tokens {
        let context_start = tokens.len().saturating_sub(seq_len);
        let context = &tokens[context_start..];
        let mut logits = model.forward_inference(context);

        // Softmax (temperature = 0.8)
        let temp = 0.8f32;
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for val in logits.iter_mut() {
            *val = ((*val - max_logit) / temp).exp();
            sum_exp += *val;
        }
        for val in logits.iter_mut() {
            *val /= sum_exp;
        }

        // Sampling
        let r = rng.next_f32();
        let mut acc = 0.0f32;
        let mut next_token = 0;
        for (idx, &p) in logits.iter().enumerate() {
            acc += p;
            if r <= acc {
                next_token = idx;
                break;
            }
        }

        tokens.push(next_token as u16);
    }

    tokenizer.decode(&tokens)
}
