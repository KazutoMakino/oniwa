//! Hardware & Latency Benchmark for oniwa-decide
//!
//! Measures inference latency distributions (p50, p95, p99),
//! Resident Set Size (RSS), energy per inference, and thermal telemetry across configurations:
//! 1. Standard Baseline (~1.26M params)
//! 2. Quaternion Head (~1.26M params, 4x head compression)
//! 3. Iso-Parameter Real Baseline (~315K params)

use oniwa_decide::{
    DecisionConfig, DecisionModel, HardwareProfileRecord, LatencyStats, MemoryProfile,
};
use oniwa_lm::power::PowerTracker;
use oniwa_lm::reproducibility::DeterministicRng;
use oniwa_lm::thermal::{ThermalConfig, ThermalController};
use oniwa_lm::tokenizer::CharTokenizer;
use std::env;
use std::fs;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!(" ⏱️ ONIWA-DECIDE: Hardware Profiling & Latency Benchmark");
    println!("    (Pure Rust, Multi-Percentile, Thermal & Energy Telemetry)");
    println!("============================================================\n");

    let args: Vec<String> = env::args().collect();
    let mut num_inferences = 100usize;
    let mut output_jsonl: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--iters" | "--num-inferences" => {
                if let Some(v) = args.get(i + 1) {
                    num_inferences = v.parse().unwrap_or(100);
                    i += 1;
                }
            }
            "--output" => {
                if let Some(v) = args.get(i + 1) {
                    output_jsonl = Some(v.clone());
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let workspace_root = oniwa_lm::find_workspace_root();
    let data_dir = workspace_root.join("data");
    let vocab_path = data_dir.join("vocab.json");
    let tokenizer = if vocab_path.exists() {
        CharTokenizer::load_vocab(&vocab_path)?
    } else {
        CharTokenizer::build_from_text("abcdefghijklmnopqrstuvwxyz 0123456789")
    };

    let sample_text = "fn main() {\n    let answer = 42;\n    println!(\"Result: {}\", answer);\n}";
    let tokens = tokenizer.encode(sample_text);

    let configs = vec![
        (
            "Standard Baseline",
            DecisionConfig::standard_baseline(tokenizer.vocab_size()),
        ),
        (
            "Quaternion Head",
            DecisionConfig::quaternion_head(tokenizer.vocab_size()),
        ),
        (
            "Iso-Parameter Real",
            DecisionConfig::iso_parameter(tokenizer.vocab_size()),
        ),
    ];

    let thermal = ThermalController::new(ThermalConfig::default());
    let mut records = Vec::new();

    println!(
        "{:<22} | {:<10} | {:<8} | {:<8} | {:<8} | {:<8} | {:<10}",
        "Configuration", "Params", "p50 (ms)", "p95 (ms)", "p99 (ms)", "Mean (ms)", "RSS (KB)"
    );
    println!("-------------------------------------------------------------------------------------------------");

    for (name, cfg) in configs {
        let mut rng = DeterministicRng::new(42);
        let model = DecisionModel::new(cfg.clone(), &mut rng);
        let mut power = PowerTracker::auto_detect();

        // Warmup (3 inferences)
        for _ in 0..3 {
            let _ = model.decide(&tokens);
        }

        let mut timings_ms = Vec::with_capacity(num_inferences);
        let bench_start = Instant::now();

        for _ in 0..num_inferences {
            let t0 = Instant::now();
            let _ = model.decide(&tokens);
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            timings_ms.push(elapsed_ms);
            let _ = power.tick(elapsed_ms as u128, 0);
        }

        let total_bench_ms = bench_start.elapsed().as_secs_f64() * 1000.0;
        let stats = LatencyStats::compute(timings_ms);
        let mem = MemoryProfile::current();
        let (cpu_temp, _) = thermal.step_throttle();

        let total_energy_joules = power.total_net_wh() * 3600.0;
        let joules_per_inf = if num_inferences > 0 {
            total_energy_joules / (num_inferences as f32)
        } else {
            0.0
        };
        let avg_power = if total_bench_ms > 0.0 {
            (total_energy_joules / (total_bench_ms as f32 / 1000.0)).max(0.0)
        } else {
            0.0
        };

        println!(
            "{:<22} | {:<10} | {:<8.2} | {:<8.2} | {:<8.2} | {:<8.2} | {:<10}",
            name,
            model.params.len(),
            stats.p50_ms,
            stats.p95_ms,
            stats.p99_ms,
            stats.mean_ms,
            mem.rss_kb,
        );

        let rec = HardwareProfileRecord {
            model_name: "oniwa-decide".to_string(),
            config_type: name.to_string(),
            params_count: model.params.len(),
            latency: stats,
            memory: mem,
            joules_per_inference: joules_per_inf,
            avg_power_watts: avg_power,
            thermal_celsius: cpu_temp,
        };
        records.push(rec);
    }

    println!("-------------------------------------------------------------------------------------------------");
    println!(
        "  ✅ Benchmark finished ({} iterations per configuration)",
        num_inferences
    );

    if let Some(out_path) = output_jsonl {
        let p = std::path::PathBuf::from(&out_path);
        if let Some(parent) = p.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let mut f_content = String::new();
        for r in &records {
            if let Ok(line) = serde_json::to_string(r) {
                f_content.push_str(&line);
                f_content.push('\n');
            }
        }
        fs::write(&p, f_content)?;
        println!("  💾 Exported hardware profiling records to {:?}", p);
    }

    Ok(())
}
