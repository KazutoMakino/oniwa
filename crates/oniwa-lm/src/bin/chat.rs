//! oniwa-lm Interactive Chat REPL
//!
//! Loads a trained model checkpoint and allows real-time interactive
//! text generation and experimentation directly in the terminal.

use oniwa_lm::logger::{InferenceLog, ProvenanceEvent, ProvenanceLedger};
use oniwa_lm::model::{LanguageModel, ModelConfig, ModelWeights, QuaternionModelWeights};
use oniwa_lm::reproducibility::{compute_checksum_f32, DeterministicRng};
use oniwa_lm::tokenizer::CharTokenizer;
use std::io::{self, Write};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!(" 💬 ONIWA (oniwa-lm): Interactive Chat REPL");
    println!("============================================================\n");

    let workspace_root = oniwa_lm::find_workspace_root();
    let data_dir = workspace_root.join("data");
    let logs_dir = workspace_root.join("logs");
    let base_dir = if workspace_root.join("crates/oniwa-lm/checkpoints").is_dir() {
        workspace_root.join("crates/oniwa-lm")
    } else {
        workspace_root.clone()
    };
    let vocab_path = data_dir.join("vocab.json");

    // Parse command-line arguments (--checkpoint best / latest, --quaternion)
    let args: Vec<String> = std::env::args().collect();
    let mut requested_checkpoint: Option<String> = None;
    let mut quaternion_mode = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" => {
                if let Some(val) = args.get(i + 1) {
                    requested_checkpoint = Some(val.clone());
                    i += 1;
                }
            }
            "--best" => {
                requested_checkpoint = Some("best".to_string());
            }
            "--latest" => {
                requested_checkpoint = Some("latest".to_string());
            }
            "--quaternion" => {
                quaternion_mode = true;
            }
            _ => {}
        }
        i += 1;
    }

    let (best_dir, latest_dir) = if quaternion_mode {
        (
            base_dir.join("checkpoints").join("quaternion").join("best"),
            base_dir
                .join("checkpoints")
                .join("quaternion")
                .join("latest"),
        )
    } else {
        (
            base_dir.join("checkpoints").join("best"),
            base_dir.join("checkpoints").join("latest"),
        )
    };

    let (checkpoint_dir, checkpoint_tag) = match requested_checkpoint.as_deref() {
        Some("best") => (
            best_dir,
            if quaternion_mode {
                "🏆 Best Quaternion Model (checkpoints/quaternion/best)"
            } else {
                "🏆 Best Model (checkpoints/best)"
            },
        ),
        Some("latest") => (
            latest_dir,
            if quaternion_mode {
                "⏱️ Latest Quaternion Model (checkpoints/quaternion/latest)"
            } else {
                "⏱️ Latest Model (checkpoints/latest)"
            },
        ),
        Some(custom) => (base_dir.join(custom), "📁 Custom Checkpoint"),
        None => {
            if best_dir.join("meta.json").exists() {
                (
                    best_dir,
                    if quaternion_mode {
                        "🏆 Best Quaternion Model (checkpoints/quaternion/best: auto-selected)"
                    } else {
                        "🏆 Best Model (checkpoints/best: auto-selected)"
                    },
                )
            } else {
                (
                    latest_dir,
                    if quaternion_mode {
                        "⏱️ Latest Quaternion Model (checkpoints/quaternion/latest: auto-selected)"
                    } else {
                        "⏱️ Latest Model (checkpoints/latest: auto-selected)"
                    },
                )
            }
        }
    };

    // 1. Load vocabulary table
    if !vocab_path.exists() {
        eprintln!("Error: Vocabulary file {:?} not found.", vocab_path);
        eprintln!("Please run training first with `cargo run --release --bin train`.");
        return Ok(());
    }
    let tokenizer = CharTokenizer::load_vocab(&vocab_path)?;
    println!("  - Vocab size: {} characters", tokenizer.vocab_size());

    // 2. Load checkpoint
    if !checkpoint_dir.join("meta.json").exists() {
        eprintln!(
            "Error: Checkpoint directory {:?} not found.",
            checkpoint_dir
        );
        eprintln!(
            "Please run training first with `cargo run --release --bin train` to save a model checkpoint."
        );
        return Ok(());
    }

    let meta_path = checkpoint_dir.join("meta.json");
    let config = ModelConfig::from_meta_json(&meta_path).unwrap_or_else(|_| ModelConfig {
        vocab_size: tokenizer.vocab_size(),
        seq_len: 128,
        dim: 128,
        num_layers: 4,
        num_heads: 4,
        head_dim: 32,
        ffn_dim: 256,
        ..Default::default()
    });

    let mut rng = DeterministicRng::new(12345);

    enum ChatModel {
        Real(ModelWeights),
        Quaternion(QuaternionModelWeights),
    }

    impl LanguageModel for ChatModel {
        fn config(&self) -> &ModelConfig {
            match self {
                Self::Real(m) => m.config(),
                Self::Quaternion(m) => m.config(),
            }
        }
        fn params(&self) -> &[f32] {
            match self {
                Self::Real(m) => m.params(),
                Self::Quaternion(m) => m.params(),
            }
        }
        fn zero_grad(&mut self) {
            match self {
                Self::Real(m) => m.zero_grad(),
                Self::Quaternion(m) => m.zero_grad(),
            }
        }
        fn forward_backward(&mut self, x: &[u16], y: &[u16], b: usize, t: usize) -> (f32, f32) {
            match self {
                Self::Real(m) => m.forward_backward(x, y, b, t),
                Self::Quaternion(m) => m.forward_backward(x, y, b, t),
            }
        }
        fn evaluate_loss(&self, x: &[u16], y: &[u16], b: usize, t: usize) -> f32 {
            match self {
                Self::Real(m) => m.evaluate_loss(x, y, b, t),
                Self::Quaternion(m) => m.evaluate_loss(x, y, b, t),
            }
        }
        fn evaluate_loss_and_top_k(
            &self,
            x: &[u16],
            y: &[u16],
            b: usize,
            t: usize,
            k: usize,
        ) -> (f32, f32) {
            match self {
                Self::Real(m) => m.evaluate_loss_and_top_k(x, y, b, t, k),
                Self::Quaternion(m) => m.evaluate_loss_and_top_k(x, y, b, t, k),
            }
        }
        fn forward_inference(&self, tokens: &[u16]) -> Vec<f32> {
            match self {
                Self::Real(m) => m.forward_inference(tokens),
                Self::Quaternion(m) => m.forward_inference(tokens),
            }
        }
        fn adamw_step(&mut self, lr: f32, wd: f32, beta1: f32, beta2: f32, eps: f32, step: usize) {
            match self {
                Self::Real(m) => m.adamw_step(lr, wd, beta1, beta2, eps, step),
                Self::Quaternion(m) => m.adamw_step(lr, wd, beta1, beta2, eps, step),
            }
        }
        fn save_checkpoint<P: AsRef<std::path::Path>>(
            &self,
            dir: P,
            step: usize,
            loss: f32,
            seed: u64,
        ) -> std::io::Result<()> {
            match self {
                Self::Real(m) => m.save_checkpoint(dir, step, loss, seed),
                Self::Quaternion(m) => m.save_checkpoint(dir, step, loss, seed),
            }
        }
    }

    let mut model = if quaternion_mode {
        ChatModel::Quaternion(QuaternionModelWeights::new(config.clone(), &mut rng))
    } else {
        ChatModel::Real(ModelWeights::new(config.clone(), &mut rng))
    };

    println!(
        "  🔄 Loading checkpoint: {} ({:?})",
        checkpoint_tag, checkpoint_dir
    );
    println!("  - Model specs: seq_len {}, dim {}, num_layers {}, vocab_size {}, params {} (~{:.2} M params)",
        config.seq_len, config.dim, config.num_layers, config.vocab_size, model.params().len(), model.params().len() as f32 / 1_000_000.0);
    let (step, loss, _) = match &mut model {
        ChatModel::Real(m) => m.load_checkpoint(&checkpoint_dir)?,
        ChatModel::Quaternion(m) => m.load_checkpoint(&checkpoint_dir)?,
    };
    let model_checksum = compute_checksum_f32(model.params());
    println!(
        "  ✅ Loaded successfully! (Step: {}, Loss: {:.4})",
        step, loss
    );
    println!("  - Model weights SHA-256: {}...\n", &model_checksum[..16]);

    // 3. Audit log ledger
    let mut ledger = ProvenanceLedger::open(logs_dir.join("ledger_index.jsonl"))?;

    println!("------------------------------------------------------------");
    println!(" [Usage]");
    println!("  - Enter a prompt or question and press [Enter].");
    println!("  - The model will autoregressively generate continuing text.");
    println!("  - Type 'quit' or 'exit' to quit.");
    println!("------------------------------------------------------------\n");

    let stdin = io::stdin();
    let mut input_buffer = String::new();

    loop {
        print!("oniwa-user > ");
        io::stdout().flush()?;
        input_buffer.clear();
        if stdin.read_line(&mut input_buffer)? == 0 {
            break;
        }

        let prompt = input_buffer.trim();
        if prompt.is_empty() {
            continue;
        }
        if prompt == "quit" || prompt == "exit" {
            println!("Exiting. Thank you for cultivating the garden! 🌱");
            break;
        }

        let gen_start = Instant::now();
        let temperature = 0.7f32;
        let max_tokens = 60;

        let generated_text = generate_response(
            &model,
            &tokenizer,
            prompt,
            max_tokens,
            temperature,
            &mut rng,
        );
        let duration_ms = gen_start.elapsed().as_millis();

        println!("\noniwa-lm > {}\n", generated_text);

        // Record inference event in provenance ledger
        let _ = ledger.record(&ProvenanceEvent::Inference(InferenceLog {
            timestamp_utc: oniwa_lm::logger::current_timestamp_utc(),
            prompt: prompt.to_string(),
            output_text: generated_text,
            tokens_generated: max_tokens,
            temperature,
            top_p: 0.9,
            random_seed: 12345,
            duration_ms,
            model_weights_sha256: model_checksum.clone(),
        }));
    }

    Ok(())
}

/// Autoregressively generate specified number of tokens from prompt
fn generate_response<M: LanguageModel>(
    model: &M,
    tokenizer: &CharTokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    rng: &mut DeterministicRng,
) -> String {
    let mut tokens = tokenizer.encode(prompt);
    // Restrict to vocabulary size (safety guard for vocab differences)
    tokens.retain(|&id| (id as usize) < model.config().vocab_size);
    if tokens.is_empty() {
        // Fallback to token 0 if all characters are unknown
        tokens.push(0);
    }
    let seq_len = model.config().seq_len;

    for _ in 0..max_tokens {
        let context_start = tokens.len().saturating_sub(seq_len);
        let context = &tokens[context_start..];
        let mut logits = model.forward_inference(context);

        // Softmax with temperature
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for val in logits.iter_mut() {
            *val = ((*val - max_logit) / temperature).exp();
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
