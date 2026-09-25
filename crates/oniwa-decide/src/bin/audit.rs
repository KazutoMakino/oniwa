//! oniwa-decide TypeSafe Audit & Inference Binary
//!
//! Evaluates natural language or code in sub-milliseconds and outputs:
//! - Choice: Language / Document Category (Rust, Python, Legal/Tech Doc, Literature)
//! - Noul: Syntactic Anomaly / Corruption Flag (True / False)
//! - Score: Syntactic Complexity (1.0 to 5.0)

use oniwa_decide::{DecisionConfig, DecisionEngine, DecisionModel};
use oniwa_lm::reproducibility::DeterministicRng;
use oniwa_lm::tokenizer::{BpeTokenizer, CharTokenizer, Tokenizer};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let text = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        "fn main() {\n    println!(\"Hello, oniwa-decide!\");\n}".to_string()
    };

    println!("============================================================");
    println!(" 🧭 oniwa-decide: TypeSafe System One Decision Engine");
    println!("    (Non-datacenter, Type-safe Decision Sensor)");
    println!("============================================================\n");

    let workspace_root = oniwa_lm::find_workspace_root();
    let data_dir = workspace_root.join("data");
    let base_dir = if workspace_root
        .join("crates/oniwa-decide/checkpoints")
        .is_dir()
    {
        workspace_root.join("crates/oniwa-decide")
    } else {
        workspace_root.clone()
    };

    let possible_ckpts = [
        base_dir.join("checkpoints/system1_mlm/best"),
        base_dir.join("checkpoints/full_quaternion/best"),
        base_dir.join("checkpoints/quaternion_head/best"),
        base_dir.join("checkpoints/best"),
        base_dir.join("checkpoints/standard_baseline/best"),
    ];
    let checkpoint_dir = possible_ckpts
        .iter()
        .find(|p| p.join("meta.json").exists())
        .cloned()
        .unwrap_or_else(|| base_dir.join("checkpoints/best"));

    let engine = if checkpoint_dir.join("meta.json").exists() {
        println!("  💾 Loaded checkpoint: {:?}", checkpoint_dir);
        let meta_str = std::fs::read_to_string(checkpoint_dir.join("meta.json"))?;
        let meta: serde_json::Value = serde_json::from_str(&meta_str)?;
        let vocab_size = meta["config"]["vocab_size"].as_u64().unwrap_or(0) as usize;

        let tokenizer: Box<dyn Tokenizer> = if vocab_size == 4096 {
            let bpe_path = data_dir.join("bpe_vocab.json");
            if bpe_path.exists() {
                Box::new(BpeTokenizer::load_vocab(&bpe_path)?)
            } else {
                return Err("Checkpoint requires data/bpe_vocab.json (vocab_size 4096)".into());
            }
        } else {
            let vocab_path = data_dir.join("vocab.json");
            if vocab_path.exists() {
                Box::new(CharTokenizer::load_vocab(&vocab_path)?)
            } else {
                Box::new(CharTokenizer::build_from_text(
                    "abcdefghijklmnopqrstuvwxyz 0123456789",
                ))
            }
        };

        DecisionEngine::load_from_dir_with_tokenizer(&checkpoint_dir, tokenizer)?
    } else {
        println!(
            "  ⚠️ No checkpoint detected: running with randomly initialized model (untrained)"
        );
        let vocab_path = data_dir.join("vocab.json");
        let tokenizer = if vocab_path.exists() {
            CharTokenizer::load_vocab(&vocab_path)?
        } else {
            CharTokenizer::build_from_text("abcdefghijklmnopqrstuvwxyz 0123456789")
        };
        let config = DecisionConfig {
            vocab_size: tokenizer.vocab_size(),
            ..Default::default()
        };
        let mut rng = DeterministicRng::new(42);
        let model = DecisionModel::new(config, &mut rng);
        DecisionEngine::new(model, tokenizer)
    };

    println!("\n🔍 Input Text:\n------------------------------------------------------------\n{}\n------------------------------------------------------------", text);

    let decision = engine.audit_text(&text);

    println!(
        "\n⚡ Decision Output (Inference Latency: {} ms):",
        decision.inference_time_ms
    );
    println!("------------------------------------------------------------");
    println!(
        "  1. 🏷️ Choice [Document Category]: {:?} (Confidence: {:.1}%)",
        decision.category.value,
        decision.category.confidence * 100.0
    );
    println!("     Probability Distribution:");
    println!(
        "       - Rust Code:       {:.1}%",
        decision.category.probabilities[0] * 100.0
    );
    println!(
        "       - Python Code:     {:.1}%",
        decision.category.probabilities[1] * 100.0
    );
    println!(
        "       - Legal/Tech Doc:  {:.1}%",
        decision.category.probabilities[2] * 100.0
    );
    println!(
        "       - Literature:      {:.1}%",
        decision.category.probabilities[3] * 100.0
    );

    println!(
        "\n  2. ⚠️ Noul   [Syntax Anomaly]: {} (Anomaly Prob: {:.1}%, Confidence: {:.1}%)",
        if decision.syntax_anomaly.value {
            "Anomaly Detected (True)"
        } else {
            "Normal Syntax (False)"
        },
        decision.syntax_anomaly.probability * 100.0,
        decision.syntax_anomaly.confidence * 100.0
    );

    println!(
        "\n  3. 📊 Score  [Syntax Complexity]: {:.2} / 5.0 (Confidence: {:.1}%)",
        decision.complexity.value,
        decision.complexity.confidence * 100.0
    );
    println!("============================================================");

    Ok(())
}
