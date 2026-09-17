//! oniwa-decide 型安全監査・推論バイナリ
//!
//! 自然言語やコードを受け取り、数ミリ秒で以下を判定・出力：
//! - Choice: 言語/文書種別 (Rust, Python, 法令/技術文書, 文学)
//! - Noul: 構文破壊・異常フラグ (True / False)
//! - Score: 構文複雑度 (1.0 〜 5.0)

use oniwa_decide::{DecisionConfig, DecisionEngine, DecisionModel};
use oniwa_lm::reproducibility::DeterministicRng;
use oniwa_lm::tokenizer::CharTokenizer;
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
    println!("    (脱データセンター・型安全意思決定センサ)");
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
    let checkpoint_dir = base_dir.join("checkpoints").join("best");

    // トークナイザの読み込み
    let vocab_path = data_dir.join("vocab.json");
    let tokenizer = if vocab_path.exists() {
        CharTokenizer::load_vocab(&vocab_path)?
    } else {
        CharTokenizer::build_from_text("abcdefghijklmnopqrstuvwxyz 0123456789")
    };

    let engine = if checkpoint_dir.join("meta.json").exists() {
        println!("  💾 チェックポイントをロード: {:?}", checkpoint_dir);
        DecisionEngine::load_from_dir(&checkpoint_dir, tokenizer)?
    } else {
        println!("  ⚠️ チェックポイント未検出: 初期化モデルで実行します（未学習）");
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
        let mut rng = DeterministicRng::new(42);
        let model = DecisionModel::new(config, &mut rng);
        DecisionEngine::new(model, tokenizer)
    };

    println!("\n🔍 入力テキスト:\n------------------------------------------------------------\n{}\n------------------------------------------------------------", text);

    let decision = engine.audit_text(&text);

    println!(
        "\n⚡ 意思決定結果 (推論時間: {} ms):",
        decision.inference_time_ms
    );
    println!("------------------------------------------------------------");
    println!(
        "  1. 🏷️ Choice [文書種別]: {:?} (確信度: {:.1}%)",
        decision.category.value,
        decision.category.confidence * 100.0
    );
    println!("     確率分布:");
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
        "\n  2. ⚠️ Noul   [構文異常]: {} (異常確率: {:.1}%, 確信度: {:.1}%)",
        if decision.syntax_anomaly.value {
            "異常あり (True)"
        } else {
            "正常 (False)"
        },
        decision.syntax_anomaly.probability * 100.0,
        decision.syntax_anomaly.confidence * 100.0
    );

    println!(
        "\n  3. 📊 Score  [構文複雑度]: {:.2} / 5.0 (確信度: {:.1}%)",
        decision.complexity.value,
        decision.complexity.confidence * 100.0
    );
    println!("============================================================");

    Ok(())
}
