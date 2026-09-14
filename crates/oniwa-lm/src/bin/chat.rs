//! oniwa-lm 対話チャット機能（Interactive REPL）
//!
//! 学習したチェックポイント（重み）を読み込み、ターミナル上で
//! モデルとリアルタイムに対話・文章生成の実験を行うことができます。

use oniwa_lm::logger::{InferenceLog, ProvenanceEvent, ProvenanceLedger};
use oniwa_lm::model::{ModelConfig, ModelWeights};
use oniwa_lm::reproducibility::{compute_checksum_f32, DeterministicRng};
use oniwa_lm::tokenizer::CharTokenizer;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!(" 💬 ONIWA (oniwa-lm): インタラクティブ対話チャット");
    println!("============================================================\n");

    let base_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = base_dir.join("data");
    let logs_dir = base_dir.join("logs");
    let vocab_path = data_dir.join("vocab.json");

    // コマンドライン引数の解析 (--checkpoint best / latest)
    let args: Vec<String> = std::env::args().collect();
    let mut requested_checkpoint: Option<String> = None;
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
            _ => {}
        }
        i += 1;
    }

    let best_dir = base_dir.join("checkpoints").join("best");
    let latest_dir = base_dir.join("checkpoints").join("latest");

    let (checkpoint_dir, checkpoint_tag) = match requested_checkpoint.as_deref() {
        Some("best") => (best_dir, "🏆 最良モデル (checkpoints/best)"),
        Some("latest") => (latest_dir, "⏱️ 最新モデル (checkpoints/latest)"),
        Some(custom) => (base_dir.join(custom), "📁 指定チェックポイント"),
        None => {
            if best_dir.join("meta.json").exists() {
                (best_dir, "🏆 最良モデル (checkpoints/best: 自動選択)")
            } else {
                (latest_dir, "⏱️ 最新モデル (checkpoints/latest: 自動選択)")
            }
        }
    };

    // 1. 語彙テーブルの読み込み
    if !vocab_path.exists() {
        eprintln!("Error: 語彙ファイル {:?} が見つかりません。", vocab_path);
        eprintln!("まずは `cargo run --release --bin train` で学習を実行してください。");
        return Ok(());
    }
    let tokenizer = CharTokenizer::load_vocab(&vocab_path)?;
    println!("  - 語彙サイズ: {} 文字", tokenizer.vocab_size());

    // 2. チェックポイントの読み込み
    if !checkpoint_dir.join("meta.json").exists() {
        eprintln!("Error: チェックポイント {:?} が見つかりません。", checkpoint_dir);
        eprintln!("まずは `cargo run --release --bin train` で学習を実行してモデルを保存してください。");
        return Ok(());
    }

    let config = ModelConfig {
        vocab_size: tokenizer.vocab_size(),
        seq_len: 32,
        dim: 64,
        num_layers: 2,
        num_heads: 2,
        head_dim: 32,
        ffn_dim: 128,
    };

    let mut rng = DeterministicRng::new(12345);
    let mut model = ModelWeights::new(config.clone(), &mut rng);

    println!("  🔄 チェックポイントをロード中: {} ({:?})", checkpoint_tag, checkpoint_dir);
    let (step, loss, _) = model.load_checkpoint(&checkpoint_dir)?;
    let model_checksum = compute_checksum_f32(&model.params);
    println!("  ✅ ロード完了！ (Step: {}, 損失: {:.4})", step, loss);
    println!("  - モデル重み SHA-256: {}...\n", &model_checksum[..16]);

    // 3. 監査ログ台帳
    let mut ledger = ProvenanceLedger::open(logs_dir.join("ledger_index.jsonl"))?;

    println!("------------------------------------------------------------");
    println!(" 【使い方】");
    println!("  ・冒頭の言葉や問いかけを入力して [Enter] を押してください。");
    println!("  ・モデルがそれに続く文章を自己回帰的に生成します。");
    println!("  ・終了するには 'quit' または 'exit' と入力してください。");
    println!("------------------------------------------------------------\n");

    let stdin = io::stdin();
    let mut input_buffer = String::new();

    loop {
        print!("あなた > ");
        io::stdout().flush()?;

        input_buffer.clear();
        if stdin.read_line(&mut input_buffer)? == 0 {
            break; // EOF
        }

        let prompt = input_buffer.trim();
        if prompt.is_empty() {
            continue;
        }
        if prompt == "quit" || prompt == "exit" {
            println!("終了します。家庭菜園のお手入れお疲れ様でした！🌱");
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

        // 推論イベントを系譜台帳に記録
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

/// プロンプトから指定文字数を Transformer 自己回帰生成
fn generate_response(
    model: &ModelWeights,
    tokenizer: &CharTokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    rng: &mut DeterministicRng,
) -> String {
    let mut tokens = tokenizer.encode(prompt);
    if tokens.is_empty() {
        // 未知文字ばかりの場合は先頭トークンを使用
        tokens.push(0);
    }
    let seq_len = model.config.seq_len;

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

        // サンプリング
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
