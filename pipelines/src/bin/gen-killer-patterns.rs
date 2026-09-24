//! Generate synthetic killer pattern contrastive dataset for oniwa-decide (System 1).
//!
//! Generates paired slices (1:1 clean vs. mutated) across 4 categories:
//! 0: RustCode, 1: PythonCode, 2: LegalOrTechDoc, 3: Literature.
//!
//! Clean slice:
//! - Noul = 0.0 (clean / valid)
//! - Score in [0.0, 1.0] (normalized complexity)
//! - mask_indices = []
//!
//! Mutated slice:
//! - Noul = 1.0 (anomalous / syntax corruption)
//! - Score in [0.0, 1.0] (elevated complexity: 3.5..4.5 on 1..5 scale -> 0.70..0.90)
//! - Exactly 1 targeted token mutation
//! - mask_indices = [mutated_position] (sorted, strictly ascending, unique, < written_tokens)

use oniwa_lm::reproducibility::DeterministicRng;
use oniwa_lm::tokenizer::{BpeTokenizer, CharTokenizer, Tokenizer};
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Serialize)]
struct JsonLabels {
    choice: usize,
    noul: f32,
    score: f32,
}

#[derive(Serialize)]
struct JsonPattern {
    id: String,
    text: String,
    labels: JsonLabels,
    mask_indices: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DocCategory {
    RustCode = 0,
    PythonCode = 1,
    LegalOrTechDoc = 2,
    Literature = 3,
}

struct CorpusStorage {
    rust_texts: Vec<String>,
    python_texts: Vec<String>,
    tech_texts: Vec<String>,
    lit_texts: Vec<String>,
}

impl CorpusStorage {
    fn load_from_dir<P: AsRef<Path>>(dir: P) -> std::io::Result<Self> {
        let mut rust_texts = Vec::new();
        let mut python_texts = Vec::new();
        let mut tech_texts = Vec::new();
        let mut lit_texts = Vec::new();

        if dir.as_ref().is_dir() {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("txt") {
                    let filename = path.file_name().unwrap().to_string_lossy();
                    if let Ok(content) = std::fs::read_to_string(&path) {
                        if content.trim().is_empty() {
                            continue;
                        }
                        if filename.starts_with("code_rust_") {
                            rust_texts.push(content);
                        } else if filename.starts_with("code_python_") {
                            python_texts.push(content);
                        } else if filename.starts_with("techdoc_") || filename.contains("法令") {
                            tech_texts.push(content);
                        } else {
                            lit_texts.push(content);
                        }
                    }
                }
            }
        }

        // Minimal fallback texts if directory is empty or sparse
        if rust_texts.is_empty() {
            rust_texts
                .push("fn main() {\n    let answer = 42;\n    println!(\"{answer}\");\n}\n".into());
        }
        if python_texts.is_empty() {
            python_texts.push(
                "def compute():\n    val = [x * 2 for x in range(10)]\n    return val\n".into(),
            );
        }
        if tech_texts.is_empty() {
            tech_texts.push("第一条 本規程は、安全なデータセット生成を目的とする。\n".into());
        }
        if lit_texts.is_empty() {
            lit_texts.push(
                "メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。\n".into(),
            );
        }

        Ok(Self {
            rust_texts,
            python_texts,
            tech_texts,
            lit_texts,
        })
    }

    fn sample_doc<'a>(&'a self, cat: DocCategory, rng: &mut DeterministicRng) -> &'a str {
        let list = match cat {
            DocCategory::RustCode => &self.rust_texts,
            DocCategory::PythonCode => &self.python_texts,
            DocCategory::LegalOrTechDoc => &self.tech_texts,
            DocCategory::Literature => &self.lit_texts,
        };
        let idx = (rng.next_f32() * list.len() as f32) as usize % list.len();
        &list[idx]
    }
}

/// Compute complexity normalized to [0.0, 1.0] (matching 1.0..5.0 scale normalized by 5.0)
fn compute_normalized_complexity(text: &str) -> f32 {
    let lines = text.lines().count();
    let mut max_indent = 0;
    let mut symbol_count = 0;

    for line in text.lines() {
        let leading_spaces = line.chars().take_while(|c| *c == ' ').count();
        let indent_level = leading_spaces / 4;
        max_indent = max_indent.max(indent_level);

        for c in line.chars() {
            if matches!(
                c,
                '{' | '}' | '(' | ')' | '[' | ']' | ';' | ':' | '<' | '>' | '=' | '&' | '|'
            ) {
                symbol_count += 1;
            }
        }
    }

    let symbol_ratio = (symbol_count as f32 / text.len().max(1) as f32) * 10.0;
    let raw_score = 1.0 + (max_indent as f32 * 0.6) + (symbol_ratio * 0.8) + (lines as f32 * 0.05);
    (raw_score.clamp(1.0, 5.0)) / 5.0
}

/// Mutate text at exactly one token position.
/// Modifies `tokens` directly and decodes back to text, guaranteeing
/// that token at `mut_pos` is within the resulting encoded text.
fn mutate_tokens(
    tokens: &[u16],
    category: DocCategory,
    tokenizer: &dyn Tokenizer,
    rng: &mut DeterministicRng,
) -> (String, usize) {
    let len = tokens.len();
    let pos = if len > 2 {
        1 + ((rng.next_f32() * (len - 2) as f32) as usize)
    } else {
        0
    };

    let mut mutated = tokens.to_vec();
    let vocab_size = tokenizer.vocab_size().max(1) as u16;

    let mode = (rng.next_f32() * 3.0) as usize;
    match mode {
        0 => {
            // Corrupt with category-specific typical bracket token or symbol
            let rep_char = match category {
                DocCategory::RustCode => '}',
                DocCategory::PythonCode => ':',
                _ => '?',
            };
            let rep_encoded = tokenizer.encode(&rep_char.to_string());
            if let Some(&t) = rep_encoded.first() {
                mutated[pos] = t;
            } else {
                mutated[pos] = (mutated[pos] + 7) % vocab_size;
            }
        }
        1 => {
            // Swap adjacent tokens
            if pos + 1 < len {
                mutated.swap(pos, pos + 1);
            } else if pos > 0 {
                mutated.swap(pos - 1, pos);
            } else {
                mutated[pos] = (mutated[pos] + 1) % vocab_size;
            }
        }
        _ => {
            // Inject unexpected token
            mutated[pos] = (mutated[pos] + 13) % vocab_size;
        }
    }

    let text = tokenizer.decode(&mutated);
    (text, pos)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!(" 🗡️ ONIWA: Killer Pattern Contrastive Mutation Generator");
    println!("    (System 1 Negative Sample & Contrastive Pair Synthesis)");
    println!("============================================================\n");

    let mut corpus_path = PathBuf::from("data/corpus");
    let mut output_path = PathBuf::from("data/killer_patterns.jsonl");
    let mut vocab_path = PathBuf::from("data/vocab.json");
    let mut bpe_path: Option<PathBuf> = None;
    let mut pairs = 5_000usize;
    let mut seed = 42u64;
    let max_seq_len = 128usize;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--corpus" => {
                if let Some(v) = args.get(i + 1) {
                    corpus_path = PathBuf::from(v);
                    i += 1;
                }
            }
            "--output" => {
                if let Some(v) = args.get(i + 1) {
                    output_path = PathBuf::from(v);
                    i += 1;
                }
            }
            "--vocab" => {
                if let Some(v) = args.get(i + 1) {
                    vocab_path = PathBuf::from(v);
                    i += 1;
                }
            }
            "--bpe" | "--bpe-vocab" => {
                if let Some(v) = args.get(i + 1) {
                    bpe_path = Some(PathBuf::from(v));
                    i += 1;
                }
            }
            "--pairs" => {
                if let Some(v) = args.get(i + 1) {
                    pairs = v.parse().unwrap_or(5_000);
                    i += 1;
                }
            }
            "--seed" => {
                if let Some(v) = args.get(i + 1) {
                    seed = v.parse().unwrap_or(42);
                    i += 1;
                }
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run -p oniwa-pipeline --bin gen-killer-patterns -- [OPTIONS]"
                );
                println!();
                println!("Options:");
                println!("  --corpus <DIR>       Path to corpus directory (default: data/corpus)");
                println!("  --output <FILE>      Path to output JSONL (default: data/killer_patterns.jsonl)");
                println!("  --vocab <FILE>       Path to vocab.json (default: data/vocab.json)");
                println!("  --bpe <FILE>         Path to BPE vocab (optional)");
                println!("  --pairs <COUNT>      Number of contrastive pairs to generate (default: 5000 -> 10000 records)");
                println!("  --seed <U64>         Random seed (default: 42)");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    println!("  - Corpus Dir:   {}", corpus_path.display());
    println!("  - Output Path:  {}", output_path.display());
    println!("  - Target Pairs: {} ({} total samples)", pairs, pairs * 2);
    println!("  - Random Seed:  {}", seed);

    let corpus = CorpusStorage::load_from_dir(&corpus_path)?;
    let mut rng = DeterministicRng::new(seed);

    let tokenizer: Box<dyn Tokenizer> = if let Some(ref p) = bpe_path {
        println!("  - Tokenizer:    BPE loaded from {}", p.display());
        Box::new(BpeTokenizer::load_vocab(p)?)
    } else if vocab_path.exists() {
        println!(
            "  - Tokenizer:    CharTokenizer loaded from {}",
            vocab_path.display()
        );
        Box::new(CharTokenizer::load_vocab(&vocab_path)?)
    } else {
        println!("  - Tokenizer:    CharTokenizer built from corpus samples");
        let sample_text = format!(
            "{}\n{}\n{}\n{}",
            corpus.rust_texts.first().cloned().unwrap_or_default(),
            corpus.python_texts.first().cloned().unwrap_or_default(),
            corpus.tech_texts.first().cloned().unwrap_or_default(),
            corpus.lit_texts.first().cloned().unwrap_or_default()
        );
        Box::new(CharTokenizer::build_from_text(&sample_text))
    };

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(&output_path)?;
    let mut writer = BufWriter::new(file);

    let categories = [
        DocCategory::RustCode,
        DocCategory::PythonCode,
        DocCategory::LegalOrTechDoc,
        DocCategory::Literature,
    ];

    let mut buf = vec![0u16; max_seq_len];

    for pair_idx in 0..pairs {
        let cat = categories[pair_idx % categories.len()];
        let doc = corpus.sample_doc(cat, &mut rng);

        // Extract a slice from the document
        let chars: Vec<char> = doc.chars().collect();
        let slice_char_len = (max_seq_len * 2).min(chars.len());
        let slice_raw = if chars.len() > slice_char_len {
            let start = (rng.next_f32() * (chars.len() - slice_char_len) as f32) as usize;
            chars[start..start + slice_char_len]
                .iter()
                .collect::<String>()
        } else {
            doc.to_string()
        };

        // Encode clean slice
        let mut clean_tokens = tokenizer.encode(&slice_raw);
        if clean_tokens.is_empty() {
            clean_tokens.push(1);
        }
        if clean_tokens.len() > max_seq_len {
            clean_tokens.truncate(max_seq_len);
        }

        let clean_text = tokenizer.decode(&clean_tokens);
        let clean_norm_score = compute_normalized_complexity(&clean_text);

        // 1. Clean record (Noul = 0.0, mask_indices = [])
        let clean_record = JsonPattern {
            id: format!("kp_{:05}_clean", pair_idx + 1),
            text: if clean_text.trim().is_empty() {
                "fn main() {}".into()
            } else {
                clean_text
            },
            labels: JsonLabels {
                choice: cat as usize,
                noul: 0.0,
                score: clean_norm_score,
            },
            mask_indices: vec![],
        };

        // 2. Mutated slice (Noul = 1.0, Score = 0.70..0.90, mask_indices = [pos])
        let (mutated_text, raw_mut_pos) = mutate_tokens(&clean_tokens, cat, &*tokenizer, &mut rng);
        let mutated_score = 0.70 + (rng.next_f32() * 0.20); // 3.5 ~ 4.5 / 5.0

        // Strict verification: ensure encode_into with `mutated_text` produces written > mut_pos
        buf.fill(0);
        let written = tokenizer.encode_into(&mutated_text, &mut buf);
        let verified_pos = if written > 0 && raw_mut_pos < written {
            raw_mut_pos
        } else if written > 0 {
            written - 1
        } else {
            0
        };

        let mutated_record = JsonPattern {
            id: format!("kp_{:05}_mutated", pair_idx + 1),
            text: if mutated_text.trim().is_empty() {
                "fn broken {".into()
            } else {
                mutated_text
            },
            labels: JsonLabels {
                choice: cat as usize,
                noul: 1.0,
                score: mutated_score.clamp(0.0, 1.0),
            },
            mask_indices: vec![verified_pos],
        };

        serde_json::to_writer(&mut writer, &clean_record)?;
        writer.write_all(b"\n")?;

        serde_json::to_writer(&mut writer, &mutated_record)?;
        writer.write_all(b"\n")?;
    }

    writer.flush()?;
    println!(
        "✅ Successfully generated {} patterns into {}",
        pairs * 2,
        output_path.display()
    );
    Ok(())
}
