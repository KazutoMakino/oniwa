//! Self-Supervised Dataset Synthesis & Batch Generator
//!
//! Ingests diverse corpora from `data/corpus` (Rust, Python, legal/tech docs, literature),
//! and automatically synthesizes ground truth for decision tasks with zero human labeling:
//! - Choice: Corpus category (0: Rust, 1: Python, 2: Legal/Doc, 3: Literature)
//! - Noul: Bracket corruption / syntax anomaly flag (true: corrupted, false: normal)
//! - Score: Syntactic complexity (1.0 to 5.0)

use oniwa_lm::reproducibility::DeterministicRng;
use oniwa_lm::tokenizer::CharTokenizer;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DocCategory {
    RustCode = 0,
    PythonCode = 1,
    LegalOrTechDoc = 2,
    Literature = 3,
}

impl DocCategory {
    pub fn name(&self) -> &'static str {
        match self {
            Self::RustCode => "Rust Code",
            Self::PythonCode => "Python Code",
            Self::LegalOrTechDoc => "Legal/Tech Doc",
            Self::Literature => "Literature",
        }
    }
}

pub struct DecisionSample {
    pub text: String,
    pub category: DocCategory,
    pub has_syntax_anomaly: bool,
    pub complexity_score: f32,
}

pub struct DatasetGenerator {
    rust_texts: Vec<String>,
    python_texts: Vec<String>,
    tech_texts: Vec<String>,
    lit_texts: Vec<String>,
}

impl DatasetGenerator {
    pub fn load_from_corpus_dir<P: AsRef<Path>>(corpus_dir: P) -> std::io::Result<Self> {
        let mut rust_texts = Vec::new();
        let mut python_texts = Vec::new();
        let mut tech_texts = Vec::new();
        let mut lit_texts = Vec::new();

        if corpus_dir.as_ref().is_dir() {
            for entry in std::fs::read_dir(corpus_dir)? {
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

        // Minimal fallback samples
        if rust_texts.is_empty() {
            rust_texts.push("fn main() {\n    println!(\"Hello, world!\");\n}\n".into());
        }
        if python_texts.is_empty() {
            python_texts.push("def hello():\n    print('Hello, world!')\n".into());
        }
        if tech_texts.is_empty() {
            tech_texts
                .push("第一条 この法律は、個人の権利利益を保護することを目的とする。\n".into());
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

    /// Dynamically synthesize a single sample
    pub fn sample_one(&self, rng: &mut DeterministicRng, max_len: usize) -> DecisionSample {
        let cat_idx = (rng.next_f32() * 4.0) as usize % 4;
        let (category, source_list) = match cat_idx {
            0 => (DocCategory::RustCode, &self.rust_texts),
            1 => (DocCategory::PythonCode, &self.python_texts),
            2 => (DocCategory::LegalOrTechDoc, &self.tech_texts),
            _ => (DocCategory::Literature, &self.lit_texts),
        };

        let doc_idx = (rng.next_f32() * source_list.len() as f32) as usize % source_list.len();
        let doc = &source_list[doc_idx];

        // Slice snippet from source
        let chars: Vec<char> = doc.chars().collect();
        let snippet = if chars.len() > max_len {
            let start = (rng.next_f32() * (chars.len() - max_len) as f32) as usize;
            chars[start..start + max_len].iter().collect::<String>()
        } else {
            doc.clone()
        };

        // Calculate syntactic complexity (1.0 to 5.0)
        let complexity = Self::compute_complexity(&snippet);

        // Corrupt brackets / inject noise with 50% probability
        let inject_anomaly = rng.next_f32() < 0.5;
        let final_text = if inject_anomaly {
            Self::corrupt_syntax(&snippet, rng)
        } else {
            snippet
        };

        DecisionSample {
            text: final_text,
            category,
            has_syntax_anomaly: inject_anomaly,
            complexity_score: complexity,
        }
    }

    /// Compute syntactic complexity score (1.0 to 5.0)
    fn compute_complexity(text: &str) -> f32 {
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
        let score = 1.0 + (max_indent as f32 * 0.6) + (symbol_ratio * 0.8) + (lines as f32 * 0.05);
        score.clamp(1.0, 5.0)
    }

    /// Corrupt brackets and inject syntactic noise
    fn corrupt_syntax(text: &str, rng: &mut DeterministicRng) -> String {
        let mut chars: Vec<char> = text.chars().collect();
        if chars.is_empty() {
            return "fn { [ ]".into();
        }

        let mode = (rng.next_f32() * 3.0) as usize;
        match mode {
            0 => {
                // Delete matching brackets
                for c in &mut chars {
                    if matches!(*c, '{' | '}' | '(' | ')' | '[' | ']') && rng.next_f32() < 0.4 {
                        *c = ' ';
                    }
                }
            }
            1 => {
                // Swap / mismatch brackets
                for c in &mut chars {
                    if *c == '{' && rng.next_f32() < 0.5 {
                        *c = ')';
                    } else if *c == '(' && rng.next_f32() < 0.5 {
                        *c = ']';
                    }
                }
            }
            _ => {
                // Inject unclosed block
                chars.extend(" { let broken = ( ; ".chars());
            }
        }

        chars.into_iter().collect()
    }

    /// Generate batch
    pub fn generate_batch(
        &self,
        tokenizer: &CharTokenizer,
        batch_size: usize,
        seq_len: usize,
        rng: &mut DeterministicRng,
    ) -> (Vec<u16>, Vec<usize>, Vec<bool>, Vec<f32>) {
        let mut batch_tokens = Vec::with_capacity(batch_size * seq_len);
        let mut choice_labels = Vec::with_capacity(batch_size);
        let mut noul_labels = Vec::with_capacity(batch_size);
        let mut score_targets = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let sample = self.sample_one(rng, seq_len);
            let mut sample_toks = tokenizer.encode(&sample.text);
            sample_toks.resize(seq_len, 0); // Padding

            batch_tokens.extend_from_slice(&sample_toks);
            choice_labels.push(sample.category as usize);
            noul_labels.push(sample.has_syntax_anomaly);
            score_targets.push(sample.complexity_score);
        }

        (batch_tokens, choice_labels, noul_labels, score_targets)
    }
}
