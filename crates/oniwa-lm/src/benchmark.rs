//! Multi-Axis Evaluation Benchmark System
//!
//! Quantitatively and qualitatively measures linguistic capabilities beyond cross-entropy validation loss:
//! 1. Next-Token Top-k accuracy (Top-5 Accuracy)
//! 2. Cloze Test Suite (Literature and code questions)
//! 3. Syntactic validity scores (bracket balance rate & repetition suppression rate)

use crate::model::LanguageModel;
use crate::reproducibility::DeterministicRng;
use crate::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Definition of a cloze test question
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClozeQuestion {
    pub category: String,
    pub prompt: String,
    pub target: char,
    pub description: String,
}

/// Individual evaluation result for a cloze test question
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClozeDetail {
    pub prompt: String,
    pub target: char,
    pub predicted: char,
    pub top1_hit: bool,
    pub top5_hit: bool,
    pub target_rank: usize,
    pub target_prob: f32,
}

/// Comprehensive results of the multi-axis benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Top-5 next-token prediction accuracy on validation dataset (%)
    pub top5_accuracy: f32,
    /// Top-1 overall accuracy across cloze tests (%)
    pub cloze_top1_accuracy: f32,
    /// Top-5 overall accuracy across cloze tests (%)
    pub cloze_top5_accuracy: f32,
    /// Syntactic validity score (%) (bracket matching rate + repetition suppression rate)
    pub syntactic_score: f32,
    /// Bracket pair matching rate (%) (all brackets)
    pub bracket_score: f32,
    /// Repetition suppression rate (%)
    pub non_repetition_score: f32,
    /// Literature cloze accuracy (%)
    #[serde(default)]
    pub lit_cloze_score: f32,
    /// Code cloze accuracy (%)
    #[serde(default)]
    pub code_cloze_score: f32,
    /// Code bracket matching rate (%) ({} () [])
    #[serde(default)]
    pub code_bracket_score: f32,
    /// Indentation matching rate (%) (4/2 spaces)
    #[serde(default)]
    pub indent_score: f32,
    /// List of detailed cloze test results
    pub cloze_details: Vec<ClozeDetail>,
}

impl BenchmarkResult {
    /// Summary string for console display
    pub fn summary_line(&self) -> String {
        format!(
            "Top-5: {:.1}% | Quiz: {:.1}% (Lit: {:.1}%, Code: {:.1}%) | Syntax: {:.1}% (Bracket: {:.1}%, CodeBracket: {:.1}%, Indent: {:.1}%)",
            self.top5_accuracy,
            self.cloze_top1_accuracy,
            self.lit_cloze_score,
            self.code_cloze_score,
            self.syntactic_score,
            self.bracket_score,
            self.code_bracket_score,
            self.indent_score,
        )
    }

    /// Short display for Markdown table columns (Top-5 / Lit / Code / Syntax)
    pub fn short_display(&self) -> String {
        format!(
            "{:.1}% / Lit:{:.1}% / Code:{:.1}% / Syntax:{:.1}%",
            self.top5_accuracy, self.lit_cloze_score, self.code_cloze_score, self.syntactic_score
        )
    }
}

impl From<&BenchmarkResult> for crate::logger::BenchmarkLog {
    fn from(r: &BenchmarkResult) -> Self {
        Self {
            top5_accuracy: r.top5_accuracy,
            cloze_top1_accuracy: r.cloze_top1_accuracy,
            cloze_top5_accuracy: r.cloze_top5_accuracy,
            syntactic_score: r.syntactic_score,
            bracket_score: r.bracket_score,
            non_repetition_score: r.non_repetition_score,
            lit_cloze_score: r.lit_cloze_score,
            code_cloze_score: r.code_cloze_score,
            code_bracket_score: r.code_bracket_score,
            indent_score: r.indent_score,
        }
    }
}

/// Default 10 classic literature cloze questions
pub fn default_cloze_questions() -> Vec<ClozeQuestion> {
    vec![
        ClozeQuestion {
            category: "Classic Literature".into(),
            prompt: "吾輩は猫で".into(),
            target: 'あ',
            description: "Natsume Soseki 'I Am a Cat'".into(),
        },
        ClozeQuestion {
            category: "Classic Literature".into(),
            prompt: "国境の長いトンネルを抜けると".into(),
            target: '雪',
            description: "Yasunari Kawabata 'Snow Country'".into(),
        },
        ClozeQuestion {
            category: "Classic Literature".into(),
            prompt: "メロスは激怒".into(),
            target: 'し',
            description: "Osamu Dazai 'Run, Melos!'".into(),
        },
        ClozeQuestion {
            category: "Classical Japanese".into(),
            prompt: "祇園精舎の鐘の".into(),
            target: '声',
            description: "'The Tale of the Heike'".into(),
        },
        ClozeQuestion {
            category: "Classical Japanese".into(),
            prompt: "色は匂へど".into(),
            target: '散',
            description: "'Iroha poem'".into(),
        },
        ClozeQuestion {
            category: "Modern Thought".into(),
            prompt: "天は人の上に人を造らず人の".into(),
            target: '下',
            description: "Yukichi Fukuzawa 'An Encouragement of Learning'".into(),
        },
        ClozeQuestion {
            category: "Classical Japanese".into(),
            prompt: "春は".into(),
            target: 'あ',
            description: "Sei Shonagon 'The Pillow Book'".into(),
        },
        ClozeQuestion {
            category: "Proverb".into(),
            prompt: "犬も歩けば".into(),
            target: '棒',
            description: "Proverb: 'A dog walking will hit a stick'".into(),
        },
        ClozeQuestion {
            category: "Proverb".into(),
            prompt: "猿も木から".into(),
            target: '落',
            description: "Proverb: 'Even monkeys fall from trees'".into(),
        },
        ClozeQuestion {
            category: "Maxim".into(),
            prompt: "時は".into(),
            target: '金',
            description: "Maxim: 'Time is money'".into(),
        },
    ]
}

/// 10 code syntax and algorithm cloze questions (Python & Rust)
pub fn code_cloze_questions() -> Vec<ClozeQuestion> {
    vec![
        ClozeQuestion {
            category: "Python Syntax".into(),
            prompt: "def fibonacci(n):\n    if n <= 1:\n        return ".into(),
            target: 'n',
            description: "Base condition return value: return n".into(),
        },
        ClozeQuestion {
            category: "Python Syntax".into(),
            prompt: "for i in range(".into(),
            target: '1',
            description: "range numeric argument".into(),
        },
        ClozeQuestion {
            category: "Python Algorithm".into(),
            prompt: "while low <= high:\n    mid = (low + high) // ".into(),
            target: '2',
            description: "Binary search midpoint calculation // 2".into(),
        },
        ClozeQuestion {
            category: "Python Data Structures".into(),
            prompt: "class Stack:\n    def __init__(self):\n        self.items = ".into(),
            target: '[',
            description: "List initialization [ ]".into(),
        },
        ClozeQuestion {
            category: "Rust Syntax".into(),
            prompt: "fn main() {\n    println".into(),
            target: '!',
            description: "Rust macro call println!".into(),
        },
        ClozeQuestion {
            category: "Rust Syntax".into(),
            prompt: "pub fn is_prime(n: ".into(),
            target: 'u',
            description: "Unsigned integer type u32/u64".into(),
        },
        ClozeQuestion {
            category: "Rust Syntax".into(),
            prompt: "let mut stack = Stack::".into(),
            target: 'n',
            description: "Constructor invocation ::new()".into(),
        },
        ClozeQuestion {
            category: "Rust Crates".into(),
            prompt: "use serde::{Serialize, ".into(),
            target: 'D',
            description: "Serde Deserialize trait".into(),
        },
        ClozeQuestion {
            category: "Rust Crates".into(),
            prompt: "let re = Regex::".into(),
            target: 'n',
            description: "Regex::new() compilation".into(),
        },
        ClozeQuestion {
            category: "Python Standard Library".into(),
            prompt: "import json\ndata = json.".into(),
            target: 'l',
            description: "json.loads() / json.load() call".into(),
        },
    ]
}

/// Combined cloze question suite (10 literature + 10 code)
pub fn all_cloze_questions() -> Vec<ClozeQuestion> {
    let mut questions = default_cloze_questions();
    questions.extend(code_cloze_questions());
    questions
}

/// Evaluate cloze test suite
pub fn evaluate_cloze_suite<M: LanguageModel, T: Tokenizer>(
    model: &M,
    tokenizer: &T,
    questions: &[ClozeQuestion],
) -> (f32, f32, Vec<ClozeDetail>) {
    let mut top1_hits = 0;
    let mut top5_hits = 0;
    let mut details = Vec::with_capacity(questions.len());

    for q in questions {
        let tokens = tokenizer.encode(&q.prompt);
        let target_str = q.target.to_string();
        let target_id_opt = tokenizer.token_to_id(&target_str);

        if tokens.is_empty() || target_id_opt.is_none() {
            continue;
        }
        let target_id = target_id_opt.unwrap() as usize;

        // Truncate context to seq_len
        let context_start = tokens.len().saturating_sub(model.config().seq_len);
        let context = &tokens[context_start..];
        let logits = model.forward_inference(context);

        // Predicted Top-1 (argmax)
        let mut max_logit = f32::NEG_INFINITY;
        let mut argmax_id = 0;
        for (i, &l) in logits.iter().enumerate() {
            if l > max_logit {
                max_logit = l;
                argmax_id = i;
            }
        }
        let predicted_char = tokenizer
            .id_to_token(argmax_id as u16)
            .and_then(|tok| tok.chars().next())
            .unwrap_or('?');

        // Calculate target token rank and probability
        let target_logit = logits.get(target_id).copied().unwrap_or(f32::NEG_INFINITY);
        let mut rank = 0;
        let mut sum_exp = 0.0f32;
        for &l in &logits {
            if l > target_logit {
                rank += 1;
            }
            sum_exp += (l - max_logit).exp();
        }
        let target_prob = if sum_exp > 0.0 {
            ((target_logit - max_logit).exp() / sum_exp).max(0.0)
        } else {
            0.0
        };

        let top1_hit = argmax_id == target_id;
        let top5_hit = rank < 5;

        if top1_hit {
            top1_hits += 1;
        }
        if top5_hit {
            top5_hits += 1;
        }

        details.push(ClozeDetail {
            prompt: q.prompt.clone(),
            target: q.target,
            predicted: predicted_char,
            top1_hit,
            top5_hit,
            target_rank: rank + 1, // 1-indexed
            target_prob,
        });
    }

    let n = details.len().max(1) as f32;
    let top1_acc = (top1_hits as f32 / n) * 100.0;
    let top5_acc = (top5_hits as f32 / n) * 100.0;
    (top1_acc, top5_acc, details)
}

/// Bracket balance score for a single text (0.0 to 100.0%)
/// Evaluates both Japanese literary brackets (「」『』（） etc.) and code brackets ({} () [])
pub fn compute_bracket_score(text: &str) -> f32 {
    let mut stack = Vec::new();
    let mut matched_pairs = 0usize;
    let mut unexpected_close = 0usize;

    for ch in text.chars() {
        match ch {
            '「' | '『' | '（' | '(' | '【' | '《' | '{' | '[' => {
                stack.push(ch);
            }
            '」' => {
                if stack.pop() == Some('「') {
                    matched_pairs += 1;
                } else {
                    unexpected_close += 1;
                }
            }
            '』' => {
                if stack.pop() == Some('『') {
                    matched_pairs += 1;
                } else {
                    unexpected_close += 1;
                }
            }
            '）' => {
                if stack.pop() == Some('（') {
                    matched_pairs += 1;
                } else {
                    unexpected_close += 1;
                }
            }
            ')' => {
                if stack.pop() == Some('(') {
                    matched_pairs += 1;
                } else {
                    unexpected_close += 1;
                }
            }
            '】' => {
                if stack.pop() == Some('【') {
                    matched_pairs += 1;
                } else {
                    unexpected_close += 1;
                }
            }
            '》' => {
                if stack.pop() == Some('《') {
                    matched_pairs += 1;
                } else {
                    unexpected_close += 1;
                }
            }
            '}' => {
                if stack.pop() == Some('{') {
                    matched_pairs += 1;
                } else {
                    unexpected_close += 1;
                }
            }
            ']' => {
                if stack.pop() == Some('[') {
                    matched_pairs += 1;
                } else {
                    unexpected_close += 1;
                }
            }
            _ => {}
        }
    }
    let unmatched_open = stack.len();

    let total_bracket_events = matched_pairs * 2 + unmatched_open + unexpected_close;
    if total_bracket_events == 0 {
        100.0 // No brackets used means no syntax violation (100%)
    } else {
        ((matched_pairs * 2) as f32 / total_bracket_events as f32) * 100.0
    }
}

/// Code-specific bracket balance score (0.0 to 100.0%) ({} () [])
pub fn compute_code_bracket_score(text: &str) -> f32 {
    let mut stack = Vec::new();
    let mut matched_pairs = 0usize;
    let mut unexpected_close = 0usize;

    for ch in text.chars() {
        match ch {
            '{' | '(' | '[' => {
                stack.push(ch);
            }
            '}' => {
                if stack.pop() == Some('{') {
                    matched_pairs += 1;
                } else {
                    unexpected_close += 1;
                }
            }
            ')' => {
                if stack.pop() == Some('(') {
                    matched_pairs += 1;
                } else {
                    unexpected_close += 1;
                }
            }
            ']' => {
                if stack.pop() == Some('[') {
                    matched_pairs += 1;
                } else {
                    unexpected_close += 1;
                }
            }
            _ => {}
        }
    }
    let unmatched_open = stack.len();
    let total = matched_pairs * 2 + unmatched_open + unexpected_close;
    if total == 0 {
        100.0
    } else {
        ((matched_pairs * 2) as f32 / total as f32) * 100.0
    }
}

/// Indentation balance rate for 4-space / 2-space indentation (0.0 to 100.0%)
pub fn compute_indent_score(text: &str) -> f32 {
    let mut total_indented_lines = 0usize;
    let mut valid_indented_lines = 0usize;

    for line in text.lines() {
        let trimmed_start = line.trim_start_matches(' ');
        let num_leading_spaces = line.len() - trimmed_start.len();
        if num_leading_spaces > 0 {
            total_indented_lines += 1;
            // Valid if multiple of 2 or 4 (standard Python/Rust indentation conventions)
            if num_leading_spaces % 2 == 0 {
                valid_indented_lines += 1;
            }
        }
    }

    if total_indented_lines == 0 {
        100.0
    } else {
        (valid_indented_lines as f32 / total_indented_lines as f32) * 100.0
    }
}

/// Repetition loop suppression rate for a single text (0.0 to 100.0%)
pub fn compute_non_repetition_score(text: &str) -> f32 {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 4 {
        return 100.0;
    }

    // Unique ratio of 2-grams
    let total_bigrams = chars.len() - 1;
    let mut bigrams = HashSet::new();
    for window in chars.windows(2) {
        bigrams.insert((window[0], window[1]));
    }
    let unique_ratio_2 = bigrams.len() as f32 / total_bigrams as f32;

    // Unique ratio of 3-grams
    let total_trigrams = chars.len() - 2;
    let mut trigrams = HashSet::new();
    for window in chars.windows(3) {
        trigrams.insert((window[0], window[1], window[2]));
    }
    let unique_ratio_3 = trigrams.len() as f32 / total_trigrams as f32;

    ((unique_ratio_2 + unique_ratio_3) / 2.0 * 100.0).clamp(0.0, 100.0)
}

/// Comprehensive syntactic validity evaluation for generated texts
/// (overall_syntax_score, all_bracket_score, loop_suppression_score, code_bracket_score, indent_score)
pub fn evaluate_syntactic_health(texts: &[String]) -> (f32, f32, f32, f32, f32) {
    if texts.is_empty() {
        return (100.0, 100.0, 100.0, 100.0, 100.0);
    }

    let mut bracket_scores = Vec::with_capacity(texts.len());
    let mut non_rep_scores = Vec::with_capacity(texts.len());
    let mut code_bracket_scores = Vec::with_capacity(texts.len());
    let mut indent_scores = Vec::with_capacity(texts.len());

    for text in texts {
        bracket_scores.push(compute_bracket_score(text));
        non_rep_scores.push(compute_non_repetition_score(text));
        code_bracket_scores.push(compute_code_bracket_score(text));
        indent_scores.push(compute_indent_score(text));
    }

    let avg_bracket = bracket_scores.iter().sum::<f32>() / bracket_scores.len() as f32;
    let avg_non_rep = non_rep_scores.iter().sum::<f32>() / non_rep_scores.len() as f32;
    let avg_code_bracket =
        code_bracket_scores.iter().sum::<f32>() / code_bracket_scores.len() as f32;
    let avg_indent = indent_scores.iter().sum::<f32>() / indent_scores.len() as f32;
    let combined = (avg_bracket + avg_non_rep + avg_code_bracket + avg_indent) / 4.0;

    (
        combined,
        avg_bracket,
        avg_non_rep,
        avg_code_bracket,
        avg_indent,
    )
}

/// Comprehensive multi-axis benchmark execution (literature + code)
pub fn run_benchmark<M: LanguageModel, T: Tokenizer>(
    model: &M,
    tokenizer: &T,
    val_tokens: &[u16],
    generated_samples: &[String],
    batch_size: usize,
    num_eval_batches: usize,
    rng: &mut DeterministicRng,
) -> (f32, BenchmarkResult) {
    // 1. Loss and Top-5 accuracy on validation split
    let (val_loss, top5_accuracy) = evaluate_validation_metrics(
        model,
        val_tokens,
        model.config().seq_len,
        batch_size,
        num_eval_batches,
        5,
        rng,
    );

    // 2. Cloze test evaluation (10 literature & 10 code questions)
    let lit_questions = default_cloze_questions();
    let code_questions = code_cloze_questions();

    let (lit_top1, _lit_top5, _lit_details) =
        evaluate_cloze_suite(model, tokenizer, &lit_questions);
    let (code_top1, _code_top5, _code_details) =
        evaluate_cloze_suite(model, tokenizer, &code_questions);

    let all_questions = all_cloze_questions();
    let (cloze_top1_accuracy, cloze_top5_accuracy, cloze_details) =
        evaluate_cloze_suite(model, tokenizer, &all_questions);

    // 3. Syntactic validity (all brackets, code brackets, indentation, repetition suppression)
    let (syntactic_score, bracket_score, non_repetition_score, code_bracket_score, indent_score) =
        evaluate_syntactic_health(generated_samples);

    let result = BenchmarkResult {
        top5_accuracy,
        cloze_top1_accuracy,
        cloze_top5_accuracy,
        syntactic_score,
        bracket_score,
        non_repetition_score,
        lit_cloze_score: lit_top1,
        code_cloze_score: code_top1,
        code_bracket_score,
        indent_score,
        cloze_details,
    };

    (val_loss, result)
}

/// Sample minibatches from validation dataset to compute validation loss and Top-k accuracy
pub fn evaluate_validation_metrics<M: LanguageModel>(
    model: &M,
    val_tokens: &[u16],
    seq_len: usize,
    batch_size: usize,
    num_eval_batches: usize,
    k: usize,
    rng: &mut DeterministicRng,
) -> (f32, f32) {
    if val_tokens.len() <= seq_len + 1 {
        return (0.0, 0.0);
    }
    let mut total_loss = 0.0f32;
    let mut total_top_k = 0.0f32;
    let max_idx = val_tokens.len() - seq_len - 1;

    for _ in 0..num_eval_batches {
        let mut x_val = Vec::with_capacity(batch_size * seq_len);
        let mut y_val = Vec::with_capacity(batch_size * seq_len);
        for _ in 0..batch_size {
            let start_idx = (rng.next_f32() * max_idx as f32) as usize;
            x_val.extend_from_slice(&val_tokens[start_idx..start_idx + seq_len]);
            y_val.extend_from_slice(&val_tokens[start_idx + 1..start_idx + 1 + seq_len]);
        }
        let (loss, top_k) = model.evaluate_loss_and_top_k(&x_val, &y_val, batch_size, seq_len, k);
        total_loss += loss;
        total_top_k += top_k;
    }

    let count = num_eval_batches.max(1) as f32;
    (total_loss / count, total_top_k / count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelConfig, ModelWeights};
    use crate::tokenizer::CharTokenizer;

    #[test]
    fn test_bracket_scoring() {
        // No brackets -> 100%
        assert_eq!(
            compute_bracket_score("吾輩は猫である。名前はまだ無い。"),
            100.0
        );

        // Correctly paired brackets -> 100%
        assert_eq!(compute_bracket_score("メロスは「走れ！」と言った。"), 100.0);
        assert_eq!(compute_bracket_score("『草枕』（夏目漱石）"), 100.0);

        // Unclosed bracket -> penalty
        let unclosed = compute_bracket_score("メロスは「走れ！と言った。");
        assert!(unclosed < 100.0);

        // Mismatched order / unexpected closing bracket -> penalty
        let mismatch = compute_bracket_score("メロスは」走れ！「と言った。");
        assert!(mismatch < 100.0);
    }

    #[test]
    fn test_non_repetition_scoring() {
        // Normal text
        let normal = compute_non_repetition_score(
            "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。",
        );
        assert!(normal > 80.0);

        // Degenerate repetition loop
        let degenerate = compute_non_repetition_score(
            "ああああああああああああああああああああああああああああああ",
        );
        assert!(degenerate < 20.0);
    }

    #[test]
    fn test_cloze_and_benchmark_execution() {
        let text = "吾輩は猫である。国境の長いトンネルを抜けると雪国であった。メロスは激怒した。祇園精舎の鐘の声。色は匂へど散りぬるを。天は人の上に人を造らず人の下に人を造らず。春はあけぼの。犬も歩けば棒に当たる。猿も木から落ちる。時は金なり。";
        let tokenizer = CharTokenizer::build_from_text(text);
        let config = ModelConfig {
            vocab_size: tokenizer.vocab_size(),
            seq_len: 16,
            dim: 16,
            num_layers: 1,
            num_heads: 1,
            head_dim: 16,
            ffn_dim: 32,
            ..Default::default()
        };
        let mut rng = DeterministicRng::new(42);
        let model = ModelWeights::new(config, &mut rng);

        let questions = default_cloze_questions();
        let (top1, top5, details) = evaluate_cloze_suite(&model, &tokenizer, &questions);

        assert!((0.0..=100.0).contains(&top1));
        assert!((0.0..=100.0).contains(&top5));
        assert_eq!(details.len(), 10);
    }

    #[test]
    fn test_code_bracket_and_indent_scoring() {
        // Valid code brackets
        assert_eq!(
            compute_code_bracket_score("fn main() { let arr = [1, 2, 3]; }"),
            100.0
        );
        // Invalid code brackets
        assert!(compute_code_bracket_score("fn main() { let arr = [1, 2, 3; }") < 100.0);

        // Valid 4-space indentation
        let clean_py =
            "def foo():\n    x = 1\n    if x > 0:\n        return True\n    return False\n";
        assert_eq!(compute_indent_score(clean_py), 100.0);

        // Odd-spaced invalid indentation
        let bad_py = "def foo():\n   x = 1\n";
        assert!(compute_indent_score(bad_py) < 100.0);

        // Verify 10 code questions are defined
        assert_eq!(code_cloze_questions().len(), 10);
        assert_eq!(all_cloze_questions().len(), 20);
    }
}
