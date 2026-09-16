//! 多軸評価ベンチマークシステム (Multi-Axis Evaluation Benchmark)
//!
//! 単一のValidation Loss（クロスエントロピー）だけでなく、
//! 以下の3つの観点からモデルの言語能力を定量的・定性的に測定します:
//! 1. Next-Token Top-k 精度 (Top-5 Accuracy)
//! 2. 定番穴埋めクイズ (Cloze Test Suite - 10問)
//! 3. 構文健全性スコア (括弧の対整合率 & 反復ループ抑制率)

use crate::model::ModelWeights;
use crate::reproducibility::DeterministicRng;
use crate::tokenizer::CharTokenizer;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// 穴埋めクイズの問題定義
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClozeQuestion {
    pub category: String,
    pub prompt: String,
    pub target: char,
    pub description: String,
}

/// 穴埋めクイズの個別回答結果
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

/// 多軸評価ベンチマークの総合結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Validation データセット上の Top-5 Next-Token 予測正解率 (%)
    pub top5_accuracy: f32,
    /// 穴埋めクイズの Top-1 正答率 (%)
    pub cloze_top1_accuracy: f32,
    /// 穴埋めクイズの Top-5 正答率 (%)
    pub cloze_top5_accuracy: f32,
    /// 構文健全性スコア (%) (括弧整合率 + 反復ループ抑制率)
    pub syntactic_score: f32,
    /// 括弧の対・閉じ整合率 (%)
    pub bracket_score: f32,
    /// 反復ループ抑制率 (%)
    pub non_repetition_score: f32,
    /// 穴埋めクイズの詳細結果リスト
    pub cloze_details: Vec<ClozeDetail>,
}

impl BenchmarkResult {
    /// コンソール表示用の要約文字列
    pub fn summary_line(&self) -> String {
        format!(
            "Top-5精度: {:.1}% | クイズ: {:.1}% (Top-5: {:.1}%) | 構文健全性: {:.1}% (括弧: {:.1}%, ループ抑制: {:.1}%)",
            self.top5_accuracy,
            self.cloze_top1_accuracy,
            self.cloze_top5_accuracy,
            self.syntactic_score,
            self.bracket_score,
            self.non_repetition_score
        )
    }

    /// Markdown テーブル列用の短縮表示 (Top-5 / クイズ / 構文)
    pub fn short_display(&self) -> String {
        format!(
            "{:.1}% / {:.1}% / {:.1}%",
            self.top5_accuracy, self.cloze_top1_accuracy, self.syntactic_score
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
        }
    }
}

/// 定番穴埋めクイズ 10問のデフォルト定義
pub fn default_cloze_questions() -> Vec<ClozeQuestion> {
    vec![
        ClozeQuestion {
            category: "名作文学".into(),
            prompt: "吾輩は猫で".into(),
            target: 'あ',
            description: "夏目漱石『吾輩は猫である』".into(),
        },
        ClozeQuestion {
            category: "名作文学".into(),
            prompt: "国境の長いトンネルを抜けると".into(),
            target: '雪',
            description: "川端康成『雪国』".into(),
        },
        ClozeQuestion {
            category: "名作文学".into(),
            prompt: "メロスは激怒".into(),
            target: 'し',
            description: "太宰治『走れメロス』".into(),
        },
        ClozeQuestion {
            category: "古典文学".into(),
            prompt: "祇園精舎の鐘の".into(),
            target: '声',
            description: "『平家物語』".into(),
        },
        ClozeQuestion {
            category: "古典文学".into(),
            prompt: "色は匂へど".into(),
            target: '散',
            description: "『いろは歌』".into(),
        },
        ClozeQuestion {
            category: "近代思想".into(),
            prompt: "天は人の上に人を造らず人の".into(),
            target: '下',
            description: "福沢諭吉『学問のすすめ』".into(),
        },
        ClozeQuestion {
            category: "古典文学".into(),
            prompt: "春は".into(),
            target: 'あ',
            description: "清少納言『枕草子』".into(),
        },
        ClozeQuestion {
            category: "ことわざ".into(),
            prompt: "犬も歩けば".into(),
            target: '棒',
            description: "「犬も歩けば棒に当たる」".into(),
        },
        ClozeQuestion {
            category: "ことわざ".into(),
            prompt: "猿も木から".into(),
            target: '落',
            description: "「猿も木から落ちる」".into(),
        },
        ClozeQuestion {
            category: "格言".into(),
            prompt: "時は".into(),
            target: '金',
            description: "「時は金なり」".into(),
        },
    ]
}

/// 穴埋めクイズスイートの評価
pub fn evaluate_cloze_suite(
    model: &ModelWeights,
    tokenizer: &CharTokenizer,
    questions: &[ClozeQuestion],
) -> (f32, f32, Vec<ClozeDetail>) {
    let mut top1_hits = 0;
    let mut top5_hits = 0;
    let mut details = Vec::with_capacity(questions.len());

    for q in questions {
        let tokens = tokenizer.encode(&q.prompt);
        let target_id_opt = tokenizer.char_to_id.get(&q.target).copied();

        if tokens.is_empty() || target_id_opt.is_none() {
            continue;
        }
        let target_id = target_id_opt.unwrap() as usize;

        // コンテキスト長を seq_len 以内に制限
        let context_start = tokens.len().saturating_sub(model.config.seq_len);
        let context = &tokens[context_start..];
        let logits = model.forward_inference(context);

        // 予測 Top-1 (argmax)
        let mut max_logit = f32::NEG_INFINITY;
        let mut argmax_id = 0;
        for (i, &l) in logits.iter().enumerate() {
            if l > max_logit {
                max_logit = l;
                argmax_id = i;
            }
        }
        let predicted_char = tokenizer.id_to_char.get(argmax_id).copied().unwrap_or('?');

        // 正解トークンの順位 (Rank) と確率の算出
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

/// 単一テキストの括弧整合性スコア (0.0 〜 100.0%)
pub fn compute_bracket_score(text: &str) -> f32 {
    let mut stack = Vec::new();
    let mut matched_pairs = 0usize;
    let mut unexpected_close = 0usize;

    for ch in text.chars() {
        match ch {
            '「' | '『' | '（' | '(' | '【' | '《' => {
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
            _ => {}
        }
    }
    let unmatched_open = stack.len();

    let total_bracket_events = matched_pairs * 2 + unmatched_open + unexpected_close;
    if total_bracket_events == 0 {
        100.0 // 括弧が使われていなければ文法違反なし (100点)
    } else {
        ((matched_pairs * 2) as f32 / total_bracket_events as f32) * 100.0
    }
}

/// 単一テキストの反復ループ抑制率 (0.0 〜 100.0%)
pub fn compute_non_repetition_score(text: &str) -> f32 {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 4 {
        return 100.0;
    }

    // 2-gram のユニーク比率
    let total_bigrams = chars.len() - 1;
    let mut bigrams = HashSet::new();
    for window in chars.windows(2) {
        bigrams.insert((window[0], window[1]));
    }
    let unique_ratio_2 = bigrams.len() as f32 / total_bigrams as f32;

    // 3-gram のユニーク比率
    let total_trigrams = chars.len() - 2;
    let mut trigrams = HashSet::new();
    for window in chars.windows(3) {
        trigrams.insert((window[0], window[1], window[2]));
    }
    let unique_ratio_3 = trigrams.len() as f32 / total_trigrams as f32;

    ((unique_ratio_2 + unique_ratio_3) / 2.0 * 100.0).clamp(0.0, 100.0)
}

/// 生成テキスト群の構文健全性総合評価
pub fn evaluate_syntactic_health(texts: &[String]) -> (f32, f32, f32) {
    if texts.is_empty() {
        return (100.0, 100.0, 100.0);
    }

    let mut bracket_scores = Vec::with_capacity(texts.len());
    let mut non_rep_scores = Vec::with_capacity(texts.len());

    for text in texts {
        bracket_scores.push(compute_bracket_score(text));
        non_rep_scores.push(compute_non_repetition_score(text));
    }

    let avg_bracket = bracket_scores.iter().sum::<f32>() / bracket_scores.len() as f32;
    let avg_non_rep = non_rep_scores.iter().sum::<f32>() / non_rep_scores.len() as f32;
    let combined = (avg_bracket + avg_non_rep) / 2.0;

    (combined, avg_bracket, avg_non_rep)
}

/// 多軸評価ベンチマークの総合実行
pub fn run_benchmark(
    model: &ModelWeights,
    tokenizer: &CharTokenizer,
    val_tokens: &[u16],
    generated_samples: &[String],
    batch_size: usize,
    num_eval_batches: usize,
    rng: &mut DeterministicRng,
) -> (f32, BenchmarkResult) {
    // 1. Validation Split 上での Loss と Top-5 精度
    let (val_loss, top5_accuracy) = evaluate_validation_metrics(
        model,
        val_tokens,
        model.config.seq_len,
        batch_size,
        num_eval_batches,
        5,
        rng,
    );

    // 2. 穴埋めクイズ 10問
    let questions = default_cloze_questions();
    let (cloze_top1_accuracy, cloze_top5_accuracy, cloze_details) =
        evaluate_cloze_suite(model, tokenizer, &questions);

    // 3. 構文健全性
    let (syntactic_score, bracket_score, non_repetition_score) =
        evaluate_syntactic_health(generated_samples);

    let result = BenchmarkResult {
        top5_accuracy,
        cloze_top1_accuracy,
        cloze_top5_accuracy,
        syntactic_score,
        bracket_score,
        non_repetition_score,
        cloze_details,
    };

    (val_loss, result)
}

/// 検証用データセットからミニバッチをサンプリングして Val Loss と Top-k 精度を算出
pub fn evaluate_validation_metrics(
    model: &ModelWeights,
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
    use crate::model::ModelConfig;

    #[test]
    fn test_bracket_scoring() {
        // 括弧なし -> 100%
        assert_eq!(compute_bracket_score("吾輩は猫である。名前はまだ無い。"), 100.0);

        // 正しい対の括弧 -> 100%
        assert_eq!(compute_bracket_score("メロスは「走れ！」と言った。"), 100.0);
        assert_eq!(compute_bracket_score("『草枕』（夏目漱石）"), 100.0);

        // 閉じ忘れ -> 減点
        let unclosed = compute_bracket_score("メロスは「走れ！と言った。");
        assert!(unclosed < 100.0);

        // 順序違い・不意の閉じ括弧 -> 減点
        let mismatch = compute_bracket_score("メロスは」走れ！「と言った。");
        assert!(mismatch < 100.0);
    }

    #[test]
    fn test_non_repetition_scoring() {
        // 正常な文
        let normal = compute_non_repetition_score("吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。");
        assert!(normal > 80.0);

        // 縮退した反復ループ
        let degenerate = compute_non_repetition_score("ああああああああああああああああああああああああああああああ");
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

        assert!(top1 >= 0.0 && top1 <= 100.0);
        assert!(top5 >= 0.0 && top5 <= 100.0);
        assert_eq!(details.len(), 10);
    }
}
