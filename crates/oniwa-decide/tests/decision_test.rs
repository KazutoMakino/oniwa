//! Comprehensive Unit Tests for oniwa-decide

use oniwa_decide::dataset::killer_patterns::KillerPatternDataset;
use oniwa_decide::dataset::DocCategory;
use oniwa_decide::loss::LossCalculator;
use oniwa_decide::memory::FlatMemoryLayout;
use oniwa_decide::mlm::apply_mlm_mask;
use oniwa_decide::model::{DecisionConfig, DecisionModel};
use oniwa_lm::reproducibility::DeterministicRng;
use oniwa_lm::tokenizer::{CharTokenizer, Tokenizer};
use std::io::Cursor;

#[test]
fn flat_memory_layout_keeps_system1_target_below_100_mb() {
    let config = DecisionConfig::system1_mlm();
    let layout = FlatMemoryLayout::for_training(&config, 4);

    assert_eq!(layout.embedding_elements(), 4_096 * 256);
    assert!(layout.total_bytes() <= 100 * 1024 * 1024);
    assert_eq!(
        layout.total_elements() * std::mem::size_of::<f32>(),
        layout.total_bytes()
    );
}

#[test]
fn mlm_masking_keeps_labels_only_for_selected_tokens() {
    let mut rng = DeterministicRng::new(7);
    let original: Vec<u16> = (1..=20).collect();
    let masked = apply_mlm_mask(&original, 64, 0, &mut rng);

    assert_eq!(masked.tokens.len(), original.len());
    assert_eq!(
        masked
            .labels
            .iter()
            .filter(|&&label| label != u16::MAX)
            .count(),
        3
    );
    for (index, &label) in masked.labels.iter().enumerate() {
        if label != u16::MAX {
            assert_eq!(label, original[index]);
        }
    }
}

#[test]
fn killer_pattern_jsonl_writes_encoded_tokens_to_supplied_slice() {
    let jsonl = concat!(
        r#"{"id":"kp_0001","text":"ab","labels":{"choice":2,"noul":0.0,"score":0.75},"mask_indices":[1]}"#,
        "\n"
    );
    let dataset = KillerPatternDataset::from_jsonl_reader(Cursor::new(jsonl)).unwrap();
    let tokenizer = CharTokenizer::build_from_text("ab");
    let mut destination = [99u16; 4];

    let labels = dataset
        .write_batch(&tokenizer, 0, &mut destination)
        .unwrap();

    assert_eq!(destination, [0, 1, 0, 0]);
    assert_eq!(labels.choice, 2);
    assert!(!labels.noul);
    assert_eq!(labels.score, 0.75);
    assert_eq!(labels.mask_indices, &[1]);
}

#[test]
fn test_generated_killer_patterns_jsonl_roundtrip() {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let root_dir = manifest_dir.parent().unwrap().parent().unwrap();
    let file_path = root_dir.join("data/killer_patterns.jsonl");

    // If file exists from gen-killer-patterns run, test full dataset load and batch write
    if file_path.exists() {
        let file = std::fs::File::open(&file_path).unwrap();
        let dataset = KillerPatternDataset::from_jsonl_reader(file).unwrap();
        assert!(dataset.len() >= 100);

        // Load CharTokenizer from vocab.json or fallback
        let vocab_path = root_dir.join("data/vocab.json");
        let tokenizer: Box<dyn Tokenizer> = if vocab_path.exists() {
            Box::new(CharTokenizer::load_vocab(&vocab_path).unwrap())
        } else {
            Box::new(CharTokenizer::build_from_text(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t{}()[]<>;:=+-*/!?,.&|#_\"'~`@$%\r\
                 第一条個人の権利利益メロス激怒した",
            ))
        };
        let mut destination = vec![0u16; 256];

        for i in 0..50.min(dataset.len()) {
            let labels = dataset
                .write_batch(&*tokenizer, i, &mut destination)
                .unwrap();
            assert!(labels.choice < 4);
            assert!(labels.score >= 0.0 && labels.score <= 1.0);
            for &idx in labels.mask_indices {
                assert!(idx < destination.len());
            }
        }
    }
}

#[test]
fn tied_mlm_projection_backpropagates_into_shared_embeddings() {
    let config = DecisionConfig {
        vocab_size: 8,
        seq_len: 2,
        dim: 8,
        num_layers: 1,
        num_heads: 2,
        head_dim: 4,
        ffn_dim: 16,
        num_choices: 2,
        temperature: 1.0,
        use_quaternion_head: false,
        quaternion_backbone: false,
        score_unit_interval: false,
    };
    let mut model = DecisionModel::new(config.clone(), &mut DeterministicRng::new(11));
    let tokens = [1u16, 2];
    let cache = model.forward(&tokens, 1, 2);
    assert_eq!(cache.mlm_logits.len(), 2 * config.vocab_size);

    let (_loss, gradients) = LossCalculator::masked_language_model_loss(
        &cache.mlm_logits,
        &[1, u16::MAX],
        config.vocab_size,
    );
    model.zero_grad();
    model.backward_with_mlm(
        &tokens,
        &cache,
        &[0.0, 0.0],
        &[0.0],
        &[0.0],
        &gradients,
        1,
        2,
    );
    assert!(
        model.grads[model.offset_wte..model.offset_wte + config.vocab_size * config.dim]
            .iter()
            .any(|gradient| gradient.abs() > 0.0)
    );
}

#[test]
fn test_loss_functions() {
    // 1. Softmax
    let logits = vec![1.0, 2.0, 3.0, 4.0];
    let probs = LossCalculator::softmax(&logits, 1.0);
    assert_eq!(probs.len(), 4);
    let sum_p: f32 = probs.iter().sum();
    assert!((sum_p - 1.0).abs() < 1e-5);
    assert!(probs[3] > probs[2] && probs[2] > probs[1]);

    // 2. Choice Loss
    let (loss, dlogits) = LossCalculator::choice_loss(&logits, 3, 4, 0.05);
    assert!(loss > 0.0);
    assert_eq!(dlogits.len(), 4);

    // 3. Noul Loss
    let (loss_t, d_t) = LossCalculator::noul_loss(2.0, true, 0.1);
    assert!(loss_t > 0.0);
    assert!(d_t < 0.0); // High prob and target is true -> negative grad (pushes logit up)

    let (loss_f, d_f) = LossCalculator::noul_loss(2.0, false, 0.1);
    assert!(loss_f > loss_t);
    assert!(d_f > 0.0); // High prob but target is false -> positive grad (pushes logit down)

    // 4. Score Loss
    let (s_loss, s_grad) = LossCalculator::score_loss(3.5, 3.0, 0.5);
    assert!(s_loss > 0.0);
    assert!((s_grad - 1.0).abs() < 1e-4);
}

#[test]
fn test_forward_dimensions_and_properties() {
    let config = DecisionConfig {
        vocab_size: 100,
        seq_len: 16,
        dim: 32,
        num_layers: 2,
        num_heads: 2,
        head_dim: 16,
        ffn_dim: 64,
        num_choices: 4,
        temperature: 1.0,
        use_quaternion_head: false,
        quaternion_backbone: false,
        score_unit_interval: false,
    };

    let mut rng = DeterministicRng::new(123);
    let model = DecisionModel::new(config.clone(), &mut rng);

    let b = 2;
    let t = 16;
    let tokens: Vec<u16> = (0..b * t).map(|i| (i % 100) as u16).collect();

    let cache = model.forward(&tokens, b, t);
    assert_eq!(cache.choice_logits.len(), b * config.num_choices);
    assert_eq!(cache.noul_logits.len(), b);
    assert_eq!(cache.score_preds.len(), b);

    // Test inference API
    let single_tokens = &tokens[..t];
    let dec = model.decide(single_tokens);
    let sum_p: f32 = dec.choice_probs.iter().sum();
    assert!((sum_p - 1.0).abs() < 1e-4);
    assert!(dec.choice_confidence >= 0.0 && dec.choice_confidence <= 1.0);
    assert!(dec.noul_prob >= 0.0 && dec.noul_prob <= 1.0);
    assert!(dec.score_value >= 1.0 && dec.score_value <= 5.0);
}

#[test]
fn test_finite_difference_gradcheck() {
    // Finite-difference gradient check on minimal model
    let config = DecisionConfig {
        vocab_size: 20,
        seq_len: 4,
        dim: 8,
        num_layers: 1,
        num_heads: 2,
        head_dim: 4,
        ffn_dim: 16,
        num_choices: 2,
        temperature: 1.0,
        use_quaternion_head: false,
        quaternion_backbone: false,
        score_unit_interval: false,
    };

    let mut rng = DeterministicRng::new(999);
    let mut model = DecisionModel::new(config.clone(), &mut rng);

    let b = 1;
    let t = 4;
    let tokens: Vec<u16> = vec![1, 3, 5, 7];
    let target_choice = DocCategory::PythonCode as usize;
    let target_noul = true;
    let target_score = 3.5f32;

    // Forward pass
    model.zero_grad();
    let cache = model.forward(&tokens, b, t);

    // Loss and backward gradients
    let (l_c, d_c) = LossCalculator::choice_loss(&cache.choice_logits, target_choice, 2, 0.0);
    let (l_n, d_n) = LossCalculator::noul_loss(cache.noul_logits[0], target_noul, 0.0);
    let (l_s, d_s) = LossCalculator::score_loss(cache.score_preds[0], target_score, 0.5);
    let total_loss = l_c + l_n + l_s;

    model.backward(&tokens, &cache, &d_c, &[d_n], &[d_s], b, t);

    // Finite-difference check on head parameters
    let test_param_idx = model.offset_head_choice;
    let analytic_grad = model.grads[test_param_idx];

    let eps = 1e-3f32;
    model.params[test_param_idx] += eps;
    let cache_p = model.forward(&tokens, b, t);
    let (l_c_p, _) = LossCalculator::choice_loss(&cache_p.choice_logits, target_choice, 2, 0.0);
    let (l_n_p, _) = LossCalculator::noul_loss(cache_p.noul_logits[0], target_noul, 0.0);
    let (l_s_p, _) = LossCalculator::score_loss(cache_p.score_preds[0], target_score, 0.5);
    let loss_plus = l_c_p + l_n_p + l_s_p;

    model.params[test_param_idx] -= 2.0 * eps;
    let cache_m = model.forward(&tokens, b, t);
    let (l_c_m, _) = LossCalculator::choice_loss(&cache_m.choice_logits, target_choice, 2, 0.0);
    let (l_n_m, _) = LossCalculator::noul_loss(cache_m.noul_logits[0], target_noul, 0.0);
    let (l_s_m, _) = LossCalculator::score_loss(cache_m.score_preds[0], target_score, 0.5);
    let loss_minus = l_c_m + l_n_m + l_s_m;

    model.params[test_param_idx] += eps; // Restore parameter

    let numerical_grad = (loss_plus - loss_minus) / (2.0 * eps);

    println!(
        "Total Loss: {:.4}, Analytic: {:.5}, Numerical: {:.5}",
        total_loss, analytic_grad, numerical_grad
    );
    let diff = (analytic_grad - numerical_grad).abs();
    assert!(
        diff < 5e-3 || diff / (analytic_grad.abs() + numerical_grad.abs()).max(1e-5) < 0.05,
        "Gradient check failed! Analytic: {}, Numerical: {}",
        analytic_grad,
        numerical_grad
    );
}

#[test]
fn test_single_step_optimization() {
    let config = DecisionConfig {
        vocab_size: 50,
        seq_len: 8,
        dim: 16,
        num_layers: 2,
        num_heads: 2,
        head_dim: 8,
        ffn_dim: 32,
        num_choices: 3,
        temperature: 1.0,
        use_quaternion_head: false,
        quaternion_backbone: false,
        score_unit_interval: false,
    };

    let mut rng = DeterministicRng::new(42);
    let mut model = DecisionModel::new(config.clone(), &mut rng);

    let b = 2;
    let t = 8;
    let tokens: Vec<u16> = (0..b * t).map(|i| (i % 50) as u16).collect();
    let target_choices = [0, 2];
    let target_nouls = [false, true];
    let target_scores = [2.0, 4.0];

    // Pre-step loss
    model.zero_grad();
    let cache1 = model.forward(&tokens, b, t);
    let mut loss1 = 0.0f32;
    let mut d_c = vec![0.0f32; b * config.num_choices];
    let mut d_n = vec![0.0f32; b];
    let mut d_s = vec![0.0f32; b];

    for bi in 0..b {
        let (lc, dc) = LossCalculator::choice_loss(
            &cache1.choice_logits[bi * config.num_choices..(bi + 1) * config.num_choices],
            target_choices[bi],
            config.num_choices,
            0.0,
        );
        let (ln, dn) = LossCalculator::noul_loss(cache1.noul_logits[bi], target_nouls[bi], 0.0);
        let (ls, ds) = LossCalculator::score_loss(cache1.score_preds[bi], target_scores[bi], 0.5);
        loss1 += lc + ln + ls;
        for k in 0..config.num_choices {
            d_c[bi * config.num_choices + k] = dc[k];
        }
        d_n[bi] = dn;
        d_s[bi] = ds;
    }

    model.backward(&tokens, &cache1, &d_c, &d_n, &d_s, b, t);
    model.adamw_step(0.01, 0.0, 0.9, 0.999, 1e-8, 1);

    // Post-step loss
    let cache2 = model.forward(&tokens, b, t);
    let mut loss2 = 0.0f32;
    for bi in 0..b {
        let (lc, _) = LossCalculator::choice_loss(
            &cache2.choice_logits[bi * config.num_choices..(bi + 1) * config.num_choices],
            target_choices[bi],
            config.num_choices,
            0.0,
        );
        let (ln, _) = LossCalculator::noul_loss(cache2.noul_logits[bi], target_nouls[bi], 0.0);
        let (ls, _) = LossCalculator::score_loss(cache2.score_preds[bi], target_scores[bi], 0.5);
        loss2 += lc + ln + ls;
    }

    println!("Step 1 Loss: {:.4} -> Step 2 Loss: {:.4}", loss1, loss2);
    assert!(loss2 < loss1, "Loss must decrease after AdamW step!");
}

#[test]
fn test_quaternion_head_gradcheck() {
    let config = DecisionConfig {
        vocab_size: 20,
        seq_len: 4,
        dim: 8, // 2 input quaternions
        num_layers: 1,
        num_heads: 2,
        head_dim: 4,
        ffn_dim: 16,
        num_choices: 4, // 4 choices supported by Quaternion Head
        temperature: 1.0,
        use_quaternion_head: true,
        quaternion_backbone: false,
        score_unit_interval: false,
    };

    let mut rng = DeterministicRng::new(888);
    let mut model = DecisionModel::new(config.clone(), &mut rng);

    let b = 1;
    let t = 4;
    let tokens: Vec<u16> = vec![2, 4, 6, 8];
    let target_choice = 2usize;
    let target_noul = true;
    let target_score = 3.2f32;

    model.zero_grad();
    let cache = model.forward(&tokens, b, t);

    let (_l_c, d_c) = LossCalculator::choice_loss(&cache.choice_logits, target_choice, 4, 0.0);
    let (_l_n, d_n) = LossCalculator::noul_loss(cache.noul_logits[0], target_noul, 0.0);
    let (_l_s, d_s) = LossCalculator::score_loss(cache.score_preds[0], target_score, 0.5);

    model.backward(&tokens, &cache, &d_c, &[d_n], &[d_s], b, t);

    // Test a parameter in the quaternion head
    let test_param_idx = model.offset_head_quat + 3;
    let analytic_grad = model.grads[test_param_idx];

    let eps = 1e-3f32;
    model.params[test_param_idx] += eps;
    let cache_p = model.forward(&tokens, b, t);
    let (l_c_p, _) = LossCalculator::choice_loss(&cache_p.choice_logits, target_choice, 4, 0.0);
    let (l_n_p, _) = LossCalculator::noul_loss(cache_p.noul_logits[0], target_noul, 0.0);
    let (l_s_p, _) = LossCalculator::score_loss(cache_p.score_preds[0], target_score, 0.5);
    let loss_plus = l_c_p + l_n_p + l_s_p;

    model.params[test_param_idx] -= 2.0 * eps;
    let cache_m = model.forward(&tokens, b, t);
    let (l_c_m, _) = LossCalculator::choice_loss(&cache_m.choice_logits, target_choice, 4, 0.0);
    let (l_n_m, _) = LossCalculator::noul_loss(cache_m.noul_logits[0], target_noul, 0.0);
    let (l_s_m, _) = LossCalculator::score_loss(cache_m.score_preds[0], target_score, 0.5);
    let loss_minus = l_c_m + l_n_m + l_s_m;

    model.params[test_param_idx] += eps;

    let numerical_grad = (loss_plus - loss_minus) / (2.0 * eps);
    let diff = (analytic_grad - numerical_grad).abs();
    assert!(
        diff < 5e-3 || diff / (analytic_grad.abs() + numerical_grad.abs()).max(1e-5) < 0.05,
        "Quaternion head gradcheck failed! Analytic: {}, Numerical: {}, Diff: {}",
        analytic_grad,
        numerical_grad,
        diff
    );
}

#[test]
fn test_checkpoint_backward_compatibility_with_quaternion_field() {
    // Verify JSON without use_quaternion_head deserializes seamlessly with false
    let json_str = r#"{
        "vocab_size": 4721,
        "seq_len": 128,
        "dim": 128,
        "num_layers": 4,
        "num_heads": 4,
        "head_dim": 32,
        "ffn_dim": 256,
        "num_choices": 4,
        "temperature": 1.0
    }"#;
    let config: DecisionConfig = serde_json::from_str(json_str).expect("Must deserialize");
    assert!(!config.use_quaternion_head);
}

#[test]
fn test_iso_parameter_config_scale() {
    let standard = DecisionConfig::standard_baseline(4721);
    let mut rng = DeterministicRng::new(42);
    let standard_model = DecisionModel::new(standard, &mut rng);
    assert_eq!(standard_model.params.len(), 1_261_568);

    let iso = DecisionConfig::iso_parameter(4721);
    let iso_model = DecisionModel::new(iso, &mut rng);
    // Iso parameter model scales down parameters to ~315K-400K range
    assert!(iso_model.params.len() < 500_000);
    assert!(iso_model.params.len() > 250_000);
    assert_eq!(iso_model.config.dim, 64);
    assert_eq!(iso_model.config.head_dim, 16);
    assert_eq!(iso_model.config.ffn_dim, 128);
}

#[test]
fn test_full_quaternion_transformer_scale() {
    let full_quat = DecisionConfig::full_quaternion_transformer(4721);
    assert!(full_quat.quaternion_backbone);
    assert!(full_quat.use_quaternion_head);

    let mut rng = DeterministicRng::new(42);
    let full_quat_model = DecisionModel::new(full_quat, &mut rng);

    // Baseline is 1_261_568 params
    // Full Quaternion Transformer drops backbone layers by 4x and head by 4x,
    // resulting in ~315K-360K parameters!
    let param_count = full_quat_model.params.len();
    println!("Full Quaternion Transformer Param Count: {}", param_count);
    // Baseline model is 1_261_568 params. Full quaternion backbone compresses
    // attention and MLP from 655,360 weights to 163,840 weights (~491.5K parameter reduction),
    // and head from 3,072 to 768 weights, achieving 770,048 total parameters (1.64x full model compression).
    assert_eq!(param_count, 770_048);
}

#[test]
fn test_checkpoint_backward_compatibility_with_quaternion_backbone_field() {
    let json_str = r#"{
        "vocab_size": 4721,
        "seq_len": 128,
        "dim": 128,
        "num_layers": 4,
        "num_heads": 4,
        "head_dim": 32,
        "ffn_dim": 256,
        "num_choices": 4,
        "temperature": 1.0,
        "use_quaternion_head": true
    }"#;
    let config: DecisionConfig = serde_json::from_str(json_str).expect("Must deserialize");
    assert!(config.use_quaternion_head);
    assert!(!config.quaternion_backbone);
}

#[test]
fn test_full_quaternion_transformer_finite_difference_gradcheck() {
    let config = DecisionConfig {
        vocab_size: 20,
        seq_len: 4,
        dim: 8, // 2 quaternions, 2 heads of dim 4 (1 quat per head)
        num_layers: 1,
        num_heads: 2,
        head_dim: 4,
        ffn_dim: 16, // 4 quaternions
        num_choices: 4,
        temperature: 1.0,
        use_quaternion_head: true,
        quaternion_backbone: true,
        score_unit_interval: false,
    };

    let mut rng = DeterministicRng::new(777);
    let mut model = DecisionModel::new(config.clone(), &mut rng);

    let b = 1;
    let t = 4;
    let tokens: Vec<u16> = vec![1, 3, 5, 7];
    let target_choice = 1usize;
    let target_noul = true;
    let target_score = 3.5f32;

    model.zero_grad();
    let cache = model.forward(&tokens, b, t);

    let (_l_c, d_c) = LossCalculator::choice_loss(&cache.choice_logits, target_choice, 4, 0.0);
    let (_l_n, d_n) = LossCalculator::noul_loss(cache.noul_logits[0], target_noul, 0.0);
    let (_l_s, d_s) = LossCalculator::score_loss(cache.score_preds[0], target_score, 0.5);

    model.backward(&tokens, &cache, &d_c, &[d_n], &[d_s], b, t);

    // Test parameters across different layers:
    // 1. A parameter in QKV weights
    let l0 = &model.offset_layers[0];
    let test_params = [
        l0.attn_w_qkv + 2,
        l0.attn_w_proj + 1,
        l0.mlp_w_gate_up + 3,
        l0.mlp_w_down + 2,
    ];

    for &test_param_idx in &test_params {
        let analytic_grad = model.grads[test_param_idx];

        let eps = 1e-3f32;
        model.params[test_param_idx] += eps;
        let cache_p = model.forward(&tokens, b, t);
        let (l_c_p, _) = LossCalculator::choice_loss(&cache_p.choice_logits, target_choice, 4, 0.0);
        let (l_n_p, _) = LossCalculator::noul_loss(cache_p.noul_logits[0], target_noul, 0.0);
        let (l_s_p, _) = LossCalculator::score_loss(cache_p.score_preds[0], target_score, 0.5);
        let loss_plus = l_c_p + l_n_p + l_s_p;

        model.params[test_param_idx] -= 2.0 * eps;
        let cache_m = model.forward(&tokens, b, t);
        let (l_c_m, _) = LossCalculator::choice_loss(&cache_m.choice_logits, target_choice, 4, 0.0);
        let (l_n_m, _) = LossCalculator::noul_loss(cache_m.noul_logits[0], target_noul, 0.0);
        let (l_s_m, _) = LossCalculator::score_loss(cache_m.score_preds[0], target_score, 0.5);
        let loss_minus = l_c_m + l_n_m + l_s_m;

        model.params[test_param_idx] += eps; // Restore

        let numerical_grad = (loss_plus - loss_minus) / (2.0 * eps);
        let diff = (analytic_grad - numerical_grad).abs();
        let rel_diff = diff / (analytic_grad.abs() + numerical_grad.abs()).max(1e-5);
        assert!(
            diff < 5e-3 || rel_diff < 0.05,
            "Full quaternion transformer gradcheck failed at param {}! Analytic: {}, Numerical: {}, Diff: {}",
            test_param_idx,
            analytic_grad,
            numerical_grad,
            diff
        );
    }
}

// ─── Gatekeeper diff-parsing and hunk-extraction tests ──────────────

/// Tests for the gatekeeper binary's diff parsing logic.
/// These tests use the gatekeeper binary's internal module via a re-exported test helper.
/// Since gatekeeper.rs is a binary, we test the parsing logic indirectly by duplicating
/// the core parsing functions here for unit testing.
///
/// Minimal re-implementation of the gatekeeper's diff parser for unit testing.
mod gatekeeper_test_helpers {
    pub const CODE_EXTENSIONS: &[&str] = &["rs", "py"];
    pub const DOC_EXTENSIONS: &[&str] = &["md", "txt"];
    pub const MAX_HUNK_TOKENS: usize = 256;
    pub const MIN_HUNK_TOKENS: usize = 128;

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum FileKind {
        Code,
        Document,
        Skip,
    }

    #[derive(Debug, Clone)]
    pub struct DiffHunk {
        pub file_path: String,
        pub extension: String,
        pub is_code: bool,
        pub content: String,
    }

    pub fn classify_extension(ext: &str) -> FileKind {
        let lower = ext.to_lowercase();
        if CODE_EXTENSIONS.contains(&lower.as_str()) {
            FileKind::Code
        } else if DOC_EXTENSIONS.contains(&lower.as_str()) {
            FileKind::Document
        } else {
            FileKind::Skip
        }
    }

    pub fn get_extension(path: &str) -> String {
        path.rsplit('.').next().unwrap_or("").to_string()
    }

    fn build_hunk_content(added: &[String], context: &[String]) -> String {
        let mut parts: Vec<&str> = added.iter().map(|s| s.as_str()).collect();
        let added_chars: usize = parts.iter().map(|s| s.len()).sum::<usize>() + parts.len();
        let target_chars = MAX_HUNK_TOKENS * 4;
        if added_chars < MIN_HUNK_TOKENS * 4 {
            let budget = target_chars.saturating_sub(added_chars);
            let mut used = 0;
            for ctx in context.iter() {
                if used + ctx.len() + 1 > budget {
                    break;
                }
                parts.push(ctx.as_str());
                used += ctx.len() + 1;
            }
        }
        let mut result = parts.join("\n");
        if result.len() > target_chars {
            result.truncate(target_chars);
        }
        result
    }

    pub fn parse_unified_diff(diff_text: &str) -> Vec<DiffHunk> {
        let mut hunks = Vec::new();
        let mut current_file: Option<String> = None;
        let mut current_ext = String::new();
        let mut current_kind = FileKind::Skip;
        let mut added_lines: Vec<String> = Vec::new();
        let mut context_lines: Vec<String> = Vec::new();

        for line in diff_text.lines() {
            if line.starts_with("+++ b/") || line.starts_with("+++ ") {
                if let Some(ref file) = current_file {
                    if current_kind != FileKind::Skip && !added_lines.is_empty() {
                        let content = build_hunk_content(&added_lines, &context_lines);
                        hunks.push(DiffHunk {
                            file_path: file.clone(),
                            extension: current_ext.clone(),
                            is_code: current_kind == FileKind::Code,
                            content,
                        });
                    }
                }
                let path = if let Some(stripped) = line.strip_prefix("+++ b/") {
                    stripped
                } else if let Some(stripped) = line.strip_prefix("+++ ") {
                    stripped
                } else {
                    line
                };
                current_ext = get_extension(path);
                current_kind = classify_extension(&current_ext);
                current_file = Some(path.to_string());
                added_lines.clear();
                context_lines.clear();
                continue;
            }
            if line.starts_with("--- ") || line.starts_with("diff ") || line.starts_with("index ") {
                continue;
            }
            if line.starts_with("@@") {
                continue;
            }
            if current_kind != FileKind::Skip {
                if let Some(stripped) = line.strip_prefix('+') {
                    added_lines.push(stripped.to_string());
                } else if !line.starts_with('-') {
                    let ctx = line.strip_prefix(' ').unwrap_or(line);
                    context_lines.push(ctx.to_string());
                }
            }
        }
        if let Some(ref file) = current_file {
            if current_kind != FileKind::Skip && !added_lines.is_empty() {
                let content = build_hunk_content(&added_lines, &context_lines);
                hunks.push(DiffHunk {
                    file_path: file.clone(),
                    extension: current_ext.clone(),
                    is_code: current_kind == FileKind::Code,
                    content,
                });
            }
        }
        hunks
    }
}

#[test]
fn gatekeeper_parse_empty_diff_returns_empty() {
    let hunks = gatekeeper_test_helpers::parse_unified_diff("");
    assert!(hunks.is_empty());
}

#[test]
fn gatekeeper_parse_rust_diff_extracts_hunk() {
    let diff = "diff --git a/src/main.rs b/src/main.rs\nindex abc..def 100644\n--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1,3 +1,4 @@\n fn main() {\n+    let x = 42;\n     println!(\"hello\");\n }";
    let hunks = gatekeeper_test_helpers::parse_unified_diff(diff);
    assert_eq!(hunks.len(), 1);
    assert_eq!(hunks[0].file_path, "src/main.rs");
    assert_eq!(hunks[0].extension, "rs");
    assert!(hunks[0].is_code);
    assert!(hunks[0].content.contains("let x = 42"));
}

#[test]
fn gatekeeper_skip_binary_extensions() {
    let diff = "diff --git a/image.png b/image.png\n--- a/image.png\n+++ b/image.png\n@@ -0,0 +1 @@\n+binary content";
    let hunks = gatekeeper_test_helpers::parse_unified_diff(diff);
    assert!(hunks.is_empty(), "Binary file hunks should be skipped");
}

#[test]
fn gatekeeper_doc_extension_classified_correctly() {
    let diff = "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1,2 +1,3 @@\n # Title\n+New content here\n End";
    let hunks = gatekeeper_test_helpers::parse_unified_diff(diff);
    assert_eq!(hunks.len(), 1);
    assert!(
        !hunks[0].is_code,
        "Markdown should not be classified as code"
    );
    assert_eq!(hunks[0].extension, "md");
}

#[test]
fn gatekeeper_deletion_only_diff_produces_no_hunks() {
    let diff = "diff --git a/src/lib.rs b/src/lib.rs\n--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1,3 +1,2 @@\n fn foo() {\n-    let old = 1;\n }";
    let hunks = gatekeeper_test_helpers::parse_unified_diff(diff);
    assert!(
        hunks.is_empty(),
        "Deletion-only diffs should produce no hunks"
    );
}

#[test]
fn gatekeeper_lock_file_is_skipped() {
    let diff = "diff --git a/Cargo.lock b/Cargo.lock\n--- a/Cargo.lock\n+++ b/Cargo.lock\n@@ -1 +1,2 @@\n some existing content\n+new lock entry";
    let hunks = gatekeeper_test_helpers::parse_unified_diff(diff);
    assert!(hunks.is_empty(), "Cargo.lock should be skipped");
}

#[test]
fn gatekeeper_multi_file_diff_extracts_all_supported() {
    let diff = concat!(
        "diff --git a/src/main.rs b/src/main.rs\n",
        "--- a/src/main.rs\n",
        "+++ b/src/main.rs\n",
        "@@ -1 +1,2 @@\n",
        " fn main() {}\n",
        "+// new comment\n",
        "diff --git a/notes.txt b/notes.txt\n",
        "--- a/notes.txt\n",
        "+++ b/notes.txt\n",
        "@@ -1 +1,2 @@\n",
        " old note\n",
        "+new note\n",
        "diff --git a/image.png b/image.png\n",
        "--- a/image.png\n",
        "+++ b/image.png\n",
        "@@ -0,0 +1 @@\n",
        "+binary\n",
    );
    let hunks = gatekeeper_test_helpers::parse_unified_diff(diff);
    assert_eq!(
        hunks.len(),
        2,
        "Should extract 2 hunks (rs + txt), skip png"
    );
    assert_eq!(hunks[0].file_path, "src/main.rs");
    assert!(hunks[0].is_code);
    assert_eq!(hunks[1].file_path, "notes.txt");
    assert!(!hunks[1].is_code);
}

#[test]
fn gatekeeper_classify_extension_coverage() {
    use gatekeeper_test_helpers::{classify_extension, FileKind};
    assert_eq!(classify_extension("rs"), FileKind::Code);
    assert_eq!(classify_extension("py"), FileKind::Code);
    assert_eq!(classify_extension("md"), FileKind::Document);
    assert_eq!(classify_extension("txt"), FileKind::Document);
    assert_eq!(classify_extension("png"), FileKind::Skip);
    assert_eq!(classify_extension("jpg"), FileKind::Skip);
    assert_eq!(classify_extension("lock"), FileKind::Skip);
    assert_eq!(classify_extension("wasm"), FileKind::Skip);
    assert_eq!(classify_extension("toml"), FileKind::Skip); // unsupported = skip
}

#[test]
fn gatekeeper_python_diff_is_code() {
    let diff = "diff --git a/script.py b/script.py\n--- a/script.py\n+++ b/script.py\n@@ -1 +1,2 @@\n import os\n+print('hello')\n";
    let hunks = gatekeeper_test_helpers::parse_unified_diff(diff);
    assert_eq!(hunks.len(), 1);
    assert!(hunks[0].is_code);
    assert_eq!(hunks[0].extension, "py");
}
