//! Short Snippet & Calibration Regression Tests for oniwa-decide (Phase B)
//!
//! Evaluates calibration under post-hoc temperature scaling ($T$)
//! and prevents overconfident misclassification on short snippets and ambiguous inputs.

use oniwa_decide::config::DecisionConfig;
use oniwa_decide::dataset::DocCategory;
use oniwa_decide::loss::LossCalculator;
use oniwa_decide::model::DecisionModel;
use oniwa_decide::router::{CascadeRouter, RouterPolicy, RoutingDecision};
use oniwa_decide::DecisionEngine;
use oniwa_lm::reproducibility::DeterministicRng;
use oniwa_lm::tokenizer::CharTokenizer;

#[test]
fn test_post_hoc_temperature_scaling_calibrates_confidence() {
    // With high temperature T=2.0, peak softmax probability decreases towards uniform
    let logits = [3.0f32, 1.0f32, 0.5f32, 0.1f32];
    let probs_t1 = LossCalculator::softmax(&logits, 1.0);
    let probs_t2 = LossCalculator::softmax(&logits, 2.0);
    let probs_t5 = LossCalculator::softmax(&logits, 5.0);

    assert!(probs_t1[0] > probs_t2[0]);
    assert!(probs_t2[0] > probs_t5[0]);

    // Check sum of probabilities equals 1.0
    let sum: f32 = probs_t2.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_decision_config_temperature_metadata_roundtrip() {
    let mut config = DecisionConfig::default();
    assert_eq!(config.temperature, 1.0);

    config = config.with_temperature(1.75);
    assert_eq!(config.temperature, 1.75);

    let temp_dir = std::env::temp_dir().join("oniwa_test_config");
    let _ = std::fs::create_dir_all(&temp_dir);
    let meta_path = temp_dir.join("meta.json");

    let meta = serde_json::json!({
        "step": 100,
        "loss": 0.5,
        "config": config,
    });
    std::fs::write(&meta_path, serde_json::to_string(&meta).unwrap()).unwrap();

    let loaded = DecisionConfig::load_from_meta(&meta_path).unwrap();
    assert_eq!(loaded.temperature, 1.75);

    // Update in-place
    DecisionConfig::update_meta_temperature(&meta_path, 2.5).unwrap();
    let updated = DecisionConfig::load_from_meta(&meta_path).unwrap();
    assert_eq!(updated.temperature, 2.5);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_short_code_snippets_classification_and_anomaly() {
    let vocab_text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _(){}[]<>;:=+-*/!?,.&|#\"'~`@$%\n\t\r";
    let tokenizer = CharTokenizer::build_from_text(vocab_text);
    let config = DecisionConfig {
        vocab_size: tokenizer.vocab_size(),
        seq_len: 128,
        dim: 64,
        num_layers: 2,
        num_heads: 2,
        head_dim: 16,
        ffn_dim: 128,
        num_choices: 4,
        temperature: 1.5, // calibrated
        ..Default::default()
    };
    let mut rng = DeterministicRng::new(1234);
    let model = DecisionModel::new(config, &mut rng);
    let engine = DecisionEngine::new(model, tokenizer);

    let short_snippets = vec![
        ("fn x() { 1 }", "Valid short Rust fn"),
        ("def f():\n    return 1", "Valid short Python fn"),
        ("fn broken( {", "Unclosed bracket short Rust"),
        ("def broken(:", "Syntax anomaly short Python"),
        ("x = [1, 2, 3", "Unclosed list"),
        ("let mut count = 0;", "Short Rust let statement"),
    ];

    let router = CascadeRouter::new(RouterPolicy::default());

    for (snippet, _desc) in short_snippets {
        let audit = engine.audit_text(snippet);
        // Valid classification properties
        assert!(audit.category.confidence >= 0.0 && audit.category.confidence <= 1.0);
        assert!(audit.syntax_anomaly.probability >= 0.0 && audit.syntax_anomaly.probability <= 1.0);
        assert!(audit.complexity.value >= 1.0 && audit.complexity.value <= 5.0);

        let routing = router.route(&audit, None, Some("snippet.rs"), true);
        match routing {
            RoutingDecision::FastPath(v) => {
                assert!(v.confidence >= 0.0);
            }
            RoutingDecision::Block { reason } => {
                assert!(!reason.reason.is_empty());
            }
            RoutingDecision::EscalateToSystemTwo { reason, .. } => {
                assert!(!reason.is_empty());
            }
        }
    }
}

#[test]
fn test_ambiguous_short_snippet_entropy_escalation() {
    // When logits are almost equal, entropy is high (> 1.6 bits)
    let uniform_probs = vec![0.25f32, 0.25f32, 0.25f32, 0.25f32];
    let entropy = CascadeRouter::entropy_bits(&uniform_probs);
    assert!((entropy - 2.0).abs() < 1e-4);

    let router = CascadeRouter::new(RouterPolicy::default());
    let audit = oniwa_decide::AuditDecision {
        category: oniwa_decide::Choice {
            value: DocCategory::RustCode,
            probabilities: uniform_probs,
            confidence: 0.25,
        },
        syntax_anomaly: oniwa_decide::Noul {
            value: false,
            probability: 0.1,
            confidence: 0.8,
        },
        complexity: oniwa_decide::Score {
            value: 2.0,
            confidence: 0.9,
        },
        inference_time_ms: 1,
    };

    let decision = router.route(&audit, None, Some("ambiguous.rs"), true);
    match decision {
        RoutingDecision::EscalateToSystemTwo { reason, .. } => {
            assert!(
                reason.contains("High categorical entropy")
                    || reason.contains("Ambiguous classification")
            );
        }
        _ => panic!("Expected escalation for ambiguous high-entropy snippet"),
    }
}
