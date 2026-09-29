//! Cascade Hybrid Router: Multi-stage TypeSafe Routing Engine
//!
//! Implements Phase 4 specification for System One Gatekeeper & Hybrid Routing:
//! - Evaluates inputs with millisecond latency via `oniwa-decide`
//! - Dispatches to:
//!   1. `FastPath`: Low-difficulty, safe, and routine queries handled instantly
//!   2. `Block`: Explicit anomaly, syntax defect, or malicious code blocked immediately
//!   3. `EscalateToSystemTwo`: High complexity, ambiguous entropy, or generative tasks routed to System 2

use crate::dataset::DocCategory;
use crate::{AuditDecision, DecisionEngine, StateEmbedding};
use serde::{Deserialize, Serialize};

/// Categorical verdict returned on fast path execution
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChoiceVerdict {
    pub category: DocCategory,
    pub confidence: f32,
    pub score: f32,
    pub latency_ms: u128,
}

/// Anomaly report returned when a query or patch is blocked
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnomalyReport {
    pub reason: String,
    pub anomaly_probability: f32,
    pub confidence: f32,
    pub complexity_score: f32,
    pub file_path: Option<String>,
}

/// Type-safe routing decision according to Phase 4 roadmap
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RoutingDecision {
    /// System 1 resolves immediately (low difficulty & safe)
    FastPath(ChoiceVerdict),
    /// Immediate block due to detected anomaly (security / syntax defect)
    Block { reason: AnomalyReport },
    /// Escalate to System 2 (oniwa-lm / external LLM) for deep reasoning or generation
    EscalateToSystemTwo {
        context_entropy: f32,
        reason: String,
        state_embedding: Option<StateEmbedding>,
    },
}

/// Policy configuration for cascade router
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RouterPolicy {
    /// Minimum anomaly confidence to trigger immediate Block (e.g. 0.80)
    pub block_anomaly_threshold: f32,
    /// Minimum choice confidence required for FastPath (e.g. 0.60)
    pub fastpath_min_choice_conf: f32,
    /// Maximum Shannon entropy (in bits) before triggering Escalation (e.g. 1.8 bits for 4 categories)
    pub max_entropy_for_fastpath: f32,
    /// Maximum complexity score before triggering Escalation to System 2 (e.g. 4.0)
    pub complexity_escalation_threshold: f32,
}

impl Default for RouterPolicy {
    fn default() -> Self {
        Self {
            block_anomaly_threshold: 0.80,
            fastpath_min_choice_conf: 0.60,
            max_entropy_for_fastpath: 1.6,
            complexity_escalation_threshold: 3.8,
        }
    }
}

/// Cascade Hybrid Router
pub struct CascadeRouter {
    pub policy: RouterPolicy,
}

impl CascadeRouter {
    /// Create a new CascadeRouter with specific policy
    pub fn new(policy: RouterPolicy) -> Self {
        Self { policy }
    }

    /// Calculate Shannon entropy (in bits) from probability distribution
    pub fn entropy_bits(probs: &[f32]) -> f32 {
        let mut h = 0.0f32;
        for &p in probs {
            if p > 1e-10 {
                h -= p * p.log2();
            }
        }
        h
    }

    /// Route a query or diff based on System 1 audit decision and optional semantic sensor embedding
    pub fn route(
        &self,
        audit: &AuditDecision,
        state_embedding: Option<StateEmbedding>,
        file_path: Option<&str>,
        is_code: bool,
    ) -> RoutingDecision {
        let noul_active = audit.syntax_anomaly.value;
        let noul_conf = audit.syntax_anomaly.confidence;
        let entropy = Self::entropy_bits(&audit.category.probabilities);
        let score = audit.complexity.value;

        // 1. Critical Anomaly Block (Code with high anomaly confidence)
        if is_code && noul_active && noul_conf >= self.policy.block_anomaly_threshold {
            return RoutingDecision::Block {
                reason: AnomalyReport {
                    reason: "High-confidence syntax anomaly or defect detected in source code"
                        .to_string(),
                    anomaly_probability: audit.syntax_anomaly.probability,
                    confidence: noul_conf,
                    complexity_score: score,
                    file_path: file_path.map(|s| s.to_string()),
                },
            };
        }

        // 2. High Complexity / High Entropy -> Escalate to System 2
        if entropy > self.policy.max_entropy_for_fastpath {
            return RoutingDecision::EscalateToSystemTwo {
                context_entropy: entropy,
                reason: format!(
                    "High categorical entropy ({:.2} bits > {:.2} bits)",
                    entropy, self.policy.max_entropy_for_fastpath
                ),
                state_embedding,
            };
        }

        if score > self.policy.complexity_escalation_threshold {
            return RoutingDecision::EscalateToSystemTwo {
                context_entropy: entropy,
                reason: format!(
                    "High complexity score ({:.2} > {:.2}) requires System 2 reasoning",
                    score, self.policy.complexity_escalation_threshold
                ),
                state_embedding,
            };
        }

        // 3. Low choice confidence -> Escalate to System 2
        if audit.category.confidence < self.policy.fastpath_min_choice_conf {
            return RoutingDecision::EscalateToSystemTwo {
                context_entropy: entropy,
                reason: format!(
                    "Ambiguous classification confidence ({:.1}% < {:.1}%)",
                    audit.category.confidence * 100.0,
                    self.policy.fastpath_min_choice_conf * 100.0
                ),
                state_embedding,
            };
        }

        // 4. Safe & Confident -> FastPath
        RoutingDecision::FastPath(ChoiceVerdict {
            category: audit.category.value,
            confidence: audit.category.confidence,
            score,
            latency_ms: audit.inference_time_ms,
        })
    }

    /// Convenience end-to-end evaluation using a DecisionEngine
    pub fn route_text(
        &self,
        engine: &DecisionEngine,
        text: &str,
        file_path: Option<&str>,
        is_code: bool,
        include_embedding: bool,
    ) -> RoutingDecision {
        let audit = engine.audit_text(text);
        let state_emb = if include_embedding {
            let sensor = engine.as_sensor();
            Some(sensor.sense(text))
        } else {
            None
        };
        self.route(&audit, state_emb, file_path, is_code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Choice, Noul, Score};

    fn make_test_audit(
        category: DocCategory,
        probs: Vec<f32>,
        choice_conf: f32,
        noul_val: bool,
        noul_prob: f32,
        noul_conf: f32,
        score: f32,
    ) -> AuditDecision {
        AuditDecision {
            category: Choice {
                value: category,
                probabilities: probs,
                confidence: choice_conf,
            },
            syntax_anomaly: Noul {
                value: noul_val,
                probability: noul_prob,
                confidence: noul_conf,
            },
            complexity: Score {
                value: score,
                confidence: 0.9,
            },
            inference_time_ms: 1,
        }
    }

    #[test]
    fn test_router_fast_path() {
        let router = CascadeRouter::new(RouterPolicy::default());
        let audit = make_test_audit(
            DocCategory::RustCode,
            vec![0.85, 0.05, 0.05, 0.05],
            0.85,
            false,
            0.1,
            0.8,
            2.0,
        );

        let decision = router.route(&audit, None, Some("src/main.rs"), true);
        match decision {
            RoutingDecision::FastPath(verdict) => {
                assert_eq!(verdict.category, DocCategory::RustCode);
                assert!((verdict.confidence - 0.85).abs() < 1e-5);
            }
            _ => panic!("Expected FastPath, got {:?}", decision),
        }
    }

    #[test]
    fn test_router_block_on_code_anomaly() {
        let router = CascadeRouter::new(RouterPolicy::default());
        let audit = make_test_audit(
            DocCategory::RustCode,
            vec![0.9, 0.03, 0.03, 0.04],
            0.9,
            true,
            0.95,
            0.90, // >= 0.80 threshold
            2.5,
        );

        let decision = router.route(&audit, None, Some("src/broken.rs"), true);
        match decision {
            RoutingDecision::Block { reason } => {
                assert!(reason.anomaly_probability >= 0.9);
                assert_eq!(reason.file_path, Some("src/broken.rs".to_string()));
            }
            _ => panic!("Expected Block, got {:?}", decision),
        }
    }

    #[test]
    fn test_router_escalate_on_high_entropy() {
        let router = CascadeRouter::new(RouterPolicy::default());
        // Uniform distribution: 4 categories -> -4 * (0.25 * log2(0.25)) = 2.0 bits > 1.6
        let audit = make_test_audit(
            DocCategory::Literature,
            vec![0.25, 0.25, 0.25, 0.25],
            0.25,
            false,
            0.2,
            0.6,
            2.0,
        );

        let decision = router.route(&audit, None, Some("notes.txt"), false);
        match decision {
            RoutingDecision::EscalateToSystemTwo {
                context_entropy,
                reason,
                ..
            } => {
                assert!((context_entropy - 2.0).abs() < 1e-4);
                assert!(reason.contains("High categorical entropy"));
            }
            _ => panic!("Expected EscalateToSystemTwo, got {:?}", decision),
        }
    }

    #[test]
    fn test_router_escalate_on_high_complexity() {
        let router = CascadeRouter::new(RouterPolicy::default());
        let audit = make_test_audit(
            DocCategory::LegalOrTechDoc,
            vec![0.05, 0.05, 0.85, 0.05],
            0.85,
            false,
            0.1,
            0.8,
            4.5, // > 3.8 threshold
        );

        let decision = router.route(&audit, None, Some("spec.md"), false);
        match decision {
            RoutingDecision::EscalateToSystemTwo { reason, .. } => {
                assert!(reason.contains("High complexity score"));
            }
            _ => panic!("Expected EscalateToSystemTwo, got {:?}", decision),
        }
    }
}
