# Implementation Plan - System 1 MLM Training Pipeline Integration (#84)

Integrate Masked Language Modeling (MLM), tied vocabulary projection, bounded flat-memory layout checking, and zero-allocation killer-pattern JSONL ingestion into `oniwa-decide` (System 1).

## Proposed Changes

### Component 1: Bounded Flat-Memory Layout Calculation & 100MB Budget Enforcement
In `crates/oniwa-decide/src/model.rs`:
- Add `MemoryBudget` struct with breakdown of:
  - Model weights bytes ($V \times C$ + backbone params)
  - Gradients bytes
  - AdamW states bytes ($m, v$)
  - Activation cache bytes ($B \times T \times C \times \dots$)
  - Total training footprint bytes
- Add method `DecisionConfig::calculate_training_memory_bytes(&self, batch_size: usize) -> usize`
- Add `DecisionConfig::assert_memory_budget(&self, batch_size: usize, budget_bytes: usize) -> Result<MemoryBudget, String>` enforcing the 100 MB ($100 \times 1024 \times 1024$ bytes) budget.

### Component 2: MLM Masking, Tied Vocabulary Projection & Weighted Multi-Task Loss
In `crates/oniwa-decide/src/loss.rs`:
- Extend `LossConfig` with `mlm_weight: f32` (default 1.0, with backward compatibility).
- Implement `LossCalculator::mlm_loss(logits: &[f32], target_token: u16, vocab_size: usize, label_smoothing: f32) -> (f32, Vec<f32>)`.
- Multi-task total loss: $\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{MLM}} + \beta \mathcal{L}_{\text{choice}} + \gamma \mathcal{L}_{\text{noul}} + \delta \mathcal{L}_{\text{score}}$.

In `crates/oniwa-decide/src/model.rs`:
- Add `mask_token_id: Option<u16>` or dedicated masking utility:
  - Randomly select $15\%$ of token positions in a sequence.
  - Replace selected tokens with `mask_token_id` (or random/identity as per standard BERT recipe).
- Implement tied vocabulary projection for MLM:
  - Project hidden state $h_i \in \mathbb{R}^C$ back to vocabulary logits $[V]$ using shared token embedding weight matrix $W_{\text{wte}} \in \mathbb{R}^{V \times C}$.
  - Support both real-valued and quaternion backbones.
  - Implement forward pass and backward pass gradient accumulation into $W_{\text{wte}}$ and hidden states $dh_i$.

### Component 3: Zero-Allocation Killer-Pattern JSONL Ingestion
In `crates/oniwa-decide/src/dataset.rs`:
- Add `KillerPatternRecord` struct deserializable from JSONL line:
  - `pattern_id: String`
  - `category: DocCategory`
  - `anomaly_type: Option<String>`
  - `has_syntax_anomaly: bool`
  - `complexity_score: f32`
  - `code_snippet: String`
- Add zero-allocation ingestion function:
  - `ingest_killer_patterns_jsonl<T: Tokenizer>(jsonl_content: &str, tokenizer: &T, seq_len: usize, batch_tokens: &mut [u16], batch_choices: &mut [usize], batch_nouls: &mut [bool], batch_scores: &mut [f32]) -> Result<usize, String>`
  - Encodes inputs directly into caller-provided slice without intermediate heap allocations per sample.

### Component 4: Crate Export and Verification Tests
In `crates/oniwa-decide/src/lib.rs`:
- Re-export `MemoryBudget`, `KillerPatternRecord`, `ingest_killer_patterns_jsonl`.

In `crates/oniwa-decide/tests/decision_test.rs`:
- Test memory budget calculation and 100MB bound enforcement.
- Test MLM forward & backward finite difference gradient check.
- Test killer-pattern JSONL ingestion into caller-provided slices.

## Verification Plan
### Automated Tests
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo fmt --all -- --check`
