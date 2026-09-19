# Implementation Plan: QuaternionLMHead & Weight Tying in oniwa-lm

## Overview
Introduce quaternion-algebra-based language model head (`QuaternionLMHead`) and tied embedding-projection autoregressive model architecture (`QuaternionModelWeights`) to `oniwa-lm`, powered by ARM NEON SIMD primitives from `oniwa-decide`.

## User Review Required
> [!NOTE]
> Backward compatibility is completely preserved. Real-valued checkpoints and existing `ModelWeights` / `ModelConfig` remain unchanged. A dedicated `--quaternion` flag in `train.rs` and `chat.rs` switches execution paths and isolates checkpoints to `checkpoints/quaternion/`.

## Architecture & Math
1. **Embedding & Tied Projection**:
   - Hidden dimension $D$ (asserted to be divisible by 4, default 128 -> $D/4 = 32$ quaternions).
   - Vocabulary weight $E \in \mathbb{H}^{V \times (D/4)}$ (parameter count: $V \times D$ floats, initialized with std $1/\sqrt{D}$).
   - Embedding lookup for token $x_t \in [0, V)$: retrieves row $E[x_t]$ of length $D$ floats.
   - Forward Logits: For hidden state $h_{b,t} \in \mathbb{H}^{D/4}$ (dim $D$) and token $v$:
     $$\text{logits}_{b,t,v} = \sum_{k=1}^{D/4} \text{Re}(E[v, k]^* \otimes h_{b,t}[k]) = \sum_{k=1}^{D/4} \text{dot\_product\_4d}(E[v, k], h_{b,t}[k])$$
   - Backward:
     $$dh_{b,t} = \sum_v d\text{logits}_{b,t,v} \cdot E[v]$$
     $$dE[v] \mathrel{+}= \sum_{b,t} d\text{logits}_{b,t,v} \cdot h_{b,t}$$
     (Accumulated alongside embedding lookup gradients $dE[x_{b,t}] \mathrel{+}= dh_{b,t}^{(0)}$).

## Implementation Phases
### Phase 1: Primitives Export & QuaternionLMHead Layer
- Export `pub mod simd;` and SIMD helpers in `crates/oniwa-decide/src/lib.rs`.
- Add `oniwa-decide = { path = "../oniwa-decide" }` to `crates/oniwa-lm/Cargo.toml`.
- Create `crates/oniwa-lm/src/layers/quaternion_head.rs`:
  - `QuaternionLMHead` struct with forward, backward, and finite difference tests.
- Export `pub mod quaternion_head;` in `crates/oniwa-lm/src/layers/mod.rs`.

### Phase 2: QuaternionModelWeights & Autoregressive Architecture
- In `crates/oniwa-lm/src/model.rs`:
  - Implement `QuaternionModelWeights` with:
    - `params`, `grads`, `m`, `v` (layout: shared embedding/LM head weight $E$, transformer layers, final RMSNorm).
    - `forward_backward`, `evaluate_loss`, `evaluate_loss_and_top_k`, `forward_inference`, `adamw_step`, `save_checkpoint`, `load_checkpoint`.
    - Checkpoint metadata contains `"model_type": "quaternion_tied"`.

### Phase 3: CLI Integration & Verification
- Update `crates/oniwa-lm/src/bin/train.rs`:
  - Support `--quaternion` flag.
  - Save checkpoints under `checkpoints/quaternion/` (both `latest` and `best`).
- Update `crates/oniwa-lm/src/bin/chat.rs`:
  - Support `--quaternion` flag.
  - Generate autoregressively from `checkpoints/quaternion/best` or specified checkpoint.
- Smoke tests:
  - `cargo test --workspace`
  - `cargo run --release -p oniwa-lm --bin train -- --quaternion --reset --steps 10`
  - `cargo run --release -p oniwa-lm --bin chat -- --quaternion`
- Run formatting and check git diff.
