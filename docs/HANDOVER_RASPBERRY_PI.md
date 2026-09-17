# 🌿 ONIWA Raspberry Pi Migration & Context Handover Guide

> **Target**: Project migration from PC (`km-gpd`) to **Raspberry Pi** physical hardware, and complete context handover for AI agents.

---

## 1. Project Status & Core Architecture

- **Project Name**: `oniwa` (Organic Non-datacenter Intelligence Without Abuse)
- **Core Philosophy**:
  - **Zero Big-GPU / Zero Black-Box**: Completely free from PyTorch or external deep learning frameworks. Built from scratch in **Pure Rust (Standard library + pure mathematics)**.
  - **Bonsai / Houseplant Edge AI**: Operating within a few watts on a Raspberry Pi at safe temperatures (~70°C), quietly running thousands of training steps, nurturing language emergence like caring for a bonsai tree.
  - **Transparency & Provenance**: Training data is not bundled in Git; instead, it is 100% reproducible via external recipes (`pipelines/config/recipes.json`) and audit ledgers (`logs/ledger_index.jsonl`).
- **Current Status**:
  - **Corpus Scale**: Public domain works from Aozora Bunko, e-Gov legal texts, official technical documentation, clean algorithms, and arXiv papers.
  - **Model Architecture**: Modern Transformer Decoder (RMSNorm, SwiGLU, RoPE, Grouped-Query Attention, Label Smoothing, Z-loss).
  - **Observability**: Multi-probe cyclic generation tests for linguistic emergence monitoring.

---

## 2. Raspberry Pi Migration Steps

### Step 1: Git Repository Synchronization
In the terminal on the Raspberry Pi:
```bash
# For existing clone
cd ~/Github/oniwa   # or your chosen directory
git pull origin main

# For a fresh clone
git clone https://github.com/KazutoMakino/oniwa.git
cd oniwa
```

### Step 2: Transferring Checkpoints and Data (Optional)
To transfer locally saved model weights or tokenized binaries from the development PC:

```bash
# Example transferring via rsync from PC to Raspberry Pi
rsync -avzP \
  crates/oniwa-lm/checkpoints/ \
  pi@raspberrypi.local:~/Github/oniwa/crates/oniwa-lm/checkpoints/

rsync -avzP \
  crates/oniwa-lm/data/ \
  pi@raspberrypi.local:~/Github/oniwa/crates/oniwa-lm/data/
```

> **Note**: If training anew from scratch on the Raspberry Pi:
> ```bash
> cargo run --release -p oniwa-pipeline -- --all
> cargo run --release -p oniwa-lm --bin train -- --reset --steps 100
> ```

### Step 3: Build & Test on Raspberry Pi
```bash
# Run all workspace unit tests
cargo test --workspace

# Test interactive chat inference
cargo run --release -p oniwa-lm --bin chat

# Continue training (automatically resumes from latest checkpoint)
cargo run --release -p oniwa-lm --bin train -- --steps 500 --interval 25
```

---

## 3. Handing Over Context to AI Agents (Antigravity) on Raspberry Pi

When starting an agent session on Raspberry Pi (or via Remote-SSH), provide the following prompt:

```markdown
Please read @docs/HANDOVER_RASPBERRY_PI.md and @AGENTS.md.
Migration from the PC development environment to physical Raspberry Pi is complete.
Please inherit our core philosophy (Pure Rust, zero external frameworks, organic edge AI, provenance ledger) and continue supporting development and training.
```

---

## 4. Directory Structure

```
oniwa/
├── crates/
│   ├── oniwa-decide/       # Compact decision Transformer & mathematical reasoning
│   └── oniwa-lm/           # Core language model, training loop, hardware trackers
│       ├── src/bin/train.rs# Training binary with thermal & power tracking
│       ├── src/bin/chat.rs # Interactive chat binary
│       └── checkpoints/    # Model weights and tokenizer state
├── pipelines/              # Ethical data ingestion pipelines (Aozora, e-Gov, arXiv, TechDocs, Code)
│   ├── config/recipes.json # Curated data recipes
│   └── src/                # Multi-source ingestion pipelines
├── logs/                   # Growth journals & provenance ledgers
│   ├── growth_journal.md   # Linguistic emergence observation logs
│   └── ledger_index.jsonl  # SHA-256 provenance audit ledger
└── docs/                   # Architectural blueprints & handoff guides
    └── HANDOVER_RASPBERRY_PI.md
```
