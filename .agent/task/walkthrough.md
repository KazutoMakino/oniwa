# Walkthrough: Align Documentation System with 10_system_integration_roadmap

Issue: #87
Task: Comprehensive alignment and modernization of the repository documentation system based on the latest implementation (Byte-level BPE 4,096 vocab, MLM multi-task learning, 100MB memory budget, quaternion algebra) and establishment of safe concurrent operations protecting background computation processes.

## Changes Overview

### 1. Root & Pipelines Documentation
- **`README.md` / `README.ja.md`**:
  - Symmetrically updated English and Japanese root documentation.
  - Added badges and architectural specifications for 100MB memory budget and pure-Rust BPE 4K tokenizer.
  - Positioned `docs/10_system_integration_roadmap.ja.md` / `10_system_integration_roadmap.md` as the Single Source of Truth (SSoT).
  - Clarified System 1 (`oniwa-decide`) role, MLM multi-task loss, killer pattern ingestion, and quaternion algebra.
- **`pipelines/README.md` / `pipelines/README.ja.md`**:
  - Documented the `train-bpe` binary (`cargo run -p oniwa-pipeline --bin train-bpe -- --input ... --vocab-size 4096 --output ...`).
  - Outlined multi-source corpus generation flow and BPE tokenization.

### 2. Crate Documentation
- **`crates/oniwa-decide/README.md` / `crates/oniwa-decide/README.ja.md` (New)**:
  - Created documentation detailing System 1 non-autoregressive decision engine.
  - Documented decision primitives (`Choice`, `Noul`, `Score`), loss functions, temperature calibration, flat 100MB memory budget, and CLI usage (`audit`, `train`).
- **`crates/oniwa-lm/docs/README.md` / `crates/oniwa-lm/docs/README.ja.md`**:
  - Updated index to reference `10_system_integration_roadmap` as current SSoT and `09_roadmap` as quaternion research document.

### 3. Historical Design Document Navigation Banners
- Symmetrically added alert navigation banners to all foundational and historical design documents:
  - `docs/00_oniwa-project-manifesto.{md,ja.md}`
  - `docs/01_llm_c_architecture.{md,ja.md}`
  - `docs/02_modern_primitives.{md,ja.md}`
  - `docs/03_rust_memory_model.{md,ja.md}`
  - `docs/04_development_handoff.{md,ja.md}`
  - `docs/05_reproducibility_and_logging.{md,ja.md}`
  - `docs/06_thermal_and_end_to_end_provenance.{md,ja.md}`
  - `docs/07_data_ingestion_charter.{md,ja.md}`
  - `docs/08_green_energy_and_power_tracking.{md,ja.md}`
  - `docs/09_roadmap.{md,ja.md}`
  - `docs/HANDOVER_RASPBERRY_PI.{md,ja.md}`
- The banners guide readers to the latest production roadmap (`10_system_integration_roadmap`) while preserving historical context and relying on Git history rather than creating separate physical archive folders.

### 4. Computation Safety & Non-Interference
- Strictly refrained from running `cargo test`, `cargo build`, or other heavy compilation or training binaries.
- Staged only the targeted Markdown documentation files and `.agent/task/` artifacts, preventing runtime files in `logs/` and `checkpoints/` from being inadvertently committed.

## Verification
- Verified all relative Markdown links in updated and historical documents programmatically; 0 broken links detected.
- Verified file symmetry between English and Japanese across all modified and newly created files.
