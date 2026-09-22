# Implementation Plan: Align Documentation System with 10_system_integration_roadmap

Issue: #87
Task: Align repository documentation with latest implementation (BPE 4K, MLM, 100MB budget, quaternion algebra) and preserve running computation processes safely.

## Background & Objectives
1. Phase 1 (`train-bpe` 4,096 vocab) and Phase 2 (MLM multi-task, killer pattern injection, flat 100MB memory budget) are completed in code.
2. Root `README`, crate documentation, and pipelines docs still reference char-level tokenizers or outdated tiny configs.
3. Align English and Japanese documentation symmetrically.
4. Establish `docs/10_system_integration_roadmap.*` as Single Source of Truth (SSoT), with `docs/09_roadmap.*` as academic quaternion monograph.
5. Add historical navigation banners to `docs/00`~`09` and `docs/HANDOVER_RASPBERRY_PI.*`.
6. Create `crates/oniwa-decide/README.md` and `README.ja.md`.
7. Computation safety: strictly avoid running `cargo test`, `cargo build`, or other heavy compilation/training commands to avoid disturbing running processes. Avoid modifying or staging runtime files (`logs/`, `checkpoints/`).

## Target Files
- Root docs: `README.md`, `README.ja.md`
- Pipelines docs: `pipelines/README.md`, `pipelines/README.ja.md`
- Crate docs:
  - `crates/oniwa-decide/README.md`, `crates/oniwa-decide/README.ja.md` (New)
  - `crates/oniwa-lm/docs/README.md`, `crates/oniwa-lm/docs/README.ja.md`
- Historical banners:
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
- Process & Task Artifacts:
  - `.agent/task/task.md`
  - `.agent/task/implement_plan.md`
  - `.agent/task/implement_plan.ja.md`
  - `.agent/task/walkthrough.md`
  - `.agent/task/walkthrough.ja.md`

## Execution Steps
1. Create implement_plan docs in `.agent/task/`.
2. Inspect current files and prepare updates.
3. Update root READMEs (`README.md`, `README.ja.md`).
4. Update pipeline READMEs (`pipelines/README.md`, `pipelines/README.ja.md`).
5. Author `crates/oniwa-decide/README.md` and `README.ja.md`.
6. Update `crates/oniwa-lm/docs/README.md` and `README.ja.md`.
7. Add banner notes to historical design docs (`docs/00`~`09`, `docs/HANDOVER_RASPBERRY_PI`).
8. Run python formatters (`uv run isort .`, `uv run ruff format .`) if any python touched, verify Markdown syntax & links.
9. Produce walkthrough reports (`.agent/task/walkthrough.*`).
10. Stage only target files, commit, push, create PR, squash merge, clean up branch.
