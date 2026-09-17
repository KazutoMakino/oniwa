# 🌿 ONIWA Contribution Guide

<p align="left">
  <b>English</b> | <a href="CONTRIBUTING.ja.md">日本語 (Japanese)</a>
</p>

Thank you for your interest in contributing to the ONIWA (お庭) project!  
ONIWA is dedicated to cultivating autonomous small language models (SLMs) in a home garden fashion—relying on clean, 100% provenance-audited open data and energy-efficient edge hardware (such as Raspberry Pi 4) rather than mega-scale GPU clusters and scraped web data.

We warmly welcome bug reports, feature suggestions, code optimizations, and documentation improvements from developers and researchers worldwide.

---

## 1. Core Principles

Before contributing, please familiarize yourself with our fundamental design principles:

1. **Pure Rust Principle**:  
   We strictly prohibit dependencies on external ML frameworks (PyTorch, TensorFlow, CUDA, etc.). All forward passes, backward passes, optimizers, data cleaning, and hardware telemetry must be implemented in Pure Rust.
2. **100% Provenance & Auditability**:  
   Every piece of training data, vocabulary table, model checkpoint, and commit hash is cryptographically fingerprinted (SHA-256) and logged into an audit ledger (`logs/ledger_index.jsonl`).
3. **Edge Hardware Compatibility (~5W Constraint)**:  
   The primary deployment target is edge hardware like Raspberry Pi 4. Code changes should prioritize CPU cache efficiency, zero unnecessary allocations, and must never disrupt thermal throttling or power tracking.
4. **Legal Clearance & Respect for Creators**:  
   Ingested corpora are strictly limited to copyright-expired public domain works, public legal texts, or permissible open-license materials (MIT, Apache-2.0, CC0). See [Data Ingestion Charter](docs/07_data_ingestion_charter.md) / [日本語](docs/07_data_ingestion_charter.ja.md) for details.

---

## 2. Development Setup

### Prerequisites
- **Rust Toolchain**: 1.75 or higher (`stable` recommended)
- **Git**

### Clone & Enable Pre-commit Hook
This repository includes a pre-commit hook that automatically formats Rust code (`cargo fmt`) and checks for Clippy lints (`cargo clippy`). Enable it once after cloning:
```bash
git clone https://github.com/KazutoMakino/oniwa.git
cd oniwa

# Enable the repository Git hooks
git config core.hooksPath .githooks
```

### Build & Verification
Ensure the workspace builds and all tests pass cleanly:
```bash
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

---

## 3. Contribution Workflow

To maintain traceability, we adopt an **Issue-Driven Development** workflow.

### Step 1: Open or Select an Issue
- Before starting work on major features or architectural refactors, please open a [GitHub Issue](https://github.com/KazutoMakino/oniwa/issues) to discuss your ideas and approach.

### Step 2: Create a Feature Branch
Use our kebab-case branch naming convention linked to the issue number:
`{issue_number}/{type}/{kebab-case-description}`
- Common `type` prefixes: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`
- Example: `42/feat/cosine-lr-warmup-min`
```bash
git checkout -b 42/feat/cosine-lr-warmup-min
```

### Step 3: Implement & Verify
- Write clean, well-tested, and idiomatic Rust code.
- **Always ensure all automated tests pass**:
  ```bash
  cargo test --workspace
  ```
- Run formatting and static analysis:
  ```bash
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets -- -D warnings
  ```

### Step 4: Commit & Push
We recommend Conventional Commits referencing the issue number:
`<type>: <description> (#<issue_number>)`
```bash
git add .
git commit -m "feat: add linear warmup to cosine learning rate schedule (#42)"
git push -u origin 42/feat/cosine-lr-warmup-min
```

### Step 5: Submit a Pull Request
- Open a Pull Request on GitHub.
- Reference the issue number in the PR description (e.g., `Closes #42`).
- GitHub Actions CI will automatically run tests, formatting, and Clippy checks.

---

## 4. AI Agent Protocol

If you are developing using autonomous AI coding assistants (such as Antigravity, Claude Code, GitHub Copilot CLI, Gemini CLI, etc.), please ensure they follow the autonomous agent protocols specified in [`AGENTS.md`](AGENTS.md).

---

## 5. Community Code of Conduct

To foster an inclusive and safe environment, all contributors are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) / [日本語](CODE_OF_CONDUCT.ja.md).
