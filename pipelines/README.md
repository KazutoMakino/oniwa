# ONIWA Data Pipelines

<p align="left">
  <b>English</b> | <a href="README.ja.md">日本語 (Japanese)</a>
</p>

Dedicated crate (`oniwa-pipeline`) responsible for clean data preparation (collecting and preprocessing 100% provenance-traceable open data) in the ONIWA project.

---

## Key Roles

1. **Compliance-First Automated Ingestion**:
   - **Aozora Bunko**: Strictly filters only works whose **copyright protection period has expired (Public Domain / CC0 equivalent)** from the official index. Completely prevents contamination from copyrighted material.
   - **e-Gov Basic Laws**: Under **Article 13 of the Japanese Copyright Act (Works not eligible for copyright protection = Public Domain)**, retrieves and cleanses core statutory texts (Constitution, Penal Code, Copyright Act, Civil Code, Court Act, etc.) via official APIs.
2. **Web Etiquette & Load Prevention**:
   - Explicit User-Agent headers, 1-second request interval pacing, and local caching (`data/raw/`) to prevent redundant downloads.
3. **High-Precision Text Cleaning**:
   - Automated removal and normalization of ruby annotations (`｜kanji《ruby》`, `漢字《ruby》`), editor notes (`［＃...］`), and statutory XML tags.
4. **Transparent Provenance Ledger Recording**:
   - Fully logs the title, source URL, legal basis, SHA-256 hash, and character count of every ingested work into `logs/ledger_index.jsonl`.
5. **Multi-Source Unified Corpus Generation**:
   - Bundles cleaned texts and regenerates `data/tokens.bin` and `data/vocab.json` for model training.

---

## Usage (`oniwa-dataset`)

```bash
# 1. Ingest preset Aozora Bunko classics + e-Gov basic laws and regenerate unified corpus
cargo run --release -p oniwa-pipeline -- --all

# 2. Ingest only e-Gov basic laws open data
cargo run --release -p oniwa-pipeline -- --laws

# 3. Batch ingest classic literature specified in recipe file (56 works)
cargo run --release -p oniwa-pipeline -- --recipe pipelines/config/recipes.json --build

# 4. Search and ingest copyright-expired works by a specific author
cargo run --release -p oniwa-pipeline -- --author "夏目漱石" --limit 5 --build

# 5. Check current ingestion status and provenance ledger
cargo run --release -p oniwa-pipeline -- --status
```
