# ONIWA Research Paper

This directory contains the LaTeX source code, bibliographies, and build configurations for the ONIWA research paper:

> **"Organic Non-Datacenter Intelligence: A Pure-Rust Full Quaternion Transformer for Edge Decision-Making via Irreversible Lossy Compression"**  
> *Author:* Kazuto Makino (*Independent Researcher, Aichi, Japan*)

---

## Directory Structure

```text
paper/
├── Makefile                 # Automated build, clean, and arXiv packaging rules
├── README.md                # Paper overview and build instructions
├── .gitignore               # Ignored build artifacts and temporary files
├── main.tex                 # Main LaTeX source document
├── main.bib                 # BibTeX references and citations
├── figures/                 # Figures, diagrams, and vector assets (future additions)
└── sections/                # Modular section drafts (reserved for extended versions)
```

---

## Building the Paper

### Requirements
- TeX Live distribution (`pdflatex`, `bibtex`)
- Required TeX packages: `amsmath`, `amsfonts`, `tikz`, `pgfplots`, `listings`, `booktabs`, `microtype`, `hyperref`, `url`

### Build PDF
Run `make` inside the `paper` directory:
```bash
make
```
This runs the full compilation pass (`pdflatex` $\rightarrow$ `bibtex` $\rightarrow$ `pdflatex` $\times 2$) to generate `main.pdf`.

### Prepare arXiv Submission Package
To generate the `.tar.gz` bundle required for uploading to arXiv:
```bash
make arxiv
```
This produces `oniwa-paper-arxiv.tar.gz` containing `main.tex`, pre-compiled `main.bbl`, and any referenced figure assets.

### Clean Build Artifacts
To remove intermediate TeX files (`.aux`, `.log`, `.bbl`, etc.):
```bash
make clean
```
