# Phase 3 Addendum 3: Clean Data Ingestion Charter

The "Kitchen Garden & Organic AI (`oniwa-lm`)" philosophy prioritizes **transparency, legality, and etiquette toward remote servers** far above data volume or acquisition speed.

This charter defines the engineering etiquette and legal compliance standards governing all data ingestion pipelines.

---

## 1. Politeness Engineering

### 1.1 Strict Compliance with `robots.txt`
All crawlers must inspect the domain's `robots.txt` before fetching and strictly honor `Disallow` directives and `Crawl-delay` requirements.

### 1.2 Rate Limiting
* Web requests must enforce a mandatory delay of **at least 1.0 second (recommended: 2.0–3.0 seconds)** between requests.
* Aggressive concurrent scraping (multi-threaded or asynchronous hammering) that stresses remote infrastructure is prohibited.

### 1.3 Prioritize Curated Bulk Archives
* Where official or verified community archives exist (e.g. Aozora Bunko bulk ZIP dumps or GitHub mirrors), use these archives rather than querying live web frontends.

### 1.4 Transparent User-Agent Identification
Anonymous or deceptive crawlers are strictly prohibited. Ingestion pipelines must identify themselves clearly with project name, purpose, and repository URL:

```http
User-Agent: oniwa-crawler/0.1.0 (+https://github.com/KazutoMakino/oniwa; Clean AI Educational Project)
```

---

## 2. Legal Clearance & Copyright Criteria

### 2.1 Authorized Data Sources (Greenlist)
Data accepted into the training corpus must satisfy one of the following criteria:

1. **Public Domain (PD)**:
   - Literary and philosophical works whose copyright protection has expired (e.g. Aozora Bunko).
2. **Permissive Open Licenses (Open Data)**:
   - **CC0**: Complete public dedication without restrictions.
   - **CC-BY**: Permissive license with author attribution preserved.
   - **Government Open Data**: Public agency sources (Digital Agency, e-Gov laws, NDL Lab) permitting educational, commercial, or non-commercial reuse.
3. **Proprietary Author-Owned Data**:
   - Original writings, technical notes, or code created directly by the project authors.

### 2.2 Prohibited Data Sources (Redlist / "Pesticides")
The following data sources are strictly banned from ingestion:

* **Synthetic Text from Commercial Third-Party LLMs (Distillation Data)**:
   - Completely excluded to prevent model collapse, inbreeding, and violations of vendor terms of service.
* **Websites explicitly prohibiting automated scraping in their Terms of Service**.
* **Uncurated web scrapes with ambiguous rights or provenance (e.g. raw Common Crawl)**.

---

## 3. Mandatory Provenance Ledger Registration

Every ingested dataset must log complete provenance metadata to the append-only ledger:

```json
{
  "event_type": "DataIngestion",
  "payload": {
    "timestamp_utc": "2026-09-14T10:15:00Z",
    "source_name": "Aozora Bunko: Atsushi Nakajima 'Sangetsuki'",
    "source_url_or_path": "https://ja.wikisource.org/wiki/山月記",
    "license": "Public Domain",
    "raw_data_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "raw_data_bytes": 14250,
    "tokenized_sha256": "5566778899aabbccddeeff0011223344...",
    "num_tokens": 4621,
    "vocab_size": 782,
    "tokenizer_type": "Character-level (UTF-8)"
  }
}
```

This permanent audit trail ensures that whenever a model checkpoint is published, its provenance is 100% legally clear and auditable by anyone.
