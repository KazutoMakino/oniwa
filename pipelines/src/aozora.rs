//! Automated ingestion & preprocessing pipeline for Aozora Bunko public domain works
//!
//! In alignment with ONIWA's philosophy, strictly filters only works with expired copyright protection
//! (public domain) while observing server etiquette (custom User-Agent, request pacing, local cache)
//! to safely download, cleanse, and log complete provenance.

#![allow(dead_code)]

use crate::cleaner::clean_aozora_text;
use oniwa_lm::logger::{DataIngestionLog, ProvenanceEvent, ProvenanceLedger};
use oniwa_lm::reproducibility::compute_checksum_bytes;
use oniwa_lm::tokenizer::CharTokenizer;
use serde::Deserialize;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread::sleep;
use std::time::Duration;

/// Target work in recipe configuration file
#[derive(Debug, Deserialize, Clone)]
pub struct RecipeWorkTarget {
    pub author: String,
    pub title: String,
}

/// Structure of overall recipe configuration file
#[derive(Debug, Deserialize, Clone)]
pub struct RecipeConfig {
    pub curated_works: Vec<RecipeWorkTarget>,
}

/// Metadata for an Aozora Bunko work
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AozoraWorkEntry {
    pub work_id: String,
    pub title: String,
    pub author_last: String,
    pub author_first: String,
    pub card_url: String,
    pub text_zip_url: String,
    pub font_type: String,      // New/old orthography, etc.
    pub copyright_work: bool,   // true: copyrighted, false: none (PD)
    pub copyright_author: bool, // true: copyrighted, false: none (PD)
}

#[allow(dead_code)]
impl AozoraWorkEntry {
    pub fn author_full(&self) -> String {
        format!("{}{}", self.author_last, self.author_first)
    }

    pub fn is_public_domain(&self) -> bool {
        !self.copyright_work && !self.copyright_author
    }
}

pub struct AozoraPipeline {
    data_dir: PathBuf,
    raw_dir: PathBuf,
    corpus_dir: PathBuf,
    logs_dir: PathBuf,
    force_download: bool,
}

impl AozoraPipeline {
    pub fn new<P: AsRef<Path>>(base_data_dir: P, logs_dir: P) -> Self {
        let data_dir = base_data_dir.as_ref().to_path_buf();
        let raw_dir = data_dir.join("raw");
        let corpus_dir = data_dir.join("corpus");
        let logs_dir = logs_dir.as_ref().to_path_buf();

        fs::create_dir_all(&raw_dir).ok();
        fs::create_dir_all(&corpus_dir).ok();
        fs::create_dir_all(&logs_dir).ok();

        Self {
            data_dir,
            raw_dir,
            corpus_dir,
            logs_dir,
            force_download: false,
        }
    }

    pub fn set_force(&mut self, force: bool) {
        self.force_download = force;
    }

    /// Cleanly reset ingested data, corpus, and provenance ledger (safe initialization)
    pub fn clean_all(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("  🧹 Initializing existing dataset, corpus, and ledger...");
        if self.corpus_dir.exists() {
            fs::remove_dir_all(&self.corpus_dir).ok();
            fs::create_dir_all(&self.corpus_dir).ok();
        }
        if self.raw_dir.exists() {
            fs::remove_dir_all(&self.raw_dir).ok();
            fs::create_dir_all(&self.raw_dir).ok();
        }
        let ledger_path = self.logs_dir.join("ledger_index.jsonl");
        if ledger_path.exists() {
            let backup_path = self.logs_dir.join("ledger_index_backup.jsonl");
            fs::copy(&ledger_path, &backup_path).ok();
            fs::remove_file(&ledger_path).ok();
            println!(
                "  📋 Archived existing ledger to {:?}; starting new ledger",
                backup_path
            );
        }
        println!("  ✅ Initialization complete!");
        Ok(())
    }

    /// Download Aozora Bunko zip archive (skips if cached)
    fn download_file(&self, url: &str, dest_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        if !self.force_download && dest_path.exists() && dest_path.metadata()?.len() > 0 {
            // Cache hit
            return Ok(());
        }

        println!("  📥 Downloading: {} ...", url);
        // Server etiquette: specify custom User-Agent
        let user_agent =
            "oniwa-lm/0.1.0 (Public Domain AI Training Pipeline; Ethical AI without abuse)";
        let status = Command::new("curl")
            .arg("-s")
            .arg("-f")
            .arg("-L")
            .arg("-A")
            .arg(user_agent)
            .arg("-o")
            .arg(dest_path)
            .arg(url)
            .status()?;

        if !status.success() {
            return Err(format!(
                "Download failed (curl exit code: {:?}): {}",
                status.code(),
                url
            )
            .into());
        }

        // Pacing wait (1 second) to prevent server overload
        sleep(Duration::from_millis(1000));
        Ok(())
    }

    /// Extract Shift_JIS text from zip file and convert to UTF-8
    fn extract_and_decode_zip(
        &self,
        zip_path: &Path,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let file = File::open(zip_path)?;
        let mut archive = zip::ZipArchive::new(file)?;

        // Target the first text file
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let name = file.name().to_string();
            if name.ends_with(".txt") {
                let mut bytes = Vec::new();
                file.read_to_end(&mut bytes)?;

                // Shift_JIS -> UTF-8 decoding
                let (cow, _, had_errors) = encoding_rs::SHIFT_JIS.decode(&bytes);
                if had_errors {
                    eprintln!(
                        "  ⚠️ Character replacement occurred during Shift_JIS decoding: {}",
                        name
                    );
                }
                return Ok(cow.into_owned());
            }
        }

        Err(format!("No .txt file found inside zip archive: {:?}", zip_path).into())
    }

    /// Ingest, cleanse, save a single work to corpus_dir, and record in ledger
    pub fn ingest_single_work(
        &self,
        author: &str,
        title: &str,
        zip_url: &str,
        card_url: &str,
        license: &str,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let file_stem = format!("{}_{}", author, title);
        let zip_dest = self.raw_dir.join(format!("{}.zip", file_stem));
        let clean_dest = self.corpus_dir.join(format!("{}.txt", file_stem));

        // 1. Download
        self.download_file(zip_url, &zip_dest)?;

        // 2. Decompress & Shift_JIS decode
        let raw_text = self.extract_and_decode_zip(&zip_dest)?;
        let raw_sha256 = compute_checksum_bytes(raw_text.as_bytes());

        // 3. Cleanse (remove ruby, notes, header, footer)
        let cleaned = clean_aozora_text(&raw_text);
        fs::write(&clean_dest, &cleaned)?;

        let cleaned_bytes = cleaned.as_bytes();
        let cleaned_sha256 = compute_checksum_bytes(cleaned_bytes);
        let char_count = cleaned.chars().count();

        // 4. Record in provenance ledger (ledger_index.jsonl)
        let ledger_path = self.logs_dir.join("ledger_index.jsonl");
        let mut ledger = ProvenanceLedger::open(&ledger_path)?;
        ledger.record(&ProvenanceEvent::DataIngestion(DataIngestionLog {
            timestamp_utc: oniwa_lm::logger::current_timestamp_utc(),
            source_name: format!("Aozora Bunko: {} \"{}\"", author, title),
            source_url_or_path: card_url.to_string(),
            license: license.to_string(),
            raw_data_sha256: raw_sha256,
            raw_data_bytes: raw_text.len(),
            tokenized_sha256: cleaned_sha256,
            num_tokens: char_count,
            vocab_size: 0, // Updated upon corpus consolidation
            tokenizer_type: "Aozora Cleaner -> Character-level UTF-8".into(),
        }))?;

        println!(
            "  ✅ Ingestion & cleansing complete: \"{}\" ({} characters)",
            title, char_count
        );
        Ok(clean_dest)
    }

    /// Batch ingest works based on recipe configuration file (JSON)
    pub fn ingest_from_recipe<P: AsRef<Path>>(
        &self,
        recipe_path: P,
    ) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        let recipe_p = recipe_path.as_ref();
        if !recipe_p.exists() {
            return Err(format!("Recipe file {:?} not found.", recipe_p).into());
        }

        let recipe_str = fs::read_to_string(recipe_p)?;
        let recipe: RecipeConfig = serde_json::from_str(&recipe_str)?;

        println!("============================================================");
        println!(" 📚 Aozora Bunko Public Domain Recipe-Driven Ingestion");
        println!(
            "    (Config file: {:?} / {} works total)",
            recipe_p,
            recipe.curated_works.len()
        );
        println!("============================================================");

        let mut paths = Vec::new();
        for target in &recipe.curated_works {
            println!("▶ \"{}\" (Author: {})", target.title, target.author);
            if let Some(entry) = self.find_work_in_index(&target.author, &target.title)? {
                let path = self.ingest_single_work(
                    &entry.author_full(),
                    &entry.title,
                    &entry.text_zip_url,
                    &entry.card_url,
                    "Public Domain (Aozora Bunko Expired Copyright)",
                )?;
                paths.push(path);
            } else {
                eprintln!(
                    "  ⚠️ \"{}\" ({}) not found in official index",
                    target.title, target.author
                );
            }
        }

        Ok(paths)
    }

    /// Preset works (auto-ingested fallback recipe)
    pub fn ingest_presets(&self) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        let default_recipe = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("config")
            .join("recipes.json");
        if default_recipe.exists() {
            self.ingest_from_recipe(&default_recipe)
        } else {
            let preset_targets = [
                ("中島敦", "山月記"),
                ("太宰治", "走れメロス"),
                ("芥川竜之介", "羅生門"),
                ("芥川竜之介", "蜘蛛の糸"),
                ("宮沢賢治", "注文の多い料理店"),
                ("宮沢賢治", "セロ弾きのゴーシュ"),
                ("森鴎外", "高瀬舟"),
                ("夏目漱石", "坊っちゃん"),
            ];

            println!("============================================================");
            println!(" 📚 Aozora Bunko Public Domain Preset Ingestion (Fallback)");
            println!(
                "    (Target: expired copyright works only / {} works total)",
                preset_targets.len()
            );
            println!("============================================================");

            let mut paths = Vec::new();
            for (author, title) in preset_targets {
                println!("▶ \"{}\" (Author: {})", title, author);
                if let Some(entry) = self.find_work_in_index(author, title)? {
                    let path = self.ingest_single_work(
                        &entry.author_full(),
                        &entry.title,
                        &entry.text_zip_url,
                        &entry.card_url,
                        "Public Domain (Aozora Bunko Expired Copyright)",
                    )?;
                    paths.push(path);
                } else {
                    eprintln!(
                        "  ⚠️ \"{}\" ({}) not found in official index",
                        title, author
                    );
                }
            }

            Ok(paths)
        }
    }

    /// Search for a work in official index CSV
    pub fn find_work_in_index(
        &self,
        target_author: &str,
        target_title: &str,
    ) -> Result<Option<AozoraWorkEntry>, Box<dyn std::error::Error>> {
        let index_zip = self.raw_dir.join("list_person_all_extended_utf8.zip");
        let index_url = "https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip";
        self.download_file(index_url, &index_zip)?;

        let file = File::open(&index_zip)?;
        let mut archive = zip::ZipArchive::new(file)?;
        let mut csv_file = archive.by_index(0)?;

        let reader = BufReader::new(&mut csv_file);

        for line_res in reader.lines() {
            let line = line_res?;
            if line.starts_with("作品ID") || line.starts_with('\u{feff}') {
                continue;
            }

            let fields = parse_csv_line(&line);
            if fields.len() < 46 {
                continue;
            }

            let work_id = fields[0].clone();
            let title = fields[1].clone();
            let font_type = fields[9].clone();
            let copyright_work = fields[10] == "あり";
            let card_url = fields[13].clone();
            let author_last = fields[15].clone();
            let author_first = fields[16].clone();
            let copyright_author = fields[26] == "あり";
            let zip_url = fields[45].clone();

            let full_author = format!("{}{}", author_last, author_first);

            if (full_author.contains(target_author) || target_author.contains(&full_author))
                && title == target_title
                && !copyright_work
                && !copyright_author
                && font_type.contains("新字")
                && zip_url.starts_with("http")
                && zip_url.ends_with(".zip")
            {
                return Ok(Some(AozoraWorkEntry {
                    work_id,
                    title,
                    author_last,
                    author_first,
                    card_url,
                    text_zip_url: zip_url,
                    font_type,
                    copyright_work,
                    copyright_author,
                }));
            }
        }

        Ok(None)
    }

    /// Download Aozora Bunko official index CSV, search and batch ingest expired works
    pub fn search_and_ingest_by_author(
        &self,
        target_author: &str,
        max_works: usize,
    ) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        let index_zip = self.raw_dir.join("list_person_all_extended_utf8.zip");
        let index_url = "https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip";

        println!("============================================================");
        println!(
            " 🔍 Searching Aozora Bunko official index for author \"{}\" (max {} works)",
            target_author, max_works
        );
        println!("============================================================");

        self.download_file(index_url, &index_zip)?;

        let file = File::open(&index_zip)?;
        let mut archive = zip::ZipArchive::new(file)?;
        let mut csv_file = archive.by_index(0)?;

        let reader = BufReader::new(&mut csv_file);
        let mut collected = Vec::new();

        for line_res in reader.lines() {
            let line = line_res?;
            if line.starts_with("作品ID") || line.starts_with('\u{feff}') {
                continue; // Header
            }

            let fields: Vec<String> = parse_csv_line(&line);
            if fields.len() < 46 {
                continue;
            }

            let title = &fields[1];
            let font_type = &fields[9];
            let copyright_work = &fields[10]; // "なし" or "あり"
            let card_url = &fields[13];
            let author_last = &fields[15];
            let author_first = &fields[16];
            let copyright_author = &fields[26]; // "なし" or "あり"
            let zip_url = &fields[45];

            let full_author = format!("{}{}", author_last, author_first);

            // Compliance check: copyright flag is none and author matches
            if full_author.contains(target_author)
                && copyright_work == "なし"
                && copyright_author == "なし"
                && font_type.contains("新字")
                && zip_url.starts_with("http")
                && zip_url.ends_with(".zip")
            {
                println!(
                    "▶ Found: \"{}\" (Author: {}) - Copyright: Expired [PD]",
                    title, full_author
                );
                match self.ingest_single_work(
                    &full_author,
                    title,
                    zip_url,
                    card_url,
                    "Public Domain (Aozora Bunko Expired Copyright)",
                ) {
                    Ok(p) => {
                        collected.push(p);
                        if collected.len() >= max_works {
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("  ⚠️ Error ingesting \"{}\": {}", title, e);
                    }
                }
            }
        }

        Ok(collected)
    }

    /// Consolidate all texts in corpus_dir and regenerate vocab.json and tokens.bin
    pub fn build_combined_corpus(&self) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        println!("\n============================================================");
        println!(" 📦 Consolidating all texts & generating vocabulary and token binary");
        println!("============================================================");

        let mut combined_text = String::new();
        let mut work_count = 0;

        let mut entries: Vec<_> = fs::read_dir(&self.corpus_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("txt"))
            .collect();
        entries.sort_by_key(|e| e.path());

        for entry in entries {
            let path = entry.path();
            let text = fs::read_to_string(&path)?;
            if !text.is_empty() {
                combined_text.push_str(&text);
                combined_text.push_str("\n\n");
                work_count += 1;
            }
        }

        let combined_file = self.data_dir.join("corpus_combined.txt");
        fs::write(&combined_file, &combined_text)?;

        let total_chars = combined_text.chars().count();
        println!("  - Total consolidated works: {}", work_count);
        println!(
            "  - Total characters: {} ({:.2} KB)",
            total_chars,
            combined_text.len() as f32 / 1024.0
        );

        // Tokenization and vocab generation
        let tokenizer = CharTokenizer::ingest_file(
            &combined_file,
            &self.data_dir,
            &self.logs_dir,
            &format!(
                "Aozora Bunko Public Domain Consolidated Corpus ({} works)",
                work_count
            ),
            "https://www.aozora.gr.jp/",
            "Public Domain (Expired Copyright)",
        )?;

        println!(
            "  - Consolidated vocab size: {} characters",
            tokenizer.vocab_size()
        );
        println!("  - tokens.bin & vocab.json successfully updated!");

        Ok((work_count, total_chars))
    }
}

/// Simple quote-aware CSV line parser
fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        if ch == '"' {
            in_quotes = !in_quotes;
        } else if ch == ',' && !in_quotes {
            fields.push(field.trim_matches('"').to_string());
            field.clear();
        } else {
            field.push(ch);
        }
        i += 1;
    }
    fields.push(field.trim_matches('"').to_string());
    fields
}
