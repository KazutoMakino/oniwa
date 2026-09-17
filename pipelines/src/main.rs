//! oniwa-pipeline: Dataset management & ethical data ingestion CLI
//!
//! ONIWA: Organic Non-datacenter Intelligence Without Abuse
//! Compliant with legal terms (public domain works only, open licenses, official APIs),
//! safely retrieves, cleanses, and logs corpora into the provenance ledger.

mod aozora;
mod arxiv;
mod cleaner;
mod code;
mod egov;
mod techdocs;

use aozora::AozoraPipeline;
use arxiv::ArxivPipeline;
use code::CodePipeline;
use egov::EgovPipeline;
use std::path::Path;
use techdocs::TechDocsPipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!(" 📚 ONIWA: Clean Open-Source Data Ingestion Pipeline");
    println!(
        "    (Aozora PD & e-Gov Laws & arXiv Open Science & Official Tech Specs & Clean Code)"
    );
    println!("============================================================\n");

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root_dir = manifest_dir.parent().unwrap_or(manifest_dir);
    let data_dir = root_dir.join("data");
    let logs_dir = root_dir.join("logs");

    let mut aozora_pipeline = AozoraPipeline::new(&data_dir, &logs_dir);
    let mut egov_pipeline = EgovPipeline::new(&data_dir, &logs_dir);
    let mut arxiv_pipeline = ArxivPipeline::new(&data_dir, &logs_dir);
    let mut techdocs_pipeline = TechDocsPipeline::new(&data_dir, &logs_dir);
    let mut code_pipeline = CodePipeline::new(&data_dir, &logs_dir);

    let args: Vec<String> = std::env::args().collect();
    let mut do_preset = false;
    let mut do_laws = false;
    let mut do_arxiv = false;
    let mut do_techdocs = false;
    let mut do_code = false;
    let mut arxiv_limit = 10usize;
    let mut arxiv_cat = "cs.AI".to_string();
    let mut do_all = false;
    let mut target_recipe: Option<String> = None;
    let mut target_author: Option<String> = None;
    let mut limit = 5usize;
    let mut do_build = false;
    let mut do_status = false;
    let mut do_force = false;
    let mut do_clean = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--recipe" => {
                if let Some(val) = args.get(i + 1) {
                    target_recipe = Some(val.clone());
                    i += 1;
                }
            }
            "--preset" => {
                do_preset = true;
            }
            "--laws" => {
                do_laws = true;
            }
            "--arxiv" => {
                do_arxiv = true;
                if let Some(val) = args.get(i + 1) {
                    if let Ok(num) = val.parse::<usize>() {
                        arxiv_limit = num;
                        i += 1;
                    }
                }
            }
            "--arxiv-cat" => {
                if let Some(val) = args.get(i + 1) {
                    arxiv_cat = val.clone();
                    i += 1;
                }
            }
            "--techdocs" => {
                do_techdocs = true;
            }
            "--code" => {
                do_code = true;
            }
            "--all" => {
                do_all = true;
                do_preset = true;
                do_laws = true;
                do_techdocs = true;
                do_code = true;
                do_arxiv = true;
                arxiv_limit = 5;
                do_build = true;
            }
            "--author" => {
                if let Some(val) = args.get(i + 1) {
                    target_author = Some(val.clone());
                    i += 1;
                }
            }
            "--limit" => {
                if let Some(val) = args.get(i + 1) {
                    limit = val.parse().unwrap_or(5);
                    i += 1;
                }
            }
            "--build" | "--build-corpus" => {
                do_build = true;
            }
            "--force" => {
                do_force = true;
            }
            "--clean" => {
                do_clean = true;
            }
            "--status" => {
                do_status = true;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    if !do_preset
        && !do_laws
        && !do_arxiv
        && !do_techdocs
        && !do_code
        && !do_all
        && target_recipe.is_none()
        && target_author.is_none()
        && !do_build
        && !do_status
        && !do_clean
    {
        print_help();
        return Ok(());
    }

    if do_force {
        aozora_pipeline.set_force(true);
        egov_pipeline.set_force(true);
        arxiv_pipeline.set_force(true);
        techdocs_pipeline.set_force(true);
        code_pipeline.set_force(true);
    }

    // 0. Clean and reset processing
    if do_clean {
        aozora_pipeline.clean_all()?;
    }

    // 1. Aozora Bunko: Fetch recipes or presets
    if let Some(ref recipe_path) = target_recipe {
        aozora_pipeline.ingest_from_recipe(recipe_path)?;
    } else if do_preset {
        aozora_pipeline.ingest_presets()?;
    }

    // 2. Aozora Bunko: Search and ingest works by author
    if let Some(author) = target_author {
        aozora_pipeline.search_and_ingest_by_author(&author, limit)?;
    }

    // 3. e-Gov: Fetch fundamental legal open data
    if do_laws {
        egov_pipeline.ingest_default_laws()?;
    }

    // 4. Official technical docs & code specifications
    if do_techdocs {
        techdocs_pipeline.ingest_default_techdocs()?;
    }

    // 5. Basic algorithms & clean code (Python/Rust)
    if do_code {
        code_pipeline.ingest_default_code()?;
    }

    // 6. arXiv: Open-access scientific paper abstracts
    if do_arxiv {
        arxiv_pipeline.ingest_category(&arxiv_cat, arxiv_limit)?;
    }

    // 7. Rebuild unified corpus (aggregate all corpus/*.txt files)
    if do_build
        || do_preset
        || do_laws
        || do_techdocs
        || do_code
        || do_arxiv
        || do_all
        || target_recipe.is_some()
    {
        aozora_pipeline.build_combined_corpus()?;
    }

    // 8. Display status
    if do_status
        || (!do_preset
            && !do_laws
            && !do_techdocs
            && !do_code
            && !do_arxiv
            && !do_all
            && target_recipe.is_none()
            && !do_build)
    {
        show_status(&data_dir, &logs_dir)?;
    }

    println!("\n✨ Ingestion pipeline execution completed successfully.");
    println!("To start training on the ingested corpus, run:");
    println!("  $ cargo run --release -p oniwa-lm --bin train -- --steps 500 --reset --prompt \"メロスは、\"");
    println!("============================================================");

    Ok(())
}

fn print_help() {
    println!("Usage:");
    println!("  cargo run --release -p oniwa-pipeline -- [options]\n");
    println!("Options:");
    println!("  --all               Ingest all sources (Aozora Bunko + Laws + Tech Docs + Code + arXiv) and rebuild the combined corpus");
    println!("  --preset            Ingest predefined classic works from Aozora Bunko (configured in recipes.json)");
    println!(
        "  --laws              Ingest fundamental Japanese laws (Constitution, Penal Code, Copyright Law, Civil Code) via e-Gov API"
    );
    println!(
        "  --techdocs          Ingest official open-source tech docs (Rust tutorials, Serde, Regex, etc.)"
    );
    println!("  --code              Ingest open-source algorithm implementations (Python/Rust: factorial, fibonacci, sort, search, etc.)");
    println!(
        "  --arxiv [count]     Ingest open-access paper abstracts from arXiv API (default: 10)"
    );
    println!(
        "  --arxiv-cat <cat>   Specify arXiv category (e.g. cs.AI, cs.CL, cs.LG / default: cs.AI)"
    );
    println!("  --recipe <path>     Ingest works specified in a JSON recipe file (e.g. pipelines/config/recipes.json)");
    println!(
        "  --author <name>     Search and ingest public domain works by author from Aozora Bunko catalog"
    );
    println!(
        "  --limit <count>     Maximum number of works to fetch per author search (default: 5)"
    );
    println!("  --build             Merge all ingested texts into combined corpus and rebuild tokens.bin & vocab.json");
    println!("  --force             Bypass cache and force re-download & re-parse");
    println!("  --clean             Reset ingested raw data, corpus, and ledger (backs up existing ledger)");
    println!(
        "  --status            Display status of current corpus files and provenance ledger records\n"
    );
    println!("Examples:");
    println!("  $ cargo run --release -p oniwa-pipeline -- --all");
    println!("  $ cargo run --release -p oniwa-pipeline -- --techdocs");
    println!("  $ cargo run --release -p oniwa-pipeline -- --arxiv 10");
    println!("  $ cargo run --release -p oniwa-pipeline -- --laws");
    println!("  $ cargo run --release -p oniwa-pipeline -- --status");
}

fn show_status(data_dir: &Path, logs_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let corpus_dir = data_dir.join("corpus");
    let ledger_path = logs_dir.join("ledger_index.jsonl");

    println!("============================================================");
    println!(" 📊 Dataset Storage Status (data/corpus/)");
    println!("============================================================");

    if corpus_dir.exists() {
        let mut entries: Vec<_> = std::fs::read_dir(&corpus_dir)?
            .filter_map(|e| e.ok())
            .collect();
        entries.sort_by_key(|e| e.path());

        let mut total_chars = 0;
        for (idx, entry) in entries.iter().enumerate() {
            let path = entry.path();
            let meta = entry.metadata()?;
            let content = std::fs::read_to_string(&path).unwrap_or_default();
            let chars = content.chars().count();
            total_chars += chars;
            println!(
                "  [{:2}] {:<35} ({:6} chars, {:.1} KB)",
                idx + 1,
                path.file_name().unwrap_or_default().to_string_lossy(),
                chars,
                meta.len() as f32 / 1024.0
            );
        }
        println!("------------------------------------------------------------");
        println!("  - Total items: {} items", entries.len());
        println!("  - Total characters: {} chars", total_chars);
    } else {
        println!("  (No works ingested yet. Ingest with --preset or --all)");
    }

    println!("\n 📜 Provenance Ledger (logs/ledger_index.jsonl):");
    if ledger_path.exists() {
        let count = std::fs::read_to_string(&ledger_path)?.lines().count();
        println!("  - Recorded provenance events: {} events", count);
    }

    Ok(())
}
