//! oniwa-decide Gatekeeper CLI — Standalone Gate-keeping Engine for Paper-Grade Evaluation
//!
//! Scans Git staging diffs (`git diff --cached`) or supplied unified-diff patches
//! at millisecond latency using the trained DecisionEngine.
//!
//! # Policy
//! - **Code files** (`.rs`, `.py`): Noul==true && confidence >= 80% → BLOCK (exit 1)
//! - **Document files** (`.md`, `.txt`): anomalies → WARNING only (exit 0)
//!
//! # Benchmark mode (`--bench`)
//! Logs per-hunk results to `logs/benchmarks/gatekeeper_eval.jsonl`

use oniwa_decide::{DecisionConfig, DecisionEngine, DecisionModel};
use oniwa_lm::reproducibility::DeterministicRng;
use oniwa_lm::tokenizer::CharTokenizer;
use serde::Serialize;
use std::env;
use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

// ─── Constants ───────────────────────────────────────────────────────

/// Supported source code extensions (BLOCK-eligible)
const CODE_EXTENSIONS: &[&str] = &["rs", "py"];
/// Supported document extensions (WARNING only)
const DOC_EXTENSIONS: &[&str] = &["md", "txt"];
/// Maximum token window for hunk extraction
const MAX_HUNK_TOKENS: usize = 256;
/// Minimum token window for hunk extraction
const MIN_HUNK_TOKENS: usize = 128;
/// Confidence threshold for blocking code anomalies
const BLOCK_CONFIDENCE_THRESHOLD: f32 = 0.80;

// ─── CLI argument parsing ────────────────────────────────────────────

struct CliArgs {
    bench: bool,
    checkpoint: Option<String>,
    help: bool,
    stdin_mode: bool,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = env::args().collect();
    let mut cli = CliArgs {
        bench: false,
        checkpoint: None,
        help: false,
        stdin_mode: false,
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--bench" => cli.bench = true,
            "--checkpoint" => {
                if let Some(v) = args.get(i + 1) {
                    cli.checkpoint = Some(v.clone());
                    i += 1;
                }
            }
            "--help" | "-h" => cli.help = true,
            "--stdin" => cli.stdin_mode = true,
            _ => {}
        }
        i += 1;
    }
    cli
}

fn print_help() {
    println!("Usage: gatekeeper [OPTIONS]");
    println!();
    println!("Standalone gate-keeping engine for oniwa-decide.");
    println!("Scans Git staging diffs or piped unified-diff patches.");
    println!();
    println!("Options:");
    println!("  --bench        Enable benchmark mode (log results to logs/benchmarks/gatekeeper_eval.jsonl)");
    println!("  --checkpoint   Path to checkpoint directory (default: auto-detect)");
    println!("  --stdin        Read diff from stdin instead of `git diff --cached`");
    println!("  -h, --help     Show this help message");
}

// ─── Unified diff parser ─────────────────────────────────────────────

/// A parsed hunk from a unified diff
#[derive(Debug, Clone)]
struct DiffHunk {
    file_path: String,
    extension: String,
    is_code: bool,
    content: String,
}

/// Classify a file extension
fn classify_extension(ext: &str) -> FileKind {
    let lower = ext.to_lowercase();
    if CODE_EXTENSIONS.contains(&lower.as_str()) {
        FileKind::Code
    } else if DOC_EXTENSIONS.contains(&lower.as_str()) {
        FileKind::Document
    } else {
        FileKind::Skip
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum FileKind {
    Code,
    Document,
    Skip,
}

/// Extract file extension from a path
fn extract_extension(path: &str) -> String {
    path.rsplit('.').next().unwrap_or("").to_string()
}

/// Parse unified diff text into hunks.
///
/// Extracts added lines (`+`) and surrounding context for each target file.
/// Skips binary files, deletion-only hunks, and unsupported extensions.
fn parse_unified_diff(diff_text: &str) -> Vec<DiffHunk> {
    let mut hunks = Vec::new();
    let mut current_file: Option<String> = None;
    let mut current_ext = String::new();
    let mut current_kind = FileKind::Skip;
    let mut added_lines: Vec<String> = Vec::new();
    let mut context_lines: Vec<String> = Vec::new();

    for line in diff_text.lines() {
        // Detect new file in diff
        if line.starts_with("+++ b/") || line.starts_with("+++ ") {
            // Flush previous file's accumulated hunk
            if let Some(ref file) = current_file {
                if current_kind != FileKind::Skip && !added_lines.is_empty() {
                    let content = build_hunk_content(&added_lines, &context_lines);
                    hunks.push(DiffHunk {
                        file_path: file.clone(),
                        extension: current_ext.clone(),
                        is_code: current_kind == FileKind::Code,
                        content,
                    });
                }
            }

            // Parse new file path
            let path = if let Some(stripped) = line.strip_prefix("+++ b/") {
                stripped
            } else if let Some(stripped) = line.strip_prefix("+++ ") {
                stripped
            } else {
                line
            };
            current_ext = extract_extension(path);
            current_kind = classify_extension(&current_ext);
            current_file = Some(path.to_string());
            added_lines.clear();
            context_lines.clear();
            continue;
        }

        // Skip diff header lines
        if line.starts_with("--- ") || line.starts_with("diff ") || line.starts_with("index ") {
            continue;
        }

        // Skip hunk headers (but don't reset context — they're part of navigation)
        if line.starts_with("@@") {
            continue;
        }

        // Collect added and context lines
        if current_kind != FileKind::Skip {
            if let Some(stripped) = line.strip_prefix('+') {
                added_lines.push(stripped.to_string());
            } else if !line.starts_with('-') {
                // Context line (no prefix, or space-prefixed)
                let ctx = line.strip_prefix(' ').unwrap_or(line);
                context_lines.push(ctx.to_string());
            }
            // Lines starting with '-' (pure deletions) are skipped
        }
    }

    // Flush last file
    if let Some(ref file) = current_file {
        if current_kind != FileKind::Skip && !added_lines.is_empty() {
            let content = build_hunk_content(&added_lines, &context_lines);
            hunks.push(DiffHunk {
                file_path: file.clone(),
                extension: current_ext.clone(),
                is_code: current_kind == FileKind::Code,
                content,
            });
        }
    }

    hunks
}

/// Build hunk content from added lines with surrounding context,
/// respecting the MIN/MAX token window constraints.
fn build_hunk_content(added: &[String], context: &[String]) -> String {
    // Start with added lines
    let mut parts: Vec<&str> = added.iter().map(|s| s.as_str()).collect();

    // Estimate rough char count (1 token ≈ 4 chars as rough heuristic)
    let added_chars: usize = parts.iter().map(|s| s.len()).sum::<usize>() + parts.len(); // +newlines
    let target_chars = MAX_HUNK_TOKENS * 4;

    // If added lines alone are below minimum, pad with context
    if added_chars < MIN_HUNK_TOKENS * 4 {
        let budget = target_chars.saturating_sub(added_chars);
        let mut used = 0;
        for ctx in context.iter() {
            if used + ctx.len() + 1 > budget {
                break;
            }
            parts.push(ctx.as_str());
            used += ctx.len() + 1;
        }
    }

    // Truncate to MAX_HUNK_TOKENS * 4 chars
    let mut result = parts.join("\n");
    if result.len() > target_chars {
        result.truncate(target_chars);
    }
    result
}

// ─── Benchmark record ────────────────────────────────────────────────

#[derive(Serialize)]
struct BenchRecord {
    file: String,
    extension: String,
    is_code: bool,
    noul: bool,
    noul_confidence: f32,
    score: f32,
    score_confidence: f32,
    choice_idx: usize,
    choice_confidence: f32,
    inference_time_ms: u128,
    entropy_bits: f32,
    verdict: String,
}

/// Calculate output entropy from probability distribution (in bits)
fn entropy_bits(probs: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for &p in probs {
        if p > 1e-10 {
            h -= p * p.log2();
        }
    }
    h
}

// ─── Main ────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = parse_args();

    if cli.help {
        print_help();
        return Ok(());
    }

    println!("============================================================");
    println!(" 🚧 ONIWA Gatekeeper: Standalone Gate-keeping Engine");
    println!("    Pure Rust · Offline · Sub-millisecond Inference");
    println!("============================================================");

    // ── Step 1: Obtain diff text ─────────────────────────────────────
    let diff_text = if cli.stdin_mode {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        buf
    } else {
        let output = std::process::Command::new("git")
            .args(["diff", "--cached"])
            .output()?;
        String::from_utf8_lossy(&output.stdout).into_owned()
    };

    // ── Step 2: Parse diff into hunks ────────────────────────────────
    let hunks = parse_unified_diff(&diff_text);

    if hunks.is_empty() {
        println!("\n  ✅ No eligible staged changes found. Nothing to scan.");
        std::process::exit(0);
    }

    println!("\n  📋 Found {} hunk(s) to scan.", hunks.len());

    // ── Step 3: Load DecisionEngine ──────────────────────────────────
    let workspace_root = oniwa_lm::find_workspace_root();
    let data_dir = workspace_root.join("data");
    let vocab_path = data_dir.join("vocab.json");
    let tokenizer = if vocab_path.exists() {
        CharTokenizer::load_vocab(&vocab_path)?
    } else {
        CharTokenizer::build_from_text("abcdefghijklmnopqrstuvwxyz 0123456789")
    };

    let base_dir = if workspace_root
        .join("crates/oniwa-decide/checkpoints")
        .is_dir()
    {
        workspace_root.join("crates/oniwa-decide")
    } else {
        workspace_root.clone()
    };

    let checkpoint_dir = if let Some(ref ckpt) = cli.checkpoint {
        PathBuf::from(ckpt)
    } else {
        let possible = [
            base_dir.join("checkpoints/quaternion_head/best"),
            base_dir.join("checkpoints/best"),
            base_dir.join("checkpoints/standard_baseline/best"),
        ];
        possible
            .iter()
            .find(|p| p.join("meta.json").exists())
            .cloned()
            .unwrap_or_else(|| base_dir.join("checkpoints/best"))
    };

    let engine = if checkpoint_dir.join("meta.json").exists() {
        println!("  💾 Loaded checkpoint: {:?}", checkpoint_dir);
        DecisionEngine::load_from_dir(&checkpoint_dir, tokenizer)?
    } else {
        println!(
            "  ⚠️  No checkpoint detected: running with randomly initialized model (untrained)"
        );
        let config = DecisionConfig {
            vocab_size: tokenizer.vocab_size(),
            ..Default::default()
        };
        let mut rng = DeterministicRng::new(42);
        let model = DecisionModel::new(config, &mut rng);
        DecisionEngine::new(model, tokenizer)
    };

    // ── Step 4: Scan each hunk ───────────────────────────────────────
    let mut blocked = false;
    let mut bench_records: Vec<BenchRecord> = Vec::new();

    for (idx, hunk) in hunks.iter().enumerate() {
        let start = Instant::now();
        let decision = engine.audit_text(&hunk.content);
        let elapsed_ms = start.elapsed().as_millis();

        let noul = decision.syntax_anomaly.value;
        let noul_conf = decision.syntax_anomaly.confidence;
        let score = decision.complexity.value;
        let ent = entropy_bits(&decision.category.probabilities);

        let verdict = if hunk.is_code && noul && noul_conf >= BLOCK_CONFIDENCE_THRESHOLD {
            blocked = true;
            "BLOCK"
        } else if noul {
            "WARNING"
        } else {
            "PASS"
        };

        // Print per-hunk result
        let icon = match verdict {
            "BLOCK" => "🚫",
            "WARNING" => "⚠️",
            _ => "✅",
        };
        println!(
            "\n  [{}/{}] {} {} — {}",
            idx + 1,
            hunks.len(),
            icon,
            verdict,
            hunk.file_path
        );
        println!(
            "         Noul={} (conf={:.1}%), Score={:.2}, Latency={}ms",
            noul,
            noul_conf * 100.0,
            score,
            elapsed_ms
        );

        if cli.bench {
            bench_records.push(BenchRecord {
                file: hunk.file_path.clone(),
                extension: hunk.extension.clone(),
                is_code: hunk.is_code,
                noul,
                noul_confidence: noul_conf,
                score,
                score_confidence: decision.complexity.confidence,
                choice_idx: decision.category.value as usize,
                choice_confidence: decision.category.confidence,
                inference_time_ms: elapsed_ms,
                entropy_bits: ent,
                verdict: verdict.to_string(),
            });
        }
    }

    // ── Step 5: Write benchmark log ──────────────────────────────────
    if cli.bench && !bench_records.is_empty() {
        let log_dir = workspace_root.join("logs/benchmarks");
        fs::create_dir_all(&log_dir)?;
        let log_path = log_dir.join("gatekeeper_eval.jsonl");
        let mut content = String::new();
        for rec in &bench_records {
            if let Ok(line) = serde_json::to_string(rec) {
                content.push_str(&line);
                content.push('\n');
            }
        }
        // Append to existing file
        use std::io::Write;
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;
        file.write_all(content.as_bytes())?;
        println!("\n  📊 Benchmark results appended to {:?}", log_path);
    }

    // ── Step 6: Exit with appropriate code ───────────────────────────
    println!("\n============================================================");
    if blocked {
        println!(" 🚫 BLOCKED: Code anomaly detected with high confidence.");
        println!("    Fix the flagged issues and re-stage.");
        println!("============================================================");
        std::process::exit(1);
    } else {
        println!(" ✅ PASSED: No blocking anomalies detected.");
        println!("============================================================");
        std::process::exit(0);
    }
}

// ─── Unit-testable functions ─────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_diff_returns_empty() {
        assert!(parse_unified_diff("").is_empty());
    }

    #[test]
    fn parse_rust_diff_extracts_hunk() {
        let diff = "diff --git a/src/main.rs b/src/main.rs\n\
                    index abc..def 100644\n\
                    --- a/src/main.rs\n\
                    +++ b/src/main.rs\n\
                    @@ -1,3 +1,4 @@\n\
                     fn main() {\n\
                    +    let x = 42;\n\
                         println!(\"hello\");\n\
                     }";
        let hunks = parse_unified_diff(diff);
        assert_eq!(hunks.len(), 1);
        assert_eq!(hunks[0].file_path, "src/main.rs");
        assert_eq!(hunks[0].extension, "rs");
        assert!(hunks[0].is_code);
        assert!(hunks[0].content.contains("let x = 42"));
    }

    #[test]
    fn skip_binary_extensions() {
        let diff = "diff --git a/image.png b/image.png\n\
                    --- a/image.png\n\
                    +++ b/image.png\n\
                    @@ -0,0 +1 @@\n\
                    +binary content";
        let hunks = parse_unified_diff(diff);
        assert!(hunks.is_empty());
    }

    #[test]
    fn doc_extension_is_not_code() {
        let diff = "diff --git a/README.md b/README.md\n\
                    --- a/README.md\n\
                    +++ b/README.md\n\
                    @@ -1,2 +1,3 @@\n\
                     # Title\n\
                    +New content here\n\
                     End";
        let hunks = parse_unified_diff(diff);
        assert_eq!(hunks.len(), 1);
        assert!(!hunks[0].is_code);
        assert_eq!(hunks[0].extension, "md");
    }

    #[test]
    fn deletion_only_diff_is_skipped() {
        let diff = "diff --git a/src/lib.rs b/src/lib.rs\n\
                    --- a/src/lib.rs\n\
                    +++ b/src/lib.rs\n\
                    @@ -1,3 +1,2 @@\n\
                     fn foo() {\n\
                    -    let old = 1;\n\
                     }";
        let hunks = parse_unified_diff(diff);
        assert!(hunks.is_empty());
    }

    #[test]
    fn entropy_uniform_distribution() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let h = entropy_bits(&probs);
        assert!((h - 2.0).abs() < 0.01); // log2(4) = 2 bits
    }

    #[test]
    fn classify_rs_as_code() {
        assert_eq!(classify_extension("rs"), FileKind::Code);
        assert_eq!(classify_extension("py"), FileKind::Code);
    }

    #[test]
    fn classify_md_as_document() {
        assert_eq!(classify_extension("md"), FileKind::Document);
        assert_eq!(classify_extension("txt"), FileKind::Document);
    }

    #[test]
    fn classify_png_as_skip() {
        assert_eq!(classify_extension("png"), FileKind::Skip);
        assert_eq!(classify_extension("lock"), FileKind::Skip);
    }
}
