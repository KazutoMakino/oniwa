//! オープンソース公式技術ドキュメント・コード仕様収集パイプライン
//!
//! ONIWA: Organic Non-datacenter Intelligence Without Abuse
//! 公式オープンソースリポジトリ（Rust公式ドキュメント等、MIT / Apache-2.0）から
//! 高品質なプログラミング解説とソースコードの実例を取得・クレンジングし、
//! 将来的なコーディング能力・アルゴリズム理解のためのクリーンな土壌として系譜台帳に記録します。

use oniwa_lm::logger::{DataIngestionLog, ProvenanceEvent, ProvenanceLedger};
use oniwa_lm::reproducibility::compute_checksum_bytes;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread::sleep;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct TechDocTarget {
    pub name: &'static str,
    pub title: &'static str,
    pub url: &'static str,
    pub license: &'static str,
    pub description: &'static str,
}

pub const DEFAULT_TECHDOCS: &[TechDocTarget] = &[
    TechDocTarget {
        name: "rust_hello_world",
        title: "Rust: Hello World & Macros",
        url: "https://raw.githubusercontent.com/rust-lang/rust-by-example/master/src/hello.md",
        license: "MIT / Apache-2.0 (Rust by Example / rust-lang)",
        description: "基本構文、コメント記法、println!マクロ仕様",
    },
    TechDocTarget {
        name: "rust_primitives",
        title: "Rust: Primitive Types",
        url: "https://raw.githubusercontent.com/rust-lang/rust-by-example/master/src/primitives.md",
        license: "MIT / Apache-2.0 (Rust by Example / rust-lang)",
        description: "スカラ型、配列、スライス、タプル型",
    },
    TechDocTarget {
        name: "rust_custom_types",
        title: "Rust: Custom Types (struct & enum)",
        url: "https://raw.githubusercontent.com/rust-lang/rust-by-example/master/src/custom_types.md",
        license: "MIT / Apache-2.0 (Rust by Example / rust-lang)",
        description: "構造体、列挙型、定数の定義とパターンマッチ",
    },
    TechDocTarget {
        name: "rust_variable_bindings",
        title: "Rust: Variable Bindings & Mutability",
        url: "https://raw.githubusercontent.com/rust-lang/rust-by-example/master/src/variable_bindings.md",
        license: "MIT / Apache-2.0 (Rust by Example / rust-lang)",
        description: "変数束縛、可変性(mut)、スコープとシャドーイング",
    },
    TechDocTarget {
        name: "rust_types",
        title: "Rust: Types & Casting",
        url: "https://raw.githubusercontent.com/rust-lang/rust-by-example/master/src/types.md",
        license: "MIT / Apache-2.0 (Rust by Example / rust-lang)",
        description: "明示的キャスト、型推論、エイリアス定義",
    },
    TechDocTarget {
        name: "rust_flow_control",
        title: "Rust: Flow of Control",
        url: "https://raw.githubusercontent.com/rust-lang/rust-by-example/master/src/flow_control.md",
        license: "MIT / Apache-2.0 (Rust by Example / rust-lang)",
        description: "if/else、loop、while、for、match分岐",
    },
    TechDocTarget {
        name: "rust_functions",
        title: "Rust: Functions & Closures",
        url: "https://raw.githubusercontent.com/rust-lang/rust-by-example/master/src/fn.md",
        license: "MIT / Apache-2.0 (Rust by Example / rust-lang)",
        description: "関数定義、メソッド、クロージャ、高階関数",
    },
    TechDocTarget {
        name: "rust_traits",
        title: "Rust: Traits & Polymorphism",
        url: "https://raw.githubusercontent.com/rust-lang/rust-by-example/master/src/trait.md",
        license: "MIT / Apache-2.0 (Rust by Example / rust-lang)",
        description: "トレイト定義、ジェネリクス境界、動的ディスパッチ",
    },
    TechDocTarget {
        name: "rust_generics",
        title: "Rust: Generics",
        url: "https://raw.githubusercontent.com/rust-lang/rust-by-example/master/src/generics.md",
        license: "MIT / Apache-2.0 (Rust by Example / rust-lang)",
        description: "ジェネリック関数・構造体・PhantomData",
    },
    TechDocTarget {
        name: "rust_error_handling",
        title: "Rust: Error Handling",
        url: "https://raw.githubusercontent.com/rust-lang/rust-by-example/master/src/error.md",
        license: "MIT / Apache-2.0 (Rust by Example / rust-lang)",
        description: "panic、Option、Result型、?演算子によるエラー伝播",
    },
];

pub struct TechDocsPipeline {
    #[allow(dead_code)]
    pub data_dir: PathBuf,
    pub logs_dir: PathBuf,
    pub corpus_dir: PathBuf,
    pub raw_dir: PathBuf,
    pub force_download: bool,
}

impl TechDocsPipeline {
    pub fn new<P: AsRef<Path>>(data_dir: P, logs_dir: P) -> Self {
        let data_p = data_dir.as_ref().to_path_buf();
        let logs_p = logs_dir.as_ref().to_path_buf();
        let corpus_dir = data_p.join("corpus");
        let raw_dir = data_p.join("raw");

        fs::create_dir_all(&corpus_dir).ok();
        fs::create_dir_all(&raw_dir).ok();

        Self {
            data_dir: data_p,
            logs_dir: logs_p,
            corpus_dir,
            raw_dir,
            force_download: false,
        }
    }

    pub fn set_force(&mut self, force: bool) {
        self.force_download = force;
    }

    /// 技術ドキュメントMarkdownをダウンロード
    pub fn download_doc(&self, target: &TechDocTarget) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let dest_path = self.raw_dir.join(format!("techdoc_{}.md", target.name));
        if !self.force_download && dest_path.exists() && dest_path.metadata()?.len() > 0 {
            return Ok(dest_path);
        }

        println!("  📥 [TechDoc] ダウンロード中: 『{}』 ...", target.title);

        let user_agent = "oniwa-pipeline/0.1.0 (Ethical Open Source Docs Ingestion; Educational AI)";
        let status = Command::new("curl")
            .arg("-s")
            .arg("-f")
            .arg("-L")
            .arg("-A")
            .arg(user_agent)
            .arg("-o")
            .arg(&dest_path)
            .arg(target.url)
            .status()?;

        if !status.success() {
            return Err(format!("技術ドキュメントダウンロード失敗: {}", target.url).into());
        }

        sleep(Duration::from_millis(1000));
        Ok(dest_path)
    }

    /// 単一の技術ドキュメントを前処理してコーパスに保存し、監査台帳に記録
    pub fn ingest_single_doc(&self, target: &TechDocTarget) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let raw_path = self.download_doc(target)?;
        let raw_bytes = fs::read(&raw_path)?;
        let raw_sha256 = compute_checksum_bytes(&raw_bytes);
        let raw_text = String::from_utf8_lossy(&raw_bytes);

        // Markdown のクレンジング（余分なHTMLタグ除去やメタデータ整理）
        let clean_text = clean_tech_markdown(&raw_text);

        let out_filename = format!("techdoc_{}.txt", target.name);
        let out_path = self.corpus_dir.join(&out_filename);
        fs::write(&out_path, clean_text.as_bytes())?;

        let clean_sha256 = compute_checksum_bytes(clean_text.as_bytes());
        let char_count = clean_text.chars().count();

        println!(
            "  💻 保存完了: 『{}』 ({}文字 / {:.2} KB)",
            target.title,
            char_count,
            clean_text.len() as f32 / 1024.0
        );

        let ledger_path = self.logs_dir.join("ledger_index.jsonl");
        let mut ledger = ProvenanceLedger::open(&ledger_path)?;

        ledger.record(&ProvenanceEvent::DataIngestion(DataIngestionLog {
            timestamp_utc: oniwa_lm::logger::current_timestamp_utc(),
            source_name: format!("TechDoc: 『{}』", target.title),
            source_url_or_path: target.url.to_string(),
            license: target.license.to_string(),
            raw_data_sha256: raw_sha256,
            raw_data_bytes: raw_bytes.len(),
            tokenized_sha256: clean_sha256,
            num_tokens: char_count,
            vocab_size: 0,
            tokenizer_type: "Raw Cleaned Text (Markdown & Code)".to_string(),
        }))?;

        Ok(out_path)
    }

    /// デフォルトの技術ドキュメント群を一括取得
    pub fn ingest_default_techdocs(&self) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        println!("============================================================");
        println!(" 💻 公式オープンソース技術ドキュメント & コード仕様一括収集");
        println!("    (根拠: MIT / Apache-2.0 / CC0 オープンライセンス)");
        println!("============================================================");

        let mut paths = Vec::new();
        for target in DEFAULT_TECHDOCS {
            println!("▶ 『{}』 ({})", target.title, target.description);
            let path = self.ingest_single_doc(target)?;
            paths.push(path);
        }

        Ok(paths)
    }
}

/// 技術系Markdownのクレンジング（リンクタグやhtmlタグの整形）
pub fn clean_tech_markdown(raw_md: &str) -> String {
    let mut cleaned = String::with_capacity(raw_md.len());
    for line in raw_md.lines() {
        let trimmed = line.trim();
        // HTMLコメントの除去
        if trimmed.starts_with("<!--") && trimmed.ends_with("-->") {
            continue;
        }
        cleaned.push_str(line);
        cleaned.push('\n');
    }
    cleaned.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_tech_markdown() {
        let sample = "# Header\n<!-- comment -->\n```rust\nfn main() {}\n```\n";
        let cleaned = clean_tech_markdown(sample);
        assert!(!cleaned.contains("<!-- comment -->"));
        assert!(cleaned.contains("fn main() {}"));
    }
}
