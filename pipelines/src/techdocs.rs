//! オープンソース公式技術ドキュメント・主要クレート仕様収集パイプライン
//!
//! ONIWA: Organic Non-datacenter Intelligence Without Abuse
//! 公式オープンソースリポジトリ（Rust公式、serde, regex, rand, Python標準ライブラリ等、MIT / Apache-2.0 / PSF）から
//! 高品質なプログラミング解説とソースコードの実例を取得・クレンジングし、
//! 小さな知性（oniwa-v2）の実践的プログラミング能力・API構文理解の土壌として系譜台帳に記録します。

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
    pub seed_content: &'static str,
}

pub const DEFAULT_TECHDOCS: &[TechDocTarget] = &[
    // --- Rust 公式チュートリアル (Rust by Example) ---
    TechDocTarget {
        name: "rust_hello_world",
        title: "Rust: Hello World & Macros",
        url: "https://raw.githubusercontent.com/rust-lang/rust-by-example/master/src/hello.md",
        license: "MIT / Apache-2.0 (Rust by Example / rust-lang)",
        description: "基本構文、コメント記法、println!マクロ仕様",
        seed_content: "# Hello World\n\n```rust\nfn main() {\n    println!(\"Hello World!\");\n}\n```\n",
    },
    TechDocTarget {
        name: "rust_primitives",
        title: "Rust: Primitive Types",
        url: "https://raw.githubusercontent.com/rust-lang/rust-by-example/master/src/primitives.md",
        license: "MIT / Apache-2.0 (Rust by Example / rust-lang)",
        description: "スカラ型、配列、スライス、タプル型",
        seed_content: "# Primitives\n\nRust provides access to a wide variety of primitives: signed integers (i8, i16, i32, i64, isize), unsigned integers (u8, u16, u32, u64, usize), floating point (f32, f64), char, bool, unit ().\n",
    },
    TechDocTarget {
        name: "rust_custom_types",
        title: "Rust: Custom Types (struct & enum)",
        url: "https://raw.githubusercontent.com/rust-lang/rust-by-example/master/src/custom_types.md",
        license: "MIT / Apache-2.0 (Rust by Example / rust-lang)",
        description: "構造体、列挙型、定数の定義とパターンマッチ",
        seed_content: "# Custom Types\n\nRust custom data types are formed mainly through the two keywords: struct and enum.\n",
    },
    TechDocTarget {
        name: "rust_variable_bindings",
        title: "Rust: Variable Bindings & Mutability",
        url: "https://raw.githubusercontent.com/rust-lang/rust-by-example/master/src/variable_bindings.md",
        license: "MIT / Apache-2.0 (Rust by Example / rust-lang)",
        description: "変数束縛、可変性(mut)、スコープとシャドーイング",
        seed_content: "# Variable Bindings\n\nRust provides type safety via static typing. Variable bindings are immutable by default, but this can be overridden using the mut modifier.\n",
    },
    TechDocTarget {
        name: "rust_types",
        title: "Rust: Types & Casting",
        url: "https://raw.githubusercontent.com/rust-lang/rust-by-example/master/src/types.md",
        license: "MIT / Apache-2.0 (Rust by Example / rust-lang)",
        description: "明示的キャスト、型推論、エイリアス定義",
        seed_content: "# Types\n\nRust provides no implicit type conversion (coercion) between primitive types. But, explicit type conversion (casting) can be performed using the as keyword.\n",
    },
    TechDocTarget {
        name: "rust_flow_control",
        title: "Rust: Flow of Control",
        url: "https://raw.githubusercontent.com/rust-lang/rust-by-example/master/src/flow_control.md",
        license: "MIT / Apache-2.0 (Rust by Example / rust-lang)",
        description: "if/else、loop、while、for、match分岐",
        seed_content: "# Flow of Control\n\nAn essential part of any programming language is branching and loop control: if/else, loop, while, for and in, match.\n",
    },
    TechDocTarget {
        name: "rust_functions",
        title: "Rust: Functions & Closures",
        url: "https://raw.githubusercontent.com/rust-lang/rust-by-example/master/src/fn.md",
        license: "MIT / Apache-2.0 (Rust by Example / rust-lang)",
        description: "関数定義、メソッド、クロージャ、高階関数",
        seed_content: "# Functions\n\nFunctions are declared using the fn keyword. Arguments are type annotated, just like variables, and if the function returns a value, the return type must be specified after an arrow ->.\n",
    },
    TechDocTarget {
        name: "rust_traits",
        title: "Rust: Traits & Polymorphism",
        url: "https://raw.githubusercontent.com/rust-lang/rust-by-example/master/src/trait.md",
        license: "MIT / Apache-2.0 (Rust by Example / rust-lang)",
        description: "トレイト定義、ジェネリクス境界、動的ディスパッチ",
        seed_content: "# Traits\n\nA trait is a collection of methods defined for an unknown type: Self. They can access other methods declared in the same trait.\n",
    },
    TechDocTarget {
        name: "rust_generics",
        title: "Rust: Generics",
        url: "https://raw.githubusercontent.com/rust-lang/rust-by-example/master/src/generics.md",
        license: "MIT / Apache-2.0 (Rust by Example / rust-lang)",
        description: "ジェネリック関数・構造体・PhantomData",
        seed_content: "# Generics\n\nGenerics is the topic of generalizing types and functionalities to broader cases. This is extremely useful for reducing code duplication.\n",
    },
    TechDocTarget {
        name: "rust_error_handling",
        title: "Rust: Error Handling",
        url: "https://raw.githubusercontent.com/rust-lang/rust-by-example/master/src/error.md",
        license: "MIT / Apache-2.0 (Rust by Example / rust-lang)",
        description: "panic、Option、Result型、?演算子によるエラー伝播",
        seed_content: "# Error Handling\n\nError handling in Rust is the process of handling the possibility of failure in a robust way: Option<T>, Result<T, E>, and the ? operator.\n",
    },

    // --- 主要オープンソースクレート (Serde, Regex, Rand) ---
    TechDocTarget {
        name: "rust_crate_serde",
        title: "Rust Crate: Serde (Serialization & Deserialization)",
        url: "https://raw.githubusercontent.com/serde-rs/serde/master/README.md",
        license: "MIT / Apache-2.0 (serde-rs)",
        description: "Serialize, Deserialize トレイトと derive マクロによるデータ構造変換",
        seed_content: r#"# Serde

Serde is a framework for serializing and deserializing Rust data structures efficiently and generically.

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct Point {
    x: i32,
    y: i32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let point = Point { x: 1, y: 2 };

    // Convert the Point to a JSON string.
    let serialized = serde_json::to_string(&point)?;
    println!("serialized = {}", serialized);

    // Convert the JSON string back to a Point.
    let deserialized: Point = serde_json::from_str(&serialized)?;
    println!("deserialized = {:?}", deserialized);

    Ok(())
}
```
"#,
    },
    TechDocTarget {
        name: "rust_crate_regex",
        title: "Rust Crate: Regex",
        url: "https://raw.githubusercontent.com/rust-lang/regex/master/README.md",
        license: "MIT / Apache-2.0 (rust-lang/regex)",
        description: "正規表現のコンパイル、マッチング、置換処理",
        seed_content: r#"# regex

A Rust library for parsing, compiling, and executing regular expressions.

```rust
use regex::Regex;

fn main() {
    let re = Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();
    assert!(re.is_match("2026-09-16"));

    let re_capture = Regex::new(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})").unwrap();
    let text = "2026-09-16";
    let caps = re_capture.captures(text).unwrap();
    assert_eq!(&caps["year"], "2026");
    assert_eq!(&caps["month"], "09");
    assert_eq!(&caps["day"], "16");
}
```
"#,
    },
    TechDocTarget {
        name: "rust_crate_rand",
        title: "Rust Crate: Rand",
        url: "https://raw.githubusercontent.com/rust-random/rand/master/README.md",
        license: "MIT / Apache-2.0 (rust-random)",
        description: "乱数生成、Rng トレイト、範囲サンプリング",
        seed_content: r#"# rand

A Rust library for random number generation.

```rust
use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();

    let n: u32 = rng.gen_range(1..=100);
    println!("Random number between 1 and 100: {}", n);

    let mut numbers = vec![1, 2, 3, 4, 5];
    use rand::seq::SliceRandom;
    numbers.shuffle(&mut rng);
    println!("Shuffled: {:?}", numbers);
}
```
"#,
    },

    // --- Python 標準ライブラリ (JSON, Pathlib) ---
    TechDocTarget {
        name: "python_stdlib_json",
        title: "Python Stdlib: JSON (JavaScript Object Notation)",
        url: "https://raw.githubusercontent.com/python/cpython/main/Doc/library/json.rst",
        license: "Python Software Foundation License (PSF-2.0)",
        description: "dumps / loads によるJSON文字列と辞書・リストの相互変換",
        seed_content: r#"# Python Standard Library: json

JSON (JavaScript Object Notation) encoder and decoder.

```python
import json

data = {
    "name": "oniwa",
    "version": "2.0.0",
    "tags": ["organic", "rust", "compact"],
    "active": True
}

# Serialize dictionary to JSON string
json_str = json.dumps(data, indent=2)
print(json_str)

# Deserialize JSON string back to dictionary
parsed = json.loads(json_str)
assert parsed["name"] == "oniwa"
assert parsed["active"] is True
```
"#,
    },
    TechDocTarget {
        name: "python_stdlib_pathlib",
        title: "Python Stdlib: Pathlib (Object-oriented filesystem paths)",
        url: "https://raw.githubusercontent.com/python/cpython/main/Doc/library/pathlib.rst",
        license: "Python Software Foundation License (PSF-2.0)",
        description: "Path オブジェクトによる安全・クロスプラットフォームなファイルシステム操作",
        seed_content: r#"# Python Standard Library: pathlib

Object-oriented filesystem paths.

```python
from pathlib import Path

# Working with file paths
data_dir = Path("data")
config_file = data_dir / "config.json"

if not data_dir.exists():
    data_dir.mkdir(parents=True, exist_ok=True)

# Reading and writing text files safely
log_path = Path("logs") / "output.txt"
log_path.parent.mkdir(parents=True, exist_ok=True)
log_path.write_text("Execution completed successfully.\n", encoding="utf-8")

content = log_path.read_text(encoding="utf-8")
print(f"Read {len(content)} characters from {log_path}")
```
"#,
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

    /// 技術ドキュメントMarkdownをダウンロード（失敗時はシードコンテンツにフォールバック）
    pub fn download_doc(&self, target: &TechDocTarget) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let dest_path = self.raw_dir.join(format!("techdoc_{}.md", target.name));
        if !self.force_download && dest_path.exists() && dest_path.metadata()?.len() > 0 {
            return Ok(dest_path);
        }

        println!("  📥 [TechDoc] ダウンロード試行: 『{}』 ...", target.title);

        let user_agent = "oniwa-pipeline/0.1.0 (Ethical Open Source Docs Ingestion; Educational AI)";
        let status = Command::new("curl")
            .arg("-s")
            .arg("-f")
            .arg("-L")
            .arg("-m")
            .arg("5") // タイムアウト5秒
            .arg("-A")
            .arg(user_agent)
            .arg("-o")
            .arg(&dest_path)
            .arg(target.url)
            .status();

        if let Ok(st) = status {
            if st.success() && dest_path.exists() && dest_path.metadata()?.len() > 0 {
                sleep(Duration::from_millis(500));
                return Ok(dest_path);
            }
        }

        println!("  ⚠️ ダウンロード失敗またはタイムアウト。公式シードドキュメントを展開します: 『{}』", target.title);
        fs::write(&dest_path, target.seed_content.as_bytes())?;
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
        println!("    (根拠: MIT / Apache-2.0 / PSF オープンライセンス)");
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

    #[test]
    fn test_default_techdocs_validity() {
        assert!(!DEFAULT_TECHDOCS.is_empty());
        for target in DEFAULT_TECHDOCS {
            assert!(!target.name.is_empty());
            assert!(!target.seed_content.is_empty());
            let cleaned = clean_tech_markdown(target.seed_content);
            assert!(!cleaned.is_empty());
        }
    }
}
