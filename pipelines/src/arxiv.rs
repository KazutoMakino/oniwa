//! arXiv オープンサイエンス論文アブストラクト収集パイプライン
//!
//! ONIWA: Organic Non-datacenter Intelligence Without Abuse
//! 公式 arXiv API (https://arxiv.org/help/api) を通じて、
//! コンピュータサイエンス・人工知能（cs.AI / cs.CL / cs.LG 等）のオープンアクセス論文
//! （CC-BY / オープンライセンス）のタイトル・要約・著者情報を取得・クレンジングし、
//! 将来の学術的・英語コンテキスト学習のためのクリーンな土壌として系譜台帳に記録します。

use oniwa_lm::logger::{DataIngestionLog, ProvenanceEvent, ProvenanceLedger};
use oniwa_lm::reproducibility::compute_checksum_bytes;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread::sleep;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct ArxivEntry {
    pub id: String,
    pub title: String,
    pub summary: String,
    pub published: String,
    pub authors: Vec<String>,
}

pub struct ArxivPipeline {
    #[allow(dead_code)]
    pub data_dir: PathBuf,
    pub logs_dir: PathBuf,
    pub corpus_dir: PathBuf,
    pub raw_dir: PathBuf,
    pub force_download: bool,
}

impl ArxivPipeline {
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

    /// arXiv API から指定カテゴリ・件数の論文フィード (Atom XML) を取得
    pub fn fetch_feed_xml(
        &self,
        category: &str,
        max_results: usize,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let clean_cat = category.replace(':', "_");
        let dest_path = self
            .raw_dir
            .join(format!("arxiv_{}_{}.xml", clean_cat, max_results));

        if !self.force_download && dest_path.exists() && dest_path.metadata()?.len() > 0 {
            return Ok(dest_path);
        }

        let url = format!(
            "https://export.arxiv.org/api/query?search_query=cat:{}&max_results={}&sortBy=submittedDate&sortOrder=descending",
            category, max_results
        );
        println!(
            "  📥 [arXiv API] 論文フィード取得中: {} (最大 {} 件) ...",
            category, max_results
        );

        let user_agent =
            "oniwa-pipeline/0.1.0 (Ethical Open Science AI Data Ingestion; Educational Project)";
        let status = Command::new("curl")
            .arg("-s")
            .arg("-f")
            .arg("-L")
            .arg("-A")
            .arg(user_agent)
            .arg("-o")
            .arg(&dest_path)
            .arg(&url)
            .status()?;

        if !status.success() {
            return Err(format!("arXiv API リクエスト失敗: {}", url).into());
        }

        // arXiv API 利用規約遵守: リクエスト間隔を確保 (最低3秒推奨)
        sleep(Duration::from_millis(3000));
        Ok(dest_path)
    }

    /// フィード XML をパースして各論文をクレンジング、個別コーパスファイルに保存
    pub fn ingest_category(
        &self,
        category: &str,
        max_results: usize,
    ) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        println!("============================================================");
        println!(
            " 🔬 arXiv オープンアクセス論文収集: カテゴリ「{}」 (最大 {} 件)",
            category, max_results
        );
        println!("    (根拠: arXiv.org API / オープンサイエンス・CCライセンス)");
        println!("============================================================");

        let xml_path = self.fetch_feed_xml(category, max_results)?;
        let xml_bytes = fs::read(&xml_path)?;
        let xml_content = String::from_utf8_lossy(&xml_bytes);

        let entries = parse_arxiv_atom(&xml_content);
        println!("  - 取得できた論文エントリー: {} 件", entries.len());

        let ledger_path = self.logs_dir.join("ledger_index.jsonl");
        let mut ledger = ProvenanceLedger::open(&ledger_path)?;
        let mut paths = Vec::new();

        for entry in entries {
            let safe_id = entry
                .id
                .split('/')
                .last()
                .unwrap_or("unknown")
                .replace(['.', ':'], "_");

            let out_filename = format!("arxiv_{}_{}.txt", safe_id, sanitize_filename(&entry.title));
            let out_path = self.corpus_dir.join(&out_filename);

            let mut body = String::new();
            body.push_str(&format!("# Title: {}\n", entry.title));
            body.push_str(&format!("Authors: {}\n", entry.authors.join(", ")));
            body.push_str(&format!("Published: {}\n", entry.published));
            body.push_str(&format!("ArXiv ID: {}\n\n", entry.id));
            body.push_str("Abstract:\n");
            body.push_str(&entry.summary);
            body.push('\n');

            fs::write(&out_path, body.as_bytes())?;

            let raw_sha256 = compute_checksum_bytes(body.as_bytes());
            let char_count = body.chars().count();

            ledger.record(&ProvenanceEvent::DataIngestion(DataIngestionLog {
                timestamp_utc: oniwa_lm::logger::current_timestamp_utc(),
                source_name: format!("arXiv: 『{}』", entry.title),
                source_url_or_path: entry.id.clone(),
                license: "Open Access (arXiv API / CC-BY or Non-Exclusive License)".to_string(),
                raw_data_sha256: raw_sha256.clone(),
                raw_data_bytes: body.len(),
                tokenized_sha256: raw_sha256,
                num_tokens: char_count,
                vocab_size: 0,
                tokenizer_type: "Raw Cleaned Text (arXiv Atom XML)".to_string(),
            }))?;

            println!("  📄 保存完了: 『{}』({} 文字)", entry.title, char_count);
            paths.push(out_path);
        }

        Ok(paths)
    }
}

/// Atom XML から各 <entry> を抽出
pub fn parse_arxiv_atom(xml: &str) -> Vec<ArxivEntry> {
    let mut entries = Vec::new();
    let mut current_pos = 0;

    while let Some(start_tag) = xml[current_pos..].find("<entry>") {
        let abs_start = current_pos + start_tag;
        if let Some(end_tag) = xml[abs_start..].find("</entry>") {
            let abs_end = abs_start + end_tag + "</entry>".len();
            let entry_xml = &xml[abs_start..abs_end];

            let id = extract_tag(entry_xml, "id").unwrap_or_default();
            let title = extract_tag(entry_xml, "title")
                .map(|t| normalize_whitespace(&t))
                .unwrap_or_default();
            let summary = extract_tag(entry_xml, "summary")
                .map(|s| normalize_whitespace(&s))
                .unwrap_or_default();
            let published = extract_tag(entry_xml, "published").unwrap_or_default();
            let authors = extract_authors(entry_xml);

            if !title.is_empty() && !summary.is_empty() {
                entries.push(ArxivEntry {
                    id,
                    title,
                    summary,
                    published,
                    authors,
                });
            }

            current_pos = abs_end;
        } else {
            break;
        }
    }

    entries
}

fn extract_tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);

    let s = xml.find(&open)?;
    let content_start = xml[s..].find('>')? + s + 1;
    let content_end = xml[content_start..].find(&close)? + content_start;

    Some(decode_xml_entities(xml[content_start..content_end].trim()))
}

fn extract_authors(xml: &str) -> Vec<String> {
    let mut authors = Vec::new();
    let mut pos = 0;
    while let Some(auth_start) = xml[pos..].find("<author>") {
        let abs_auth_start = pos + auth_start;
        if let Some(auth_end) = xml[abs_auth_start..].find("</author>") {
            let abs_auth_end = abs_auth_start + auth_end + "</author>".len();
            let auth_block = &xml[abs_auth_start..abs_auth_end];
            if let Some(name) = extract_tag(auth_block, "name") {
                authors.push(name);
            }
            pos = abs_auth_end;
        } else {
            break;
        }
    }
    authors
}

fn normalize_whitespace(s: &str) -> String {
    s.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn decode_xml_entities(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

fn sanitize_filename(s: &str) -> String {
    let sanitized: String = s
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(40)
        .collect();
    sanitized.trim_matches('_').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_arxiv_atom_entry() {
        let sample = r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>A Simple Transformer Model</title>
    <summary>This is a paper about
pure Rust implementations of language models.</summary>
    <published>2023-01-01T00:00:00Z</published>
    <author><name>Alan Turing</name></author>
    <author><name>Ada Lovelace</name></author>
  </entry>
</feed>"#;

        let entries = parse_arxiv_atom(sample);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].title, "A Simple Transformer Model");
        assert_eq!(entries[0].authors, vec!["Alan Turing", "Ada Lovelace"]);
        assert!(entries[0].summary.contains("pure Rust implementations"));
    }
}
