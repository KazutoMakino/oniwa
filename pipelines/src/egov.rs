//! e-Gov Legal Open Data Ingestion Pipeline
//!
//! ONIWA: Organic Non-datacenter Intelligence Without Abuse
//! Under Article 13 of the Japanese Copyright Act (Works not eligible for copyright),
//! safely retrieves and cleanses basic statutory laws via public APIs
//! and logs complete provenance for high-quality normative/logical training data.

use oniwa_lm::logger::{DataIngestionLog, ProvenanceEvent, ProvenanceLedger};
use oniwa_lm::reproducibility::compute_checksum_bytes;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread::sleep;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct LawTarget {
    pub law_id: &'static str,
    pub title: &'static str,
    pub description: &'static str,
}

pub const DEFAULT_LAWS: &[LawTarget] = &[
    LawTarget {
        law_id: "321CONSTITUTION",
        title: "日本国憲法",
        description: "Supreme law of Japan; fundamental human rights and governance",
    },
    LawTarget {
        law_id: "140AC0000000045",
        title: "刑法",
        description: "General norms concerning crimes and penalties",
    },
    LawTarget {
        law_id: "345AC0000000048",
        title: "著作権法",
        description: "Norms concerning protection and fair use of intellectual works",
    },
    LawTarget {
        law_id: "129AC0000000089",
        title: "民法",
        description: "Basic law of civil society, property, and family relations",
    },
    LawTarget {
        law_id: "322AC0000000059",
        title: "裁判所法",
        description: "Basic organization of the judiciary and judicial proceedings",
    },
];

pub struct EgovPipeline {
    #[allow(dead_code)]
    pub data_dir: PathBuf,
    pub logs_dir: PathBuf,
    pub corpus_dir: PathBuf,
    pub raw_dir: PathBuf,
    pub force_download: bool,
}

impl EgovPipeline {
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

    /// Download statutory XML from e-Gov API
    pub fn download_law_xml(&self, law_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let dest_path = self.raw_dir.join(format!("egov_{}.xml", law_id));
        if !self.force_download && dest_path.exists() && dest_path.metadata()?.len() > 0 {
            return Ok(dest_path);
        }

        let url = format!("https://elaws.e-gov.go.jp/api/1/lawdata/{}", law_id);
        println!("  📥 [e-Gov] Downloading: {} ({}) ...", law_id, url);

        let user_agent = "oniwa-lm/0.1.0 (Public Domain Legal AI Pipeline; Clean Open Data)";
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
            return Err(format!(
                "e-Gov download failed (curl exit code: {:?}): {}",
                status.code(),
                url
            )
            .into());
        }

        // Pacing wait (1 second) to prevent server overload
        sleep(Duration::from_millis(1000));
        Ok(dest_path)
    }

    /// Download, parse, cleanse a single statute, store in corpus, and log provenance
    pub fn ingest_single_law(
        &self,
        target: &LawTarget,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let xml_path = self.download_law_xml(target.law_id)?;
        let raw_bytes = fs::read(&xml_path)?;
        let raw_sha256 = compute_checksum_bytes(&raw_bytes);
        let xml_content = String::from_utf8_lossy(&raw_bytes);

        // Extract and normalize statutory text from XML
        let (extracted_title, clean_text) = parse_egov_xml(&xml_content, target.title);

        let out_filename = format!("法令_{}.txt", extracted_title);
        let out_path = self.corpus_dir.join(&out_filename);
        fs::write(&out_path, clean_text.as_bytes())?;

        let clean_sha256 = compute_checksum_bytes(clean_text.as_bytes());
        let char_count = clean_text.chars().count();

        println!(
            "  📜 Saved: \"{}\" ({} characters / {:.2} KB)",
            extracted_title,
            char_count,
            clean_text.len() as f32 / 1024.0
        );

        // Mandatory record in provenance ledger
        let ledger_path = self.logs_dir.join("ledger_index.jsonl");
        let mut ledger = ProvenanceLedger::open(&ledger_path)?;

        let source_url = format!("https://elaws.e-gov.go.jp/api/1/lawdata/{}", target.law_id);
        ledger.record(&ProvenanceEvent::DataIngestion(DataIngestionLog {
            timestamp_utc: oniwa_lm::logger::current_timestamp_utc(),
            source_name: format!("e-Gov Law: \"{}\"", extracted_title),
            source_url_or_path: source_url,
            license: "Public Domain (Article 13, Copyright Act of Japan)".to_string(),
            raw_data_sha256: raw_sha256,
            raw_data_bytes: raw_bytes.len(),
            tokenized_sha256: clean_sha256,
            num_tokens: char_count,
            vocab_size: 0,
            tokenizer_type: "Raw Cleaned Text (e-Gov XML Strip)".to_string(),
        }))?;

        Ok(out_path)
    }

    /// Batch ingest default basic statutory laws
    pub fn ingest_default_laws(&self) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        println!("============================================================");
        println!(" ⚖️ e-Gov Public Statutory Open Data Batch Ingestion");
        println!("    (Basis: Article 13 Copyright Act of Japan / Public Domain)");
        println!("============================================================");

        let mut paths = Vec::new();
        for target in DEFAULT_LAWS {
            println!("▶ \"{}\" ({})", target.title, target.description);
            let path = self.ingest_single_law(target)?;
            paths.push(path);
        }

        Ok(paths)
    }
}

/// Extract body text (preamble, articles, paragraphs, items) from e-Gov XML into natural prose
pub fn parse_egov_xml(xml: &str, fallback_title: &str) -> (String, String) {
    let title = extract_tag_content(xml, "LawTitle")
        .map(|s| decode_xml_entities(&s))
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| fallback_title.to_string());

    let mut body = String::new();
    body.push_str(&title);
    body.push_str("\n\n");

    // Extract preamble
    if let Some(preamble) = extract_block(xml, "Preamble") {
        for sentence in extract_all_sentences(&preamble) {
            body.push_str(&sentence);
            body.push('\n');
        }
        body.push('\n');
    }

    // Main provision block (<MainProvision> or entire text)
    let content_block = extract_block(xml, "MainProvision").unwrap_or_else(|| xml.to_string());

    // Traverse each article (<Article>) or sentences sequentially
    let mut current_pos = 0;
    let bytes = content_block.as_bytes();
    let n = bytes.len();

    while current_pos < n {
        if let Some(art_start) = content_block[current_pos..].find("<Article ") {
            let abs_art_start = current_pos + art_start;
            if let Some(art_end) = content_block[abs_art_start..].find("</Article>") {
                let abs_art_end = abs_art_start + art_end + "</Article>".len();
                let art_xml = &content_block[abs_art_start..abs_art_end];

                // Article number/title (e.g., Article 1)
                let art_title = extract_tag_content(art_xml, "ArticleTitle")
                    .map(|s| decode_xml_entities(&s))
                    .unwrap_or_default();

                let sentences = extract_all_sentences(art_xml);
                if !sentences.is_empty() {
                    if !art_title.is_empty() {
                        body.push_str(&art_title);
                        body.push('　');
                    }
                    for (idx, sent) in sentences.iter().enumerate() {
                        if idx > 0 {
                            body.push('\n');
                        }
                        body.push_str(sent);
                    }
                    body.push_str("\n\n");
                }

                current_pos = abs_art_end;
                continue;
            }
        }
        break;
    }

    // Fallback: extract all sentences if no article division is found
    if body.trim() == title {
        for sentence in extract_all_sentences(&content_block) {
            body.push_str(&sentence);
            body.push('\n');
        }
    }

    (title, body.trim().to_string())
}

/// Extract text within a single tag (first occurrence)
fn extract_tag_content(xml: &str, tag_name: &str) -> Option<String> {
    let open_prefix = format!("<{}", tag_name);
    let close_tag = format!("</{}>", tag_name);

    let start = xml.find(&open_prefix)?;
    let content_start = xml[start..].find('>')? + start + 1;
    let end = xml[content_start..].find(&close_tag)? + content_start;

    Some(xml[content_start..end].trim().to_string())
}

/// Extract entire block tag (<Tag...> ... </Tag>)
fn extract_block(xml: &str, tag_name: &str) -> Option<String> {
    let open_prefix = format!("<{}", tag_name);
    let close_tag = format!("</{}>", tag_name);

    let start = xml.find(&open_prefix)?;
    let end = xml[start..].find(&close_tag)? + start + close_tag.len();

    Some(xml[start..end].to_string())
}

/// Extract all <Sentence>...</Sentence> within XML
fn extract_all_sentences(xml: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current_pos = 0;

    while current_pos < xml.len() {
        if let Some(sent_start) = xml[current_pos..].find("<Sentence") {
            let abs_sent_start = current_pos + sent_start;
            if let Some(bracket_close) = xml[abs_sent_start..].find('>') {
                let content_start = abs_sent_start + bracket_close + 1;
                if let Some(sent_end) = xml[content_start..].find("</Sentence>") {
                    let content_end = content_start + sent_end;
                    let raw_sentence = &xml[content_start..content_end];
                    // Strip inner XML tags (ruby, bold, etc.) and decode XML entities
                    let cleaned = strip_inner_xml_tags(raw_sentence);
                    let decoded = decode_xml_entities(&cleaned);
                    let trimmed = decoded.trim();
                    if !trimmed.is_empty() {
                        sentences.push(trimmed.to_string());
                    }
                    current_pos = content_end + "</Sentence>".len();
                    continue;
                }
            }
        }
        break;
    }

    sentences
}

/// Strip nested XML tags (<Ruby>, etc.)
fn strip_inner_xml_tags(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        if c == '<' {
            in_tag = true;
        } else if c == '>' {
            in_tag = false;
        } else if !in_tag {
            result.push(c);
        }
    }
    result
}

/// Decode XML special character entities
fn decode_xml_entities(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&#10;", "\n")
        .replace("&#13;", "")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_egov_xml_constitution_snippet() {
        let sample_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<DataRoot>
  <ApplData>
    <LawFullText>
      <LawBody>
        <LawTitle>日本国憲法</LawTitle>
        <Preamble>
          <Paragraph Num="1">
            <ParagraphSentence>
              <Sentence>日本国民は、正当に選挙された国会における代表者を通じて行動し、この憲法を確定する。</Sentence>
            </ParagraphSentence>
          </Paragraph>
        </Preamble>
        <MainProvision>
          <Article Num="1">
            <ArticleTitle>第一条</ArticleTitle>
            <Paragraph Num="1">
              <ParagraphSentence>
                <Sentence>天皇は、日本国の象徴であり日本国民統合の象徴であつて、主権の存する日本国民の総意に基く。</Sentence>
              </ParagraphSentence>
            </Paragraph>
          </Article>
        </MainProvision>
      </LawBody>
    </LawFullText>
  </ApplData>
</DataRoot>"#;

        let (title, text) = parse_egov_xml(sample_xml, "憲法");
        assert_eq!(title, "日本国憲法");
        assert!(text.contains("日本国憲法"));
        assert!(text.contains("日本国民は、正当に選挙された国会における代表者を通じて行動し"));
        assert!(text.contains("第一条　天皇は、日本国の象徴であり"));
    }
}
