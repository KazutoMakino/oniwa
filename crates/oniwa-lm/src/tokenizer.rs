//! 文字単位トークナイザー（Character-level Tokenizer）
//!
//! 外部の巨大な辞書やPythonに依存せず、ピュアRustで日本語テキストから
//! ユニーク文字を抽出して語彙テーブル（vocab.json）とトークンバイナリ（tokens.bin）を生成します。

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CharTokenizer {
    /// 文字 -> トークンID (決定論的順序のため BTreeMap)
    pub char_to_id: BTreeMap<char, u16>,
    /// トークンID -> 文字
    pub id_to_char: Vec<char>,
}

impl CharTokenizer {
    /// テキストから語彙テーブルを構築
    pub fn build_from_text(text: &str) -> Self {
        let mut unique_chars: Vec<char> = text.chars().collect();
        unique_chars.sort();
        unique_chars.dedup();

        let mut char_to_id = BTreeMap::new();
        let mut id_to_char = Vec::with_capacity(unique_chars.len());

        for (id, ch) in unique_chars.into_iter().enumerate() {
            let id = id as u16;
            char_to_id.insert(ch, id);
            id_to_char.push(ch);
        }

        Self {
            char_to_id,
            id_to_char,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_char.len()
    }

    /// テキストをトークン列（u16）にエンコード
    pub fn encode(&self, text: &str) -> Vec<u16> {
        text.chars()
            .filter_map(|ch| self.char_to_id.get(&ch).copied())
            .collect()
    }

    /// トークン列（u16）を文字列にデコード
    pub fn decode(&self, tokens: &[u16]) -> String {
        tokens
            .iter()
            .filter_map(|&tid| self.id_to_char.get(tid as usize).copied())
            .collect()
    }

    /// 語彙テーブルを JSON ファイルとして保存
    pub fn save_vocab<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// 語彙テーブルを JSON ファイルから読み込み
    pub fn load_vocab<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let tokenizer = serde_json::from_reader(reader)?;
        Ok(tokenizer)
    }

    /// トークン列をリトルエンディアン u16 バイナリとして保存
    pub fn save_tokens_bin<P: AsRef<Path>>(tokens: &[u16], path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        for &t in tokens {
            writer.write_all(&t.to_le_bytes())?;
        }
        writer.flush()?;
        Ok(())
    }

    /// トークンバイナリファイルを読み込み
    pub fn load_tokens_bin<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<u16>> {
        let mut file = File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        let count = bytes.len() / 2;
        let mut tokens = Vec::with_capacity(count);
        for i in 0..count {
            let b0 = bytes[i * 2];
            let b1 = bytes[i * 2 + 1];
            tokens.push(u16::from_le_bytes([b0, b1]));
        }
        Ok(tokens)
    }
    /// テキストファイルを読み込んでトークナイズ・語彙保存・バイナリ保存・台帳記録を一気通貫で実行
    pub fn ingest_file<P: AsRef<Path>>(
        raw_text_path: P,
        data_dir: P,
        logs_dir: P,
        source_name: &str,
        source_url: &str,
        license: &str,
    ) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(raw_text_path.as_ref())?;
        let raw_sha256 = crate::reproducibility::compute_checksum_bytes(text.as_bytes());

        let tokenizer = Self::build_from_text(&text);
        let tokens = tokenizer.encode(&text);

        let data_dir_p = data_dir.as_ref();
        let logs_dir_p = logs_dir.as_ref();
        std::fs::create_dir_all(data_dir_p)?;
        std::fs::create_dir_all(logs_dir_p)?;

        // 語彙とバイナリの保存
        tokenizer.save_vocab(data_dir_p.join("vocab.json"))?;
        Self::save_tokens_bin(&tokens, data_dir_p.join("tokens.bin"))?;

        // トークンバイナリのハッシュ
        let bin_bytes = std::fs::read(data_dir_p.join("tokens.bin"))?;
        let tokenized_sha256 = crate::reproducibility::compute_checksum_bytes(&bin_bytes);

        // 系譜台帳 (ledger_index.jsonl) への記録
        let mut ledger = crate::logger::ProvenanceLedger::open(logs_dir_p.join("ledger_index.jsonl"))?;
        ledger.record(&crate::logger::ProvenanceEvent::DataIngestion(
            crate::logger::DataIngestionLog {
                timestamp_utc: crate::logger::current_timestamp_utc(),
                source_name: source_name.to_string(),
                source_url_or_path: source_url.to_string(),
                license: license.to_string(),
                raw_data_sha256: raw_sha256,
                raw_data_bytes: text.len(),
                tokenized_sha256,
                num_tokens: tokens.len(),
                vocab_size: tokenizer.vocab_size(),
                tokenizer_type: "Character-level (UTF-8)".to_string(),
            },
        ))?;

        Ok(tokenizer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_tokenizer_roundtrip() {
        let text = "山月記。隴西の李徴は博学才頴。";
        let tokenizer = CharTokenizer::build_from_text(text);
        assert_eq!(tokenizer.vocab_size(), 14); // 重複除外

        let encoded = tokenizer.encode(text);
        assert_eq!(encoded.len(), text.chars().count());

        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_ingest_sangetsuki_dataset() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let raw_path = manifest_dir.join("data/sangetsuki_clean.txt");
        if raw_path.exists() {
            let data_dir = manifest_dir.join("data");
            let logs_dir = manifest_dir.join("logs");

            let tokenizer = CharTokenizer::ingest_file(
                &raw_path,
                &data_dir,
                &logs_dir,
                "青空文庫/Wikisource: 中島敦『山月記』",
                "https://ja.wikisource.org/wiki/山月記",
                "Public Domain",
            )
            .unwrap();

            assert!(tokenizer.vocab_size() > 500);
            assert!(data_dir.join("vocab.json").exists());
            assert!(data_dir.join("tokens.bin").exists());
            assert!(logs_dir.join("ledger_index.jsonl").exists());
        }
    }
}
