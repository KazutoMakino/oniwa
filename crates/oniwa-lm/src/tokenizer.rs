//! Character-level Tokenizer
//!
//! Generates a vocabulary table (`vocab.json`) and token binary (`tokens.bin`) from
//! text in pure Rust without depending on external massive dictionaries or Python.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CharTokenizer {
    /// Character -> Token ID (BTreeMap for deterministic ordering)
    pub char_to_id: BTreeMap<char, u16>,
    /// Token ID -> Character
    pub id_to_char: Vec<char>,
}

impl CharTokenizer {
    /// Build vocabulary table from text
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

    /// Encode text into token IDs (u16)
    pub fn encode(&self, text: &str) -> Vec<u16> {
        text.chars()
            .filter_map(|ch| self.char_to_id.get(&ch).copied())
            .collect()
    }

    /// Decode token IDs (u16) into String
    pub fn decode(&self, tokens: &[u16]) -> String {
        tokens
            .iter()
            .filter_map(|&tid| self.id_to_char.get(tid as usize).copied())
            .collect()
    }

    /// Save vocabulary table as JSON file
    pub fn save_vocab<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Load vocabulary table from JSON file
    pub fn load_vocab<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let tokenizer = serde_json::from_reader(reader)?;
        Ok(tokenizer)
    }

    /// Save token sequence as little-endian u16 binary
    pub fn save_tokens_bin<P: AsRef<Path>>(tokens: &[u16], path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        for &t in tokens {
            writer.write_all(&t.to_le_bytes())?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Load token sequence from little-endian u16 binary file
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

    /// Read raw text file, tokenize, save vocab, save binary, and record in provenance ledger end-to-end.
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

        // Save vocabulary and binary
        tokenizer.save_vocab(data_dir_p.join("vocab.json"))?;
        Self::save_tokens_bin(&tokens, data_dir_p.join("tokens.bin"))?;

        // Token binary hash
        let bin_bytes = std::fs::read(data_dir_p.join("tokens.bin"))?;
        let tokenized_sha256 = crate::reproducibility::compute_checksum_bytes(&bin_bytes);

        // Record in provenance ledger (ledger_index.jsonl)
        let mut ledger =
            crate::logger::ProvenanceLedger::open(logs_dir_p.join("ledger_index.jsonl"))?;
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

    /// Special token IDs
    pub const UNK_TOKEN: &'static str = "<unk>";
    pub const BOS_TOKEN: &'static str = "<bos>";
    pub const EOS_TOKEN: &'static str = "<eos>";
    pub const PAD_TOKEN: &'static str = "<pad>";
}

/// Common tokenizer interface shared by character-level and BPE subword tokenizers
pub trait Tokenizer: Send + Sync {
    fn vocab_size(&self) -> usize;
    fn encode(&self, text: &str) -> Vec<u16>;
    fn encode_into(&self, text: &str, destination: &mut [u16]) -> usize {
        let tokens = self.encode(text);
        let written = tokens.len().min(destination.len());
        destination[..written].copy_from_slice(&tokens[..written]);
        written
    }
    fn decode(&self, tokens: &[u16]) -> String;
    fn token_to_id(&self, token: &str) -> Option<u16>;
    fn id_to_token(&self, id: u16) -> Option<String>;
}

impl Tokenizer for CharTokenizer {
    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }

    fn encode(&self, text: &str) -> Vec<u16> {
        self.encode(text)
    }

    fn encode_into(&self, text: &str, destination: &mut [u16]) -> usize {
        let mut written = 0;
        for ch in text.chars() {
            let Some(&token) = self.char_to_id.get(&ch) else {
                continue;
            };
            if written == destination.len() {
                break;
            }
            destination[written] = token;
            written += 1;
        }
        written
    }

    fn decode(&self, tokens: &[u16]) -> String {
        self.decode(tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u16> {
        if token.chars().count() == 1 {
            let ch = token.chars().next().unwrap();
            self.char_to_id.get(&ch).copied()
        } else {
            None
        }
    }

    fn id_to_token(&self, id: u16) -> Option<String> {
        self.id_to_char.get(id as usize).map(|&ch| ch.to_string())
    }
}

/// Pure Rust Byte-Pair Encoding (BPE) Subword Tokenizer
///
/// Features:
/// - 100% Pure Rust implementation without external Python or HuggingFace dependencies.
/// - Base byte vocabulary + deterministic pair merge iterations.
/// - Special tokens (`<unk>`, `<bos>`, `<eos>`, `<pad>`).
/// - Deterministic tie-breaking for identical pair frequencies (lexicographic).
/// - Fast subword tokenization and lossless UTF-8 reconstruction.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BpeTokenizer {
    /// Token string -> Token ID
    pub token_to_id: BTreeMap<String, u16>,
    /// Token ID -> Token string
    pub id_to_token: Vec<String>,
    /// Ordered merge rules: (token_a, token_b) -> merged_token
    pub merges: Vec<(String, String)>,
    /// Quick lookup for merge priority: (token_a, token_b) -> merge_index
    #[serde(skip)]
    pub merge_ranks: BTreeMap<(String, String), usize>,
    /// Special token IDs
    pub unk_id: u16,
    pub bos_id: u16,
    pub eos_id: u16,
    pub pad_id: u16,
}

impl BpeTokenizer {
    pub const UNK: &'static str = "<unk>";
    pub const BOS: &'static str = "<bos>";
    pub const EOS: &'static str = "<eos>";
    pub const PAD: &'static str = "<pad>";
    pub const MASK: &'static str = "<mask>";

    /// Train a BPE vocabulary and merge rules from a text corpus
    /// Note: `target_vocab_size` is the goal vocabulary size. The base vocabulary will include
    /// special tokens (5) plus all unique UTF-8 bytes in `text`. If `target_vocab_size` is greater than
    /// the base vocabulary, BPE merges are executed until reaching `target_vocab_size` or until
    /// no further frequent pairs exist.
    pub fn train_from_text(text: &str, target_vocab_size: usize) -> Self {
        assert!(
            target_vocab_size >= 5,
            "target_vocab_size must be at least 5 for special tokens"
        );

        let mut token_to_id = BTreeMap::new();
        let mut id_to_token = Vec::new();

        // 1. Register special tokens
        let special_tokens = [Self::UNK, Self::BOS, Self::EOS, Self::PAD, Self::MASK];
        for (i, &st) in special_tokens.iter().enumerate() {
            let id = i as u16;
            token_to_id.insert(st.to_string(), id);
            id_to_token.push(st.to_string());
        }
        let unk_id = 0u16;
        let bos_id = 1u16;
        let eos_id = 2u16;
        let pad_id = 3u16;

        // 2. Collect every UTF-8 byte as the base alphabet. A byte is represented by the
        // corresponding U+0000..U+00FF scalar only inside the serializable vocabulary table.
        let mut unique_bytes = text.as_bytes().to_vec();
        unique_bytes.sort_unstable();
        unique_bytes.dedup();

        for byte in unique_bytes {
            let s = (byte as char).to_string();
            if !token_to_id.contains_key(&s) {
                let id = id_to_token.len() as u16;
                token_to_id.insert(s.clone(), id);
                id_to_token.push(s);
            }
        }

        // Represent the corpus as byte symbols. Newlines are ordinary bytes, so round-trips are exact.
        let mut sequences: Vec<Vec<String>> = text
            .as_bytes()
            .split_inclusive(|&byte| byte == b'\n')
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|&byte| (byte as char).to_string())
                    .collect()
            })
            .filter(|v: &Vec<String>| !v.is_empty())
            .collect();

        let mut merges = Vec::new();
        let mut merge_ranks = BTreeMap::new();

        // 3. Iteratively find the most frequent adjacent pair and merge
        while id_to_token.len() < target_vocab_size {
            let mut pair_counts: BTreeMap<(String, String), usize> = BTreeMap::new();

            for seq in &sequences {
                if seq.len() < 2 {
                    continue;
                }
                for i in 0..seq.len() - 1 {
                    let pair = (seq[i].clone(), seq[i + 1].clone());
                    *pair_counts.entry(pair).or_insert(0) += 1;
                }
            }

            if pair_counts.is_empty() {
                break;
            }

            // Find best pair: highest count, deterministic tie-breaking (lexicographical order of pair)
            let mut best_pair: Option<((String, String), usize)> = None;
            for (pair, count) in pair_counts {
                if count < 2 {
                    // Stop merging if no pair occurs at least twice
                    continue;
                }
                match &best_pair {
                    None => best_pair = Some((pair, count)),
                    Some((best_p, best_c)) => {
                        if count > *best_c || (count == *best_c && pair < *best_p) {
                            best_pair = Some((pair, count));
                        }
                    }
                }
            }

            let ((first, second), _) = match best_pair {
                Some(p) => p,
                None => break, // No more pairs with frequency >= 2
            };

            let merged = format!("{}{}", first, second);
            let merge_idx = merges.len();
            merge_ranks.insert((first.clone(), second.clone()), merge_idx);
            merges.push((first.clone(), second.clone()));

            if !token_to_id.contains_key(&merged) {
                let id = id_to_token.len() as u16;
                token_to_id.insert(merged.clone(), id);
                id_to_token.push(merged.clone());
            }

            // Apply merge across all sequences
            for seq in &mut sequences {
                if seq.len() < 2 {
                    continue;
                }
                let mut new_seq = Vec::with_capacity(seq.len());
                let mut i = 0;
                while i < seq.len() {
                    if i < seq.len() - 1 && seq[i] == first && seq[i + 1] == second {
                        new_seq.push(merged.clone());
                        i += 2;
                    } else {
                        new_seq.push(seq[i].clone());
                        i += 1;
                    }
                }
                *seq = new_seq;
            }
        }

        Self {
            token_to_id,
            id_to_token,
            merges,
            merge_ranks,
            unk_id,
            bos_id,
            eos_id,
            pad_id,
        }
    }

    /// Total vocabulary size including special tokens, base characters, and merged subwords
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Encode text using BPE subword tokens
    pub fn encode(&self, text: &str) -> Vec<u16> {
        <Self as Tokenizer>::encode(self, text)
    }

    /// Decode BPE subword tokens back to UTF-8 text
    pub fn decode(&self, tokens: &[u16]) -> String {
        <Self as Tokenizer>::decode(self, tokens)
    }

    /// Rebuild merge_ranks lookup after deserialization
    pub fn rebuild_merge_ranks(&mut self) {
        self.merge_ranks.clear();
        for (i, pair) in self.merges.iter().enumerate() {
            self.merge_ranks.insert(pair.clone(), i);
        }
    }

    /// Save BPE tokenizer model to JSON file
    pub fn save_vocab<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Load BPE tokenizer model from JSON file
    pub fn load_vocab<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut tokenizer: Self = serde_json::from_reader(reader)?;
        tokenizer.rebuild_merge_ranks();
        Ok(tokenizer)
    }

    /// Ingest a raw text file using BpeTokenizer and record in provenance ledger
    pub fn ingest_file<P: AsRef<Path>>(
        raw_text_path: P,
        data_dir: P,
        logs_dir: P,
        target_vocab_size: usize,
        source_name: &str,
        source_url: &str,
        license: &str,
    ) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(raw_text_path.as_ref())?;
        let raw_sha256 = crate::reproducibility::compute_checksum_bytes(text.as_bytes());

        let mut tokenizer = Self::train_from_text(&text, target_vocab_size);
        tokenizer.rebuild_merge_ranks();
        let tokens = tokenizer.encode(&text);

        let data_dir_p = data_dir.as_ref();
        let logs_dir_p = logs_dir.as_ref();
        std::fs::create_dir_all(data_dir_p)?;
        std::fs::create_dir_all(logs_dir_p)?;

        // Save BPE vocab and binary
        tokenizer.save_vocab(data_dir_p.join("bpe_vocab.json"))?;
        CharTokenizer::save_tokens_bin(&tokens, data_dir_p.join("bpe_tokens.bin"))?;

        // Token binary hash
        let bin_bytes = std::fs::read(data_dir_p.join("bpe_tokens.bin"))?;
        let tokenized_sha256 = crate::reproducibility::compute_checksum_bytes(&bin_bytes);

        // Record in provenance ledger
        let mut ledger =
            crate::logger::ProvenanceLedger::open(logs_dir_p.join("ledger_index.jsonl"))?;
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
                tokenizer_type: "Pure Rust BPE (Byte-Pair Encoding)".to_string(),
            },
        ))?;

        Ok(tokenizer)
    }

    /// Ingest a list of document strings, inserting `<eos>` between documents, and record in provenance ledger
    pub fn ingest_documents<P: AsRef<Path>>(
        documents: &[String],
        data_dir: P,
        logs_dir: P,
        target_vocab_size: usize,
        source_name: &str,
        source_url: &str,
        license: &str,
    ) -> std::io::Result<Self> {
        // Concatenate documents for BPE training
        let mut full_text = String::new();
        for (i, doc) in documents.iter().enumerate() {
            if i > 0 {
                full_text.push('\n');
            }
            full_text.push_str(doc);
        }

        let raw_sha256 = crate::reproducibility::compute_checksum_bytes(full_text.as_bytes());

        let mut tokenizer = Self::train_from_text(&full_text, target_vocab_size);
        tokenizer.rebuild_merge_ranks();

        // Encode documents with <eos> delimiter between them
        let mut all_tokens = Vec::new();
        for (i, doc) in documents.iter().enumerate() {
            let doc_tokens = tokenizer.encode(doc);
            all_tokens.extend_from_slice(&doc_tokens);
            // Append EOS token delimiter at the end of each document
            all_tokens.push(tokenizer.eos_id);
            let _ = i;
        }

        let data_dir_p = data_dir.as_ref();
        let logs_dir_p = logs_dir.as_ref();
        std::fs::create_dir_all(data_dir_p)?;
        std::fs::create_dir_all(logs_dir_p)?;

        // Save BPE vocab and binary
        tokenizer.save_vocab(data_dir_p.join("bpe_vocab.json"))?;
        CharTokenizer::save_tokens_bin(&all_tokens, data_dir_p.join("bpe_tokens.bin"))?;

        // Token binary hash
        let bin_bytes = std::fs::read(data_dir_p.join("bpe_tokens.bin"))?;
        let tokenized_sha256 = crate::reproducibility::compute_checksum_bytes(&bin_bytes);

        // Record in provenance ledger
        let mut ledger =
            crate::logger::ProvenanceLedger::open(logs_dir_p.join("ledger_index.jsonl"))?;
        ledger.record(&crate::logger::ProvenanceEvent::DataIngestion(
            crate::logger::DataIngestionLog {
                timestamp_utc: crate::logger::current_timestamp_utc(),
                source_name: source_name.to_string(),
                source_url_or_path: source_url.to_string(),
                license: license.to_string(),
                raw_data_sha256: raw_sha256,
                raw_data_bytes: full_text.len(),
                tokenized_sha256,
                num_tokens: all_tokens.len(),
                vocab_size: tokenizer.vocab_size(),
                tokenizer_type: "Pure Rust BPE with <eos> Delimiters".to_string(),
            },
        ))?;

        Ok(tokenizer)
    }

    /// Internal helper to tokenize a byte chunk with BPE merge rules.
    fn tokenize_chunk(&self, chunk: &str) -> Vec<String> {
        if chunk.is_empty() {
            return Vec::new();
        }

        let mut tokens: Vec<String> = chunk
            .bytes()
            .map(|byte| (byte as char).to_string())
            .collect();

        if tokens.len() < 2 {
            return tokens;
        }

        loop {
            // Find pair with lowest merge rank (highest priority)
            let mut best_pair: Option<((String, String), usize)> = None;

            for i in 0..tokens.len() - 1 {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    match best_pair {
                        None => best_pair = Some((pair, rank)),
                        Some((_, best_rank)) => {
                            if rank < best_rank {
                                best_pair = Some((pair, rank));
                            }
                        }
                    }
                }
            }

            let ((first, second), _) = match best_pair {
                Some(p) => p,
                None => break, // No more mergeable pairs
            };

            let merged = format!("{}{}", first, second);
            let mut new_tokens = Vec::with_capacity(tokens.len());
            let mut i = 0;
            while i < tokens.len() {
                if i < tokens.len() - 1 && tokens[i] == first && tokens[i + 1] == second {
                    new_tokens.push(merged.clone());
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            tokens = new_tokens;
            if tokens.len() < 2 {
                break;
            }
        }

        tokens
    }
}

impl Tokenizer for BpeTokenizer {
    fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    fn encode(&self, text: &str) -> Vec<u16> {
        let mut result = Vec::new();

        // Process line by line to keep memory bounded while keeping newline bytes.
        for chunk in text.as_bytes().split_inclusive(|&byte| byte == b'\n') {
            let chunk = std::str::from_utf8(chunk).expect("text bytes are valid UTF-8");
            let tokens = self.tokenize_chunk(chunk);
            for t in tokens {
                if let Some(&id) = self.token_to_id.get(&t) {
                    result.push(id);
                } else {
                    // Fallback to byte-level or unknown token.
                    for ch in t.chars() {
                        let ch_s = ch.to_string();
                        if let Some(&id) = self.token_to_id.get(&ch_s) {
                            result.push(id);
                        } else {
                            result.push(self.unk_id);
                        }
                    }
                }
            }
        }

        result
    }

    fn encode_into(&self, text: &str, destination: &mut [u16]) -> usize {
        let mut written = 0;
        for chunk in text.as_bytes().split_inclusive(|&byte| byte == b'\n') {
            let chunk = std::str::from_utf8(chunk).expect("text bytes are valid UTF-8");
            for token in self.tokenize_chunk(chunk) {
                if let Some(&id) = self.token_to_id.get(&token) {
                    if written == destination.len() {
                        return written;
                    }
                    destination[written] = id;
                    written += 1;
                } else {
                    for ch in token.chars() {
                        if written == destination.len() {
                            return written;
                        }
                        destination[written] = self
                            .token_to_id
                            .get(&ch.to_string())
                            .copied()
                            .unwrap_or(self.unk_id);
                        written += 1;
                    }
                }
            }
        }
        written
    }

    fn decode(&self, tokens: &[u16]) -> String {
        let mut bytes = Vec::new();
        let mut out = String::new();
        for &tid in tokens {
            if let Some(tok) = self.id_to_token.get(tid as usize) {
                if tok == Self::BOS || tok == Self::EOS || tok == Self::PAD || tok == Self::MASK {
                    continue;
                }
                if tok == Self::UNK {
                    out.push_str(&String::from_utf8_lossy(&bytes));
                    bytes.clear();
                    out.push_str(Self::UNK);
                } else {
                    bytes.extend(tok.chars().map(|ch| ch as u8));
                }
            }
        }
        out.push_str(&String::from_utf8_lossy(&bytes));
        out
    }

    fn token_to_id(&self, token: &str) -> Option<u16> {
        self.token_to_id.get(token).copied()
    }

    fn id_to_token(&self, id: u16) -> Option<String> {
        self.id_to_token.get(id as usize).cloned()
    }
}

/// Unified Tokenizer Enum supporting both Character-level and BPE subwords
#[derive(Debug, Clone)]
pub enum AnyTokenizer {
    Char(CharTokenizer),
    Bpe(BpeTokenizer),
}

impl Tokenizer for AnyTokenizer {
    fn vocab_size(&self) -> usize {
        match self {
            AnyTokenizer::Char(t) => t.vocab_size(),
            AnyTokenizer::Bpe(t) => t.vocab_size(),
        }
    }

    fn encode(&self, text: &str) -> Vec<u16> {
        match self {
            AnyTokenizer::Char(t) => t.encode(text),
            AnyTokenizer::Bpe(t) => t.encode(text),
        }
    }

    fn decode(&self, tokens: &[u16]) -> String {
        match self {
            AnyTokenizer::Char(t) => t.decode(tokens),
            AnyTokenizer::Bpe(t) => t.decode(tokens),
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u16> {
        match self {
            AnyTokenizer::Char(t) => t.token_to_id(token),
            AnyTokenizer::Bpe(t) => t.token_to_id(token),
        }
    }

    fn id_to_token(&self, id: u16) -> Option<String> {
        match self {
            AnyTokenizer::Char(t) => t.id_to_token(id),
            AnyTokenizer::Bpe(t) => t.id_to_token(id),
        }
    }
}

impl AnyTokenizer {
    pub fn vocab_size(&self) -> usize {
        <Self as Tokenizer>::vocab_size(self)
    }

    pub fn encode(&self, text: &str) -> Vec<u16> {
        <Self as Tokenizer>::encode(self, text)
    }

    pub fn decode(&self, tokens: &[u16]) -> String {
        <Self as Tokenizer>::decode(self, tokens)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u16> {
        <Self as Tokenizer>::token_to_id(self, token)
    }

    pub fn id_to_token(&self, id: u16) -> Option<String> {
        <Self as Tokenizer>::id_to_token(self, id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_tokenizer_roundtrip() {
        let text = "山月記。隴西の李徴は博学才頴。";
        let tokenizer = CharTokenizer::build_from_text(text);
        assert_eq!(tokenizer.vocab_size(), 14); // Unique characters

        let encoded = tokenizer.encode(text);
        assert_eq!(encoded.len(), text.chars().count());

        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_bpe_tokenizer_training_and_roundtrip() {
        let text = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。";
        let target_vocab_size = 60;
        let mut tokenizer = BpeTokenizer::train_from_text(text, target_vocab_size);
        tokenizer.rebuild_merge_ranks();

        assert!(tokenizer.vocab_size() <= target_vocab_size);
        assert!(tokenizer.vocab_size() >= 10);
        assert_eq!(tokenizer.unk_id, 0);
        assert_eq!(tokenizer.bos_id, 1);
        assert_eq!(tokenizer.eos_id, 2);
        assert_eq!(tokenizer.pad_id, 3);

        // Encoding compression check against the byte-level source representation.
        let sample = "ニャーニャー泣いていた事だけは記憶している。";
        let encoded = tokenizer.encode(sample);
        let char_count = sample.len();
        assert!(
            encoded.len() <= char_count,
            "encoded len {} should be <= char count {}",
            encoded.len(),
            char_count
        );

        // Lossless roundtrip reconstruction
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, sample);
    }

    #[test]
    fn test_bpe_tokenizer_save_load() {
        let text = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n";
        let tokenizer = BpeTokenizer::train_from_text(text, 35);

        let temp_dir = std::env::temp_dir().join("oniwa_bpe_test");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let vocab_path = temp_dir.join("test_bpe_vocab.json");

        tokenizer.save_vocab(&vocab_path).unwrap();
        let loaded = BpeTokenizer::load_vocab(&vocab_path).unwrap();

        assert_eq!(tokenizer.vocab_size(), loaded.vocab_size());
        assert_eq!(tokenizer.merges, loaded.merges);

        let sample = "def fibonacci(n):\n    if n <= 1:\n        return n\n";
        let enc1 = tokenizer.encode(sample);
        let enc2 = loaded.encode(sample);
        assert_eq!(enc1, enc2);
        assert_eq!(tokenizer.decode(&enc1), sample);
        assert_eq!(loaded.decode(&enc2), sample);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_bpe_unknown_character_fallback() {
        let text = "こんにちは世界";
        let mut tokenizer = BpeTokenizer::train_from_text(text, 15);
        tokenizer.rebuild_merge_ranks();

        // Character 'Z' is not in training text
        let encoded = tokenizer.encode("こんにちはZ世界");
        let decoded = tokenizer.decode(&encoded);

        // 'Z' should map to unk_id (0) and decode into "<unk>"
        assert!(encoded.contains(&tokenizer.unk_id));
        assert!(decoded.contains("<unk>"));
    }

    #[test]
    fn test_tokenizer_trait_polymorphism() {
        let text = "Rust言語によるONIWAアーキテクチャ";
        let char_tok = CharTokenizer::build_from_text(text);
        let bpe_tok = BpeTokenizer::train_from_text(text, 25);

        let tokenizers: Vec<Box<dyn Tokenizer>> = vec![Box::new(char_tok), Box::new(bpe_tok)];

        for tok in tokenizers {
            let encoded = tok.encode(text);
            let decoded = tok.decode(&encoded);
            assert_eq!(decoded, text);
            assert!(tok.vocab_size() > 0);
        }
    }

    #[test]
    fn test_ingest_sangetsuki_dataset() {
        let workspace_root = crate::find_workspace_root();
        let raw_path = workspace_root.join("data/sangetsuki_clean.txt");
        if raw_path.exists() {
            let data_dir = workspace_root.join("data");
            let logs_dir = workspace_root.join("logs");

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
