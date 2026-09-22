use oniwa_lm::tokenizer::Tokenizer;
use serde::Deserialize;
use std::io::{self, BufRead, BufReader, Read};

#[derive(Deserialize)]
struct JsonLabels {
    choice: usize,
    noul: f32,
    score: f32,
}

#[derive(Deserialize)]
struct JsonPattern {
    id: String,
    text: String,
    labels: JsonLabels,
    mask_indices: Vec<usize>,
}

pub struct KillerPattern {
    text: String,
    pub choice: usize,
    pub noul: bool,
    pub score: f32,
    pub mask_indices: Vec<usize>,
}

pub struct KillerPatternDataset {
    patterns: Vec<KillerPattern>,
}

pub struct KillerPatternLabels<'a> {
    pub choice: usize,
    pub noul: bool,
    pub score: f32,
    pub mask_indices: &'a [usize],
}

impl KillerPatternDataset {
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    pub fn from_jsonl_reader(reader: impl Read) -> io::Result<Self> {
        let mut patterns = Vec::new();
        for (line_number, line) in BufReader::new(reader).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let raw: JsonPattern = serde_json::from_str(&line).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("line {}: {error}", line_number + 1),
                )
            })?;
            if raw.id.trim().is_empty()
                || raw.text.trim().is_empty()
                || raw.labels.choice >= 4
                || !(0.0..=1.0).contains(&raw.labels.noul)
                || !(0.0..=1.0).contains(&raw.labels.score)
                || raw.mask_indices.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("line {}: invalid killer pattern", line_number + 1),
                ));
            }
            patterns.push(KillerPattern {
                text: raw.text,
                choice: raw.labels.choice,
                noul: raw.labels.noul >= 0.5,
                score: raw.labels.score,
                mask_indices: raw.mask_indices,
            });
        }
        if patterns.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "no killer patterns",
            ));
        }
        Ok(Self { patterns })
    }

    pub fn write_batch<'a>(
        &'a self,
        tokenizer: &dyn Tokenizer,
        index: usize,
        destination: &mut [u16],
    ) -> io::Result<KillerPatternLabels<'a>> {
        let pattern = self.patterns.get(index).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "killer pattern index out of range",
            )
        })?;
        destination.fill(0);
        let written = tokenizer.encode_into(&pattern.text, destination);
        if pattern.mask_indices.iter().any(|&index| index >= written) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "killer pattern mask index is outside encoded text",
            ));
        }
        Ok(KillerPatternLabels {
            choice: pattern.choice,
            noul: pattern.noul,
            score: pattern.score,
            mask_indices: &pattern.mask_indices,
        })
    }
}
