//! Train and save ONIWA's dependency-free BPE vocabulary.

use oniwa_lm::tokenizer::BpeTokenizer;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = None;
    let mut output = None;
    let mut vocab_size = 4_096usize;
    let args: Vec<_> = std::env::args().collect();
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--input" => input = args.get(index + 1).map(PathBuf::from),
            "--output" => output = args.get(index + 1).map(PathBuf::from),
            "--vocab-size" => {
                vocab_size = args
                    .get(index + 1)
                    .ok_or("--vocab-size requires a value")?
                    .parse()?
            }
            "--help" | "-h" => {
                println!("cargo run -p oniwa-pipeline --bin train-bpe -- --input CORPUS --vocab-size 4096 --output bpe_vocab.json");
                return Ok(());
            }
            _ => {}
        }
        index += 1;
    }
    let input = input.ok_or("--input is required")?;
    let output = output.ok_or("--output is required")?;
    let text = std::fs::read_to_string(input)?;
    let tokenizer = BpeTokenizer::train_from_text(&text, vocab_size);
    tokenizer.save_vocab(output)?;
    Ok(())
}
