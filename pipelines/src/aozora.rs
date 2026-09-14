//! 青空文庫パブリックドメイン（著作権満了）作品の自動収集 & 前処理パイプライン
//!
//! ONIWAの理念に基づき、著作権保護期間が満了した作品（パブリックドメイン）のみを
//! 厳格にフィルタリングし、サーバー負荷を防ぐエチケット（User-Agent、ウェイト、キャッシュ）
//! を遵守して安全に取得・クレンジング・系譜台帳記録を行います。

use crate::cleaner::clean_aozora_text;
use oniwa_lm::logger::{DataIngestionLog, ProvenanceEvent, ProvenanceLedger};
use oniwa_lm::reproducibility::compute_checksum_bytes;
use oniwa_lm::tokenizer::CharTokenizer;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread::sleep;
use std::time::Duration;

/// 青空文庫の作品メタデータ
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AozoraWorkEntry {
    pub work_id: String,
    pub title: String,
    pub author_last: String,
    pub author_first: String,
    pub card_url: String,
    pub text_zip_url: String,
    pub font_type: String,     // 新字新仮名, 旧字旧仮名 等
    pub copyright_work: bool,  // true: 著作権あり, false: なし(PD)
    pub copyright_author: bool,// true: 著作権あり, false: なし(PD)
}

#[allow(dead_code)]
impl AozoraWorkEntry {
    pub fn author_full(&self) -> String {
        format!("{}{}", self.author_last, self.author_first)
    }

    pub fn is_public_domain(&self) -> bool {
        !self.copyright_work && !self.copyright_author
    }
}

pub struct AozoraPipeline {
    data_dir: PathBuf,
    raw_dir: PathBuf,
    corpus_dir: PathBuf,
    logs_dir: PathBuf,
    force_download: bool,
}

impl AozoraPipeline {
    pub fn new<P: AsRef<Path>>(base_data_dir: P, logs_dir: P) -> Self {
        let data_dir = base_data_dir.as_ref().to_path_buf();
        let raw_dir = data_dir.join("raw");
        let corpus_dir = data_dir.join("corpus");
        let logs_dir = logs_dir.as_ref().to_path_buf();

        fs::create_dir_all(&raw_dir).ok();
        fs::create_dir_all(&corpus_dir).ok();
        fs::create_dir_all(&logs_dir).ok();

        Self {
            data_dir,
            raw_dir,
            corpus_dir,
            logs_dir,
            force_download: false,
        }
    }

    pub fn set_force(&mut self, force: bool) {
        self.force_download = force;
    }

    /// 取得データ・コーパス・監査台帳をまっさらにリセット（安全に初期化）
    pub fn clean_all(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("  🧹 既存のデータセット・コーパス・台帳を初期化中...");
        if self.corpus_dir.exists() {
            fs::remove_dir_all(&self.corpus_dir).ok();
            fs::create_dir_all(&self.corpus_dir).ok();
        }
        if self.raw_dir.exists() {
            fs::remove_dir_all(&self.raw_dir).ok();
            fs::create_dir_all(&self.raw_dir).ok();
        }
        let ledger_path = self.logs_dir.join("ledger_index.jsonl");
        if ledger_path.exists() {
            let backup_path = self.logs_dir.join("ledger_index_backup.jsonl");
            fs::copy(&ledger_path, &backup_path).ok();
            fs::remove_file(&ledger_path).ok();
            println!("  📋 既存の台帳は {:?} に退避し、新規台帳を開設します", backup_path);
        }
        println!("  ✅ 初期化完了！");
        Ok(())
    }

    /// 青空文庫zipアーカイブをダウンロード（キャッシュがあればスキップ）
    fn download_file(&self, url: &str, dest_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        if !self.force_download && dest_path.exists() && dest_path.metadata()?.len() > 0 {
            // キャッシュヒット
            return Ok(());
        }

        println!("  📥 ダウンロード中: {} ...", url);
        // 青空文庫サーバーへのエチケット: User-Agentを明記
        let user_agent = "oniwa-lm/0.1.0 (Public Domain AI Training Pipeline; Ethical AI without abuse)";
        let status = Command::new("curl")
            .arg("-s")
            .arg("-f")
            .arg("-L")
            .arg("-A")
            .arg(user_agent)
            .arg("-o")
            .arg(dest_path)
            .arg(url)
            .status()?;

        if !status.success() {
            return Err(format!("ダウンロード失敗 (curl exit code: {:?}): {}", status.code(), url).into());
        }

        // サーバー負荷軽減のためのウェイト（1秒）
        sleep(Duration::from_millis(1000));
        Ok(())
    }

    /// zipファイルからShift_JISテキストを取り出し、UTF-8に変換
    fn extract_and_decode_zip(&self, zip_path: &Path) -> Result<String, Box<dyn std::error::Error>> {
        let file = File::open(zip_path)?;
        let mut archive = zip::ZipArchive::new(file)?;

        // 通常1番目のテキストファイルを対象とする
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let name = file.name().to_string();
            if name.ends_with(".txt") {
                let mut bytes = Vec::new();
                file.read_to_end(&mut bytes)?;

                // Shift_JIS -> UTF-8 デコード
                let (cow, _, had_errors) = encoding_rs::SHIFT_JIS.decode(&bytes);
                if had_errors {
                    eprintln!("  ⚠️ Shift_JISデコード中に一部文字の置換が発生しました: {}", name);
                }
                return Ok(cow.into_owned());
            }
        }

        Err(format!("zipファイル内に .txt が見つかりませんでした: {:?}", zip_path).into())
    }

    /// 単一の作品を取得・クレンジングして corpus_dir に保存し、台帳に記録
    pub fn ingest_single_work(
        &self,
        author: &str,
        title: &str,
        zip_url: &str,
        card_url: &str,
        license: &str,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let file_stem = format!("{}_{}", author, title);
        let zip_dest = self.raw_dir.join(format!("{}.zip", file_stem));
        let clean_dest = self.corpus_dir.join(format!("{}.txt", file_stem));

        // 1. ダウンロード
        self.download_file(zip_url, &zip_dest)?;

        // 2. 解凍 & Shift_JISデコード
        let raw_text = self.extract_and_decode_zip(&zip_dest)?;
        let raw_sha256 = compute_checksum_bytes(raw_text.as_bytes());

        // 3. クレンジング（ルビ・注記・ヘッダー・フッター除去）
        let cleaned = clean_aozora_text(&raw_text);
        fs::write(&clean_dest, &cleaned)?;

        let cleaned_bytes = cleaned.as_bytes();
        let cleaned_sha256 = compute_checksum_bytes(cleaned_bytes);
        let char_count = cleaned.chars().count();

        // 4. データ系譜台帳 (ledger_index.jsonl) への記録
        let ledger_path = self.logs_dir.join("ledger_index.jsonl");
        let mut ledger = ProvenanceLedger::open(&ledger_path)?;
        ledger.record(&ProvenanceEvent::DataIngestion(DataIngestionLog {
            timestamp_utc: oniwa_lm::logger::current_timestamp_utc(),
            source_name: format!("青空文庫: {}『{}』", author, title),
            source_url_or_path: card_url.to_string(),
            license: license.to_string(),
            raw_data_sha256: raw_sha256,
            raw_data_bytes: raw_text.len(),
            tokenized_sha256: cleaned_sha256,
            num_tokens: char_count,
            vocab_size: 0, // 全体統合時に更新
            tokenizer_type: "Aozora Cleaner -> Character-level UTF-8".into(),
        }))?;

        println!("  ✅ 取得 & クレンジング完了: 『{}』({} 文字)", title, char_count);
        Ok(clean_dest)
    }

    /// プリセット作品群（代表的な著作権満了の名作）を自動収集
    pub fn ingest_presets(&self) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        let preset_targets = [
            ("中島敦", "山月記"),
            ("太宰治", "走れメロス"),
            ("芥川竜之介", "羅生門"),
            ("芥川竜之介", "蜘蛛の糸"),
            ("宮沢賢治", "注文の多い料理店"),
            ("宮沢賢治", "セロ弾きのゴーシュ"),
            ("森鴎外", "高瀬舟"),
            ("夏目漱石", "坊っちゃん"),
        ];

        println!("============================================================");
        println!(" 📚 青空文庫パブリックドメイン・プリセット作品の自動収集");
        println!("    (対象: 著作権保護期間満了作品のみ / 全 {} 作品)", preset_targets.len());
        println!("============================================================");

        let mut paths = Vec::new();
        for (author, title) in preset_targets {
            println!("▶ 『{}』（著: {}）", title, author);
            if let Some(entry) = self.find_work_in_index(author, title)? {
                let path = self.ingest_single_work(
                    &entry.author_full(),
                    &entry.title,
                    &entry.text_zip_url,
                    &entry.card_url,
                    "Public Domain (青空文庫 著作権満了)",
                )?;
                paths.push(path);
            } else {
                eprintln!("  ⚠️ 『{}』（{}）が公式インデックスで見つかりませんでした", title, author);
            }
        }

        Ok(paths)
    }

    /// 公式インデックスCSVから作品を検索
    pub fn find_work_in_index(
        &self,
        target_author: &str,
        target_title: &str,
    ) -> Result<Option<AozoraWorkEntry>, Box<dyn std::error::Error>> {
        let index_zip = self.raw_dir.join("list_person_all_extended_utf8.zip");
        let index_url = "https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip";
        self.download_file(index_url, &index_zip)?;

        let file = File::open(&index_zip)?;
        let mut archive = zip::ZipArchive::new(file)?;
        let mut csv_file = archive.by_index(0)?;

        let reader = BufReader::new(&mut csv_file);

        for line_res in reader.lines() {
            let line = line_res?;
            if line.starts_with("作品ID") || line.starts_with('\u{feff}') {
                continue;
            }

            let fields = parse_csv_line(&line);
            if fields.len() < 46 {
                continue;
            }

            let work_id = fields[0].clone();
            let title = fields[1].clone();
            let font_type = fields[9].clone();
            let copyright_work = fields[10] == "あり";
            let card_url = fields[13].clone();
            let author_last = fields[15].clone();
            let author_first = fields[16].clone();
            let copyright_author = fields[26] == "あり";
            let zip_url = fields[45].clone();

            let full_author = format!("{}{}", author_last, author_first);

            if (full_author.contains(target_author) || target_author.contains(&full_author))
                && title == target_title
                && !copyright_work
                && !copyright_author
                && font_type.contains("新字")
                && zip_url.starts_with("http")
                && zip_url.ends_with(".zip")
            {
                return Ok(Some(AozoraWorkEntry {
                    work_id,
                    title,
                    author_last,
                    author_first,
                    card_url,
                    text_zip_url: zip_url,
                    font_type,
                    copyright_work,
                    copyright_author,
                }));
            }
        }

        Ok(None)
    }

    /// 青空文庫公式拡張インデックスCSVをダウンロードして、著作権満了作品を検索・一括収集
    pub fn search_and_ingest_by_author(
        &self,
        target_author: &str,
        max_works: usize,
    ) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        let index_zip = self.raw_dir.join("list_person_all_extended_utf8.zip");
        let index_url = "https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip";

        println!("============================================================");
        println!(" 🔍 青空文庫公式インデックス検索: 著者「{}」 (最大 {} 作品)", target_author, max_works);
        println!("============================================================");

        self.download_file(index_url, &index_zip)?;

        let file = File::open(&index_zip)?;
        let mut archive = zip::ZipArchive::new(file)?;
        let mut csv_file = archive.by_index(0)?;

        let reader = BufReader::new(&mut csv_file);
        let mut collected = Vec::new();

        for line_res in reader.lines() {
            let line = line_res?;
            if line.starts_with("作品ID") || line.starts_with('\u{feff}') {
                continue; // ヘッダー
            }

            let fields: Vec<String> = parse_csv_line(&line);
            if fields.len() < 46 {
                continue;
            }

            let title = &fields[1];
            let font_type = &fields[9];
            let copyright_work = &fields[10]; // "なし" or "あり"
            let card_url = &fields[13];
            let author_last = &fields[15];
            let author_first = &fields[16];
            let copyright_author = &fields[26]; // "なし" or "あり"
            let zip_url = &fields[45];

            let full_author = format!("{}{}", author_last, author_first);

            // コンプライアンスチェック: 著作権フラグが「なし」かつ著者名一致
            if full_author.contains(target_author)
                && copyright_work == "なし"
                && copyright_author == "なし"
                && font_type.contains("新字")
                && zip_url.starts_with("http")
                && zip_url.ends_with(".zip")
            {
                println!("▶ 検出: 『{}』（著: {}）- 著作権: 満了 [PD]", title, full_author);
                match self.ingest_single_work(
                    &full_author,
                    title,
                    zip_url,
                    card_url,
                    "Public Domain (青空文庫 著作権満了)",
                ) {
                    Ok(p) => {
                        collected.push(p);
                        if collected.len() >= max_works {
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("  ⚠️ 『{}』の取得エラー: {}", title, e);
                    }
                }
            }
        }

        Ok(collected)
    }

    /// corpus_dir 内の全テキストを統合し、語彙テーブル (vocab.json) とトークン列 (tokens.bin) を再生成
    pub fn build_combined_corpus(&self) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        println!("\n============================================================");
        println!(" 📦 全作品テキストの統合 & 語彙・トークンバイナリ生成");
        println!("============================================================");

        let mut combined_text = String::new();
        let mut work_count = 0;

        let mut entries: Vec<_> = fs::read_dir(&self.corpus_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("txt"))
            .collect();
        entries.sort_by_key(|e| e.path());

        for entry in entries {
            let path = entry.path();
            let text = fs::read_to_string(&path)?;
            if !text.is_empty() {
                combined_text.push_str(&text);
                combined_text.push_str("\n\n");
                work_count += 1;
            }
        }

        let combined_file = self.data_dir.join("corpus_combined.txt");
        fs::write(&combined_file, &combined_text)?;

        let total_chars = combined_text.chars().count();
        println!("  - 統合作品数: {} 作品", work_count);
        println!("  - 統合文字数: {} 文字 ({:.2} KB)", total_chars, combined_text.len() as f32 / 1024.0);

        // トークナイズと語彙生成
        let tokenizer = CharTokenizer::ingest_file(
            &combined_file,
            &self.data_dir,
            &self.logs_dir,
            &format!("青空文庫パブリックドメイン統合コーパス ({}作品)", work_count),
            "https://www.aozora.gr.jp/",
            "Public Domain (著作権満了)",
        )?;

        println!("  - 統合語彙サイズ: {} 文字", tokenizer.vocab_size());
        println!("  - tokens.bin & vocab.json 更新完了！");

        Ok((work_count, total_chars))
    }
}

/// CSV1行の簡易クオート対応パーサー
fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        if ch == '"' {
            in_quotes = !in_quotes;
        } else if ch == ',' && !in_quotes {
            fields.push(field.trim_matches('"').to_string());
            field.clear();
        } else {
            field.push(ch);
        }
        i += 1;
    }
    fields.push(field.trim_matches('"').to_string());
    fields
}
