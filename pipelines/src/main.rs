//! oniwa-pipeline: データセット管理 & 青空文庫自動収集CLI
//!
//! ONIWA: Organic Non-datacenter Intelligence Without Abuse
//! 法・コンプライアンス（著作権満了作品のみ）を厳格に守りながら、
//! 青空文庫の名作群を安全に取得・クレンジング・系譜台帳に記録します。

mod aozora;
mod cleaner;

use aozora::AozoraPipeline;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!(" 📚 ONIWA: 青空文庫パブリックドメイン・データ収集パイプライン");
    println!("    (脱データセンター・無断搾取なきオーガニック知性)");
    println!("============================================================\n");

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root_dir = manifest_dir.parent().unwrap_or(manifest_dir);
    let data_dir = root_dir.join("data");
    let logs_dir = root_dir.join("logs");

    let mut pipeline = AozoraPipeline::new(&data_dir, &logs_dir);

    let args: Vec<String> = std::env::args().collect();
    let mut do_preset = false;
    let mut target_author: Option<String> = None;
    let mut limit = 5usize;
    let mut do_build = false;
    let mut do_status = false;
    let mut do_force = false;
    let mut do_clean = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--preset" => {
                do_preset = true;
            }
            "--author" => {
                if let Some(val) = args.get(i + 1) {
                    target_author = Some(val.clone());
                    i += 1;
                }
            }
            "--limit" => {
                if let Some(val) = args.get(i + 1) {
                    limit = val.parse().unwrap_or(5);
                    i += 1;
                }
            }
            "--build" | "--build-corpus" => {
                do_build = true;
            }
            "--force" => {
                do_force = true;
            }
            "--clean" => {
                do_clean = true;
            }
            "--status" => {
                do_status = true;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    if !do_preset && target_author.is_none() && !do_build && !do_status && !do_clean {
        print_help();
        return Ok(());
    }

    if do_force {
        pipeline.set_force(true);
    }

    // 0. クリーン・リセット処理
    if do_clean {
        pipeline.clean_all()?;
    }

    // 1. プリセット作品群の取得
    if do_preset {
        pipeline.ingest_presets()?;
    }

    // 2. 指定著者の作品収集
    if let Some(author) = target_author {
        pipeline.search_and_ingest_by_author(&author, limit)?;
    }

    // 3. 統合コーパスの再生成
    if do_build || do_preset {
        pipeline.build_combined_corpus()?;
    }

    // 4. ステータス表示
    if do_status || (!do_preset && !do_build) {
        show_status(&data_dir, &logs_dir)?;
    }

    println!("\n✨ すべての処理が完了しました。");
    println!("学習を実行するには以下のコマンドを実行してください:");
    println!("  $ cargo run --release -p oniwa-lm --bin train -- --steps 500 --reset --prompt \"メロスは、\"");
    println!("============================================================");

    Ok(())
}

fn print_help() {
    println!("【使い方】");
    println!("  cargo run --release -p oniwa-pipeline -- [オプション]\n");
    println!("【オプション】");
    println!("  --preset            代表的な名作（太宰治、芥川龍之介、中島敦、宮沢賢治、夏目漱石、森鴎外など）を一括取得");
    println!("  --author <名前>     指定した著者の著作権満了作品を青空文庫全作品リストから検索して取得");
    println!("  --limit <数>        著者検索時の取得上限作品数 (デフォルト: 5)");
    println!("  --build             収集済みテキストを統合して tokens.bin & vocab.json を生成");
    println!("  --force             キャッシュを無視して強制的に再ダウンロード & 再解析");
    println!("  --clean             収集データ・コーパス・台帳を初期化（既存台帳はバックアップ）");
    println!("  --status            現在収集されている作品一覧と系譜台帳の状況を表示\n");
    println!("【使用例】");
    println!("  $ cargo run --release -p oniwa-pipeline -- --preset");
    println!("  $ cargo run --release -p oniwa-pipeline -- --author \"夏目漱石\" --limit 3 --build");
    println!("  $ cargo run --release -p oniwa-pipeline -- --clean --preset --build");
    println!("  $ cargo run --release -p oniwa-pipeline -- --status");
}

fn show_status(data_dir: &Path, logs_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let corpus_dir = data_dir.join("corpus");
    let ledger_path = logs_dir.join("ledger_index.jsonl");

    println!("============================================================");
    println!(" 📊 現在のデータセット保管状況 (data/corpus/)");
    println!("============================================================");

    if corpus_dir.exists() {
        let mut entries: Vec<_> = std::fs::read_dir(&corpus_dir)?
            .filter_map(|e| e.ok())
            .collect();
        entries.sort_by_key(|e| e.path());

        let mut total_chars = 0;
        for (idx, entry) in entries.iter().enumerate() {
            let path = entry.path();
            let meta = entry.metadata()?;
            let content = std::fs::read_to_string(&path).unwrap_or_default();
            let chars = content.chars().count();
            total_chars += chars;
            println!("  [{:2}] {:<35} ({:6} 文字, {:.1} KB)",
                idx + 1,
                path.file_name().unwrap_or_default().to_string_lossy(),
                chars,
                meta.len() as f32 / 1024.0
            );
        }
        println!("------------------------------------------------------------");
        println!("  - 合計作品数: {} 作品", entries.len());
        println!("  - 合計文字数: {} 文字", total_chars);
    } else {
        println!("  （まだ収集された作品はありません。--preset で取得できます）");
    }

    println!("\n 📜 監査台帳 (logs/ledger_index.jsonl):");
    if ledger_path.exists() {
        let count = std::fs::read_to_string(&ledger_path)?
            .lines()
            .count();
        println!("  - 記録された監査・系譜イベント数: {} 件", count);
    }

    Ok(())
}
