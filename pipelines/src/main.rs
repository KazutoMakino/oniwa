//! oniwa-pipeline: データセット管理 & 青空文庫自動収集CLI
//!
//! ONIWA: Organic Non-datacenter Intelligence Without Abuse
//! 法・コンプライアンス（著作権満了作品のみ）を厳格に守りながら、
//! 青空文庫の名作群を安全に取得・クレンジング・系譜台帳に記録します。

mod aozora;
mod arxiv;
mod cleaner;
mod code;
mod egov;
mod techdocs;

use aozora::AozoraPipeline;
use arxiv::ArxivPipeline;
use code::CodePipeline;
use egov::EgovPipeline;
use std::path::Path;
use techdocs::TechDocsPipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!(" 📚 ONIWA: クリーン・オープンデータ収集パイプライン");
    println!("    (青空文庫PD ＆ e-Gov法令 ＆ arXivオープンサイエンス ＆ 公式技術仕様 ＆ クリーンコード)");
    println!("============================================================\n");

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root_dir = manifest_dir.parent().unwrap_or(manifest_dir);
    let data_dir = root_dir.join("data");
    let logs_dir = root_dir.join("logs");

    let mut aozora_pipeline = AozoraPipeline::new(&data_dir, &logs_dir);
    let mut egov_pipeline = EgovPipeline::new(&data_dir, &logs_dir);
    let mut arxiv_pipeline = ArxivPipeline::new(&data_dir, &logs_dir);
    let mut techdocs_pipeline = TechDocsPipeline::new(&data_dir, &logs_dir);
    let mut code_pipeline = CodePipeline::new(&data_dir, &logs_dir);

    let args: Vec<String> = std::env::args().collect();
    let mut do_preset = false;
    let mut do_laws = false;
    let mut do_arxiv = false;
    let mut do_techdocs = false;
    let mut do_code = false;
    let mut arxiv_limit = 10usize;
    let mut arxiv_cat = "cs.AI".to_string();
    let mut do_all = false;
    let mut target_recipe: Option<String> = None;
    let mut target_author: Option<String> = None;
    let mut limit = 5usize;
    let mut do_build = false;
    let mut do_status = false;
    let mut do_force = false;
    let mut do_clean = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--recipe" => {
                if let Some(val) = args.get(i + 1) {
                    target_recipe = Some(val.clone());
                    i += 1;
                }
            }
            "--preset" => {
                do_preset = true;
            }
            "--laws" => {
                do_laws = true;
            }
            "--arxiv" => {
                do_arxiv = true;
                if let Some(val) = args.get(i + 1) {
                    if let Ok(num) = val.parse::<usize>() {
                        arxiv_limit = num;
                        i += 1;
                    }
                }
            }
            "--arxiv-cat" => {
                if let Some(val) = args.get(i + 1) {
                    arxiv_cat = val.clone();
                    i += 1;
                }
            }
            "--techdocs" => {
                do_techdocs = true;
            }
            "--code" => {
                do_code = true;
            }
            "--all" => {
                do_all = true;
                do_preset = true;
                do_laws = true;
                do_techdocs = true;
                do_code = true;
                do_arxiv = true;
                arxiv_limit = 5;
                do_build = true;
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

    if !do_preset
        && !do_laws
        && !do_arxiv
        && !do_techdocs
        && !do_code
        && !do_all
        && target_recipe.is_none()
        && target_author.is_none()
        && !do_build
        && !do_status
        && !do_clean
    {
        print_help();
        return Ok(());
    }

    if do_force {
        aozora_pipeline.set_force(true);
        egov_pipeline.set_force(true);
        arxiv_pipeline.set_force(true);
        techdocs_pipeline.set_force(true);
        code_pipeline.set_force(true);
    }

    // 0. クリーン・リセット処理
    if do_clean {
        aozora_pipeline.clean_all()?;
    }

    // 1. 青空文庫: レシピまたはプリセット作品群の取得
    if let Some(ref recipe_path) = target_recipe {
        aozora_pipeline.ingest_from_recipe(recipe_path)?;
    } else if do_preset {
        aozora_pipeline.ingest_presets()?;
    }

    // 2. 青空文庫: 指定著者の作品収集
    if let Some(author) = target_author {
        aozora_pipeline.search_and_ingest_by_author(&author, limit)?;
    }

    // 3. e-Gov: 基本法令オープンデータの取得
    if do_laws {
        egov_pipeline.ingest_default_laws()?;
    }

    // 4. 公式技術ドキュメント・コード仕様の取得
    if do_techdocs {
        techdocs_pipeline.ingest_default_techdocs()?;
    }

    // 5. 基本アルゴリズム・クリーンコード（Python/Rust）の取得
    if do_code {
        code_pipeline.ingest_default_code()?;
    }

    // 6. arXiv: オープンサイエンス論文アブストラクトの取得
    if do_arxiv {
        arxiv_pipeline.ingest_category(&arxiv_cat, arxiv_limit)?;
    }

    // 7. 統合コーパスの再生成 (全ソースの corpus/*.txt を一括結合)
    if do_build
        || do_preset
        || do_laws
        || do_techdocs
        || do_code
        || do_arxiv
        || do_all
        || target_recipe.is_some()
    {
        aozora_pipeline.build_combined_corpus()?;
    }

    // 8. ステータス表示
    if do_status
        || (!do_preset
            && !do_laws
            && !do_techdocs
            && !do_code
            && !do_arxiv
            && !do_all
            && target_recipe.is_none()
            && !do_build)
    {
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
    println!("  --all               全ソース（青空文庫＋法令＋技術＋コード＋arXiv）を一括取得し、統合コーパスを再生成");
    println!("  --preset            青空文庫の代表的な名作群（recipes.json 設定作品群）を一括取得");
    println!(
        "  --laws              e-Gov APIから日本国憲法・刑法・著作権法・民法等の基本法令を一括取得"
    );
    println!(
        "  --techdocs          公式オープンソース技術ドキュメント（Rust公式解説等）を一括取得"
    );
    println!("  --code              オープンソース基本アルゴリズムコード（Python/Rust、階乗/フィボナッチ/探索/ソート等）を一括取得");
    println!("  --arxiv [件数]      arXiv APIから人工知能・自然言語処理等のオープンアクセス論文要約を取得 (デフォルト: 10)");
    println!(
        "  --arxiv-cat <分野>  arXiv検索カテゴリ指定 (例: cs.AI, cs.CL, cs.LG / デフォルト: cs.AI)"
    );
    println!("  --recipe <パス>     JSONレシピファイルに基づいて指定作品群を一括取得（例: pipelines/config/recipes.json）");
    println!(
        "  --author <名前>     指定した著者の著作権満了作品を青空文庫全作品リストから検索して取得"
    );
    println!("  --limit <数>        著者検索時の取得上限作品数 (デフォルト: 5)");
    println!("  --build             収集済みテキスト群を統合して tokens.bin & vocab.json を生成");
    println!("  --force             キャッシュを無視して強制的に再ダウンロード & 再解析");
    println!("  --clean             収集データ・コーパス・台帳を初期化（既存台帳はバックアップ）");
    println!(
        "  --status            現在収集されている作品・法令・論文一覧と系譜台帳の状況を表示\n"
    );
    println!("【使用例】");
    println!("  $ cargo run --release -p oniwa-pipeline -- --all");
    println!("  $ cargo run --release -p oniwa-pipeline -- --techdocs");
    println!("  $ cargo run --release -p oniwa-pipeline -- --arxiv 10");
    println!("  $ cargo run --release -p oniwa-pipeline -- --laws");
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
            println!(
                "  [{:2}] {:<35} ({:6} 文字, {:.1} KB)",
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
        let count = std::fs::read_to_string(&ledger_path)?.lines().count();
        println!("  - 記録された監査・系譜イベント数: {} 件", count);
    }

    Ok(())
}
