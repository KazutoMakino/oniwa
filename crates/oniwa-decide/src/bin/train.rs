//! oniwa-decide 決定モデル学習実行バイナリ
//!
//! 自己教師あり自動生成データセットを用いて、双方向 Transformer エンコーダと
//! 3つの決定ヘッド（Choice, Noul, Score）を同時に学習します。
//! 熱制御と消費電力測定（ピュアRust）を完全搭載。

use oniwa_decide::{DatasetGenerator, DecisionConfig, DecisionModel, LossCalculator, LossConfig};
use oniwa_lm::power::PowerTracker;
use oniwa_lm::reproducibility::DeterministicRng;
use oniwa_lm::thermal::{ThermalConfig, ThermalController};
use oniwa_lm::tokenizer::CharTokenizer;
use std::env;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!(" 🧭 ONIWA-DECIDE: TypeSafe System One Training Engine");
    println!("    (型安全意思決定モデル学習バイナリ)");
    println!("============================================================\n");

    let args: Vec<String> = env::args().collect();
    let mut num_steps = 300usize;
    let mut batch_size = 8usize;
    let mut lr = 0.001f32;
    let mut seed = 42u64;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => {
                if let Some(v) = args.get(i + 1) {
                    num_steps = v.parse().unwrap_or(300);
                    i += 1;
                }
            }
            "--batch-size" => {
                if let Some(v) = args.get(i + 1) {
                    batch_size = v.parse().unwrap_or(8);
                    i += 1;
                }
            }
            "--lr" => {
                if let Some(v) = args.get(i + 1) {
                    lr = v.parse().unwrap_or(0.001);
                    i += 1;
                }
            }
            "--seed" => {
                if let Some(v) = args.get(i + 1) {
                    seed = v.parse().unwrap_or(42);
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let workspace_root = oniwa_lm::find_workspace_root();
    let data_dir = workspace_root.join("data");
    let corpus_dir = data_dir.join("corpus");
    let base_dir = workspace_root.join("crates/oniwa-decide");
    let checkpoint_dir = base_dir.join("checkpoints").join("latest");
    let best_checkpoint_dir = base_dir.join("checkpoints").join("best");

    // トークナイザとデータ生成器の準備
    let vocab_path = data_dir.join("vocab.json");
    let tokenizer = if vocab_path.exists() {
        CharTokenizer::load_vocab(&vocab_path)?
    } else {
        CharTokenizer::build_from_text("abcdefghijklmnopqrstuvwxyz 0123456789")
    };
    let dataset = DatasetGenerator::load_from_corpus_dir(&corpus_dir)?;

    let config = DecisionConfig {
        vocab_size: tokenizer.vocab_size(),
        seq_len: 128,
        dim: 128,
        num_layers: 4,
        num_heads: 4,
        head_dim: 32,
        ffn_dim: 256,
        num_choices: 4,
        temperature: 1.0,
    };

    let mut rng = DeterministicRng::new(seed);
    let mut model = DecisionModel::new(config.clone(), &mut rng);
    let loss_config = LossConfig::default();

    let thermal = ThermalController::new(ThermalConfig::default());
    let mut power_tracker = PowerTracker::auto_detect();

    println!(
        "  - モデル規模: {} layers, {} heads, dim {} (パラメータ数: {})",
        config.num_layers,
        config.num_heads,
        config.dim,
        model.params.len()
    );
    println!(
        "  - 学習ステップ数: {}, バッチサイズ: {}, 初期学習率: {}",
        num_steps, batch_size, lr
    );
    println!("  - 決定タスク: Choice (4種別), Noul (構文異常), Score (複雑度)\n");

    let mut best_loss = f32::INFINITY;
    let start_time = Instant::now();

    for step in 1..=num_steps {
        let step_start = Instant::now();

        // 1. バッチ生成
        let (tokens, target_choices, target_nouls, target_scores) =
            dataset.generate_batch(&tokenizer, batch_size, config.seq_len, &mut rng);

        // 2. 順伝播
        model.zero_grad();
        let cache = model.forward(&tokens, batch_size, config.seq_len);

        // 3. 損失計算
        let mut total_loss = 0.0f32;
        let mut dchoice_logits = vec![0.0f32; batch_size * config.num_choices];
        let mut dnoul_logits = vec![0.0f32; batch_size];
        let mut dscore_preds = vec![0.0f32; batch_size];

        let mut correct_choice = 0;
        let mut correct_noul = 0;
        let mut total_score_err = 0.0f32;

        for bi in 0..batch_size {
            // Choice Loss
            let logits =
                &cache.choice_logits[bi * config.num_choices..(bi + 1) * config.num_choices];
            let (l_c, d_c) = LossCalculator::choice_loss(
                logits,
                target_choices[bi],
                config.num_choices,
                loss_config.label_smoothing,
            );
            total_loss += loss_config.choice_weight * l_c;
            for k in 0..config.num_choices {
                dchoice_logits[bi * config.num_choices + k] =
                    loss_config.choice_weight * d_c[k] / (batch_size as f32);
            }
            let pred_choice = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(k, _)| k)
                .unwrap();
            if pred_choice == target_choices[bi] {
                correct_choice += 1;
            }

            // Noul Loss
            let noul_logit = cache.noul_logits[bi];
            let (l_n, d_n) =
                LossCalculator::noul_loss(noul_logit, target_nouls[bi], loss_config.brier_weight);
            total_loss += loss_config.noul_weight * l_n;
            dnoul_logits[bi] = loss_config.noul_weight * d_n / (batch_size as f32);
            let pred_noul = LossCalculator::sigmoid(noul_logit) >= 0.5;
            if pred_noul == target_nouls[bi] {
                correct_noul += 1;
            }

            // Score Loss
            let score_pred = cache.score_preds[bi];
            let (l_s, d_s) =
                LossCalculator::score_loss(score_pred, target_scores[bi], loss_config.huber_delta);
            total_loss += loss_config.score_weight * l_s;
            dscore_preds[bi] = loss_config.score_weight * d_s / (batch_size as f32);
            total_score_err += (score_pred - target_scores[bi]).abs();
        }

        let mean_loss = total_loss / (batch_size as f32);

        // 4. 逆伝播
        model.backward(
            &tokens,
            &cache,
            &dchoice_logits,
            &dnoul_logits,
            &dscore_preds,
            batch_size,
            config.seq_len,
        );

        // 5. AdamW 最適化
        let cur_lr = lr * (1.0 - (step as f32 / num_steps as f32) * 0.8); // 緩やかなLinear Decay
        model.adamw_step(cur_lr, 0.01, 0.9, 0.999, 1e-8, step);

        // 6. 熱制御 & 電力追跡
        let calc_time = step_start.elapsed().as_millis();
        let (cpu_temp, throttle_ms) = thermal.step_throttle();
        let reading = power_tracker.tick(calc_time, throttle_ms);

        // 定期ログ出力 (25ステップごと、または初回/最終)
        if step % 25 == 0 || step == 1 || step == num_steps {
            let choice_acc = (correct_choice as f32 / batch_size as f32) * 100.0;
            let noul_acc = (correct_noul as f32 / batch_size as f32) * 100.0;
            let score_mae = total_score_err / batch_size as f32;

            println!(
                "Step {:4}/{} | Loss: {:.4} | Choice Acc: {:5.1}% | Noul Acc: {:5.1}% | Score MAE: {:.3} | Temp: {} | Power: {:.1}W",
                step,
                num_steps,
                mean_loss,
                choice_acc,
                noul_acc,
                score_mae,
                cpu_temp.map(|t| format!("{:.1}℃", t)).unwrap_or_else(|| "N/A".into()),
                reading.net_watts,
            );

            // ベスト保存
            if mean_loss < best_loss {
                best_loss = mean_loss;
                let _ = model.save_checkpoint(&best_checkpoint_dir, step, mean_loss);
            }
            let _ = model.save_checkpoint(&checkpoint_dir, step, mean_loss);
        }
    }

    let elapsed = start_time.elapsed();
    println!("\n============================================================");
    println!(" 🎉 oniwa-decide 学習完了！");
    println!("  - 所要時間: {:.2?}", elapsed);
    println!("  - 最小 Loss: {:.4}", best_loss);
    println!(
        "  - ベストチェックポイント保存先: {:?}",
        best_checkpoint_dir
    );
    println!(
        "  - ⚡ 累積消費電力量 (Net): {:.4} Wh",
        power_tracker.total_net_wh()
    );
    println!("============================================================");

    Ok(())
}
