//! oniwa-decide 包括的ユニットテスト

use oniwa_decide::dataset::DocCategory;
use oniwa_decide::loss::LossCalculator;
use oniwa_decide::model::{DecisionConfig, DecisionModel};
use oniwa_lm::reproducibility::DeterministicRng;

#[test]
fn test_loss_functions() {
    // 1. Softmax
    let logits = vec![1.0, 2.0, 3.0, 4.0];
    let probs = LossCalculator::softmax(&logits, 1.0);
    assert_eq!(probs.len(), 4);
    let sum_p: f32 = probs.iter().sum();
    assert!((sum_p - 1.0).abs() < 1e-5);
    assert!(probs[3] > probs[2] && probs[2] > probs[1]);

    // 2. Choice Loss
    let (loss, dlogits) = LossCalculator::choice_loss(&logits, 3, 4, 0.05);
    assert!(loss > 0.0);
    assert_eq!(dlogits.len(), 4);

    // 3. Noul Loss
    let (loss_t, d_t) = LossCalculator::noul_loss(2.0, true, 0.1);
    assert!(loss_t > 0.0);
    assert!(d_t < 0.0); // 確率が高く正解がtrueなので勾配は負（ロジットを上げる方向）

    let (loss_f, d_f) = LossCalculator::noul_loss(2.0, false, 0.1);
    assert!(loss_f > loss_t);
    assert!(d_f > 0.0); // 確率が高いのに正解がfalseなので勾配は正（ロジットを下げる方向）

    // 4. Score Loss
    let (s_loss, s_grad) = LossCalculator::score_loss(3.5, 3.0, 0.5);
    assert!(s_loss > 0.0);
    assert!((s_grad - 1.0).abs() < 1e-4);
}

#[test]
fn test_forward_dimensions_and_properties() {
    let config = DecisionConfig {
        vocab_size: 100,
        seq_len: 16,
        dim: 32,
        num_layers: 2,
        num_heads: 2,
        head_dim: 16,
        ffn_dim: 64,
        num_choices: 4,
        temperature: 1.0,
    };

    let mut rng = DeterministicRng::new(123);
    let model = DecisionModel::new(config.clone(), &mut rng);

    let b = 2;
    let t = 16;
    let tokens: Vec<u16> = (0..b * t).map(|i| (i % 100) as u16).collect();

    let cache = model.forward(&tokens, b, t);
    assert_eq!(cache.choice_logits.len(), b * config.num_choices);
    assert_eq!(cache.noul_logits.len(), b);
    assert_eq!(cache.score_preds.len(), b);

    // 推論 API のテスト
    let single_tokens = &tokens[..t];
    let dec = model.decide(single_tokens);
    let sum_p: f32 = dec.choice_probs.iter().sum();
    assert!((sum_p - 1.0).abs() < 1e-4);
    assert!(dec.choice_confidence >= 0.0 && dec.choice_confidence <= 1.0);
    assert!(dec.noul_prob >= 0.0 && dec.noul_prob <= 1.0);
    assert!(dec.score_value >= 1.0 && dec.score_value <= 5.0);
}

#[test]
fn test_finite_difference_gradcheck() {
    // 最小規模モデルでの有限差分勾配チェック
    let config = DecisionConfig {
        vocab_size: 20,
        seq_len: 4,
        dim: 8,
        num_layers: 1,
        num_heads: 2,
        head_dim: 4,
        ffn_dim: 16,
        num_choices: 2,
        temperature: 1.0,
    };

    let mut rng = DeterministicRng::new(999);
    let mut model = DecisionModel::new(config.clone(), &mut rng);

    let b = 1;
    let t = 4;
    let tokens: Vec<u16> = vec![1, 3, 5, 7];
    let target_choice = DocCategory::PythonCode as usize;
    let target_noul = true;
    let target_score = 3.5f32;

    // 順伝播
    model.zero_grad();
    let cache = model.forward(&tokens, b, t);

    // 損失と逆伝播用勾配
    let (l_c, d_c) = LossCalculator::choice_loss(&cache.choice_logits, target_choice, 2, 0.0);
    let (l_n, d_n) = LossCalculator::noul_loss(cache.noul_logits[0], target_noul, 0.0);
    let (l_s, d_s) = LossCalculator::score_loss(cache.score_preds[0], target_score, 0.5);
    let total_loss = l_c + l_n + l_s;

    model.backward(&tokens, &cache, &d_c, &[d_n], &[d_s], b, t);

    // ヘッドパラメータの有限差分チェック
    let test_param_idx = model.offset_head_choice;
    let analytic_grad = model.grads[test_param_idx];

    let eps = 1e-3f32;
    model.params[test_param_idx] += eps;
    let cache_p = model.forward(&tokens, b, t);
    let (l_c_p, _) = LossCalculator::choice_loss(&cache_p.choice_logits, target_choice, 2, 0.0);
    let (l_n_p, _) = LossCalculator::noul_loss(cache_p.noul_logits[0], target_noul, 0.0);
    let (l_s_p, _) = LossCalculator::score_loss(cache_p.score_preds[0], target_score, 0.5);
    let loss_plus = l_c_p + l_n_p + l_s_p;

    model.params[test_param_idx] -= 2.0 * eps;
    let cache_m = model.forward(&tokens, b, t);
    let (l_c_m, _) = LossCalculator::choice_loss(&cache_m.choice_logits, target_choice, 2, 0.0);
    let (l_n_m, _) = LossCalculator::noul_loss(cache_m.noul_logits[0], target_noul, 0.0);
    let (l_s_m, _) = LossCalculator::score_loss(cache_m.score_preds[0], target_score, 0.5);
    let loss_minus = l_c_m + l_n_m + l_s_m;

    model.params[test_param_idx] += eps; // 復元

    let numerical_grad = (loss_plus - loss_minus) / (2.0 * eps);

    println!(
        "Total Loss: {:.4}, Analytic: {:.5}, Numerical: {:.5}",
        total_loss, analytic_grad, numerical_grad
    );
    let diff = (analytic_grad - numerical_grad).abs();
    assert!(
        diff < 5e-3 || diff / (analytic_grad.abs() + numerical_grad.abs()).max(1e-5) < 0.05,
        "Gradient check failed! Analytic: {}, Numerical: {}",
        analytic_grad,
        numerical_grad
    );
}

#[test]
fn test_single_step_optimization() {
    let config = DecisionConfig {
        vocab_size: 50,
        seq_len: 8,
        dim: 16,
        num_layers: 2,
        num_heads: 2,
        head_dim: 8,
        ffn_dim: 32,
        num_choices: 3,
        temperature: 1.0,
    };

    let mut rng = DeterministicRng::new(42);
    let mut model = DecisionModel::new(config.clone(), &mut rng);

    let b = 2;
    let t = 8;
    let tokens: Vec<u16> = (0..b * t).map(|i| (i % 50) as u16).collect();
    let target_choices = [0, 2];
    let target_nouls = [false, true];
    let target_scores = [2.0, 4.0];

    // ステップ前 Loss
    model.zero_grad();
    let cache1 = model.forward(&tokens, b, t);
    let mut loss1 = 0.0f32;
    let mut d_c = vec![0.0f32; b * config.num_choices];
    let mut d_n = vec![0.0f32; b];
    let mut d_s = vec![0.0f32; b];

    for bi in 0..b {
        let (lc, dc) = LossCalculator::choice_loss(
            &cache1.choice_logits[bi * config.num_choices..(bi + 1) * config.num_choices],
            target_choices[bi],
            config.num_choices,
            0.0,
        );
        let (ln, dn) = LossCalculator::noul_loss(cache1.noul_logits[bi], target_nouls[bi], 0.0);
        let (ls, ds) = LossCalculator::score_loss(cache1.score_preds[bi], target_scores[bi], 0.5);
        loss1 += lc + ln + ls;
        for k in 0..config.num_choices {
            d_c[bi * config.num_choices + k] = dc[k];
        }
        d_n[bi] = dn;
        d_s[bi] = ds;
    }

    model.backward(&tokens, &cache1, &d_c, &d_n, &d_s, b, t);
    model.adamw_step(0.01, 0.0, 0.9, 0.999, 1e-8, 1);

    // ステップ後 Loss
    let cache2 = model.forward(&tokens, b, t);
    let mut loss2 = 0.0f32;
    for bi in 0..b {
        let (lc, _) = LossCalculator::choice_loss(
            &cache2.choice_logits[bi * config.num_choices..(bi + 1) * config.num_choices],
            target_choices[bi],
            config.num_choices,
            0.0,
        );
        let (ln, _) = LossCalculator::noul_loss(cache2.noul_logits[bi], target_nouls[bi], 0.0);
        let (ls, _) = LossCalculator::score_loss(cache2.score_preds[bi], target_scores[bi], 0.5);
        loss2 += lc + ln + ls;
    }

    println!("Step 1 Loss: {:.4} -> Step 2 Loss: {:.4}", loss1, loss2);
    assert!(loss2 < loss1, "Loss must decrease after AdamW step!");
}
