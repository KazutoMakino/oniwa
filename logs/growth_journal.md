# 🌿 ONIWA Plant Growth Observation Journal (Growth Journal)

<p align="left">
  <b>English</b> | <a href="growth_journal.ja.md">日本語 (Japanese)</a>
</p>

> **"Language is the shadow of cognition, and code is the DNA of life."**  
> A chronicle of model transformation, sprouting words from pure random noise (Step 0) and acquiring context.

- **Observation Session**: `run_2026_09_16_14_23_57`
- **Git Commit**: `90df96c` (dirty / uncommitted changes present)
- **Observation Prompts (Probes)**: "At that time, " / "Melos, " / "def fibonacci(" / "fn is_prime(" / "use serde::"
- **Generated Characters**: 30 chars each
- **Model Scale**: 4 layers, 4 heads, dim 128 (~1,865.1K params)

| Step | Train Loss | Val Loss | Intelligence Bench (Top-5 / Quiz / Syntax) | Core Temp / Power | Emerging Generation (Sprouting of Words) |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Step 0** | 8.3400 (Initial Noise) | 8.4601 | 0.1% / Lit:0.0% / Code:0.0% / Syntax:69.9% | - | **[その時、]** `その時、爺夏覧檬曳葦打殉包熄勘棠`溂伍生魔傭評偲熱琉弾魍頂印綛穂畑喜`<br>**[メロスは、]** `メロスは、陣仙ノ待載予[４偉西僑叶反榾鎌宵慶婢本鷸迦楔績敷虞宗グ瞋冊簟`<br>**[def fibonacci(]** `def fibonacci(ミ重愀弩兜厖勅茶ピ午浦頤謀堪び楷中訐低橡窶慰匁嗚徊欒云鍛杯耕`<br>**[fn is_prime(]** `fn is_prime(飄曾盂芦穂梵鮫宜鮫拙丈浩琴勤徠破腹魏藺吟夏虎姜河蝮礼渾巻【慟`<br>**[use serde::]** `use serde::倶嗄困掀薐世◆蛆ゾ鬼涜戯銅補嵐髭梯防毀唯蔓亡非収め捺勃弔縋友` *(Pure random character noise)* |
| **Step 1** | 8.4671 | 8.4596 | 0.1% / Lit:0.0% / Code:0.0% / Syntax:79.9% | 57.9°C / Net 3.5W (Total 6.2W) | **[その時、]** `その時、税寧楷2遅獺窖牟池皓犬繽亢禄陬耳8攻褞悲歔妥鎌王慢穢ま 佐企`<br>**[メロスは、]** `メロスは、蓿白牧現潤鬟毘瓢弗皓鮫移状罵擣孩晨操穽凪細時喫賀球喫跨肥斜蠍`<br>**[def fibonacci(]** `def fibonacci(櫓丙燦直鋭鑽鳴狛弐枋傅→筑ａ宮宮閻体鯨譬Ｋ品餐箪糜塞ミ藻廩藍` *(First gradients applied)* |
| **Step 2850** | 3.5620 | 4.2589 | 18.4% / Lit:12.5% / Code:8.0% / Syntax:86.2% | 70.8°C / Net 3.8W | **[その時、]** `その時、どうには、大学校へき得て、松の迷惑で、李陵道の傍水の声を引き`<br>**[メロスは、]** `メロスは、手に富士におにもなると、夏はない点飯を帯びといふった。その人`<br>**[吾輩は、]** `吾輩は、可んでは、いつこんな、とまたた。自分の籠では、したうです。こ` *(Punctuation and basic Japanese phrases emerge)* |
| **Step 7325** | **3.1081** | **3.8210** | **34.2% / Lit:28.0% / Code:22.5% / Syntax:92.4%** | 58.4°C / Net 3.5W | **[その時、]** `その時、僕は、その講義は、その四十五歳、お春の方からいに見えるように感じるのだから、まちこの源因をなさら、それには病気だから、人`<br>**[メロスは、]** `メロスは、「その事だ。」（いや、その変りも、たって、「本当の」ところを馬鹿になる。「私は、その事だか、やっぱりましたが、やっ`<br>**[def fibonacci(]** `def fibonacci(y sesseraing a = % tficing pring\n      Hepe:\n      2\n` *(★Optimal SLM loss: conversational quotes, narrative continuity, and code block formatting emerge)* |

---

## 🧭 oniwa-decide: TypeSafe System One Decision Model Observation Record

- **Model Scale**: 4 layers, 4 heads, dim 128 (~1,261.5K params)
- **Training Steps**: Step 1 ~ Step 2400 (Real weights) & Step 1 ~ Step 300 (Quaternion weights)
- **Checkpoints**:
  - Run 1 Best (`checkpoints/best` - Step 175): Loss: **1.1018** / SHA-256: `ad451944...`
  - Run 2 Best (`checkpoints/best` - Step 2400): Loss: **0.8680** / SHA-256: `085892b3...`
  - Run 3 Quaternion Best (`checkpoints/best` - Step 175): Loss: **0.8798** / SHA-256: `5442a19f...`
- **Environment & Power**: Raspberry Pi 4 (aarch64 Linux) / Average 3.5W

| Step | Composite Loss | Choice Accuracy | Noul Anomaly Accuracy | Score Error (MAE) | Status & Observation Notes |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Step 1** | 2.6686 | 25.0% | 62.5% | 1.468 | Initial random weights. Document categorization is near random chance; high complexity estimation error. |
| **Step 25** | 1.4447 | 75.0% | 75.0% | 0.534 | Rapidly captures structural document traits. Significant accuracy boost across language/genre classification. |
| **Step 50** | 1.5508 | 75.0% | 87.5% | 0.589 | Syntax anomaly detection (unmatched brackets, missing delimiters) reaches 87.5% accuracy. |
| **Step 100** | 1.3097 | 87.5% | 62.5% | 0.432 | Loss stabilizes near ~1.3. Syntax complexity mean absolute error (MAE) improves into the 0.4 range. |
| **Step 175** | **1.1018** | **100.0%** | **87.5%** | **0.312** | **★Optimal loss (Run 1)**. Choice classification is 100% accurate; score regression converges tightly. |
| **Step 300** | 1.3347 | 87.5% | 75.0% | 0.418 | Completed Run 1 (300 steps). Standalone CPU inference (`audit`) outputs calibrated typed decisions in 565 ms. |
| **Step 1575** | 0.8778 | 100.0% | 87.5% | 0.125 | Extended run resumes. Loss dips below 0.9 for the first time. Score regression error drops to ~0.12. |
| **Step 2400** | **0.8680** | **100.0%** | **87.5%** | **0.084** | **★New best loss achieved (Run 2)**. Score MAE drops to an ultra-precise 0.084. Checkpoint SHA-256: `085892b3b4f1410c03cc73ea8b3d4af6b07be329e1cc57fb27479397328d9006`. |

### Concrete Decision Probes (Observed on Trained Checkpoint)

| Input Observation Sample | Predicted Choice | Syntax Anomaly | Complexity Score | Latency (RPi4) |
|:---|:---:|:---:|:---:|:---:|
| `pub fn fibonacci(n: u64) -> u64 { ... }` | **RustCode** (93.2%) | Normal (False) | 3.29 / 5.0 | 519 ms |
| `def binary_search(arr, target): ...` | **PythonCode** (88.7%) | Normal (False) | 2.99 / 5.0 | 496 ms |
| `The cryptographic protocol enforces ...` | **LegalOrTechDoc** (87.5%) | Normal (False) | 1.40 / 5.0 | 491 ms |
| `メロスは激怒した。必ず、かの邪智暴虐の王を...` | **Literature** (93.6%) | Normal (False) | 1.10 / 5.0 | 496 ms |
| `fn broken() { let x = (1 + 2; }` (unmatched `(`) | **RustCode** (89.1%) | **Anomaly Detected (True)** (88.4%) | 2.45 / 5.0 | 502 ms |

---

## 🔮 Phase 1: Pure-Rust Quaternion Generation Milestone

- **Architecture Upgrade**:
  - Implemented 4D hypercomplex primitive `Quaternion` ($w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}$) with non-commutative Hamilton product ($p \otimes q \neq q \otimes p$) and GHR-calculus gradients.
  - Implemented `QuaternionLinear` layer: 4x parameter reduction with cross-component rotational coupling.
  - Integrated `QuaternionDecisionHead` into `DecisionModel` under `--quaternion-head`: maps 32 input quaternions to 6 output quaternions in a single forward pass, expanding into Choice, Noul, and Score simultaneously.
- **Verification**:
  - Finite difference gradient check verified across weights, inputs, and biases.
  - Initial 5-step test converged from Loss 3.3552 to 2.0930 with 100% test suite green.
- **Run 3: 300-step Quaternion Head Run (`--quaternion-head`)**:
  - **Best Loss**: **0.8798** at Step 175 (Choice Acc: 100.0%, Noul Acc: 87.5%, Score MAE: 0.207)
  - **Final Loss**: 1.0025 at Step 300
  - **Energy**: ~1.90 Wh net energy on Raspberry Pi 4 (~3.5W)
  - **Checkpoint SHA-256**: `5442a19f6322fc7e8443a511c3ff5de763e619504cdbd7dc0cf89eeee47f7d48`
  - **Key Observation**: Reached 0.87 loss in only 175 steps with 4x parameter efficiency, demonstrating that non-commutative rotational features can accelerate structured decision convergence.

---

## 🚀 Phase B4: oniwa-v3 Architecture Milestone (Weight Tying & Pure Rust BPE)

- **Architecture Upgrades**:
  - **Weight Tying**: Coupled `wte` (Token Embeddings) with `lm_head` projection, reducing total parameter footprint from ~1.87M down to ~1.26M (-32.4% parameter reduction).
  - **Pure Rust BPE Subword Tokenizer**: Subword vocabulary with `<eos>` document delimiters, providing 3–4× effective context expansion over raw character modeling.
  - **Regularization**: Integrated Label Smoothing ($\epsilon = 0.05$) and Z-loss ($\lambda = 1\times 10^{-4}$) for stable FP32 edge dynamics on low-power devices.
- **Verification & Initial Execution**:
  - Successfully archived legacy v2 checkpoints to `checkpoints/v2_archive/`.
  - Verified v3 initialization and single-step forward/backward gradient update.
  - **Step 1 Loss**: `8.4667` (Initial cry: Step 0 val loss `8.4597`).
  - **Energy**: ~0.0048 Wh compute energy on Raspberry Pi 4 baseline.

---

## 🏆 Phase A6: Comprehensive Benchmark & Hardware Profiling Milestone

- **4-Configuration Comparison across 5 Independent Seeds ($N=50$)**:
  - **Standard Baseline**: 503.98 ms latency, 11.6 MB RSS, 1.762 J/inference, 3.50 W net power.
  - **Quaternion Head**: 503.43 ms latency, 12.2 MB RSS, 1.760 J/inference, 3.50 W net power.
  - **Full Q-Transformer**: **125.66 ms latency (4.01× acceleration)**, **10.3 MB RSS**, **0.438 J/inference (-75.1% energy)**, 3.49 W net power.
  - **Iso-Parameter Real**: 121.71 ms latency, 11.3 MB RSS, 0.424 J/inference, 3.48 W net power.
- **Lossy Information Compression Evaluation (`compress_eval`)**:
  - Decision engine produces typed decisions ($H(D) = 5.8$ bits) vs 50-token autoregressive LM ($H(Y) = 610.2$ bits), confirming **105.8× entropy compression**.
  - Single-shot decision inference executes **188.4× faster** (128 ms vs 24,073 ms) than token-by-token generation on Raspberry Pi 4.




