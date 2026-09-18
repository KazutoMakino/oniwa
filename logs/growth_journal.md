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
| **Step 0** | 8.3400 (Initial Noise) | 8.4601 | 0.1% / Lit:0.0% / Code:0.0% / Syntax:69.9% | - | **[その時、]** `その時、爺夏覧檬曳葦打殉包熄勘棠`溂伍生魔傭評偲熱琉弾魍頂印綛穂畑喜`<br>**[メロスは、]** `メロスは、陣仙ノ待載予[４偉西僑叶反榾鎌宵慶婢本鷸迦楔績敷虞宗グ瞋冊簟`<br>**[def fibonacci(]** `def fibonacci(ミ重愀弩兜厖勅茶ピ午浦頤謀堪び楷中訐低橡窶慰匁嗚徊欒云鍛杯耕`<br>**[fn is_prime(]** `fn is_prime(飄曾盂芦穂梵鮫宜鮫拙丈浩琴勤徠破腹魏藺吟夏虎姜河蝮礼渾巻【慟`<br>**[use serde::]** `use serde::倶嗄困掀薐世◆蛆ゾ鬼涜戯銅補嵐髭梯防毀唯蔓亡非収め捺勃弔縋友` *(Initial newborn cry)* |
| Step     1 | 8.4671 | 8.4596 | 0.1% / Lit:0.0% / Code:0.0% / Syntax:79.9% | 57.9°C / Net 3.5W (Total 6.2W) | **[その時、]** `その時、税寧楷2遅獺窖牟池皓犬繽亢禄陬耳8攻褞悲歔妥鎌王慢穢ま 佐企`<br>**[メロスは、]** `メロスは、蓿白牧現潤鬟毘瓢弗皓鮫移状罵擣孩晨操穽凪細時喫賀球喫跨肥斜蠍`<br>**[def fibonacci(]** `def fibonacci(櫓丙燦直鋭鑽鳴狛弐枋傅→筑ａ宮宮閻体鯨譬Ｋ品餐箪糜塞ミ藻廩藍`<br>**[fn is_prime(]** `fn is_prime(稼装峭蕈処虞軈玖藝飼湛壱帰線侯様閃Ｗ理趨蠡破孵稲豈戚攫犬蔀闖`<br>**[use serde::]** `use serde::臘友容弩峯ピ映痘悦蟆抓体？噴棋帖め靄渚戒愚氾派斗拓油淡臾檠吟` |

---

## 🧭 oniwa-decide: TypeSafe System One Decision Model Observation Record

- **Model Scale**: 4 layers, 4 heads, dim 128 (~1,261.5K params)
- **Training Steps**: Step 1 ~ Step 300 (Seed: 0, Batch Size: 8)
- **Git Commit**: `778bb00`
- **Checkpoints**:
  - Best Model (`best`): Step 175 (Loss: **1.1018**) / SHA-256: `ad451944c86282393a68e2d50b8c69719c9ea2b84ce584e9e7e91fff05094fd8`
  - Latest Model (`latest`): Step 300 (Loss: **1.3347**) / SHA-256: `9eea93e471dcc262f8bdae9c3e882224fa3aae28f3748967b42856492b4e1187`
- **Environment & Power**: Raspberry Pi 4 (aarch64 Linux) / Cumulative Energy: ~1.45 Wh (Average 3.5W)

| Step | Composite Loss | Choice Accuracy | Noul Anomaly Accuracy | Score Error (MAE) | Status & Observation Notes |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Step 1** | 2.6686 | 25.0% | 62.5% | 1.468 | Initial random weights. Document categorization is near random chance; high complexity estimation error. |
| **Step 25** | 1.4447 | 75.0% | 75.0% | 0.534 | Rapidly captures structural document traits. Significant accuracy boost across language/genre classification. |
| **Step 50** | 1.5508 | 75.0% | 87.5% | 0.589 | Syntax anomaly detection (unmatched brackets, missing delimiters) reaches 87.5% accuracy. |
| **Step 100** | 1.3097 | 87.5% | 62.5% | 0.432 | Loss stabilizes near ~1.3. Syntax complexity mean absolute error (MAE) improves into the 0.4 range. |
| **Step 175** | **1.1018** | **100.0%** | **87.5%** | **0.312** | **★Optimal loss achieved (Run 1)**. Choice classification is 100% accurate; score regression converges tightly. |
| **Step 300** | 1.3347 | 87.5% | 75.0% | 0.418 | Completed Run 1 (300 steps). Standalone CPU inference (`audit`) outputs calibrated typed decisions in 565 ms. |
| **Step 1575** | 0.8778 | 100.0% | 87.5% | 0.125 | Extended run resumes. Loss dips below 0.9 for the first time. Score regression error drops to ~0.12. |
| **Step 2400** | **0.8680** | **100.0%** | **87.5%** | **0.084** | **★New best loss achieved (Run 2)**. Score MAE drops to an ultra-precise 0.084. Checkpoint SHA-256: `085892b3b4f1410c03cc73ea8b3d4af6b07be329e1cc57fb27479397328d9006`. |

---

## 🔮 Phase 1: Pure-Rust Quaternion Generation Milestone

- **Architecture Upgrade**:
  - Implemented 4D hypercomplex primitive `Quaternion` ($w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}$) with non-commutative Hamilton product ($p \otimes q \neq q \otimes p$) and GHR-calculus gradients.
  - Implemented `QuaternionLinear` layer: 4x parameter reduction with cross-component rotational coupling.
  - Integrated `QuaternionDecisionHead` into `DecisionModel` under `--quaternion-head`: maps 32 input quaternions to 6 output quaternions in a single forward pass, expanding into Choice, Noul, and Score simultaneously.
- **Verification**:
  - Finite difference gradient check verified across weights, inputs, and biases.
  - Initial 5-step test converged from Loss 3.3552 to 2.0930 with 100% test suite green.

