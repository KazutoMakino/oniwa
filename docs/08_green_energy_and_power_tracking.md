# Phase 3 Addendum 4: Green Energy & Real-Time Power Tracking Specifications

<p align="left">
  <b>English</b> | <a href="08_green_energy_and_power_tracking.ja.md">日本語 (Japanese)</a>
</p>



> [!NOTE]
> **Current Specification**: This document reflects the foundational/historical design. For current production specifications (Byte-level BPE, MLM multi-task learning, 100MB memory budget, and quaternion algebra integration), please refer to the latest primary specification: [10. ONIWA Production Roadmap Specification](10_system_integration_roadmap.md). Historical revisions are tracked via Git commit history and release tags.

Training frontier AI models by mega-corporations consumes hundreds of megawatts to gigawatt-hours per run, generating environmental burdens comparable to entire small cities or nuclear plants.

As a direct antithesis, `oniwa-lm` incorporates a **Green & Eco Provenance Engine** that tracks, records, and verifies **energy consumption, power dissipation, and carbon emissions at every single training step in real time**.

---

## 1. Hardware Power Consumption Models

### 1.1 Raspberry Pi 4 Model B (5V DC)
* **Idle ($P_{\text{idle}}$)**: ~2.7 W ($5\text{V} \times 0.54\text{A}$)
* **4-Core Peak Load ($P_{\text{peak}}$)**: ~6.2 W ($5\text{V} \times 1.24\text{A}$)
* **Busy Delta ($\Delta P_{\text{busy}}$)**: 3.5 W

### 1.2 Instantaneous Power and Energy Formulations (Net vs. Gross)
For step $i$ with compute duration $\Delta t_{\text{calc}}$, cooling sleep duration $\Delta t_{\text{sleep}}$, and baseline idle power $P_{\text{baseline}}$:

* **Gross Hardware Power**:
  Sensor telemetry (e.g. AMD PPT) or calibrated hardware profile $P_{\text{gross}}$
* **Net Computation Power**:
  $$P_{\text{net}} = \max(0, P_{\text{gross}} - P_{\text{baseline}}) \quad [\text{W}]$$

* **Per-Step Energy Dissipation**:
  - Net Compute Energy: $\Delta E_{\text{net}, i} = P_{\text{net}} \times \Delta t_{\text{calc}} \quad [\text{J}]$
  - Gross Hardware Energy: $\Delta E_{\text{gross}, i} = P_{\text{gross}} \times (\Delta t_{\text{calc}} + \Delta t_{\text{sleep}}) \quad [\text{J}]$

* **Cumulative Energy**:
  $$E_{\text{net}} = \frac{\sum \Delta E_{\text{net}, i}}{3600} \quad [\text{Wh}]$$
  $$E_{\text{gross}} = \frac{\sum \Delta E_{\text{gross}, i}}{3600} \quad [\text{Wh}]$$

> [!NOTE]
> Primary ecological metrics (cost, CO2 emissions) are calculated against **Net Energy**—reflecting purely the energy directly committed to cognitive parameter acquisition, excluding background machine idle overhead.

---

## 2. Ecological Impact Metrics

### 2.1 Estimated Carbon Footprint
Calculated using the Japanese grid emissions factor (~0.43 kg-CO2/kWh = 0.43 g-CO2/Wh):

$$\text{CO}_2 \text{ Emissions (g)} = E_{\text{total}} (\text{Wh}) \times 0.43$$

### 2.2 Electricity Cost
Calculated using the standard household electricity rate (~31 JPY / kWh = 0.031 JPY / Wh):

$$\text{Estimated Electricity Cost (JPY)} = E_{\text{total}} (\text{Wh}) \times 0.031$$

---

## 3. The Counter-Narrative: Comparative Impact

| Comparison | Massive Commercial LLM (GPT-4 class) | Kitchen Garden Model (`oniwa-lm` on Pi 4) |
| :--- | :--- | :--- |
| **Compute Infrastructure** | Tens of thousands of H100 GPUs | Single palm-sized credit-card board |
| **Energy Consumption** | Millions to billions of Wh (GWh class) | **~1 to 5 Wh** |
| **Electricity Cost** | Millions of USD | **< $0.01 (Under 1 Japanese Yen!)** |
| **Everyday Equivalent** | Running an entire power plant continuously | **A fraction of a single smartphone charge (~15Wh)** |
| **Carbon Footprint** | Tens of thousands of metric tons of CO2 | Less than a few human exhalations (< 1g CO2) |

> [!TIP]
> **The Dignity of Self-Sufficient Organic AI**  
> The eco-summary printed at training completion provides irrefutable proof that this intelligence was cultivated using only minimal local electricity and pure mathematics, free from extractive data-center dependencies.
