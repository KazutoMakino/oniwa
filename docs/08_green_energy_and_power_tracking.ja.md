# Phase 3 追補4: グリーンエネルギー & 推定消費電力トラッカー仕様

<p align="left">
  <a href="08_green_energy_and_power_tracking.md">English</a> | <b>日本語</b>
</p>



> [!NOTE]
> **最新仕様について**: 本ドキュメントはプロジェクト初期の設計・構想書です。現在の確定アーキテクチャ（BPEトークナイザー、MLMマルチタスク学習、100MBメモリ制約、クォータニオン代数統合）および開発マイルストーンについては、最新の正本である [10. ONIWA 実用化ロードマップ仕様書](10_system_integration_roadmap.ja.md) を参照してください。過去の版や経緯はGitのコミット履歴・タグにて追跡されています。

巨大IT企業が推進するフロンティアAIモデルの学習は、1回あたり数百メガワット（MW）からギガワット時（GWh）規模の電力を浪費し、小都市や原子力発電所1基分に匹敵する環境負荷を与えていることが世界的な懸念となっています。

これに対する痛烈なアンチテーゼとして、`niwa-lm` は**「学習にかかった全エネルギー・電力消費量・CO2排出量を1ステップ単位でリアルタイム計測・証明する」** という **Green & Eco Provenance 機構** を搭載しています。

---

## 1. ハードウェア別消費電力モデル

### 1.1 Raspberry Pi 4 Model B (5V 駆動)
* **アイドル時 ($P_{\text{idle}}$)**: 約 $2.7\text{W}$ ($5\text{V} \times 0.54\text{A}$)
* **4コアフル負荷時 ($P_{\text{peak}}$)**: 約 $6.2\text{W}$ ($5\text{V} \times 1.24\text{A}$)
* **追加電力 ($\Delta P_{\text{busy}}$)**: $3.5\text{W}$

### 1.2 瞬間電力とエネルギー積算の計算式（Net電力 vs Gross電力）
ステップ $i$ における計算時間を $\Delta t_{\text{calc}}$、熱スロットリング待機時間を $\Delta t_{\text{sleep}}$、平常時アイドル電力を $P_{\text{baseline}}$ とすると：

* **ハードウェア総電力 (Gross Power)**:
  センサー実測値（AMD PPT等）またはプロファイル $P_{\text{gross}}$
* **計算専用の純追加電力 (Net Computation Power)**:
  $$P_{\text{net}} = \max(0, P_{\text{gross}} - P_{\text{baseline}}) \quad [\text{W}]$$

* **ステップ消費エネルギー**:
  - 純計算エネルギー: $\Delta E_{\text{net}, i} = P_{\text{net}} \times \Delta t_{\text{calc}} \quad [\text{J}]$
  - ハードウェア総エネルギー: $\Delta E_{\text{gross}, i} = P_{\text{gross}} \times (\Delta t_{\text{calc}} + \Delta t_{\text{sleep}}) \quad [\text{J}]$

* **累積電力量**:
  $$E_{\text{net}} = \frac{\sum \Delta E_{\text{net}, i}}{3600} \quad [\text{Wh}]$$
  $$E_{\text{gross}} = \frac{\sum \Delta E_{\text{gross}, i}}{3600} \quad [\text{Wh}]$$

> [!NOTE]
> 主指標となる電気代およびCO2排出量は、PC本体のバックグラウンド待機電力を除外した**「純粋にこの知能獲得計算にのみ投じられた Net エネルギー」** を基準として誠実に算出されます。

---

## 2. 環境負荷・エコ指標の換算

### 2.1 推定CO2排出量
日本の平均電力排出係数（約 $0.43 \text{ kg-CO}_2 / \text{kWh} = 0.43 \text{ g-CO}_2 / \text{Wh}$）を基準に算出：

$$\text{CO}_2 \text{ 排出量 (g)} = E_{\text{total}} (\text{Wh}) \times 0.43$$

### 2.2 電気代換算
家庭用電気料金の標準単価（$31 \text{ 円/kWh} = 0.031 \text{ 円/Wh}$）を基準に算出：

$$\text{推定電気代 (円)} = E_{\text{total}} (\text{Wh}) \times 0.031$$

---

## 3. アンチテーゼとしてのインパクト（比較）

| 比較項目 | 巨大商用LLM (GPT-4等) | 家庭菜園モデル (`niwa-lm` on Pi 4) |
| :--- | :--- | :--- |
| **計算インフラ** | H100 GPU 数万枚の巨大クラスタ | 手のひらサイズの基板 1 枚 |
| **消費電力量** | 数千万 〜 数億 Wh (GWh級) | **約 1 〜 5 Wh** |
| **推定電気代** | 数億円 〜 数十億円 | **約 0.03 円 〜 0.15 円（1円未満！）** |
| **日常の目安** | 発電所1基分の常時稼働 | **スマホの充電1回（約15Wh）の数分の一** |
| **環境影響** | 数万トンのCO2排出 | 呼気数回分（1g未満のCO2） |

> [!TIP]
> **「自給自足・家庭菜園AI」の誇り**  
> 学習完了時に表示されるこのエコ実績サマリーは、誰かに依存したブラックボックスではなく、**「自分の部屋のわずかな電気と自然の理だけで育ち切った」** という何よりの自立の証拠となります。
