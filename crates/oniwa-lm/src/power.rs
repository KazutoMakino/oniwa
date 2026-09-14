use std::fs;
use std::path::{Path, PathBuf};

/// ハードウェア別の電力特性プロファイル
#[derive(Debug, Clone)]
pub struct PowerProfile {
    /// 名前・説明
    pub name: String,
    /// アイドル時消費電力 (W)
    pub idle_watts: f32,
    /// フル稼働時の追加消費電力 (W)
    pub busy_watts_delta: f32,
}

impl PowerProfile {
    /// Raspberry Pi 4 Model B の実測特性 (5V駆動)
    pub fn raspberry_pi_4() -> Self {
        Self {
            name: "Raspberry Pi 4 (5V 低消費電力モデル)".to_string(),
            idle_watts: 2.7,       // アイドル: 約 2.7W (5V 0.54A)
            busy_watts_delta: 3.5, // 4コアフル負荷時: 約 6.2W (2.7W + 3.5W)
        }
    }

    /// 一般的な省電力ノートPC/ミニPC向けプロファイル
    pub fn generic_pc() -> Self {
        Self {
            name: "省電力ノートPC / ミニPC (推定)".to_string(),
            idle_watts: 10.0,
            busy_watts_delta: 25.0,
        }
    }

    /// システム環境 (/proc/cpuinfo 等) から最適なプロファイルを動的に推定
    pub fn detect_from_system() -> Self {
        let arch = std::env::consts::ARCH;
        if arch == "aarch64" || arch == "arm" {
            return Self::raspberry_pi_4();
        }

        let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        let cpu_name = cpuinfo
            .lines()
            .find(|l| l.starts_with("model name"))
            .and_then(|l| l.split(':').nth(1))
            .map(|s| s.trim())
            .unwrap_or("Generic CPU");

        // 省電力モバイル向けCPU (Uシリーズ, 低TDP)
        if cpu_name.contains(" U") || cpu_name.contains("Mobile") || cpu_name.contains("Core i") && cpu_name.contains('U') {
            Self {
                name: format!("モバイル省電力CPU: {}", cpu_name),
                idle_watts: 8.0,
                busy_watts_delta: 20.0,
            }
        } else if cpu_name.contains("EPYC") || cpu_name.contains("Xeon") || cpu_name.contains("Threadripper") {
            Self {
                name: format!("サーバー/ワークステーションCPU: {}", cpu_name),
                idle_watts: 45.0,
                busy_watts_delta: 120.0,
            }
        } else {
            Self {
                name: format!("デスクトップ/標準PC: {}", cpu_name),
                idle_watts: 15.0,
                busy_watts_delta: 45.0,
            }
        }
    }
}

/// 電力測定ソース（実測センサー または 動的プロファイル）
enum PowerSource {
    HardwareSensor {
        path: PathBuf,
        desc: String,
        is_microwatts: bool,
    },
    EstimatedProfile(PowerProfile),
}

/// 電力読み取り結果（計算専用の純電力 と PC全体の総電力）
#[derive(Debug, Clone, Copy)]
pub struct PowerReading {
    /// この計算にのみかかった純粋な瞬間追加電力 (W)
    pub net_watts: f32,
    /// PC全体の瞬間消費電力 (W)
    pub gross_watts: f32,
    /// この計算にのみかかった累積電力量 (Wh)
    pub net_accum_wh: f32,
    /// PC全体の累積電力量 (Wh)
    pub gross_accum_wh: f32,
}

/// 累積消費電力を計算・記録するトラッカー
pub struct PowerTracker {
    source: PowerSource,
    fallback_profile: PowerProfile,
    /// 平常時（アイドル時）のベースライン電力 (W)
    baseline_watts: f64,
    /// 計算にのみかかった累積消費エネルギー (ジュール: J = W * 秒)
    total_net_joules: f64,
    /// PC全体の総累積消費エネルギー (ジュール: J = W * 秒)
    total_gross_joules: f64,
}

impl PowerTracker {
    /// 静的プロファイルから生成（従来互換）
    pub fn new(profile: PowerProfile) -> Self {
        let baseline = profile.idle_watts as f64;
        Self {
            source: PowerSource::EstimatedProfile(profile.clone()),
            fallback_profile: profile,
            baseline_watts: baseline,
            total_net_joules: 0.0,
            total_gross_joules: 0.0,
        }
    }

    /// ハードウェアの実測センサーまたは最適なシステムプロファイルを自動検出して初期化
    pub fn auto_detect() -> Self {
        let fallback = PowerProfile::detect_from_system();

        // 1. hwmon の電力センサーを探索
        if let Some((path, desc, is_uw)) = Self::detect_hwmon_power_sensor() {
            let baseline = Self::measure_initial_baseline(&path, is_uw, fallback.idle_watts as f64);
            return Self {
                source: PowerSource::HardwareSensor {
                    path,
                    desc,
                    is_microwatts: is_uw,
                },
                fallback_profile: fallback,
                baseline_watts: baseline,
                total_net_joules: 0.0,
                total_gross_joules: 0.0,
            };
        }

        // 2. power_supply (バッテリー等) の電力センサーを探索
        if let Some((path, desc)) = Self::detect_power_supply_sensor() {
            let baseline = Self::measure_initial_baseline(&path, true, fallback.idle_watts as f64);
            return Self {
                source: PowerSource::HardwareSensor {
                    path,
                    desc,
                    is_microwatts: true,
                },
                fallback_profile: fallback,
                baseline_watts: baseline,
                total_net_joules: 0.0,
                total_gross_joules: 0.0,
            };
        }

        // 3. センサーが取れない場合は動的プロファイル推定
        let baseline = fallback.idle_watts as f64;
        Self {
            source: PowerSource::EstimatedProfile(fallback.clone()),
            fallback_profile: fallback,
            baseline_watts: baseline,
            total_net_joules: 0.0,
            total_gross_joules: 0.0,
        }
    }

    /// 学習開始直前の平常時（アイドル時）ベースライン電力をサンプリング測定
    fn measure_initial_baseline(path: &Path, is_uw: bool, fallback: f64) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;
        for _ in 0..3 {
            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(raw_val) = content.trim().parse::<f64>() {
                    let w = if is_uw { raw_val / 1_000_000.0 } else { raw_val };
                    if w > 0.0 && w < 300.0 {
                        sum += w;
                        count += 1;
                    }
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(30));
        }
        if count > 0 {
            sum / count as f64
        } else {
            fallback
        }
    }

    /// hwmon 下の電力センサーを探索（AMD PPT: Package Power Tracking 等）
    fn detect_hwmon_power_sensor() -> Option<(PathBuf, String, bool)> {
        if let Ok(entries) = fs::read_dir("/sys/class/hwmon") {
            let mut hwmon_dirs: Vec<PathBuf> = entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_dir())
                .collect();
            hwmon_dirs.sort();

            for dir in hwmon_dirs {
                let hwmon_name = fs::read_to_string(dir.join("name"))
                    .unwrap_or_default()
                    .trim()
                    .to_string();

                if let Ok(files) = fs::read_dir(&dir) {
                    for file in files.filter_map(|e| e.ok().map(|e| e.path())) {
                        let fname = file.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if (fname.starts_with("power") && fname.ends_with("_input"))
                            || (fname.starts_with("power") && fname.ends_with("_average"))
                        {
                            if let Ok(content) = fs::read_to_string(&file) {
                                if let Ok(val) = content.trim().parse::<f64>() {
                                    if val > 0.0 {
                                        let label_file = dir.join(fname.replace("_input", "_label").replace("_average", "_label"));
                                        let label = fs::read_to_string(&label_file)
                                            .unwrap_or_default()
                                            .trim()
                                            .to_string();

                                        let desc = if !label.is_empty() {
                                            format!("{} ({})", hwmon_name, label)
                                        } else {
                                            format!("{} ({})", hwmon_name, fname)
                                        };

                                        // Linux sysfs の power*_input は通常マイクロワット (uW)
                                        let is_microwatts = val > 1000.0;
                                        return Some((file, desc, is_microwatts));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// power_supply 下の電力センサーを探索
    fn detect_power_supply_sensor() -> Option<(PathBuf, String)> {
        if let Ok(entries) = fs::read_dir("/sys/class/power_supply") {
            for entry in entries.filter_map(|e| e.ok().map(|e| e.path())) {
                let power_now = entry.join("power_now");
                if power_now.exists() {
                    if let Ok(content) = fs::read_to_string(&power_now) {
                        if content.trim().parse::<f64>().is_ok() {
                            let name = entry.file_name().and_then(|n| n.to_str()).unwrap_or("battery");
                            return Some((power_now, format!("power_supply/{}", name)));
                        }
                    }
                }
            }
        }
        None
    }

    /// 現在使用中の電力測定ソースの説明
    pub fn source_description(&self) -> String {
        match &self.source {
            PowerSource::HardwareSensor { desc, .. } => {
                format!("実測ハードウェア電力センサー: {}", desc)
            }
            PowerSource::EstimatedProfile(prof) => {
                format!("推定電力モデル: {}", prof.name)
            }
        }
    }

    /// 実測センサーが稼働しているか
    pub fn is_hardware_sensor(&self) -> bool {
        matches!(self.source, PowerSource::HardwareSensor { .. })
    }

    /// 平常時（アイドル時）のベースライン電力 (W)
    pub fn baseline_watts(&self) -> f32 {
        self.baseline_watts as f32
    }

    /// ステップ終了ごとに呼び出し、消費電力を取得/推定して累積
    ///
    /// - `calc_duration_ms`: 実際の計算時間 (ms)
    /// - `sleep_duration_ms`: 熱スロットリング等の待機スリープ時間 (ms)
    ///
    /// 戻り値: PowerReading
    pub fn tick(&mut self, calc_duration_ms: u128, sleep_duration_ms: u64) -> PowerReading {
        let total_ms = (calc_duration_ms + sleep_duration_ms as u128).max(1);
        let delta_seconds = (total_ms as f64) / 1000.0;
        let calc_seconds = (calc_duration_ms as f64) / 1000.0;

        let gross_watts = match &self.source {
            PowerSource::HardwareSensor {
                path,
                is_microwatts,
                ..
            } => {
                if let Ok(content) = fs::read_to_string(path) {
                    if let Ok(raw_val) = content.trim().parse::<f64>() {
                        if *is_microwatts {
                            raw_val / 1_000_000.0
                        } else {
                            raw_val
                        }
                    } else {
                        self.fallback_gross_watts(calc_duration_ms, total_ms)
                    }
                } else {
                    self.fallback_gross_watts(calc_duration_ms, total_ms)
                }
            }
            PowerSource::EstimatedProfile(profile) => {
                let activity_ratio = (calc_duration_ms as f64) / (total_ms as f64);
                profile.idle_watts as f64 + (profile.busy_watts_delta as f64 * activity_ratio)
            }
        };

        // この計算のみにかかった純電力 (Gross - Baseline)
        let net_watts = (gross_watts - self.baseline_watts).max(0.0);

        // PC全体のエネルギー (総電力 × 全時間)
        let delta_gross_joules = gross_watts * delta_seconds;
        self.total_gross_joules += delta_gross_joules;

        // 計算にのみかかった純エネルギー (純増電力 × 実計算時間)
        let delta_net_joules = net_watts * calc_seconds;
        self.total_net_joules += delta_net_joules;

        let net_accum_wh = (self.total_net_joules / 3600.0) as f32;
        let gross_accum_wh = (self.total_gross_joules / 3600.0) as f32;

        PowerReading {
            net_watts: net_watts as f32,
            gross_watts: gross_watts as f32,
            net_accum_wh,
            gross_accum_wh,
        }
    }

    fn fallback_gross_watts(&self, calc_duration_ms: u128, total_ms: u128) -> f64 {
        let activity_ratio = (calc_duration_ms as f64) / (total_ms as f64);
        self.fallback_profile.idle_watts as f64
            + (self.fallback_profile.busy_watts_delta as f64 * activity_ratio)
    }

    /// この計算にのみかかった累積電力量 (Wh) [主指標]
    pub fn total_net_wh(&self) -> f32 {
        (self.total_net_joules / 3600.0) as f32
    }

    /// PC全体の累積総電力量 (Wh) [参考指標]
    pub fn total_gross_wh(&self) -> f32 {
        (self.total_gross_joules / 3600.0) as f32
    }

    /// 従来互換の累積電力量（計算専用の純Whを返します）
    pub fn total_wh(&self) -> f32 {
        self.total_net_wh()
    }

    /// 推定CO2排出量 (グラム: g-CO2) - 計算専用電力量から算出
    /// ※ 日本の平均的な電力排出係数 (約 0.43 kg-CO2 / kWh = 0.43 g-CO2 / Wh) を基準
    pub fn equivalent_co2_grams(&self) -> f32 {
        self.total_net_wh() * 0.43
    }

    /// 推定電気代 (円) - 計算専用電力量から算出 (1kWh = 31円換算)
    pub fn cost_yen(&self) -> f32 {
        self.total_net_wh() * 0.031
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_tracker_pi4() {
        let mut tracker = PowerTracker::new(PowerProfile::raspberry_pi_4());

        // 1秒間フル稼働 (スリープなし)
        let reading = tracker.tick(1000, 0);
        // 総電力: 6.2W, 純電力: 3.5W
        assert!((reading.gross_watts - 6.2).abs() < 0.1);
        assert!((reading.net_watts - 3.5).abs() < 0.1);
        assert!(reading.net_accum_wh > 0.0);
        assert!(reading.gross_accum_wh > reading.net_accum_wh);

        // 累積Wh検証: 3.5W * 1秒 = 3.5 J = 3.5 / 3600 Wh ≈ 0.00097 Wh
        let expected_net_wh = 3.5 / 3600.0;
        assert!((tracker.total_net_wh() - expected_net_wh).abs() < 1e-4);

        // CO2排出量検証
        assert!(tracker.equivalent_co2_grams() > 0.0);
    }
}
