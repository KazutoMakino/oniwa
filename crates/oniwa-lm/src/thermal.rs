//! プラットフォーム透過的な実温度監視と動的熱制御
//!
//! - Raspberry Pi 4 (Linux ARM64): /sys/class/thermal/thermal_zone0/temp などを監視
//! - 一般的な Linux PC (x86_64): hwmon や thermal_zone の温度センサーを自動検出
//! - macOS / Windows / センサー非搭載環境: 自動フォールバック（温度監視をスキップし、フルスピードで安全に稼働）

use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

/// サーマル管理の設定
#[derive(Debug, Clone)]
pub struct ThermalConfig {
    /// 監視対象の sysfs パス (None の場合は自動検出)
    pub thermal_zone_path: Option<PathBuf>,
    /// 冷却制御を開始する目標温度 (℃) (デフォルト: 70.0)
    pub target_temp_c: f32,
    /// 強制冷却（長めスリープ）を行う危険温度 (℃) (デフォルト: 78.0)
    pub critical_temp_c: f32,
    /// 通常の冷却スリープ時間 (ミリ秒) (デフォルト: 50ms)
    pub throttle_sleep_ms: u64,
    /// 危険温度時の緊急スリープ時間 (ミリ秒) (デフォルト: 300ms)
    pub critical_sleep_ms: u64,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            thermal_zone_path: None, // 自動検出
            target_temp_c: 70.0,
            critical_temp_c: 78.0,
            throttle_sleep_ms: 50,
            critical_sleep_ms: 300,
        }
    }
}

/// 動的熱制御マネージャー
pub struct ThermalController {
    config: ThermalConfig,
    active_path: Option<PathBuf>,
    sensor_name: Option<String>,
}

impl ThermalController {
    /// 利用可能な温度センサーを自動検出し、コントローラーを初期化
    pub fn new(config: ThermalConfig) -> Self {
        let (active_path, sensor_name) = if let Some(ref p) = config.thermal_zone_path {
            if p.exists() {
                (Some(p.clone()), Some(p.to_string_lossy().to_string()))
            } else {
                (None, None)
            }
        } else {
            Self::detect_cpu_thermal_sensor()
        };

        Self {
            config,
            active_path,
            sensor_name,
        }
    }

    /// CPU / SoC 温度センサーを高精度に自動検出
    ///
    /// 1. hwmon (k10temp, zenpower, coretemp など CPU 直結センサー)
    /// 2. thermal_zone (x86_pkg_temp, cpu-thermal など)
    /// 3. 周辺機器 (Wi-Fi, NVMe 等) を除外した汎用フォールバック
    fn detect_cpu_thermal_sensor() -> (Option<PathBuf>, Option<String>) {
        // --- 1. /sys/class/hwmon のスキャン ---
        let mut fallback_hwmon: Option<(PathBuf, String)> = None;

        if let Ok(entries) = fs::read_dir("/sys/class/hwmon") {
            let mut hwmon_dirs: Vec<PathBuf> = entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_dir())
                .collect();
            hwmon_dirs.sort();

            for dir in hwmon_dirs {
                let name = fs::read_to_string(dir.join("name"))
                    .unwrap_or_default()
                    .trim()
                    .to_string();

                // Wi-Fi 等の明らかに CPU でないものは除外
                if name.contains("iwl") || name.contains("wifi") || name.contains("wireless") {
                    continue;
                }

                // temp*_input を探索
                if let Ok(files) = fs::read_dir(&dir) {
                    for file in files.filter_map(|e| e.ok().map(|e| e.path())) {
                        let fname = file.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if fname.starts_with("temp")
                            && fname.ends_with("_input")
                            && Self::read_temp_from_path(&file).is_some()
                        {
                            let label_file = dir.join(fname.replace("_input", "_label"));
                            let label = fs::read_to_string(&label_file)
                                .unwrap_or_default()
                                .trim()
                                .to_string();

                            let desc = if !label.is_empty() {
                                format!("{} ({})", name, label)
                            } else {
                                name.clone()
                            };

                            // 最優先: AMD k10temp / Intel coretemp / Zenpower
                            if (name == "k10temp" || name == "coretemp" || name == "zenpower")
                                && (label == "Tctl"
                                    || label == "Package id 0"
                                    || label.contains("CPU")
                                    || label.is_empty())
                            {
                                return (Some(file), Some(desc));
                            }

                            if fallback_hwmon.is_none() && !name.is_empty() {
                                fallback_hwmon = Some((file, desc));
                            }
                        }
                    }
                }
            }
        }

        // --- 2. /sys/class/thermal のスキャン ---
        if let Ok(entries) = fs::read_dir("/sys/class/thermal") {
            let mut tz_dirs: Vec<PathBuf> = entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|s| s.starts_with("thermal_zone"))
                })
                .collect();
            tz_dirs.sort();

            for dir in tz_dirs {
                let tz_type = fs::read_to_string(dir.join("type"))
                    .unwrap_or_default()
                    .trim()
                    .to_string();

                // 明らかに CPU 系の thermal_zone
                if tz_type.contains("cpu")
                    || tz_type.contains("x86_pkg")
                    || tz_type.contains("soc")
                    || tz_type == "cpu-thermal"
                {
                    let temp_file = dir.join("temp");
                    if Self::read_temp_from_path(&temp_file).is_some() {
                        return (Some(temp_file), Some(format!("thermal_zone ({})", tz_type)));
                    }
                }
            }
        }

        // --- 3. フォールバック ---
        if let Some((path, desc)) = fallback_hwmon {
            return (Some(path), Some(desc));
        }

        (None, None)
    }

    /// 温度センサーが利用可能かどうか
    pub fn is_available(&self) -> bool {
        self.active_path.is_some()
    }

    /// 検出されたセンサー名・種類
    pub fn sensor_name(&self) -> Option<&str> {
        self.sensor_name.as_deref()
    }

    /// 監視中のファイルパス
    pub fn sensor_path(&self) -> Option<&Path> {
        self.active_path.as_deref()
    }

    /// 現在の CPU 実温度 (℃) を取得 (センサーなし/非対応OSの場合は None)
    pub fn read_temperature(&self) -> Option<f32> {
        self.active_path
            .as_ref()
            .and_then(|p| Self::read_temp_from_path(p))
    }

    /// 指定パスから温度をミリ度（または度）単位で読み取って ℃ に変換
    pub fn read_temp_from_path(path: &Path) -> Option<f32> {
        if !path.exists() {
            return None;
        }
        let content = fs::read_to_string(path).ok()?;
        let val = content.trim().parse::<f32>().ok()?;

        // Linux の sysfs は通常 1000倍のミリ度 (例: 55000 = 55.0℃)
        // すでに度単位の場合は 100未満の値が入る
        if val > 1000.0 {
            Some(val / 1000.0)
        } else {
            Some(val)
        }
    }

    /// ステップ終了ごとに呼び出し、温度に応じて動的に負荷調整（スリープ）を適用
    ///
    /// センサー非搭載PCやmacOS等では、自動的にスリープなし（0ms）で最高速動作します。
    /// 戻り値: (現在の温度℃, 適用されたスリープ時間ms)
    pub fn step_throttle(&self) -> (Option<f32>, u64) {
        let temp = self.read_temperature();

        let sleep_ms = match temp {
            Some(t) if t >= self.config.critical_temp_c => {
                // 危険温度: 緊急スリープ
                thread::sleep(Duration::from_millis(self.config.critical_sleep_ms));
                self.config.critical_sleep_ms
            }
            Some(t) if t >= self.config.target_temp_c => {
                // 目標超過: 比例スリープ
                let over = t - self.config.target_temp_c;
                let scale =
                    (over / (self.config.critical_temp_c - self.config.target_temp_c)).min(1.0);
                let wait = (self.config.throttle_sleep_ms as f32 * (1.0 + scale)) as u64;
                thread::sleep(Duration::from_millis(wait));
                wait
            }
            _ => 0, // 安全温度帯 または センサーなし環境: スリープなしで最高速度で実行
        };

        (temp, sleep_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_thermal_fallback_on_regular_pc() {
        // センサーが存在しない環境（macOSやWindows、仮想マシン等）でも安全に動作することを検証
        let config = ThermalConfig {
            thermal_zone_path: Some(PathBuf::from("/non/existent/path/temp")),
            ..Default::default()
        };
        let controller = ThermalController::new(config);

        assert!(!controller.is_available());
        let (temp, sleep_ms) = controller.step_throttle();
        assert_eq!(temp, None);
        assert_eq!(sleep_ms, 0); // パニックせず、スリープゼロでフル稼働
    }

    #[test]
    fn test_thermal_reading_and_throttling() {
        let tmp_dir = std::env::temp_dir();
        let mock_temp_file = tmp_dir.join("mock_thermal_zone_temp");

        // 1. 安全温度: 55℃ (55000ミリ度)
        {
            let mut f = fs::File::create(&mock_temp_file).unwrap();
            writeln!(f, "55000").unwrap();
        }

        let config = ThermalConfig {
            thermal_zone_path: Some(mock_temp_file.clone()),
            target_temp_c: 70.0,
            critical_temp_c: 78.0,
            throttle_sleep_ms: 10,
            critical_sleep_ms: 20,
        };
        let controller = ThermalController::new(config);
        assert!(controller.is_available());

        let (temp, sleep_ms) = controller.step_throttle();
        assert_eq!(temp, Some(55.0));
        assert_eq!(sleep_ms, 0);

        // 2. 目標超過: 72℃
        {
            let mut f = fs::File::create(&mock_temp_file).unwrap();
            writeln!(f, "72000").unwrap();
        }
        let (temp, sleep_ms) = controller.step_throttle();
        assert_eq!(temp, Some(72.0));
        assert!(sleep_ms >= 10);

        // クリーンアップ
        let _ = fs::remove_file(mock_temp_file);
    }
}
