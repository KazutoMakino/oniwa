//! Platform-transparent real temperature monitoring and dynamic thermal throttling.
//!
//! - Raspberry Pi 4 (Linux ARM64): monitors `/sys/class/thermal/thermal_zone0/temp`, etc.
//! - General Linux PC (x86_64): auto-detects CPU temperature sensors via hwmon or thermal_zone.
//! - macOS / Windows / systems without sensors: automatic fallback (skips monitoring, runs safely at full speed).

use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

/// Configuration for thermal management.
#[derive(Debug, Clone)]
pub struct ThermalConfig {
    /// Target sysfs path to monitor (None for auto-detection)
    pub thermal_zone_path: Option<PathBuf>,
    /// Target temperature (Celsius) to start throttling (default: 70.0)
    pub target_temp_c: f32,
    /// Critical temperature (Celsius) to trigger emergency cooling sleep (default: 78.0)
    pub critical_temp_c: f32,
    /// Normal throttling sleep duration (ms) (default: 50ms)
    pub throttle_sleep_ms: u64,
    /// Emergency cooling sleep duration (ms) (default: 300ms)
    pub critical_sleep_ms: u64,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            thermal_zone_path: None, // Auto-detect
            target_temp_c: 70.0,
            critical_temp_c: 78.0,
            throttle_sleep_ms: 50,
            critical_sleep_ms: 300,
        }
    }
}

/// Dynamic thermal management controller.
pub struct ThermalController {
    config: ThermalConfig,
    active_path: Option<PathBuf>,
    sensor_name: Option<String>,
}

impl ThermalController {
    /// Auto-detect available temperature sensors and initialize controller.
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

    /// Auto-detect CPU / SoC thermal sensors with high precision.
    ///
    /// 1. hwmon (CPU-direct sensors such as k10temp, zenpower, coretemp)
    /// 2. thermal_zone (x86_pkg_temp, cpu-thermal, etc.)
    /// 3. Generic fallback excluding peripherals (Wi-Fi, NVMe, etc.)
    fn detect_cpu_thermal_sensor() -> (Option<PathBuf>, Option<String>) {
        // --- 1. Scan /sys/class/hwmon ---
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

                // Exclude obvious non-CPU peripherals such as Wi-Fi
                if name.contains("iwl") || name.contains("wifi") || name.contains("wireless") {
                    continue;
                }

                // Search for temp*_input
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

                            // Top priority: AMD k10temp / Intel coretemp / Zenpower
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

        // --- 2. Scan /sys/class/thermal ---
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

                // Obvious CPU thermal_zone
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

        // --- 3. Fallback ---
        if let Some((path, desc)) = fallback_hwmon {
            return (Some(path), Some(desc));
        }

        (None, None)
    }

    /// Check if temperature sensor is available.
    pub fn is_available(&self) -> bool {
        self.active_path.is_some()
    }

    /// Detected sensor name / type.
    pub fn sensor_name(&self) -> Option<&str> {
        self.sensor_name.as_deref()
    }

    /// Monitored file path.
    pub fn sensor_path(&self) -> Option<&Path> {
        self.active_path.as_deref()
    }

    /// Read current CPU actual temperature in Celsius (None if sensor unavailable or unsupported OS).
    pub fn read_temperature(&self) -> Option<f32> {
        self.active_path
            .as_ref()
            .and_then(|p| Self::read_temp_from_path(p))
    }

    /// Read temperature in millidegrees (or degrees) from specified path and convert to Celsius.
    pub fn read_temp_from_path(path: &Path) -> Option<f32> {
        if !path.exists() {
            return None;
        }
        let content = fs::read_to_string(path).ok()?;
        let val = content.trim().parse::<f32>().ok()?;

        // Linux sysfs is usually in millidegrees (e.g. 55000 = 55.0 C).
        // If already in degrees, value is typically under 100.
        if val > 1000.0 {
            Some(val / 1000.0)
        } else {
            Some(val)
        }
    }

    /// Call at the end of each step to dynamically apply throttling sleep based on temperature.
    ///
    /// On PCs without sensors or on macOS, runs safely at full speed with 0ms sleep.
    /// Returns: (current_temp_c, applied_sleep_ms)
    pub fn step_throttle(&self) -> (Option<f32>, u64) {
        let temp = self.read_temperature();

        let sleep_ms = match temp {
            Some(t) if t >= self.config.critical_temp_c => {
                // Critical temperature: emergency sleep
                thread::sleep(Duration::from_millis(self.config.critical_sleep_ms));
                self.config.critical_sleep_ms
            }
            Some(t) if t >= self.config.target_temp_c => {
                // Target exceeded: proportional throttling sleep
                let over = t - self.config.target_temp_c;
                let scale =
                    (over / (self.config.critical_temp_c - self.config.target_temp_c)).min(1.0);
                let wait = (self.config.throttle_sleep_ms as f32 * (1.0 + scale)) as u64;
                thread::sleep(Duration::from_millis(wait));
                wait
            }
            _ => 0, // Safe temperature range or no sensor: execute at full speed without sleep
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
        // Verify safe operation even on systems without sensors (macOS, Windows, VMs, etc.)
        let config = ThermalConfig {
            thermal_zone_path: Some(PathBuf::from("/non/existent/path/temp")),
            ..Default::default()
        };
        let controller = ThermalController::new(config);

        assert!(!controller.is_available());
        let (temp, sleep_ms) = controller.step_throttle();
        assert_eq!(temp, None);
        assert_eq!(sleep_ms, 0); // Runs at full speed with zero sleep without panicking
    }

    #[test]
    fn test_thermal_reading_and_throttling() {
        let tmp_dir = std::env::temp_dir();
        let mock_temp_file = tmp_dir.join("mock_thermal_zone_temp");

        // 1. Safe temperature: 55 C (55000 millidegrees)
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

        // 2. Exceeded target: 72 C
        {
            let mut f = fs::File::create(&mock_temp_file).unwrap();
            writeln!(f, "72000").unwrap();
        }
        let (temp, sleep_ms) = controller.step_throttle();
        assert_eq!(temp, Some(72.0));
        assert!(sleep_ms >= 10);

        // Cleanup
        let _ = fs::remove_file(mock_temp_file);
    }
}
