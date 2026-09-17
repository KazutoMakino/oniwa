use std::fs;
use std::path::{Path, PathBuf};

/// Hardware-specific power profile.
#[derive(Debug, Clone)]
pub struct PowerProfile {
    /// Profile name or description
    pub name: String,
    /// Idle power consumption (W)
    pub idle_watts: f32,
    /// Additional power consumption under full load (W)
    pub busy_watts_delta: f32,
}

impl PowerProfile {
    /// Measured characteristics for Raspberry Pi 4 Model B (5V power supply)
    pub fn raspberry_pi_4() -> Self {
        Self {
            name: "Raspberry Pi 4 (5V low-power model)".to_string(),
            idle_watts: 2.7,       // Idle: ~2.7W (5V 0.54A)
            busy_watts_delta: 3.5, // 4-core full load: ~6.2W (2.7W + 3.5W)
        }
    }

    /// Profile for general low-power laptop / mini PC
    pub fn generic_pc() -> Self {
        Self {
            name: "Low-Power Laptop / Mini PC (Estimated)".to_string(),
            idle_watts: 10.0,
            busy_watts_delta: 25.0,
        }
    }

    /// Dynamically estimate optimal profile from system environment (/proc/cpuinfo, etc.)
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

        // Low-power mobile CPUs (U series, low TDP)
        if cpu_name.contains(" U")
            || cpu_name.contains("Mobile")
            || cpu_name.contains("Core i") && cpu_name.contains('U')
        {
            Self {
                name: format!("Mobile Low-Power CPU: {}", cpu_name),
                idle_watts: 8.0,
                busy_watts_delta: 20.0,
            }
        } else if cpu_name.contains("EPYC")
            || cpu_name.contains("Xeon")
            || cpu_name.contains("Threadripper")
        {
            Self {
                name: format!("Server/Workstation CPU: {}", cpu_name),
                idle_watts: 45.0,
                busy_watts_delta: 120.0,
            }
        } else {
            Self {
                name: format!("Desktop/Standard PC: {}", cpu_name),
                idle_watts: 15.0,
                busy_watts_delta: 45.0,
            }
        }
    }
}

/// Power measurement source (hardware sensor or dynamic profile)
enum PowerSource {
    HardwareSensor {
        path: PathBuf,
        desc: String,
        is_microwatts: bool,
    },
    EstimatedProfile(PowerProfile),
}

/// Power reading result (net computation power and gross system power)
#[derive(Debug, Clone, Copy)]
pub struct PowerReading {
    /// Instantaneous net power consumed purely by this computation (W)
    pub net_watts: f32,
    /// Instantaneous gross power consumed by entire system (W)
    pub gross_watts: f32,
    /// Cumulative net energy consumed purely by this computation (Wh)
    pub net_accum_wh: f32,
    /// Cumulative gross energy consumed by entire system (Wh)
    pub gross_accum_wh: f32,
}

/// Tracker for calculating and logging cumulative power consumption.
pub struct PowerTracker {
    source: PowerSource,
    fallback_profile: PowerProfile,
    /// Baseline idle power consumption (W)
    baseline_watts: f64,
    /// Cumulative net energy consumed purely by computation (Joules: J = W * s)
    total_net_joules: f64,
    /// Cumulative gross energy consumed by entire system (Joules: J = W * s)
    total_gross_joules: f64,
}

impl PowerTracker {
    /// Create from static profile (backward compatibility)
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

    /// Auto-detect hardware power sensors or optimal system profile and initialize.
    pub fn auto_detect() -> Self {
        let fallback = PowerProfile::detect_from_system();

        // 1. Search for hwmon power sensors
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

        // 2. Search for power_supply (battery, etc.) power sensors
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

        // 3. Fallback to dynamic profile estimation if no sensor found
        let baseline = fallback.idle_watts as f64;
        Self {
            source: PowerSource::EstimatedProfile(fallback.clone()),
            fallback_profile: fallback,
            baseline_watts: baseline,
            total_net_joules: 0.0,
            total_gross_joules: 0.0,
        }
    }

    /// Sample baseline idle power immediately before training starts.
    fn measure_initial_baseline(path: &Path, is_uw: bool, fallback: f64) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;
        for _ in 0..3 {
            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(raw_val) = content.trim().parse::<f64>() {
                    let w = if is_uw {
                        raw_val / 1_000_000.0
                    } else {
                        raw_val
                    };
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

    /// Search for power sensors under hwmon (e.g. AMD PPT: Package Power Tracking)
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
                                        let label_file = dir.join(
                                            fname
                                                .replace("_input", "_label")
                                                .replace("_average", "_label"),
                                        );
                                        let label = fs::read_to_string(&label_file)
                                            .unwrap_or_default()
                                            .trim()
                                            .to_string();

                                        let desc = if !label.is_empty() {
                                            format!("{} ({})", hwmon_name, label)
                                        } else {
                                            format!("{} ({})", hwmon_name, fname)
                                        };

                                        // Linux sysfs power*_input is usually in microwatts (uW)
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

    /// Search for power sensors under power_supply
    fn detect_power_supply_sensor() -> Option<(PathBuf, String)> {
        if let Ok(entries) = fs::read_dir("/sys/class/power_supply") {
            for entry in entries.filter_map(|e| e.ok().map(|e| e.path())) {
                let power_now = entry.join("power_now");
                if power_now.exists() {
                    if let Ok(content) = fs::read_to_string(&power_now) {
                        if content.trim().parse::<f64>().is_ok() {
                            let name = entry
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("battery");
                            return Some((power_now, format!("power_supply/{}", name)));
                        }
                    }
                }
            }
        }
        None
    }

    /// Description of current power measurement source
    pub fn source_description(&self) -> String {
        match &self.source {
            PowerSource::HardwareSensor { desc, .. } => {
                format!("Hardware power sensor: {}", desc)
            }
            PowerSource::EstimatedProfile(prof) => {
                format!("Estimated power model: {}", prof.name)
            }
        }
    }

    /// Whether a hardware power sensor is active
    pub fn is_hardware_sensor(&self) -> bool {
        matches!(self.source, PowerSource::HardwareSensor { .. })
    }

    /// Baseline idle power (W)
    pub fn baseline_watts(&self) -> f32 {
        self.baseline_watts as f32
    }

    /// Call at the end of each step to record/estimate power consumption and accumulate energy.
    ///
    /// - `calc_duration_ms`: Actual computation time (ms)
    /// - `sleep_duration_ms`: Idle/throttling sleep time (ms)
    ///
    /// Returns: PowerReading
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

        // Net power consumed purely by this computation (Gross - Baseline)
        let net_watts = (gross_watts - self.baseline_watts).max(0.0);

        // Entire system energy (gross power * total duration)
        let delta_gross_joules = gross_watts * delta_seconds;
        self.total_gross_joules += delta_gross_joules;

        // Energy consumed purely by computation (net power * actual computation time)
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

    /// Cumulative energy consumed purely by this computation (Wh) [Primary Metric]
    pub fn total_net_wh(&self) -> f32 {
        (self.total_net_joules / 3600.0) as f32
    }

    /// Cumulative gross energy consumed by the entire system (Wh) [Reference Metric]
    pub fn total_gross_wh(&self) -> f32 {
        (self.total_gross_joules / 3600.0) as f32
    }

    /// Backward-compatible cumulative Wh (returns net computation Wh)
    pub fn total_wh(&self) -> f32 {
        self.total_net_wh()
    }

    /// Estimated CO2 emissions in grams (g-CO2) derived from net computation Wh.
    /// Uses standard grid emission factor (~0.43 kg-CO2 / kWh = 0.43 g-CO2 / Wh).
    pub fn equivalent_co2_grams(&self) -> f32 {
        self.total_net_wh() * 0.43
    }

    /// Estimated electricity cost in Japanese Yen derived from net computation Wh (~31 JPY / kWh).
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

        // 1 second full load (no sleep)
        let reading = tracker.tick(1000, 0);
        // Gross power: 6.2W, Net power: 3.5W
        assert!((reading.gross_watts - 6.2).abs() < 0.1);
        assert!((reading.net_watts - 3.5).abs() < 0.1);
        assert!(reading.net_accum_wh > 0.0);
        assert!(reading.gross_accum_wh > reading.net_accum_wh);

        // Cumulative Wh verification: 3.5W * 1s = 3.5 J = 3.5 / 3600 Wh ~ 0.00097 Wh
        let expected_net_wh = 3.5 / 3600.0;
        assert!((tracker.total_net_wh() - expected_net_wh).abs() < 1e-4);

        // CO2 emission verification
        assert!(tracker.equivalent_co2_grams() > 0.0);
    }
}
