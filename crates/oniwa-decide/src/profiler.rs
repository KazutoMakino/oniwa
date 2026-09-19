//! Hardware profiling and telemetry module for edge intelligence (Pure Rust).
//!
//! Provides utilities for:
//! - Measuring inference latency distributions (p50, p95, p99)
//! - Resident Set Size (RSS / VmRSS) extraction via `/proc/self/status`
//! - Thermal tracking and energy per inference calculation

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Memory usage profile read from system status
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryProfile {
    /// Resident Set Size in Kilobytes
    pub rss_kb: usize,
    /// Peak Resident Set Size in Kilobytes
    pub peak_rss_kb: usize,
}

impl MemoryProfile {
    /// Read current RSS memory usage from `/proc/self/status` (Linux only, fallback to 0)
    pub fn current() -> Self {
        Self::from_status_path(Path::new("/proc/self/status"))
    }

    /// Read RSS from a specified status path
    pub fn from_status_path(path: &Path) -> Self {
        let mut rss = 0;
        let mut peak_rss = 0;

        if let Ok(content) = fs::read_to_string(path) {
            for line in content.lines() {
                if line.starts_with("VmRSS:") {
                    rss = parse_kb_line(line);
                } else if line.starts_with("VmHWM:") {
                    peak_rss = parse_kb_line(line);
                }
            }
        }

        Self {
            rss_kb: rss,
            peak_rss_kb: peak_rss,
        }
    }
}

fn parse_kb_line(line: &str) -> usize {
    line.split_whitespace()
        .nth(1)
        .and_then(|val| val.parse::<usize>().ok())
        .unwrap_or(0)
}

/// Latency statistics summary for a batch of inferences
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatencyStats {
    pub count: usize,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub mean_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

impl LatencyStats {
    /// Compute latency statistics from an array of millisecond timings
    pub fn compute(mut timings_ms: Vec<f64>) -> Self {
        if timings_ms.is_empty() {
            return Self {
                count: 0,
                p50_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
                mean_ms: 0.0,
                min_ms: 0.0,
                max_ms: 0.0,
            };
        }

        timings_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let count = timings_ms.len();
        let sum: f64 = timings_ms.iter().sum();
        let mean_ms = sum / count as f64;
        let min_ms = timings_ms[0];
        let max_ms = timings_ms[count - 1];

        let percentile = |p: f64| -> f64 {
            let rank = (p / 100.0) * (count as f64 - 1.0);
            let lower = rank.floor() as usize;
            let upper = rank.ceil() as usize;
            if lower == upper {
                timings_ms[lower]
            } else {
                let weight = rank - lower as f64;
                timings_ms[lower] * (1.0 - weight) + timings_ms[upper] * weight
            }
        };

        Self {
            count,
            p50_ms: percentile(50.0),
            p95_ms: percentile(95.0),
            p99_ms: percentile(99.0),
            mean_ms,
            min_ms,
            max_ms,
        }
    }
}

/// Comprehensive hardware profiling record for decision models
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HardwareProfileRecord {
    pub model_name: String,
    pub config_type: String,
    pub params_count: usize,
    pub latency: LatencyStats,
    pub memory: MemoryProfile,
    pub joules_per_inference: f32,
    pub avg_power_watts: f32,
    pub thermal_celsius: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_stats_calculation() {
        let timings = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = LatencyStats::compute(timings);
        assert_eq!(stats.count, 10);
        assert_eq!(stats.min_ms, 1.0);
        assert_eq!(stats.max_ms, 10.0);
        assert_eq!(stats.mean_ms, 5.5);
        assert!((stats.p50_ms - 5.5).abs() < 1e-5);
    }

    #[test]
    fn test_memory_profile_fallback_or_read() {
        let mem = MemoryProfile::current();
        let _ = mem.rss_kb;
    }
}
