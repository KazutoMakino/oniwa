use std::process::Command;

fn main() {
    // 1. Retrieve full 40-character Git commit hash
    let commit_hash = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                String::from_utf8(out.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string());

    let short_hash = if commit_hash.len() >= 7 {
        commit_hash[..7].to_string()
    } else {
        commit_hash.clone()
    };

    // 2. Check if working tree has unstaged/uncommitted changes (git dirty)
    let is_dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .map(|out| out.status.success() && !out.stdout.is_empty())
        .unwrap_or(false);

    println!("cargo:rustc-env=ONIWA_GIT_HASH={}", commit_hash);
    println!("cargo:rustc-env=ONIWA_GIT_SHORT_HASH={}", short_hash);
    println!("cargo:rustc-env=ONIWA_GIT_DIRTY={}", is_dirty);

    // Re-run build script if Git state changes
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");
}
