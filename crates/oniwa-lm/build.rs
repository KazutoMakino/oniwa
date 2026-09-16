use std::process::Command;

fn main() {
    // 1. Git コミットハッシュの取得 (40桁完全ハッシュ)
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

    // 2. 作業ツリーの変更有無 (git dirty) の取得
    let is_dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .map(|out| out.status.success() && !out.stdout.is_empty())
        .unwrap_or(false);

    println!("cargo:rustc-env=ONIWA_GIT_HASH={}", commit_hash);
    println!("cargo:rustc-env=ONIWA_GIT_SHORT_HASH={}", short_hash);
    println!("cargo:rustc-env=ONIWA_GIT_DIRTY={}", is_dirty);

    // .git 参照の変更を検知して環境変数を再評価
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");
}
