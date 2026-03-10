// Copyright 2025 Stoolap Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::process::Command;

fn main() {
    // Embed git commit hash at compile time
    if std::env::var("STOOLAP_GIT_COMMIT").is_err() {
        if let Ok(output) = Command::new("git").args(["rev-parse", "HEAD"]).output() {
            if output.status.success() {
                let commit = String::from_utf8_lossy(&output.stdout).trim().to_string();
                println!("cargo:rustc-env=STOOLAP_GIT_COMMIT={}", commit);
            }
        }
    }

    // Set dylib version on macOS (current_version and compatibility_version)
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        let version = env!("CARGO_PKG_VERSION");
        // compatibility_version: minor-level (consumers linked against 0.3.x work with any 0.3.y)
        let parts: Vec<&str> = version.split('.').collect();
        let compat = if parts.len() >= 2 {
            format!("{}.{}.0", parts[0], parts[1])
        } else {
            version.to_string()
        };
        println!("cargo:rustc-cdylib-link-arg=-Wl,-current_version,{version}");
        println!("cargo:rustc-cdylib-link-arg=-Wl,-compatibility_version,{compat}");
    }

    // Only re-run if HEAD changes or env var is set
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs/heads/");
    println!("cargo:rerun-if-env-changed=STOOLAP_GIT_COMMIT");
}
