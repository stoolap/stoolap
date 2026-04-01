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
use std::time::SystemTime;

/// Generate UTC timestamp without external dependencies.
fn chrono_free_timestamp() -> String {
    let dur = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Convert to UTC date-time components
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Civil date from days since epoch (algorithm from Howard Hinnant)
    let z = days as i64 + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y, m, d, hours, minutes, seconds
    )
}

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

    // Embed build timestamp at compile time
    if std::env::var("STOOLAP_BUILD_TIME").is_err() {
        let now = chrono_free_timestamp();
        println!("cargo:rustc-env=STOOLAP_BUILD_TIME={}", now);
    }
    println!("cargo:rerun-if-env-changed=STOOLAP_BUILD_TIME");

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
