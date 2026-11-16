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

//! Version information for Stoolap
//!
//!
//! This module provides version constants and build information.

/// Full version string from Cargo.toml (single source of truth)
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Major version number (parsed at runtime for compatibility)
pub const MAJOR: u32 = parse_version_component(env!("CARGO_PKG_VERSION_MAJOR"));

/// Minor version number (parsed at runtime for compatibility)
pub const MINOR: u32 = parse_version_component(env!("CARGO_PKG_VERSION_MINOR"));

/// Patch version number (parsed at runtime for compatibility)
pub const PATCH: u32 = parse_version_component(env!("CARGO_PKG_VERSION_PATCH"));

/// Parse version component at compile time
const fn parse_version_component(s: &str) -> u32 {
    let bytes = s.as_bytes();
    let mut result: u32 = 0;
    let mut i = 0;
    while i < bytes.len() {
        let digit = bytes[i] - b'0';
        result = result * 10 + digit as u32;
        i += 1;
    }
    result
}

/// Git commit hash at build time
/// Set via STOOLAP_GIT_COMMIT environment variable during compilation
pub const GIT_COMMIT: &str = match option_env!("STOOLAP_GIT_COMMIT") {
    Some(commit) => commit,
    None => "unknown",
};

/// Build timestamp
/// Set via STOOLAP_BUILD_TIME environment variable during compilation
pub const BUILD_TIME: &str = match option_env!("STOOLAP_BUILD_TIME") {
    Some(time) => time,
    None => "unknown",
};

/// Returns the full version string
pub fn version() -> &'static str {
    VERSION
}

/// Returns version info as a formatted string
pub fn version_info() -> String {
    format!(
        "stoolap {} (commit: {}, built: {})",
        VERSION, GIT_COMMIT, BUILD_TIME
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_string() {
        // Version string should match the constants
        let expected = format!("{}.{}.{}", MAJOR, MINOR, PATCH);
        assert_eq!(version(), expected);
    }

    #[test]
    fn test_version_info() {
        let info = version_info();
        assert!(info.contains("stoolap"));
        assert!(info.contains(version()));
    }

    #[test]
    fn test_git_commit_default() {
        // Without env var set, should be "unknown"
        // This tests the default case
        assert!(!GIT_COMMIT.is_empty());
    }

    #[test]
    fn test_build_time_default() {
        // Without env var set, should be "unknown"
        assert!(!BUILD_TIME.is_empty());
    }
}
