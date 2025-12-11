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

/// Major version number
pub const MAJOR: u32 = 0;

/// Minor version number
pub const MINOR: u32 = 1;

/// Patch version number
pub const PATCH: u32 = 0;

use std::sync::OnceLock;

/// Full version string in semver format (e.g., "0.1.0")
static VERSION: OnceLock<String> = OnceLock::new();

/// Get the version string
fn get_version() -> &'static String {
    VERSION.get_or_init(|| format!("{}.{}.{}", MAJOR, MINOR, PATCH))
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
    get_version()
}

/// Returns version info as a formatted string
pub fn version_info() -> String {
    format!(
        "stoolap {} (commit: {}, built: {})",
        get_version(),
        GIT_COMMIT,
        BUILD_TIME
    )
}

/// Semantic version components
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemVer {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl SemVer {
    /// Create a new SemVer from components
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Get the current version as SemVer
    pub const fn current() -> Self {
        Self::new(MAJOR, MINOR, PATCH)
    }

    /// Check if this version is compatible with another version
    /// Compatible means same major version and minor >= other.minor
    pub fn is_compatible_with(&self, other: &SemVer) -> bool {
        self.major == other.major && self.minor >= other.minor
    }
}

impl std::fmt::Display for SemVer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl std::str::FromStr for SemVer {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return Err(format!("invalid version format: {}", s));
        }

        let major = parts[0]
            .parse()
            .map_err(|_| format!("invalid major version: {}", parts[0]))?;
        let minor = parts[1]
            .parse()
            .map_err(|_| format!("invalid minor version: {}", parts[1]))?;
        let patch = parts[2]
            .parse()
            .map_err(|_| format!("invalid patch version: {}", parts[2]))?;

        Ok(SemVer::new(major, minor, patch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_constants() {
        assert_eq!(MAJOR, 0);
        assert_eq!(MINOR, 1);
        assert_eq!(PATCH, 0);
    }

    #[test]
    fn test_version_string() {
        assert_eq!(version(), "0.1.0");
    }

    #[test]
    fn test_version_info() {
        let info = version_info();
        assert!(info.contains("stoolap"));
        assert!(info.contains("0.1.0"));
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

    #[test]
    fn test_semver_new() {
        let v = SemVer::new(1, 2, 3);
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn test_semver_current() {
        let v = SemVer::current();
        assert_eq!(v.major, MAJOR);
        assert_eq!(v.minor, MINOR);
        assert_eq!(v.patch, PATCH);
    }

    #[test]
    fn test_semver_display() {
        let v = SemVer::new(1, 2, 3);
        assert_eq!(v.to_string(), "1.2.3");
    }

    #[test]
    fn test_semver_from_str() {
        let v: SemVer = "1.2.3".parse().unwrap();
        assert_eq!(v, SemVer::new(1, 2, 3));

        let v: SemVer = "0.1.0".parse().unwrap();
        assert_eq!(v, SemVer::current());
    }

    #[test]
    fn test_semver_from_str_invalid() {
        assert!("1.2".parse::<SemVer>().is_err());
        assert!("1.2.3.4".parse::<SemVer>().is_err());
        assert!("a.b.c".parse::<SemVer>().is_err());
        assert!("".parse::<SemVer>().is_err());
    }

    #[test]
    fn test_semver_compatibility() {
        let v1 = SemVer::new(1, 2, 0);
        let v2 = SemVer::new(1, 1, 0);
        let v3 = SemVer::new(1, 3, 0);
        let v4 = SemVer::new(2, 0, 0);

        // v1 (1.2.0) is compatible with v2 (1.1.0) - same major, higher minor
        assert!(v1.is_compatible_with(&v2));

        // v1 (1.2.0) is NOT compatible with v3 (1.3.0) - needs higher minor
        assert!(!v1.is_compatible_with(&v3));

        // v1 (1.2.0) is NOT compatible with v4 (2.0.0) - different major
        assert!(!v1.is_compatible_with(&v4));

        // Same version is always compatible
        assert!(v1.is_compatible_with(&v1));
    }
}
