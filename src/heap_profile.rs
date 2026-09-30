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

//! Heap profiles from jemalloc's sampling profiler, in pprof format, for a
//! Linux build with the `jemalloc-prof` feature

use crate::core::{Error, Result};

/// Writes the live heap profile to `path`; returns its size in bytes
#[cfg(all(
    feature = "jemalloc-prof",
    target_os = "linux",
    not(feature = "dhat-heap")
))]
pub(crate) fn write_heap_profile(path: &str) -> Result<usize> {
    let ctl = jemalloc_pprof::PROF_CTL
        .as_ref()
        .ok_or_else(|| Error::internal("jemalloc heap profiling is not enabled"))?;
    let mut ctl = ctl
        .try_lock()
        .map_err(|_| Error::internal("another heap profile is being written"))?;
    let profile = ctl
        .dump_pprof()
        .map_err(|e| Error::internal(format!("heap profile: {e}")))?;
    std::fs::write(path, &profile)
        .map_err(|e| Error::internal(format!("heap profile {path}: {e}")))?;
    Ok(profile.len())
}

#[cfg(not(all(
    feature = "jemalloc-prof",
    target_os = "linux",
    not(feature = "dhat-heap")
)))]
pub(crate) fn write_heap_profile(_path: &str) -> Result<usize> {
    Err(Error::internal(
        "heap profiles need a Linux build with the jemalloc-prof feature",
    ))
}
