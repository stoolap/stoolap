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

//! Heap profiles from stoolap-jemalloc's sampling profiler, in gzipped
//! pprof format, for a build with the `heap-profile` feature

use crate::core::{Error, Result};

/// Writes the live heap profile to `path`; returns its size in bytes
#[cfg(all(feature = "heap-profile", not(feature = "dhat-heap")))]
pub(crate) fn write_heap_profile(path: &str) -> Result<usize> {
    use std::io::Write;

    let profile = stoolap_jemalloc::prof::dump_pprof()
        .map_err(|e| Error::internal(format!("heap profile: {e}")))?;
    let mut gzip = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    let gzipped = gzip
        .write_all(&profile)
        .and_then(|()| gzip.finish())
        .map_err(|e| Error::internal(format!("heap profile: {e}")))?;
    std::fs::write(path, &gzipped)
        .map_err(|e| Error::internal(format!("heap profile {path}: {e}")))?;
    Ok(gzipped.len())
}

#[cfg(not(all(feature = "heap-profile", not(feature = "dhat-heap"))))]
pub(crate) fn write_heap_profile(_path: &str) -> Result<usize> {
    Err(Error::internal(
        "heap profiles need a build with the heap-profile feature",
    ))
}
