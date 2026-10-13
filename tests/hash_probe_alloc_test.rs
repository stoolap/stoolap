// Copyright 2026 Stoolap Contributors
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

//! Probing a hash join's table allocates nothing per probe row, whether
//! the table was built sequentially or in parallel

#![cfg(not(any(feature = "jemalloc", feature = "dhat-heap")))]

use stoolap::core::{Row, Value};
use stoolap::executor::{parallel_hash_build, parallel_hash_probe, ParallelConfig};

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

const ROWS: i64 = 20_000;

fn rows() -> Vec<Row> {
    (1..=ROWS)
        .map(|k| Row::from_values(vec![Value::Integer(k)]))
        .collect()
}

/// Allocations made while every probe row finds its one build row
fn probe_allocations(build_config: &ParallelConfig) -> u64 {
    let build = rows();
    let probe = rows();
    let table = parallel_hash_build(&build, &[0], build_config);
    let profiler = dhat::Profiler::builder().testing().build();
    let before = dhat::HeapStats::get().total_blocks;
    let matches = parallel_hash_probe(
        &probe,
        &[0],
        &table,
        &build,
        |p, b| p.get(0) == b.get(0),
        &ParallelConfig::disabled(),
    );
    let allocated = dhat::HeapStats::get().total_blocks - before;
    drop(profiler);
    assert_eq!(matches.len(), ROWS as usize);
    allocated
}

// One test: dhat counts for the whole process, so both builds are measured
// one after the other, never beside another test
#[test]
fn probing_a_built_table_allocates_nothing_per_row() {
    let sequential = probe_allocations(&ParallelConfig::disabled());
    assert!(
        sequential < 64,
        "probe of a sequential table made {sequential} allocations"
    );

    #[cfg(feature = "parallel")]
    {
        let config = ParallelConfig {
            min_rows_for_parallel_join: 1000,
            ..ParallelConfig::default()
        };
        assert!(config.should_parallel_join(ROWS as usize));
        let parallel = probe_allocations(&config);
        assert!(
            parallel < 64,
            "probe of a parallel table made {parallel} allocations"
        );
    }
}
