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

use stoolap::storage::mvcc::version_store::clear_version_map_pools;
use stoolap::Database;

#[test]
fn transaction_map_capacity_follows_growth_reuse_and_pool_release() {
    let db =
        Database::open("memory://transaction_map_capacity_follows_growth_reuse_and_pool_release")
            .unwrap();
    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();
    let bytes = || {
        db.engine()
            .memory_stats()
            .last()
            .unwrap()
            .transaction_map_bytes
    };
    clear_version_map_pools();
    let empty = bytes();
    db.execute("BEGIN", ()).unwrap();
    db.execute(
        "INSERT INTO items SELECT n, n FROM generate_series(1, 2048) AS g(n)",
        (),
    )
    .unwrap();
    let active = bytes();
    assert!(active > empty);
    db.execute("ROLLBACK", ()).unwrap();
    let pooled = bytes();
    assert!(
        pooled >= active,
        "cleared transaction maps keep their capacity in the pools"
    );
    db.execute("BEGIN", ()).unwrap();
    db.execute(
        "INSERT INTO items SELECT n, n FROM generate_series(1, 2048) AS g(n)",
        (),
    )
    .unwrap();
    assert_eq!(
        bytes(),
        pooled,
        "checkout transfers the existing allocation"
    );
    db.execute("ROLLBACK", ()).unwrap();
    assert_eq!(bytes(), pooled);
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO items VALUES (1, 1)", ()).unwrap();
    db.execute("COMMIT", ()).unwrap();
    let drained = bytes();
    assert!(drained < pooled, "commit drains shrink oversized maps");
    clear_version_map_pools();
    assert!(bytes() < drained, "pool clearing releases map allocations");
    let empty_pool_vectors = bytes();
    clear_version_map_pools();
    assert_eq!(bytes(), empty_pool_vectors);
}
