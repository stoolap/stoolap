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

//! The indexes a table is created with come back from the log alone, with
//! no checkpoint taken, whether the engine or a CREATE TABLE made them

use std::path::Path;

use stoolap::core::{DataType, SchemaBuilder};
use stoolap::storage::config::Config;
use stoolap::storage::mvcc::MVCCEngine;
use stoolap::storage::traits::{Engine, NewIndex};
use stoolap::Database;

fn no_checkpoint(path: &Path) -> Config {
    let mut config = Config::with_path(path.to_str().unwrap());
    config.cleanup.enabled = false;
    config.persistence.checkpoint_on_close = false;
    config.persistence.checkpoint_interval = 3600;
    config
}

/// The database files as they stand, as a crash would leave them
fn crash_image(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            crash_image(&entry.path(), &target);
        } else if entry.file_name() != "db.lock" {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

#[test]
fn an_index_the_engine_creates_a_table_with_is_replayed() {
    let dir = tempfile::tempdir().unwrap();
    let schema = SchemaBuilder::new("t")
        .column("id", DataType::Integer, false, true)
        .column("b", DataType::Integer, true, false)
        .build();
    let index = NewIndex {
        name: "unique_t_b".to_string(),
        columns: vec!["b".to_string()],
        unique: true,
    };
    let engine = MVCCEngine::new(no_checkpoint(dir.path()));
    engine.open_engine().unwrap();
    engine.create_table_with_indexes(schema, &[index]).unwrap();
    engine.close_engine().unwrap();
    drop(engine);

    let engine = MVCCEngine::new(no_checkpoint(dir.path()));
    engine.open_engine().unwrap();
    let tx = engine.begin_transaction().unwrap();
    let replayed = tx.get_table("t").unwrap().get_index("unique_t_b");
    assert!(
        replayed.is_some_and(|index| index.is_unique()),
        "unique_t_b is not back after the replay"
    );
    drop(tx);
    engine.close_engine().unwrap();
}

#[test]
fn an_index_create_table_makes_is_replayed() {
    let dir = tempfile::tempdir().unwrap();
    let image = tempfile::tempdir().unwrap();
    let dsn = |path: &Path| format!("file://{}?checkpoint_interval=3600", path.display());
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, b INTEGER UNIQUE, c INTEGER, d INTEGER, \
         UNIQUE (c, d))",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1, 1, 1)", ()).unwrap();
    crash_image(dir.path(), image.path());
    db.close().unwrap();

    let db = Database::open(&dsn(image.path())).unwrap();
    let mut wrong = Vec::new();
    for sql in [
        "INSERT INTO t VALUES (2, 1, 2, 2)",
        "INSERT INTO t VALUES (3, 3, 1, 1)",
    ] {
        if db.execute(sql, ()).is_ok() {
            wrong.push(sql);
        }
    }
    db.close().unwrap();
    assert!(wrong.is_empty(), "accepted after the replay: {wrong:?}");
}
