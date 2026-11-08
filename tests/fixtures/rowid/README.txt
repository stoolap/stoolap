Databases written by a build from before _rowid_ was reserved for the row
identity, read by tests/persistence/rowid_fixture_test.rs. The current build
cannot write them: it refuses a column of that name.

Generator: commit 7b798823, program generator.rs.txt placed at
examples/rowid_fixture.rs and run as

    cargo run --example rowid_fixture -- <out>

db.lock files were left out.

1. column_named_rowid: table t (a INTEGER, _rowid_ INTEGER), rows (1, 10)
   and (2, 20), sealed into a volume as the database closed.
2. snapshot_named_rowid: the same table and rows, PRAGMA SNAPSHOT, then
   ALTER TABLE t RENAME COLUMN _rowid_ TO b, closed. The table opens; its
   snapshot holds the column.
