Databases written by a build from before CREATE TABLE refused
AUTO_INCREMENT on a column other than the INTEGER PRIMARY KEY, read by
tests/persistence/auto_increment_column_test.rs. The current build cannot
write them: it refuses that CREATE TABLE.

Generator: commit c0ef5522, program generator.rs.txt placed at
examples/auto_increment_fixture.rs and run as

    cargo run --example auto_increment_fixture -- <out>

db.lock files were left out.

1. wal_column: table t (id INTEGER PRIMARY KEY, n INTEGER AUTO_INCREMENT,
   v TEXT), rows (1, NULL, 'a') and (2, NULL, 'b'), closed with
   checkpoint_on_close=off, so the table is in the WAL alone.
2. snapshot_column: the same table and rows, PRAGMA SNAPSHOT, then closed,
   so the table's schema is read back from the snapshot.
