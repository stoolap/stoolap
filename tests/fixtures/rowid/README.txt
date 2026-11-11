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
3. column_named_rowid_beside: the table and rows of 1, a view
   v AS SELECT _rowid_ AS x FROM t reading the column, and a table
   u (id INTEGER PRIMARY KEY, v INTEGER) holding (1, 100), both tables
   sealed into volumes as the database closed. Written at the same commit by
   generator_beside.rs.txt, placed at examples/rowid_fixture2.rs and run
   as cargo run --example rowid_fixture2 -- <out>.
4. snapshot_named_rowid_twice: tables t and t2, each (a INTEGER,
   _rowid_ INTEGER) with rows (1, 10) and (2, 20), PRAGMA SNAPSHOT, then
   both columns renamed to b, closed. Written at the same commit by
   generator_twice.rs.txt, placed at examples/rowid_fixture3.rs.
