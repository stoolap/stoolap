A database written by a build from before _rowid_ was reserved for the row
identity, read by tests/persistence/rowid_fixture_test.rs. The current build
cannot write it: it refuses a column of that name.

Generator: commit 7b798823, program generator.rs.txt placed at
examples/rowid_fixture.rs and run as

    cargo run --example rowid_fixture -- <out>

db.lock was left out.

column_named_rowid: table t (a INTEGER, _rowid_ INTEGER), rows (1, 10) and
(2, 20), sealed into a volume as the database closed.
