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

use stoolap::Database;

fn ints(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect()
}

/// `t (id, a)` with ids 1..=20 and `a = id`, and `u (id, a)` with `a = 21 - id`
fn tables(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    for i in 1..=20 {
        db.execute(&format!("INSERT INTO t VALUES ({i}, {i})"), ())
            .unwrap();
        db.execute(&format!("INSERT INTO u VALUES ({i}, {})", 21 - i), ())
            .unwrap();
    }
    db
}

#[test]
fn test_a_keyset_walk_starts_at_the_tightest_bound() {
    let db = tables("range_keyset");
    for (sql, want) in [
        (
            "SELECT id FROM t WHERE id > 5 AND 10 < id ORDER BY id LIMIT 3",
            vec![11, 12, 13],
        ),
        (
            "SELECT id FROM t WHERE id > 5 AND id >= 8 ORDER BY id LIMIT 2",
            vec![8, 9],
        ),
        (
            "SELECT id FROM t WHERE id >= 2.5 ORDER BY id LIMIT 2",
            vec![3, 4],
        ),
        (
            "SELECT id FROM t WHERE id > 5 AND id > 10 ORDER BY id DESC LIMIT 2",
            vec![20, 19],
        ),
    ] {
        assert_eq!(ints(&db, sql), want, "{sql}");
    }
}

#[test]
fn test_no_key_lies_above_the_largest_one() {
    let db = Database::open("memory://range_pk_max").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (9223372036854775807, 1), (1, 1)", ())
        .unwrap();
    let sql = "DELETE FROM t WHERE id > 9223372036854775807 AND id <= 9223372036854775807";
    assert_eq!(db.execute(sql, ()).unwrap(), 0, "{sql}");
    assert_eq!(ints(&db, "SELECT COUNT(*) FROM t"), vec![2]);
}

#[test]
fn test_an_outer_conjunct_beside_an_inner_one_of_the_same_name_is_kept() {
    let db = tables("range_outer_name");
    assert_eq!(
        ints(
            &db,
            "SELECT t.id FROM t WHERE t.id IN (SELECT u.id FROM u WHERE a > 15 AND t.a > 5) \
             ORDER BY t.id"
        ),
        Vec::<i64>::new(),
        "u.a > 15 holds for ids 1..5, t.a > 5 for none of them"
    );
    assert_eq!(
        ints(
            &db,
            "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a > 15 AND t.a > 5) > 0 \
             ORDER BY t.id"
        ),
        (6..=20).collect::<Vec<i64>>(),
    );
}

#[test]
fn test_a_quoted_column_is_no_constant_bound() {
    let db = Database::open("memory://range_quoted_column").unwrap();
    db.execute(
        "CREATE TABLE s (id INTEGER PRIMARY KEY, x TEXT, y TEXT)",
        (),
    )
    .unwrap();
    for i in 1..=5 {
        db.execute(&format!("INSERT INTO s VALUES ({i}, 'p{i}', 'p{i}')"), ())
            .unwrap();
    }
    for sql in [
        "SELECT id FROM s WHERE x = \"y\" || '' ORDER BY id",
        "SELECT id FROM s WHERE \"y\" || '' = x ORDER BY id",
    ] {
        assert_eq!(ints(&db, sql), vec![1, 2, 3, 4, 5], "{sql}");
    }
    assert_eq!(
        ints(
            &db,
            "SELECT \"1\" FROM generate_series(1, 3) AS g(\"1\") WHERE \"1\" = CAST(\"1\" AS INTEGER)"
        ),
        vec![1, 2, 3],
        "both sides read the series' own column"
    );
}

#[test]
fn test_a_narrowed_series_keeps_its_direction() {
    let db = Database::open("memory://range_series_direction").unwrap();
    for (sql, want) in [
        (
            "SELECT value FROM generate_series(1, 3) WHERE value >= 4",
            vec![],
        ),
        (
            "SELECT value FROM generate_series(1, 3) WHERE value >= 3.5",
            vec![],
        ),
        (
            "SELECT value FROM generate_series(3, 1) WHERE value <= 0",
            vec![],
        ),
        (
            "SELECT value FROM generate_series(1, 5) WHERE value < 3.5",
            vec![1, 2, 3],
        ),
        (
            "SELECT value FROM generate_series(5, 1) WHERE value >= 3",
            vec![5, 4, 3],
        ),
        (
            "SELECT value FROM generate_series(1, 10) WHERE value >= 2 AND value <= 1",
            vec![],
        ),
    ] {
        assert_eq!(ints(&db, sql), want, "{sql}");
    }
}

/// Two rows, its two arguments; any other count of arguments is an error
struct PairRange;

impl stoolap::functions::tvf::TableValuedFunction for PairRange {
    fn name(&self) -> &str {
        "PAIR_RANGE"
    }
    fn column_names(&self) -> Vec<String> {
        vec!["value".to_string()]
    }
    fn generate(
        &self,
        args: &[stoolap::core::Value],
        _limit: Option<usize>,
    ) -> stoolap::core::Result<stoolap::core::RowVec> {
        let [first, second] = args else {
            return Err(stoolap::core::Error::invalid_argument(
                "PAIR_RANGE takes two arguments",
            ));
        };
        let mut rows = stoolap::core::RowVec::new();
        rows.push((0, stoolap::core::Row::from_values(vec![first.clone()])));
        rows.push((1, stoolap::core::Row::from_values(vec![second.clone()])));
        Ok(rows)
    }
}

#[test]
fn test_only_a_series_is_narrowed() {
    stoolap::functions::global_registry()
        .register_tvf("PAIR_RANGE", std::sync::Arc::new(|| Box::new(PairRange)));
    let db = Database::open("memory://range_other_tvf").unwrap();
    assert_eq!(
        ints(&db, "SELECT value FROM pair_range(1, 3) WHERE value >= 0"),
        vec![1, 3]
    );
    assert_eq!(
        ints(&db, "SELECT value FROM pair_range(1, 3) WHERE value >= 2"),
        vec![3],
        "the bound is no new first argument for a function that is not a series"
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn test_a_narrowed_series_generates_only_its_range() {
    use stoolap::test_failpoints::tvf_rows;
    let db = Database::open("memory://range_series_generated").unwrap();
    for (sql, rows, generated) in [
        (
            "SELECT value FROM generate_series(1, 1000000) WHERE value >= 2 AND value <= 1",
            0,
            0,
        ),
        (
            "SELECT value FROM generate_series(1, 1000000) WHERE value BETWEEN 10 AND 12",
            3,
            3,
        ),
        (
            "SELECT value FROM generate_series(1000000, 1) WHERE value < 3.5",
            3,
            3,
        ),
    ] {
        let before = tvf_rows();
        assert_eq!(ints(&db, sql).len(), rows, "{sql}");
        assert_eq!(tvf_rows() - before, generated, "rows generated for {sql}");
    }
}
