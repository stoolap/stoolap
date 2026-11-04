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

//! Every path that builds an aggregate result answers as recorded, each
//! answer confirmed by another path when it was recorded, and every
//! path is exercised: a path added without a case here fails the coverage
//! check

#![cfg(feature = "test-failpoints")]

use stoolap::core::Value;
use stoolap::functions::{
    global_registry, FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, NativeFn1,
    ScalarFunction,
};
use stoolap::test_failpoints::{aggregate_paths, aggregate_paths_taken, reset_aggregate_paths};
use stoolap::Database;

mod oracle;

/// The uncorrelated subqueries the references hold, each written in as its
/// answer before a reference runs, an `IN` list when marked, so the
/// reference reads no subquery
const WRITTEN_IN: &[(&str, bool)] = &[
    ("SELECT a FROM t WHERE id = 3", false),
    ("SELECT a FROM t WHERE id = 1", false),
    ("SELECT g FROM t WHERE id = 1", false),
    ("SELECT a FROM t WHERE id > 1", true),
    ("SELECT 0", false),
    ("SELECT 1", false),
    ("SELECT 2", false),
    ("SELECT 5", false),
];

/// `reference` with each subquery `WRITTEN_IN` names written in over `db`
fn written_in(db: &Database, reference: &str) -> String {
    WRITTEN_IN
        .iter()
        .fold(reference.to_string(), |sql, &(subquery, list)| {
            match sql.contains(&format!("({subquery})")) {
                true => oracle::literal_twin(db, &sql, subquery, list),
                false => sql,
            }
        })
}

/// Where the first space outside every parenthesis stands in `text`
fn word_end(text: &str) -> usize {
    let mut depth = 0i32;
    text.char_indices()
        .find(|&(_, c)| {
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                _ => {}
            }
            c == ' ' && depth == 0
        })
        .map_or(text.len(), |(at, _)| at)
}

/// `sql` with its first source, at the top level, joined to one row of a
/// column no statement names, which leaves its rows as they are
fn joined(sql: &str) -> String {
    let (mut depth, mut from) = (0i32, None);
    for (at, c) in sql.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            _ if depth == 0 && sql[at..].starts_with(" FROM ") => {
                from = Some(at + " FROM ".len());
                break;
            }
            _ => {}
        }
    }
    let from = from.unwrap_or_else(|| panic!("{sql}: no FROM"));
    // The source ends at the first space outside its parentheses, and after
    // its name, the lowercase word after it, when it has one
    let end = from + word_end(&sql[from..]);
    let rest = sql[end..].strip_prefix(" AS ").unwrap_or(&sql[end..]);
    let rest = rest.trim_start();
    let end = match rest.starts_with(|c: char| c.is_ascii_lowercase()) {
        true => sql.len() - rest.len() + word_end(rest),
        false => end,
    };
    let (head, tail) = sql.split_at(end);
    format!("{head} CROSS JOIN (VALUES (0)) AS oracle_one(oracle_x){tail}")
}

/// (context, path the query takes there, query, the same query written so
/// that another path answers it, run to confirm an answer as it is
/// recorded; a correlated subquery written as a join)
const CASES: &[(&str, &str, &str, &str)] = &[
    (
        "memory",
        "stats",
        "SELECT SUM(a) AS x, COUNT(*), SUM(a) AS y, MIN(a), MAX(a) FROM t",
        "SELECT SUM(a) AS x, COUNT(*), SUM(a) AS y, MIN(a), MAX(a) FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "stats",
        "SELECT SUM(a) FROM t ORDER BY COUNT(*)",
        "SELECT SUM(a) FROM t WHERE id + 0 = id ORDER BY COUNT(*)",
    ),
    (
        "sealed",
        "storage_filtered",
        "SELECT SUM(a) AS x, COUNT(*), SUM(a) AS y, MIN(a) FROM t WHERE a > 0",
        "SELECT SUM(a) AS x, COUNT(*), SUM(a) AS y, MIN(a) FROM t WHERE a > 0 AND id + 0 = id",
    ),
    (
        "sealed",
        "storage_filtered",
        "SELECT MAX(a) FROM t WHERE a > 0 ORDER BY COUNT(*)",
        "SELECT MAX(a) FROM t WHERE a > 0 AND id + 0 = id ORDER BY COUNT(*)",
    ),
    (
        "sealed",
        "storage_filtered",
        "SELECT SUM(a) * 2 FROM t WHERE a > 0",
        "SELECT SUM(a) * 2 FROM t WHERE a > 0 AND id + 0 = id",
    ),
    (
        "memory",
        "stats",
        "SELECT SUM(a) * 2 AS x, SUM(a), SUM(a) * 2 AS y FROM t",
        "SELECT SUM(a) * 2 AS x, SUM(a), SUM(a) * 2 AS y FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "global_scan",
        "SELECT COUNT(a) * 2, COUNT(DISTINCT a), COUNT(a) * 2 AS z FROM t",
        "SELECT COUNT(a) * 2, COUNT(DISTINCT a), COUNT(a) * 2 AS z FROM t WHERE id + 0 = id",
    ),
    // A WHERE hot rows hold: the runner feeds the rows its read passes
    (
        "memory",
        "global_scan",
        "SELECT SUM(a) AS x, COUNT(*), SUM(a) AS y, MIN(a), MAX(a) FROM t WHERE a > 0",
        "SELECT SUM(a) AS x, COUNT(*), SUM(a) AS y, MIN(a), MAX(a) FROM t WHERE a > 0 \
         AND id + 0 = id",
    ),
    (
        "memory",
        "global_scan",
        "SELECT SUM(a) FROM t WHERE g = 2 ORDER BY COUNT(*)",
        "SELECT SUM(a) FROM t WHERE g = 2 AND id + 0 = id ORDER BY COUNT(*)",
    ),
    (
        "memory",
        "global_scan",
        "SELECT COUNT(DISTINCT a), SUM(a + 1), COUNT(*) FILTER (WHERE a > 1) FROM t WHERE g > 0",
        "SELECT COUNT(DISTINCT a), SUM(a + 1), COUNT(*) FILTER (WHERE a > 1) FROM t \
         WHERE g > 0 AND id + 0 = id",
    ),
    (
        "memory",
        "global_scan",
        "SELECT SUM(a), COUNT(a), MAX(s) FROM t WHERE a > 100",
        "SELECT SUM(a), COUNT(a), MAX(s) FROM t WHERE a > 100 AND id + 0 = id",
    ),
    (
        "memory",
        "global_scan",
        "SELECT SUM(a), COUNT(*), MAX(s) FROM t WHERE id = 4",
        "SELECT SUM(a), COUNT(*), MAX(s) FROM t WHERE id = 4 AND id + 0 = id",
    ),
    (
        "memory",
        "global_scan",
        "SELECT COUNT(*), SUM(a) FROM t WHERE 1 = 0",
        "SELECT COUNT(*), SUM(a) FROM t WHERE 1 = 0 AND id + 0 = id",
    ),
    (
        "memory",
        "global_scan",
        "SELECT 1 + COUNT(*), MIN(s) FROM t WHERE g = 1",
        "SELECT 1 + COUNT(*), MIN(s) FROM t WHERE g = 1 AND id + 0 = id",
    ),
    (
        "memory",
        "storage_grouped",
        "SELECT SUM(a) AS x, g, SUM(a) AS y, COUNT(*) FROM t GROUP BY g",
        "SELECT SUM(a) AS x, g, SUM(a) AS y, COUNT(*) FROM t WHERE id + 0 = id GROUP BY g",
    ),
    // An ORDER BY naming a SELECT alias reads the grouped slots all the same
    (
        "memory",
        "storage_grouped",
        "SELECT g, COUNT(*) AS n FROM t GROUP BY g ORDER BY n, g",
        "SELECT g, COUNT(*) AS n FROM t WHERE id + 0 = id GROUP BY g ORDER BY n, g",
    ),
    // A WHERE holding subqueries passes the rows the runner aggregates
    (
        "memory",
        "hash_groups",
        "SELECT g, COUNT(*), SUM(a) FROM t WHERE a IN (SELECT a FROM t WHERE id > 1) GROUP BY g",
        "SELECT g, COUNT(*), SUM(a) FROM t WHERE id + 0 = id \
         AND a IN (SELECT a FROM t WHERE id > 1) GROUP BY g",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT g, MAX(a) FROM t AS o \
         WHERE EXISTS (SELECT 1 FROM t WHERE t.g = o.g AND t.id <> o.id) GROUP BY g",
        "SELECT o.g AS g, MAX(o.a) AS \"MAX(a)\" FROM t AS o \
         JOIN (SELECT g FROM t WHERE id + 0 = id GROUP BY g HAVING COUNT(*) > 1) k \
         ON k.g = o.g WHERE o.id + 0 = o.id GROUP BY o.g",
    ),
    (
        "sealed",
        "global_scan",
        "SELECT COUNT(*), SUM(a) FROM t WHERE a IN (SELECT a FROM t WHERE id > 1)",
        "SELECT COUNT(*), SUM(a) FROM t WHERE id + 0 = id \
         AND a IN (SELECT a FROM t WHERE id > 1)",
    ),
    (
        "empty",
        "hash_groups",
        "SELECT g, COUNT(*) FROM t WHERE a IN (SELECT a FROM t) GROUP BY g",
        "SELECT g, COUNT(*) FROM t WHERE id + 0 = id AND a IS NOT NULL GROUP BY g",
    ),
    // Only COUNT counts the rows of a star; any other call of one is grouped
    // from the rows the read passes
    (
        "memory",
        "hash_groups",
        "SELECT g, SUM(*) FROM t GROUP BY g LIMIT 1",
        "SELECT g, SUM(*) FROM t WHERE id + 0 = id GROUP BY g LIMIT 1",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g, MAX(*) FROM t GROUP BY g LIMIT 1",
        "SELECT g, MAX(*) FROM t WHERE id + 0 = id GROUP BY g LIMIT 1",
    ),
    (
        "sealed",
        "storage_grouped",
        "SELECT g, MAX(a), MAX(a) AS m FROM t GROUP BY g ORDER BY COUNT(*)",
        "SELECT g, MAX(a), MAX(a) AS m FROM t WHERE id + 0 = id GROUP BY g ORDER BY COUNT(*)",
    ),
    // A HAVING the index walk does not judge judges the groups the table gives
    (
        "memory",
        "storage_grouped",
        "SELECT g, SUM(a) FROM t GROUP BY g HAVING SUM(a) > 3",
        "SELECT g, SUM(a) FROM t WHERE id + 0 = id GROUP BY g HAVING SUM(a) > 3",
    ),
    (
        "sealed",
        "storage_grouped",
        "SELECT g, SUM(a) FROM t GROUP BY g HAVING COUNT(*) > 1 AND g > 1",
        "SELECT g, SUM(a) FROM t WHERE id + 0 = id GROUP BY g HAVING COUNT(*) > 1 AND g > 1",
    ),
    (
        "empty",
        "storage_grouped",
        "SELECT g, MAX(a) FROM t GROUP BY g HAVING MAX(a) IS NULL",
        "SELECT g, MAX(a) FROM t WHERE id + 0 = id GROUP BY g HAVING MAX(a) IS NULL",
    ),
    (
        "memory",
        "streaming_group_by",
        "SELECT COUNT(*) AS n, g, COUNT(*) FROM t GROUP BY g HAVING COUNT(*) > 0",
        "SELECT COUNT(*) AS n, g, COUNT(*) FROM t WHERE id + 0 = id GROUP BY g \
         HAVING COUNT(*) > 0",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT COUNT(*) AS n, s, COUNT(*) FROM (SELECT s FROM t) d GROUP BY s",
        "SELECT COUNT(*) AS n, s, COUNT(*) FROM (SELECT s FROM t) d WHERE s = s GROUP BY s",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT s, MIN(a), MIN(a) AS lo FROM (SELECT s, a FROM t) d GROUP BY s",
        "SELECT s, MIN(a), MIN(a) AS lo FROM (SELECT s, a FROM t) d WHERE s = s GROUP BY s",
    ),
    (
        "memory",
        "storage_grouped",
        "SELECT g AS n, COUNT(*) AS n FROM t GROUP BY g",
        "SELECT g AS n, COUNT(*) AS n FROM t WHERE id + 0 = id GROUP BY g",
    ),
    (
        "memory",
        "streaming_group_by",
        "SELECT g, COUNT(*) FROM t GROUP BY g HAVING MIN(s) > 0 LIMIT 10",
        "SELECT g, COUNT(*) FROM t WHERE id + 0 = id GROUP BY g HAVING MIN(s) > 0 LIMIT 10",
    ),
    (
        "memory",
        "streaming_group_by",
        "SELECT g, COUNT(*) FROM t GROUP BY g HAVING g > 1 OR COUNT(*) + 0 >= 2 * MIN(s) LIMIT 10",
        "SELECT g, COUNT(*) FROM t WHERE id + 0 = id GROUP BY g \
         HAVING g > 1 OR COUNT(*) + 0 >= 2 * MIN(s) LIMIT 10",
    ),
    (
        "memory",
        "streaming_group_by",
        "SELECT g, MAX(s), MIN(a) FROM t GROUP BY g HAVING COUNT(*) > 0 LIMIT 10",
        "SELECT g, MAX(s), MIN(a) FROM t WHERE id + 0 = id GROUP BY g HAVING COUNT(*) > 0 \
         LIMIT 10",
    ),
    (
        "memory",
        "storage_grouped",
        "SELECT g AS n, COUNT(*) AS g FROM t GROUP BY g",
        "SELECT g AS n, COUNT(*) AS g FROM t WHERE id + 0 = id GROUP BY g",
    ),
    (
        "memory",
        "streaming_group_by",
        "SELECT g AS n, COUNT(*) AS g FROM t GROUP BY g HAVING COUNT(*) > 0 LIMIT 10",
        "SELECT g AS n, COUNT(*) AS g FROM t WHERE id + 0 = id GROUP BY g \
         HAVING COUNT(*) > 0 LIMIT 10",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT s AS n, COUNT(*) AS s FROM (SELECT s FROM t) d GROUP BY s",
        "SELECT s AS n, COUNT(*) AS s FROM (SELECT s FROM t) d WHERE s = s GROUP BY s",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT COUNT(*) FROM (SELECT CAST(a AS TEXT) AS \"x.s\", s FROM t) d GROUP BY \"x.s\"",
        "SELECT COUNT(*) FROM (SELECT CAST(a AS TEXT) AS \"x.s\", s FROM t) d WHERE s = s \
         GROUP BY \"x.s\"",
    ),
    (
        "memory",
        "storage_grouped",
        "SELECT a AS g, COUNT(*) FROM t GROUP BY g",
        "SELECT a AS g, COUNT(*) FROM t WHERE id + 0 = id GROUP BY g",
    ),
    (
        "memory",
        "index_bound",
        "SELECT MIN(g) FROM t",
        "SELECT MIN(g) FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "index_bound",
        "SELECT MAX(t.g) FROM t",
        "SELECT MAX(t.g) FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "index_bound",
        "SELECT MAX(g) + 1 AS m FROM t",
        "SELECT MAX(g) + 1 AS m FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "stats",
        "SELECT COUNT(*) AS n FROM t",
        "SELECT COUNT(*) AS n FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "stats",
        "SELECT COUNT(1), COUNT('x') AS n FROM t",
        "SELECT COUNT(1), COUNT('x') AS n FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "stats",
        "SELECT COUNT(DISTINCT g) FROM t",
        "SELECT COUNT(DISTINCT g) FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "stats",
        "SELECT COUNT(DISTINCT g), SUM(a) FROM t",
        "SELECT COUNT(DISTINCT g), SUM(a) FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "global_scan",
        "SELECT COUNT(DISTINCT s), COUNT(a) * 2 FROM t",
        "SELECT COUNT(DISTINCT s), COUNT(a) * 2 FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "prepared_block",
        "SELECT id FROM t WHERE a <= (SELECT MAX(a) FROM t AS u WHERE u.g = t.g AND u.id + 0 = u.id)",
        "SELECT t.id AS id FROM t JOIN (SELECT g, MAX(a) AS m FROM t WHERE id + 0 = id GROUP BY g) u \
         ON u.g = t.g WHERE t.a <= u.m",
    ),
    (
        "sealed",
        "prepared_block",
        "SELECT id FROM t WHERE 1 < (SELECT COUNT(a) FROM t AS u WHERE u.g = t.g AND u.id + 0 = u.id)",
        "SELECT t.id AS id FROM t JOIN (SELECT g, COUNT(a) AS c FROM t WHERE id + 0 = id GROUP BY g) u \
         ON u.g = t.g WHERE 1 < u.c",
    ),
    (
        "memory",
        "index_count",
        "SELECT id FROM t WHERE 1 < (SELECT COUNT(*) FROM t AS u WHERE u.g = t.g) LIMIT 10",
        "SELECT t.id AS id FROM t JOIN (SELECT g, COUNT(*) AS c FROM t WHERE id + 0 = id GROUP BY g) u \
         ON u.g = t.g WHERE 1 < u.c LIMIT 10",
    ),
    (
        "memory",
        "index_count",
        "SELECT id, (SELECT COUNT(*) FROM t AS u WHERE u.g = t.a) AS n FROM t LIMIT 10",
        "SELECT t.id AS id, COALESCE(u.c, 0) AS n FROM t \
         LEFT JOIN (SELECT g, COUNT(*) AS c FROM t WHERE id + 0 = id GROUP BY g) u ON u.g = t.a LIMIT 10",
    ),
    (
        "memory",
        "index_count",
        "SELECT id FROM t WHERE 4 = (SELECT COUNT(*) + COUNT(*) FROM t AS u WHERE u.g = t.g \
         HAVING COUNT(*) > 1) LIMIT 10",
        "SELECT t.id AS id FROM t JOIN (SELECT g, COUNT(*) AS c FROM t WHERE id + 0 = id GROUP BY g) u \
         ON u.g = t.g WHERE u.c = 2 LIMIT 10",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g, s, COUNT(*), SUM(a) FROM t WHERE a > 0 GROUP BY g, s",
        "SELECT g, s, COUNT(*), SUM(a) FROM t WHERE a > 0 AND id + 0 = id GROUP BY g, s",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT g, s, COUNT(*), SUM(a) FROM t WHERE a > 0 GROUP BY g, s",
        "SELECT g, s, COUNT(*), SUM(a) FROM t WHERE a > 0 AND id + 0 = id GROUP BY g, s",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g + 1 AS k, AVG(a), MIN(s) FROM t GROUP BY g + 1 HAVING COUNT(a) > 0",
        "SELECT g + 1 AS k, AVG(a), MIN(s) FROM t WHERE id + 0 = id GROUP BY g + 1 \
         HAVING COUNT(a) > 0",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT g, SUM(a) FROM t GROUP BY ROLLUP (g, s) ORDER BY 1, 2",
        "SELECT g, SUM(a) FROM t WHERE id + 0 = id GROUP BY ROLLUP (g, s) ORDER BY 1, 2",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g, s, COUNT(*), GROUPING(g) FROM t GROUP BY CUBE (g, s)",
        "SELECT g, s, COUNT(*), GROUPING(g) FROM t WHERE id + 0 = id GROUP BY CUBE (g, s)",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g, s, SUM(a) FROM t GROUP BY GROUPING SETS ((g), (s), ())",
        "SELECT g, s, SUM(a) FROM t WHERE id + 0 = id GROUP BY GROUPING SETS ((g), (s), ())",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT DISTINCT COUNT(*) FROM t WHERE a > 0 GROUP BY g",
        "SELECT DISTINCT COUNT(*) FROM t WHERE a > 0 AND id + 0 = id GROUP BY g",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g FROM t WHERE a IS NOT NULL GROUP BY g ORDER BY SUM(a) DESC LIMIT 1 OFFSET 1",
        "SELECT g FROM t WHERE a IS NOT NULL AND id + 0 = id GROUP BY g ORDER BY SUM(a) DESC \
         LIMIT 1 OFFSET 1",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT s, COUNT(DISTINCT a), STRING_AGG(s, ',' ORDER BY id) FROM t GROUP BY s",
        "SELECT s, COUNT(DISTINCT a), STRING_AGG(s, ',' ORDER BY id) FROM t \
         WHERE id + 0 = id GROUP BY s",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT COUNT(*), SUM(a) FROM t HAVING SUM(a) > 1",
        "SELECT COUNT(*), SUM(a) FROM t WHERE id + 0 = id HAVING SUM(a) > 1",
    ),
    (
        "memory",
        "stats",
        "SELECT DISTINCT SUM(a) > 0 FROM t",
        "SELECT DISTINCT SUM(a) > 0 FROM t WHERE id + 0 = id",
    ),
    (
        "empty",
        "hash_groups",
        "SELECT COUNT(*), MAX(a) FROM t HAVING COUNT(*) = 0",
        "SELECT COUNT(*), MAX(a) FROM t WHERE id + 0 = id HAVING COUNT(*) = 0",
    ),
    (
        "empty",
        "hash_groups",
        "SELECT g, COUNT(*) FROM t WHERE a > 0 GROUP BY g",
        "SELECT g, COUNT(*) FROM t WHERE a > 0 AND id + 0 = id GROUP BY g",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT 42 AS answer, COUNT(*) FROM t WHERE id = 1 GROUP BY GROUPING SETS (())",
        "SELECT 42 AS answer, COUNT(*) FROM t WHERE id = 1 AND id + 0 = id \
         GROUP BY GROUPING SETS (())",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g, COUNT(*) FROM t WHERE FALSE GROUP BY g",
        "SELECT g, COUNT(*) FROM t WHERE FALSE AND id + 0 = id GROUP BY g",
    ),
    (
        "empty",
        "hash_groups",
        "SELECT COUNT(*) AS n, s, COUNT(*) AS m FROM (SELECT s FROM t) d GROUP BY s",
        "SELECT COUNT(*) AS n, s, COUNT(*) AS m FROM (SELECT s FROM t) d WHERE s = s GROUP BY s",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT COUNT(*), SUM(a), MAX(s) FROM (SELECT s, a FROM t WHERE a > 0) d",
        "SELECT COUNT(*), SUM(a), MAX(s) FROM (SELECT s, a FROM t WHERE a > 0) d WHERE s = s",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT g, SUM(a) FROM (SELECT g, a, s FROM t) d WHERE a > 0 GROUP BY g \
         HAVING COUNT(*) > 0 ORDER BY g",
        "SELECT g, SUM(a) FROM (SELECT g, a, s FROM t) d WHERE a > 0 AND s = s GROUP BY g \
         HAVING COUNT(*) > 0 ORDER BY g",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g, COUNT(*), MIN(s) FROM tv GROUP BY g",
        "SELECT g, COUNT(*), MIN(s) FROM tv WHERE s = s GROUP BY g",
    ),
    (
        "sealed",
        "hash_groups",
        "WITH c AS (SELECT g, s, a FROM t) SELECT g, s, SUM(a) FROM c GROUP BY ROLLUP (g, s)",
        "WITH c AS (SELECT g, s, a FROM t) SELECT g, s, SUM(a) FROM c WHERE s = s \
         GROUP BY ROLLUP (g, s)",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT x, COUNT(*), SUM(y) FROM (VALUES (1, 2), (1, 3), (2, NULL)) AS v(x, y) GROUP BY x",
        "SELECT x, COUNT(*), SUM(y) FROM (VALUES (1, 2), (1, 3), (2, NULL)) AS v(x, y) \
         WHERE x = x GROUP BY x",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT value % 3 AS r, COUNT(*) FROM generate_series(1, 10) GROUP BY value % 3",
        "SELECT value % 3 AS r, COUNT(*) FROM generate_series(1, 10) WHERE value = value \
         GROUP BY value % 3",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT DISTINCT COUNT(*) FROM (SELECT g FROM t) d GROUP BY g",
        "SELECT DISTINCT COUNT(*) FROM (SELECT g FROM t) d WHERE g = g GROUP BY g",
    ),
    (
        "empty",
        "hash_groups",
        "SELECT COUNT(*), SUM(a) FROM (SELECT a FROM t) d",
        "SELECT COUNT(*), SUM(a) FROM (SELECT a FROM t) d WHERE a = a",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g, SUM(a), RANK() OVER (ORDER BY SUM(a) DESC) FROM t GROUP BY g",
        "SELECT g, SUM(a), RANK() OVER (ORDER BY SUM(a) DESC) FROM t WHERE id + 0 = id \
         GROUP BY g",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT g, s, SUM(a), ROW_NUMBER() OVER (ORDER BY g, s) FROM t GROUP BY ROLLUP (g, s)",
        "SELECT g, s, SUM(a), ROW_NUMBER() OVER (ORDER BY g, s) FROM t WHERE id + 0 = id \
         GROUP BY ROLLUP (g, s)",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g, s, COUNT(*), COUNT(*) OVER (PARTITION BY GROUPING(s)) FROM t \
         GROUP BY CUBE (g, s)",
        "SELECT g, s, COUNT(*), COUNT(*) OVER (PARTITION BY GROUPING(s)) FROM t \
         WHERE id + 0 = id GROUP BY CUBE (g, s)",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT g, SUM(a), SUM(SUM(a)) OVER () FROM (SELECT g, a, s FROM t) d \
         WHERE a > 0 GROUP BY g HAVING COUNT(*) > 0 ORDER BY g LIMIT 1",
        "SELECT g, SUM(a), SUM(SUM(a)) OVER () FROM (SELECT g, a, s FROM t) d \
         WHERE a > 0 AND s = s GROUP BY g HAVING COUNT(*) > 0 ORDER BY g LIMIT 1",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT COUNT(*), ROW_NUMBER() OVER () FROM t WHERE a > 0",
        "SELECT COUNT(*), ROW_NUMBER() OVER () FROM t WHERE a > 0 AND id + 0 = id",
    ),
    (
        "empty",
        "hash_groups",
        "SELECT g, COUNT(*), RANK() OVER (ORDER BY g) FROM t GROUP BY g",
        "SELECT g, COUNT(*), RANK() OVER (ORDER BY g) FROM t WHERE id + 0 = id GROUP BY g",
    ),
    // A statement whose output asks its subqueries, over its table
    (
        "memory",
        "hash_groups",
        "SELECT g, SUM(a), (SELECT a FROM t WHERE id = 3) AS n FROM t GROUP BY g",
        "SELECT g, SUM(a), (SELECT a FROM t WHERE id = 3) AS n FROM t GROUP BY g",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT g, SUM(a), (SELECT a FROM t WHERE id = 3) AS n FROM t GROUP BY g",
        "SELECT g, SUM(a), (SELECT a FROM t WHERE id = 3) AS n FROM t GROUP BY g",
    ),
    (
        "sealed",
        "storage_grouped",
        "SELECT g, SUM(a) FROM t GROUP BY g HAVING SUM(a) > (SELECT a FROM t WHERE id = 1)",
        "SELECT g, SUM(a) FROM t WHERE id + 0 = id GROUP BY g \
         HAVING SUM(a) > (SELECT a FROM t WHERE id = 1)",
    ),
    (
        "memory",
        "storage_grouped",
        "SELECT g, SUM(a) FROM t GROUP BY g HAVING SUM(a) > (SELECT a FROM t WHERE id = 1)",
        "SELECT g, SUM(a) FROM t WHERE id + 0 = id GROUP BY g \
         HAVING SUM(a) > (SELECT a FROM t WHERE id = 1)",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT g, SUM(a) AS s FROM t GROUP BY g \
         ORDER BY (SELECT a FROM t WHERE id = 3) - SUM(a)",
        "SELECT g, SUM(a) AS s FROM t WHERE id + 0 = id GROUP BY g \
         ORDER BY (SELECT a FROM t WHERE id = 3) - SUM(a)",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g, SUM(a), RANK() OVER (ORDER BY SUM(a) * (SELECT 2)) FROM t GROUP BY g",
        "SELECT g, SUM(a), RANK() OVER (ORDER BY SUM(a) * (SELECT 2)) FROM t GROUP BY g",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT g, SUM(a + (SELECT 1)) FROM t WHERE a > 0 GROUP BY g",
        "SELECT g, SUM(a + (SELECT 1)) FROM t WHERE a > 0 GROUP BY g",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g, COUNT(*), (SELECT 5) FROM t WHERE a < (SELECT a FROM t WHERE id = 3) \
         GROUP BY g",
        "SELECT g, COUNT(*), (SELECT 5) FROM t WHERE a < (SELECT a FROM t WHERE id = 3) \
         GROUP BY g",
    ),
    (
        "empty",
        "hash_groups",
        "SELECT g, COUNT(*), (SELECT a FROM t WHERE id = 3) FROM t GROUP BY g",
        "SELECT g, COUNT(*), (SELECT a FROM t WHERE id = 3) FROM t GROUP BY g",
    ),
    (
        "memory",
        "global_scan",
        "SELECT COUNT(*), (SELECT a FROM t WHERE id = 3) FROM t WHERE a > 0",
        "SELECT COUNT(*), (SELECT a FROM t WHERE id = 3) FROM t WHERE a > 0",
    ),
    (
        "sealed",
        "stats",
        "SELECT COUNT(*), SUM(a), (SELECT g FROM t WHERE id = 1) FROM t",
        "SELECT COUNT(*), SUM(a), (SELECT g FROM t WHERE id = 1) FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "global_scan",
        "SELECT COUNT(*), (SELECT 1) FROM t WHERE 1 = 0",
        "SELECT COUNT(*), (SELECT 1) FROM t WHERE 1 = 0",
    ),
    // The same over another source's rows, and over a join's
    (
        "memory",
        "hash_groups",
        "SELECT g, SUM(a), (SELECT a FROM t WHERE id = 3) FROM (SELECT g, a FROM t) d GROUP BY g",
        "SELECT g, SUM(a), (SELECT a FROM t WHERE id = 3) FROM (SELECT g, a FROM t) d GROUP BY g",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT g, COUNT(*), (SELECT 5) FROM (SELECT g, a FROM t) d \
         WHERE a < (SELECT a FROM t WHERE id = 3) GROUP BY g",
        "SELECT g, COUNT(*), (SELECT 5) FROM (SELECT g, a FROM t) d \
         WHERE a < (SELECT a FROM t WHERE id = 3) GROUP BY g",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT g, COUNT(*), (SELECT a FROM t WHERE id = 3) FROM tv GROUP BY g \
         HAVING COUNT(*) > (SELECT 0)",
        "SELECT g, COUNT(*), (SELECT a FROM t WHERE id = 3) FROM tv GROUP BY g \
         HAVING COUNT(*) > (SELECT 0)",
    ),
    (
        "memory",
        "hash_groups",
        "WITH c AS (SELECT g, a FROM t) SELECT g, SUM(a), (SELECT 5) FROM c GROUP BY g",
        "WITH c AS (SELECT g, a FROM t) SELECT g, SUM(a), (SELECT 5) FROM c GROUP BY g",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT x, COUNT(*), (SELECT 5) FROM (VALUES (1), (1), (2)) v(x) GROUP BY x",
        "SELECT x, COUNT(*), (SELECT 5) FROM (VALUES (1), (1), (2)) v(x) GROUP BY x",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT value % 2, COUNT(*), (SELECT 5) FROM generate_series(1, 6) GROUP BY value % 2",
        "SELECT value % 2, COUNT(*), (SELECT 5) FROM generate_series(1, 6) GROUP BY value % 2",
    ),
    (
        "empty",
        "hash_groups",
        "SELECT COUNT(*), (SELECT 5) FROM (SELECT a FROM t) d",
        "SELECT COUNT(*), (SELECT 5) FROM (SELECT a FROM t) d",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT t.g, SUM(t.a), (SELECT a FROM t WHERE id = 3) FROM t JOIN t t2 ON t2.id = t.id \
         GROUP BY t.g",
        "SELECT t.g, SUM(t.a), (SELECT a FROM t WHERE id = 3) FROM t JOIN t t2 ON t2.id = t.id \
         GROUP BY t.g",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT t.g, COUNT(*), RANK() OVER (ORDER BY COUNT(*) * (SELECT 2)) FROM t \
         JOIN t t2 ON t2.id = t.id GROUP BY t.g",
        "SELECT t.g, COUNT(*), RANK() OVER (ORDER BY COUNT(*) * (SELECT 2)) FROM t \
         JOIN t t2 ON t2.id = t.id GROUP BY t.g",
    ),
    // A DISTINCT ON over the groups, its keys computed over them
    (
        "memory",
        "hash_groups",
        "SELECT DISTINCT ON (s) s, g, SUM(a) FROM t GROUP BY s, g ORDER BY s, g DESC",
        "SELECT DISTINCT ON (s) s, g, SUM(a) FROM t WHERE id + 0 = id GROUP BY s, g \
         ORDER BY s, g DESC",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT DISTINCT ON (SUM(a) > 2) g FROM t GROUP BY g",
        "SELECT DISTINCT ON (SUM(a) > 2) g FROM t WHERE id + 0 = id GROUP BY g",
    ),
    (
        "sealed",
        "storage_grouped",
        "SELECT DISTINCT ON (COUNT(*)) g, COUNT(*) FROM t GROUP BY g ORDER BY COUNT(*), g",
        "SELECT DISTINCT ON (COUNT(*)) g, COUNT(*) FROM t WHERE id + 0 = id GROUP BY g \
         ORDER BY COUNT(*), g",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT DISTINCT ON (s) s, COUNT(*), RANK() OVER (ORDER BY s) FROM (SELECT s, a FROM t) d \
         GROUP BY s ORDER BY s",
        "SELECT DISTINCT ON (s) s, COUNT(*), RANK() OVER (ORDER BY s) FROM (SELECT s, a FROM t) d \
         WHERE s = s GROUP BY s ORDER BY s",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT DISTINCT ON (g % 2) g, MAX(a) FROM tv GROUP BY g HAVING COUNT(*) > 0 \
         ORDER BY g % 2, g DESC LIMIT 1",
        "SELECT DISTINCT ON (g % 2) g, MAX(a) FROM tv WHERE g = g GROUP BY g HAVING COUNT(*) > 0 \
         ORDER BY g % 2, g DESC LIMIT 1",
    ),
    (
        "empty",
        "storage_grouped",
        "SELECT DISTINCT ON (s) s, COUNT(*) FROM t GROUP BY s",
        "SELECT DISTINCT ON (s) s, COUNT(*) FROM t WHERE id + 0 = id GROUP BY s",
    ),
];

fn open(db: Database, seal: bool, rows: bool) -> Database {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, s TEXT, a INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_g ON t(g)", ()).unwrap();
    db.execute("CREATE VIEW tv AS SELECT g, s, a FROM t WHERE id > 0", ())
        .unwrap();
    if rows {
        db.execute(
            "INSERT INTO t VALUES (1, 1, '1', 1), (2, 1, '1', 2), (3, 2, '2', 10), \
             (4, 2, '2', NULL)",
            (),
        )
        .unwrap();
    }
    if seal {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// A value with its type, so INTEGER 3 and FLOAT 3.0 differ
fn render(value: &Value) -> String {
    match value {
        Value::Integer(n) => format!("i:{n}"),
        Value::Float(f) => format!("f:{f:?}"),
        other => format!("{other}"),
    }
}

/// The column names and the rows sorted, or the error
type Answer = Result<(Vec<String>, Vec<String>), String>;

/// The answer to `sql` and the paths that answered it
fn run(db: &Database, sql: &str) -> (Answer, Vec<&'static str>) {
    reset_aggregate_paths();
    let answer = db
        .query(sql, ())
        .map_err(|e| e.to_string())
        .and_then(|rows| {
            let columns = rows.columns().to_vec();
            let mut out = Vec::new();
            for row in rows {
                let row = row.map_err(|e| e.to_string())?;
                out.push(
                    (0..row.len())
                        .map(|i| render(row.get_value(i).unwrap()))
                        .collect::<Vec<_>>()
                        .join("|"),
                );
            }
            out.sort();
            Ok((columns, out))
        });
    (answer, aggregate_paths_taken())
}

const RECORDED: oracle::Answers = answers!("aggregate_path_equivalence");

#[test]
fn every_aggregate_path_answers_as_recorded() {
    let memory = open(
        Database::open("memory://aggregate_path_equivalence").unwrap(),
        false,
        true,
    );
    let empty = open(
        Database::open("memory://aggregate_path_equivalence_empty").unwrap(),
        false,
        false,
    );
    let dir = tempfile::tempdir().unwrap();
    let sealed = open(
        Database::open(&format!(
            "file://{}?checkpoint_interval=0",
            dir.path().display()
        ))
        .unwrap(),
        true,
        true,
    );
    let mut wrong = Vec::new();
    let mut covered: Vec<&str> = Vec::new();
    for &(ctx, path, sql, reference) in CASES {
        let db = match ctx {
            "sealed" => &sealed,
            "empty" => &empty,
            _ => &memory,
        };
        let (got, taken) = run(db, sql);
        // A value written in names its column by the value: the rows compare
        let named = !WRITTEN_IN
            .iter()
            .any(|(subquery, _)| reference.contains(&format!("({subquery})")));
        let lines = |answer: Answer| {
            answer.map(|(columns, rows)| {
                let names = named.then(|| format!("columns: {}", columns.join("|")));
                names.into_iter().chain(rows).collect::<Vec<_>>()
            })
        };
        if taken != [path] {
            wrong.push(format!("{ctx}: {sql}: took {taken:?}, expected [{path}]"));
        }
        let key = format!("{ctx}: {sql}");
        // Recorded only as the reference, aggregated on another path,
        // answers it; a reference on the same path alone is read over a
        // join with one row
        let confirm = || {
            let other = |taken: &[&str]| taken.iter().any(|p| *p != path);
            let mut written = written_in(db, reference);
            let (mut want, mut reference_taken) = run(db, &written);
            if !other(&reference_taken) {
                written = joined(&written);
                (want, reference_taken) = run(db, &written);
            }
            assert!(
                other(&reference_taken),
                "{ctx}: {written}: took {reference_taken:?}"
            );
            lines(want)
        };
        if let Some(error) = oracle::check_confirmed(&RECORDED, &key, &lines(got), confirm) {
            wrong.push(error);
        }
        covered.extend(taken);
    }
    // The general path answers the references, and only when recording
    for path in aggregate_paths().into_iter().filter(|p| *p != "general") {
        if !covered.contains(&path) {
            wrong.push(format!("no case takes the {path} path"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_filtered_aggregate_keeps_every_conjunct_with_literals_and_parameters() {
    let int = Value::Integer;
    // Each WHERE with its parameters; the literal form writes them in
    let cases: &[(&str, Vec<Value>, &str)] = &[
        (
            "a > $1 AND a NOT BETWEEN $2 AND $3",
            vec![int(0), int(1), int(2)],
            "a > 0 AND a NOT BETWEEN 1 AND 2",
        ),
        (
            "id > $1 AND id NOT BETWEEN $2 AND $3",
            vec![int(0), int(1), int(2)],
            "id > 0 AND id NOT BETWEEN 1 AND 2",
        ),
        (
            "a > $1 AND a BETWEEN $2 AND $3",
            vec![int(0), int(2), int(10)],
            "a > 0 AND a BETWEEN 2 AND 10",
        ),
        (
            "a > $1 AND a NOT IN ($2, $3)",
            vec![int(0), int(1), int(10)],
            "a > 0 AND a NOT IN (1, 10)",
        ),
        (
            "a > $1 AND a IN ($2, $3)",
            vec![int(0), int(1), int(10)],
            "a > 0 AND a IN (1, 10)",
        ),
        ("a > $1 AND g IS NULL", vec![int(0)], "a > 0 AND g IS NULL"),
        (
            "a > $1 AND a <> $2",
            vec![int(0), int(2)],
            "a > 0 AND a <> 2",
        ),
    ];
    let dir = tempfile::tempdir().unwrap();
    let sealed = open(
        Database::open(&format!(
            "file://{}?checkpoint_interval=0",
            dir.path().display()
        ))
        .unwrap(),
        true,
        true,
    );
    let hot = open(
        Database::open("memory://aggregate_conjuncts_hot").unwrap(),
        false,
        true,
    );
    let mut wrong = Vec::new();
    for (ctx, db) in [("hot", &hot), ("sealed", &sealed)] {
        for (with_params, params, literal) in cases {
            let select = "SELECT COUNT(*), SUM(a), MIN(id) FROM t WHERE";
            let general = run(db, &format!("{select} {literal} AND id + 0 = id")).0;
            let written = run(db, &format!("{select} {literal}")).0;
            let prepared = db
                .query(&format!("{select} {with_params}"), params.clone())
                .map_err(|e| e.to_string())
                .map(|rows| {
                    let rows: Vec<String> = rows
                        .map(|row| {
                            let row = row.unwrap();
                            (0..row.len())
                                .map(|i| render(row.get_value(i).unwrap()))
                                .collect::<Vec<_>>()
                                .join("|")
                        })
                        .collect();
                    rows
                });
            let general_rows = general.clone().map(|(_, rows)| rows);
            if written != general {
                wrong.push(format!(
                    "{ctx}: {literal}: {written:?}, general {general:?}"
                ));
            }
            if prepared != general_rows {
                wrong.push(format!(
                    "{ctx}: {with_params} {params:?}: {prepared:?}, general {general_rows:?}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_having_that_calls_a_scalar_function_is_not_judged_inside_the_index_walk() {
    let db = open(
        Database::open("memory://aggregate_path_having_scalar_call").unwrap(),
        false,
        true,
    );
    let sql = "SELECT g, COUNT(*) FROM t GROUP BY g HAVING SLEEP(0) IS NOT NULL LIMIT 10";
    let (got, taken) = run(&db, sql);
    assert_eq!(
        taken,
        ["hash_groups"],
        "a memory table's B-tree walk holds its index lock while it judges HAVING"
    );
    let (plain, _) = run(&db, "SELECT g, COUNT(*) FROM t GROUP BY g LIMIT 10");
    assert_eq!(got, plain);
}

#[derive(Default)]
struct NativeHaving;

impl ScalarFunction for NativeHaving {
    fn name(&self) -> &str {
        "NATIVE_HAVING_PROBE"
    }
    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "NATIVE_HAVING_PROBE",
            FunctionType::Scalar,
            "a one-argument function with a native pointer",
            FunctionSignature::new(FunctionDataType::Any, vec![FunctionDataType::Any], 1, 1),
        )
    }
    fn evaluate(&self, args: &[Value]) -> stoolap::core::Result<Value> {
        Ok(args[0].clone())
    }
    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(NativeHaving)
    }
    fn native_fn1(&self) -> Option<NativeFn1> {
        Some(|_| {})
    }
}

#[test]
fn a_having_that_calls_a_native_function_is_not_judged_inside_the_index_walk() {
    global_registry().register_scalar::<NativeHaving>().unwrap();
    let db = open(
        Database::open("memory://aggregate_path_having_native_call").unwrap(),
        false,
        true,
    );
    let sql =
        "SELECT g, COUNT(*) FROM t GROUP BY g HAVING NATIVE_HAVING_PROBE(g) IS NOT NULL LIMIT 10";
    let (got, taken) = run(&db, sql);
    assert_eq!(
        taken,
        ["hash_groups"],
        "a native pointer runs user code under the walk's index lock as well"
    );
    let (plain, _) = run(&db, "SELECT g, COUNT(*) FROM t GROUP BY g LIMIT 10");
    assert_eq!(got, plain);
}
