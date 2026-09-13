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

//! A NOT IN over a subquery whose result holds a NULL keeps nothing:
//! no value is known to be outside such a set.

use stoolap::Database;

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect()
}

fn tables(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE c (id INTEGER PRIMARY KEY, p_id INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO p VALUES (1, 1), (2, 2), (3, 3)", ())
        .unwrap();
    db.execute("INSERT INTO c VALUES (1, 1), (2, NULL)", ())
        .unwrap();
    db
}

#[test]
fn a_negated_subquery_set_holding_null_keeps_nothing_on_the_primary_key() {
    let db = tables("not_in_subquery_null_pk");
    for i in 0..3 {
        let rows = ids(
            &db,
            "SELECT id FROM p WHERE id NOT IN (SELECT p_id FROM c) ORDER BY id",
        );
        assert!(rows.is_empty(), "execution {i} returned {rows:?}");
    }
}

#[test]
fn a_negated_subquery_set_holding_null_keeps_nothing_on_a_plain_column() {
    let db = tables("not_in_subquery_null_col");
    assert!(ids(
        &db,
        "SELECT id FROM p WHERE k NOT IN (SELECT p_id FROM c) ORDER BY id"
    )
    .is_empty());
}

#[test]
fn a_negated_subquery_set_without_null_keeps_the_rest() {
    let db = tables("not_in_subquery_no_null");
    db.execute("DELETE FROM c WHERE p_id IS NULL", ()).unwrap();
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM p WHERE id NOT IN (SELECT p_id FROM c) ORDER BY id"
        ),
        vec![2, 3]
    );
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM p WHERE id IN (SELECT p_id FROM c) ORDER BY id"
        ),
        vec![1]
    );
}

#[test]
fn a_subquery_set_holding_null_still_answers_in() {
    let db = tables("in_subquery_null");
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM p WHERE id IN (SELECT p_id FROM c) ORDER BY id"
        ),
        vec![1]
    );
}

#[test]
fn a_negated_subquery_set_holding_null_keeps_nothing_under_a_limit() {
    let db = tables("not_in_subquery_null_limit");
    assert!(ids(
        &db,
        "SELECT id FROM p WHERE id NOT IN (SELECT p_id FROM c) ORDER BY id LIMIT 10"
    )
    .is_empty());
}

#[test]
fn a_negated_subquery_set_holding_null_keeps_nothing_after_the_in_form_ran() {
    let db = tables("not_in_subquery_null_after_in");
    // The IN form runs the same subquery first and may keep its values
    // for the next execution; the negated form must still see the NULL
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM p WHERE id IN (SELECT p_id FROM c) ORDER BY id"
        ),
        vec![1]
    );
    for i in 0..2 {
        let rows = ids(
            &db,
            "SELECT id FROM p WHERE id NOT IN (SELECT p_id FROM c) ORDER BY id",
        );
        assert!(rows.is_empty(), "execution {i} returned {rows:?}");
    }
}

#[test]
fn a_negated_subquery_set_holding_null_keeps_nothing_beside_an_exists() {
    let db = tables("not_in_subquery_null_or_exists");
    // The OR turns the NOT IN into a semi-join rewrite; the NULL member
    // must survive that, so only the row with a child comes back
    for limit in ["", " LIMIT 10"] {
        assert_eq!(
            ids(
                &db,
                &format!("SELECT id FROM p WHERE id NOT IN (SELECT p_id FROM c) OR EXISTS (SELECT 1 FROM c WHERE c.p_id = p.id) ORDER BY id{limit}")
            ),
            vec![1],
            "with{limit:?}"
        );
    }
}

#[test]
fn a_positive_in_through_a_secondary_index_skips_null() {
    let db = tables("in_subquery_null_secondary_index");
    db.execute("INSERT INTO p VALUES (4, NULL)", ()).unwrap();
    db.execute("CREATE INDEX idx_p_k ON p(k)", ()).unwrap();
    // A NULL member of the set equals nothing, so the row whose k is NULL
    // does not match it
    for limit in ["", " LIMIT 10"] {
        assert_eq!(
            ids(
                &db,
                &format!("SELECT id FROM p WHERE k IN (SELECT p_id FROM c) ORDER BY id{limit}")
            ),
            vec![1],
            "with{limit:?}"
        );
    }
}

#[test]
fn a_negated_subquery_with_a_limit_is_read_as_written_beside_an_exists() {
    let db = tables("not_in_subquery_limit_or_exists");
    // The subquery keeps only the row with id 1, whose p_id is 1: the NULL
    // is not in its result, so every id but 1 is outside the set
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM p WHERE id NOT IN (SELECT p_id FROM c ORDER BY id LIMIT 1) OR EXISTS (SELECT 1 FROM c WHERE c.p_id = p.id) ORDER BY id"
        ),
        vec![1, 2, 3]
    );
    // An empty result leaves everything outside the set
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM p WHERE id NOT IN (SELECT p_id FROM c LIMIT 0) OR EXISTS (SELECT 1 FROM c WHERE c.p_id = p.id) ORDER BY id"
        ),
        vec![1, 2, 3]
    );
    // And the positive form reads the same result
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM p WHERE id IN (SELECT p_id FROM c ORDER BY id DESC LIMIT 1) OR EXISTS (SELECT 1 FROM c WHERE c.p_id = p.id) ORDER BY id"
        ),
        vec![1]
    );
}
