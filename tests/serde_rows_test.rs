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

//! Rows deserialized through serde: structs by column name, tuples by
//! position, every column type, NULL handling, and the errors a mismatch
//! produces.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::Deserialize;
use stoolap::{named_params, Database, Error};

fn db() -> Database {
    let db = Database::open_in_memory().unwrap();
    db.execute(
        "CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            score FLOAT,
            active BOOLEAN,
            created_at TIMESTAMP,
            profile JSON,
            embedding VECTOR(3)
        )",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO users VALUES
            (1, 'Alice', 'alice@example.com', 9.5, true, '2026-09-26 10:30:00',
             '{\"city\": \"Istanbul\", \"tags\": [\"a\", \"b\"], \"age\": 30}', '[1.0, 2.5, -3.0]'),
            (2, 'Bob', NULL, 4, false, NULL, NULL, NULL)",
        (),
    )
    .unwrap();
    db
}

#[derive(Debug, Deserialize, PartialEq)]
struct User {
    id: i64,
    name: String,
    email: Option<String>,
    score: f64,
    active: bool,
}

#[test]
fn a_struct_fills_its_fields_by_column_name() {
    let db = db();
    let users: Vec<User> = db
        .query_as_serde(
            "SELECT id, name, email, score, active FROM users ORDER BY id",
            (),
        )
        .unwrap();
    assert_eq!(
        users,
        vec![
            User {
                id: 1,
                name: "Alice".into(),
                email: Some("alice@example.com".into()),
                score: 9.5,
                active: true,
            },
            User {
                id: 2,
                name: "Bob".into(),
                email: None,
                score: 4.0,
                active: false,
            },
        ]
    );
}

#[test]
fn column_order_and_extra_columns_do_not_matter_to_a_struct() {
    let db = db();
    let users: Vec<User> = db
        .query_as_serde(
            "SELECT active, score, created_at, email, name, id FROM users WHERE id = 1",
            (),
        )
        .unwrap();
    assert_eq!(users[0].id, 1);
    assert_eq!(users[0].name, "Alice");
}

#[test]
fn a_tuple_takes_the_columns_by_position() {
    let db = db();
    let rows: Vec<(i64, String, Option<String>)> = db
        .query_as_serde("SELECT id, name, email FROM users ORDER BY id", ())
        .unwrap();
    assert_eq!(
        rows[0],
        (1, "Alice".into(), Some("alice@example.com".into()))
    );
    assert_eq!(rows[1], (2, "Bob".into(), None));
}

#[test]
fn a_row_can_be_deserialized_from_the_rows_iterator_and_borrow_its_text() {
    let db = db();
    let mut seen = Vec::new();
    for row in db
        .query("SELECT name, id FROM users ORDER BY id", ())
        .unwrap()
    {
        let row = row.unwrap();
        #[derive(Deserialize)]
        struct Named<'a> {
            name: &'a str,
            id: i32,
        }
        let named: Named<'_> = row.deserialize().unwrap();
        seen.push((named.name.to_string(), named.id));
    }
    assert_eq!(seen, vec![("Alice".to_string(), 1), ("Bob".to_string(), 2)]);

    let ids: Vec<i64> = db
        .query("SELECT id FROM users ORDER BY id DESC", ())
        .unwrap()
        .deserialize::<(i64,)>()
        .map(|r| r.map(|(id,)| id))
        .collect::<stoolap::Result<_>>()
        .unwrap();
    assert_eq!(ids, vec![2, 1]);
}

#[test]
fn named_parameters_deserialize_too() {
    let db = db();
    let users: Vec<User> = db
        .query_as_serde_named(
            "SELECT id, name, email, score, active FROM users WHERE id = :id",
            named_params! { id: 2 },
        )
        .unwrap();
    assert_eq!(users.len(), 1);
    assert_eq!(users[0].name, "Bob");
}

#[derive(Debug, Deserialize, PartialEq)]
struct Profile {
    city: String,
    tags: Vec<String>,
    age: u8,
}

#[derive(Debug, Deserialize)]
struct Rich {
    id: i64,
    created_at: Option<DateTime<Utc>>,
    profile: Option<Profile>,
    embedding: Option<Vec<f32>>,
    raw: Option<serde_json::Value>,
}

#[test]
fn timestamps_json_and_vectors_deserialize_into_their_natural_types() {
    let db = db();
    let rows: Vec<Rich> = db
        .query_as_serde(
            "SELECT id, created_at, profile, embedding, profile AS raw FROM users ORDER BY id",
            (),
        )
        .unwrap();
    assert_eq!(rows.iter().map(|r| r.id).collect::<Vec<_>>(), vec![1, 2]);
    let alice = &rows[0];
    assert_eq!(
        alice.created_at.unwrap().to_rfc3339(),
        "2026-09-26T10:30:00+00:00"
    );
    assert_eq!(
        alice.profile,
        Some(Profile {
            city: "Istanbul".into(),
            tags: vec!["a".into(), "b".into()],
            age: 30,
        })
    );
    assert_eq!(alice.embedding, Some(vec![1.0, 2.5, -3.0]));
    assert_eq!(alice.raw.as_ref().unwrap()["city"], "Istanbul");

    let bob = &rows[1];
    assert!(bob.created_at.is_none());
    assert!(bob.profile.is_none());
    assert!(bob.embedding.is_none());
    assert!(bob.raw.is_none());
}

#[test]
fn a_map_takes_every_column_by_name() {
    let db = db();
    let rows: Vec<HashMap<String, serde_json::Value>> = db
        .query_as_serde("SELECT id, name, score, active FROM users WHERE id = 1", ())
        .unwrap();
    let row = &rows[0];
    assert_eq!(row["id"], 1);
    assert_eq!(row["name"], "Alice");
    assert_eq!(row["score"], 9.5);
    assert_eq!(row["active"], true);
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Role {
    Admin,
    Member,
}

#[derive(Debug, Deserialize)]
struct WithEnumAndRename {
    #[serde(rename = "name")]
    display_name: String,
    role: Role,
    #[serde(default)]
    missing: Option<i64>,
}

#[test]
fn text_names_a_unit_variant_and_serde_attributes_apply() {
    let db = db();
    let rows: Vec<WithEnumAndRename> = db
        .query_as_serde(
            "SELECT name, CASE WHEN id = 1 THEN 'admin' ELSE 'member' END AS role FROM users ORDER BY id",
            (),
        )
        .unwrap();
    assert_eq!(rows[0].display_name, "Alice");
    assert_eq!(rows[0].role, Role::Admin);
    assert_eq!(rows[1].role, Role::Member);
    assert_eq!(rows[0].missing, None);
}

#[test]
fn integers_widen_and_narrow_within_range_and_fail_outside_it() {
    let db = db();
    let (small, wide, float): (u8, u64, f64) = db
        .query("SELECT 200, 5000000000, 7 FROM users WHERE id = 1", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .deserialize()
        .unwrap();
    assert_eq!((small, wide, float), (200, 5_000_000_000, 7.0));

    let too_big = db
        .query("SELECT 300 FROM users WHERE id = 1", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .deserialize::<(u8,)>();
    let message = match too_big {
        Err(Error::InvalidArgument(m)) => m,
        other => panic!("expected InvalidArgument, got {:?}", other),
    };
    assert!(message.contains("(u8,)"), "{}", message);
    assert!(message.contains("300"), "{}", message);
}

#[test]
fn a_null_in_a_required_field_and_a_missing_column_are_errors_that_name_the_type() {
    let db = db();
    let null_email =
        db.query_as_serde::<(i64, String), _>("SELECT id, email FROM users WHERE id = 2", ());
    match null_email {
        Err(Error::InvalidArgument(m)) => {
            assert!(m.contains("(i64, alloc::string::String)"), "{}", m);
            assert!(m.contains("invalid type: null, expected a string"), "{}", m);
        }
        other => panic!("expected InvalidArgument, got {:?}", other),
    }

    let missing = db.query_as_serde::<User, _>("SELECT id, name FROM users WHERE id = 1", ());
    match missing {
        Err(Error::InvalidArgument(m)) => {
            assert!(m.contains("User"), "{}", m);
            assert!(m.contains("missing field `score`"), "{}", m);
        }
        other => panic!("expected InvalidArgument, got {:?}", other),
    }

    let wrong_type = db.query_as_serde::<(String,), _>("SELECT active FROM users WHERE id = 1", ());
    match wrong_type {
        Err(Error::InvalidArgument(m)) => {
            assert!(m.contains("boolean `true`"), "{}", m);
        }
        other => panic!("expected InvalidArgument, got {:?}", other),
    }
}

#[test]
fn a_json_column_read_as_a_string_is_the_raw_document_and_a_json_null_is_none() {
    let db = db();
    let (raw,): (String,) = db
        .query("SELECT profile FROM users WHERE id = 1", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .deserialize()
        .unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(parsed["city"], "Istanbul");

    db.execute(
        "INSERT INTO users (id, name, profile) VALUES (3, 'Cem', 'null')",
        (),
    )
    .unwrap();
    let rows: Vec<(Option<Profile>, Option<serde_json::Value>, Option<String>)> = db
        .query_as_serde(
            "SELECT profile, profile, profile FROM users WHERE id = 3",
            (),
        )
        .unwrap();
    assert_eq!(rows, vec![(None, None, None)]);

    // The JSON *string* "null" is a value, not an absent document
    db.execute(
        "INSERT INTO users (id, name, profile) VALUES (4, 'Dee', '\"null\"')",
        (),
    )
    .unwrap();
    let rows: Vec<(Option<String>, Option<serde_json::Value>)> = db
        .query_as_serde("SELECT profile, profile FROM users WHERE id = 4", ())
        .unwrap();
    assert_eq!(
        rows,
        vec![(
            Some("\"null\"".to_string()),
            Some(serde_json::Value::String("null".to_string()))
        )]
    );
}

#[test]
fn a_vector_fills_an_array_of_its_own_length_only() {
    let db = db();
    let (fixed,): ([f32; 3],) = db
        .query("SELECT embedding FROM users WHERE id = 1", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .deserialize()
        .unwrap();
    assert_eq!(fixed, [1.0, 2.5, -3.0]);

    let too_short =
        db.query_as_serde::<([f32; 2],), _>("SELECT embedding FROM users WHERE id = 1", ());
    let too_long =
        db.query_as_serde::<([f32; 4],), _>("SELECT embedding FROM users WHERE id = 1", ());
    for result in [too_short.map(|_| ()), too_long.map(|_| ())] {
        match result {
            Err(Error::InvalidArgument(m)) => assert!(m.contains("invalid length 3"), "{}", m),
            other => panic!("expected InvalidArgument, got {:?}", other),
        }
    }
}

#[derive(Debug, Deserialize, PartialEq)]
struct UserId(i64);

#[test]
fn a_single_column_row_is_also_one_bare_value() {
    let db = db();
    let ids: Vec<i64> = db
        .query_as_serde("SELECT id FROM users ORDER BY id", ())
        .unwrap();
    assert_eq!(ids, vec![1, 2]);
    let ids: Vec<UserId> = db
        .query_as_serde("SELECT id FROM users ORDER BY id", ())
        .unwrap();
    assert_eq!(ids, vec![UserId(1), UserId(2)]);
    let emails: Vec<Option<String>> = db
        .query_as_serde("SELECT email FROM users ORDER BY id", ())
        .unwrap();
    assert_eq!(emails, vec![Some("alice@example.com".to_string()), None]);
    let count: Vec<u32> = db.query_as_serde("SELECT COUNT(*) FROM users", ()).unwrap();
    assert_eq!(count, vec![2]);

    // A sequence target always takes the columns, so a vector column is a 1-tuple
    let (embedding,): (Vec<f32>,) = db
        .query("SELECT embedding FROM users WHERE id = 1", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .deserialize()
        .unwrap();
    assert_eq!(embedding, vec![1.0, 2.5, -3.0]);

    let two_columns = db.query_as_serde::<i64, _>("SELECT id, name FROM users", ());
    match two_columns {
        Err(Error::InvalidArgument(m)) => {
            assert!(
                m.contains("a row of 2 columns is not a single value"),
                "{}",
                m
            )
        }
        other => panic!("expected InvalidArgument, got {:?}", other),
    }
}

#[test]
fn a_query_error_surfaces_through_the_deserializing_iterator() {
    let db = db();
    let result = db.query_as_serde::<User, _>("SELECT * FROM missing_table", ());
    let err = result.expect_err("the query fails");
    assert!(
        !matches!(err, Error::InvalidArgument(_)),
        "the query error must not be wrapped as a deserialization error: {:?}",
        err
    );
    assert!(
        format!("{}", err).to_lowercase().contains("missing_table"),
        "{}",
        err
    );
}
