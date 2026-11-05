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
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::Deserialize;
use stoolap::core::Value;
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
            Some("null".to_string()),
            Some(serde_json::Value::String("null".to_string()))
        )]
    );

    // The document `null` is a unit too, and at row level an Option looks
    // only at SQL NULL: a lone JSON column is still the row for a Value
    let units: Vec<((),)> = db
        .query_as_serde("SELECT profile FROM users WHERE id = 3", ())
        .unwrap();
    assert_eq!(units, vec![((),)]);
    let whole: Vec<Option<serde_json::Value>> = db
        .query_as_serde(
            "SELECT profile FROM users WHERE id IN (1, 3) ORDER BY id",
            (),
        )
        .unwrap();
    assert_eq!(whole[0].as_ref().unwrap()["profile"]["city"], "Istanbul");
    assert_eq!(whole[1], Some(serde_json::json!({ "profile": null })));
}

#[derive(Debug, Deserialize, PartialEq)]
struct Located {
    path: PathBuf,
    at: DateTime<Utc>,
}

#[test]
fn a_json_string_document_is_its_value_and_any_other_document_is_raw_text() {
    let db = db();
    db.execute(
        "INSERT INTO users (id, name, profile) VALUES
            (13, 'Mia', '\"/tmp/a b\"'),
            (14, 'Ned', '\"2026-09-26T10:30:00Z\"'),
            (15, 'Ola', '\"tab\\\\tsep\"'),
            (16, 'Pat', '{\"path\": \"/tmp/x\", \"at\": \"2026-09-26T10:30:00Z\"}')",
        (),
    )
    .unwrap();
    let paths: Vec<PathBuf> = db
        .query_as_serde("SELECT profile FROM users WHERE id = 13", ())
        .unwrap();
    assert_eq!(paths, vec![PathBuf::from("/tmp/a b")]);
    let at: Vec<DateTime<Utc>> = db
        .query_as_serde("SELECT profile FROM users WHERE id = 14", ())
        .unwrap();
    assert_eq!(at[0].to_rfc3339(), "2026-09-26T10:30:00+00:00");

    // An escape is decoded, so that string is owned; a plain one can be borrowed
    let row = db
        .query("SELECT profile FROM users WHERE id = 15", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let (escaped,): (String,) = row.deserialize().unwrap();
    assert_eq!(escaped, "tab\tsep");
    assert!(row.deserialize::<(&str,)>().is_err());
    let row = db
        .query("SELECT profile FROM users WHERE id = 13", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let (borrowed,): (&str,) = row.deserialize().unwrap();
    assert_eq!(borrowed, "/tmp/a b");

    // Any other document is raw text for a String, and a struct parses it
    let (raw,): (String,) = db
        .query("SELECT profile FROM users WHERE id = 16", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .deserialize()
        .unwrap();
    assert!(raw.starts_with('{'), "{}", raw);
    let located: Vec<(Located,)> = db
        .query_as_serde("SELECT profile FROM users WHERE id = 16", ())
        .unwrap();
    assert_eq!(located[0].0.path, PathBuf::from("/tmp/x"));
}

#[derive(Debug)]
struct NonNegative(u64);

impl<'de> Deserialize<'de> for NonNegative {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct OnlyU64;
        impl<'de> serde::de::Visitor<'de> for OnlyU64 {
            type Value = NonNegative;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a non-negative integer")
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<NonNegative, E> {
                Ok(NonNegative(v))
            }
        }
        d.deserialize_u64(OnlyU64)
    }
}

#[test]
fn an_integer_is_offered_as_u64_unless_it_is_negative_like_serde_json() {
    let db = db();
    let ids: Vec<NonNegative> = db
        .query_as_serde("SELECT id FROM users ORDER BY id", ())
        .unwrap();
    assert_eq!(ids.iter().map(|n| n.0).collect::<Vec<_>>(), vec![1, 2]);
    match db.query_as_serde::<NonNegative, _>("SELECT id - 3 FROM users WHERE id = 1", ()) {
        Err(Error::InvalidArgument(m)) => assert!(
            m.contains("invalid type: integer `-2`, expected a non-negative integer"),
            "{}",
            m
        ),
        other => panic!("expected InvalidArgument, got {:?}", other),
    }
}

#[test]
fn a_timestamp_reads_as_the_same_text_to_rfc3339_writes() {
    let db = db();
    db.execute(
        "CREATE TABLE stamps (id INTEGER PRIMARY KEY, at TIMESTAMP)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO stamps VALUES
            (1, '2026-09-26T10:30:00Z'),
            (2, '2026-09-26T10:30:00.5Z'),
            (3, '2026-09-26T10:30:00.123456789Z'),
            (4, '1969-12-31T23:59:59Z')",
        (),
    )
    .unwrap();
    for row in db.query("SELECT at FROM stamps ORDER BY id", ()).unwrap() {
        let row = row.unwrap();
        let Some(Value::Timestamp(ts)) = row.get_value(0) else {
            panic!("a timestamp column")
        };
        let (text,): (String,) = row.deserialize().unwrap();
        assert_eq!(text, ts.to_rfc3339());
        let (parsed,): (DateTime<Utc>,) = row.deserialize().unwrap();
        assert_eq!(&parsed, ts);
    }
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

    match db.query_as_serde::<Vec<f32>, _>("SELECT embedding FROM users WHERE id = 1", ()) {
        Err(Error::InvalidArgument(m)) => assert!(
            m.contains("read the column itself as `(T,)` or as a struct field"),
            "{}",
            m
        ),
        other => panic!("expected InvalidArgument, got {:?}", other),
    }

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

#[derive(Debug, Deserialize, PartialEq)]
struct Big {
    n: u128,
}

#[test]
fn json_integers_beyond_u64_keep_their_requested_width() {
    let db = db();
    db.execute(
        "INSERT INTO users (id, name, profile) VALUES
            (5, 'Eve', '18446744073709551616'),
            (6, 'Fay', '-18446744073709551616'),
            (7, 'Gus', '{\"n\": 18446744073709551616}')",
        (),
    )
    .unwrap();
    let big: Vec<u128> = db
        .query_as_serde("SELECT profile FROM users WHERE id = 5", ())
        .unwrap();
    assert_eq!(big, vec![18_446_744_073_709_551_616]);
    let negative: Vec<(i128,)> = db
        .query_as_serde("SELECT profile FROM users WHERE id = 6", ())
        .unwrap();
    assert_eq!(negative, vec![(-18_446_744_073_709_551_616,)]);
    let optional: Vec<Option<u128>> = db
        .query_as_serde("SELECT profile FROM users WHERE id = 5", ())
        .unwrap();
    assert_eq!(optional, vec![Some(18_446_744_073_709_551_616)]);
    let nested: Vec<(Big,)> = db
        .query_as_serde("SELECT profile FROM users WHERE id = 7", ())
        .unwrap();
    assert_eq!(
        nested,
        vec![(Big {
            n: 18_446_744_073_709_551_616
        },)]
    );

    // A narrower target still refuses it
    match db.query_as_serde::<i64, _>("SELECT profile FROM users WHERE id = 5", ()) {
        Err(Error::InvalidArgument(m)) => assert!(m.contains("expected i64"), "{}", m),
        other => panic!("expected InvalidArgument, got {:?}", other),
    }

    // A document that is not an integer gets the usual wording, not a
    // number-scanner error
    db.execute(
        "INSERT INTO users (id, name, profile) VALUES (17, 'Quinn', '1.5'), (18, 'Rae', 'null')",
        (),
    )
    .unwrap();
    for (sql, expected) in [
        (
            "SELECT profile FROM users WHERE id = 17",
            "invalid type: floating point `1.5`, expected i128",
        ),
        (
            "SELECT profile FROM users WHERE id = 18",
            "invalid type: null, expected i128",
        ),
    ] {
        match db.query_as_serde::<i128, _>(sql, ()) {
            Err(Error::InvalidArgument(m)) => assert!(m.contains(expected), "{}: {}", sql, m),
            other => panic!("expected InvalidArgument for {}, got {:?}", sql, other),
        }
    }
    match db.query_as_serde::<u128, _>("SELECT profile FROM users WHERE id = 18", ()) {
        Err(Error::InvalidArgument(m)) => {
            assert!(m.contains("invalid type: null, expected u128"), "{}", m)
        }
        other => panic!("expected InvalidArgument, got {:?}", other),
    }
}

#[derive(Debug, Deserialize, PartialEq)]
enum Shape {
    Dot,
    Circle { r: f64 },
}

#[derive(Debug, Deserialize, PartialEq)]
struct Nothing;

#[derive(Debug, Deserialize, PartialEq)]
struct Email(String);

#[derive(Debug, Deserialize, PartialEq)]
struct Wrapped(User);

#[derive(Debug, Deserialize, PartialEq)]
struct IdName(i64, String);

#[derive(Debug, Deserialize, PartialEq)]
struct WithEmail {
    email: Option<Email>,
}

#[test]
fn enums_units_newtypes_and_whole_rows_deserialize_too() {
    let db = db();
    db.execute(
        "INSERT INTO users (id, name, profile) VALUES
            (8, 'Hal', '\"Dot\"'),
            (9, 'Ida', '{\"Circle\": {\"r\": 2.0}}')",
        (),
    )
    .unwrap();

    // An enum in a JSON column follows serde's usual tagging
    let shapes: Vec<(Shape,)> = db
        .query_as_serde(
            "SELECT profile FROM users WHERE id IN (8, 9) ORDER BY id",
            (),
        )
        .unwrap();
    assert_eq!(shapes, vec![(Shape::Dot,), (Shape::Circle { r: 2.0 },)]);
    match db.query_as_serde::<Shape, _>("SELECT id FROM users WHERE id = 1", ()) {
        Err(Error::InvalidArgument(m)) => {
            assert!(m.contains("integer `1`, expected enum Shape"), "{}", m)
        }
        other => panic!("expected InvalidArgument, got {:?}", other),
    }

    // NULL is a unit or a unit struct; a value is not
    let units: Vec<((), Nothing)> = db
        .query_as_serde("SELECT email, profile FROM users WHERE id = 2", ())
        .unwrap();
    assert_eq!(units, vec![((), Nothing)]);
    match db.query_as_serde::<(), _>("SELECT name FROM users WHERE id = 1", ()) {
        Err(Error::InvalidArgument(m)) => {
            assert!(m.contains("string \"Alice\", expected unit"), "{}", m)
        }
        other => panic!("expected InvalidArgument, got {:?}", other),
    }

    // A newtype wraps a column or the whole row, and so does an Option
    let emails: Vec<Option<Email>> = db
        .query_as_serde("SELECT email FROM users WHERE id <= 2 ORDER BY id", ())
        .unwrap();
    assert_eq!(emails, vec![Some(Email("alice@example.com".into())), None]);
    let columns = "SELECT id, name, email, score, active FROM users WHERE id = 1";
    let wrapped: Vec<Wrapped> = db.query_as_serde(columns, ()).unwrap();
    assert_eq!(wrapped[0].0.name, "Alice");
    let some: Vec<Option<User>> = db.query_as_serde(columns, ()).unwrap();
    assert_eq!(some[0].as_ref().map(|u| u.id), Some(1));
    let with_email: Vec<WithEmail> = db
        .query_as_serde("SELECT email FROM users WHERE id <= 2 ORDER BY id", ())
        .unwrap();
    assert_eq!(
        with_email,
        vec![
            WithEmail {
                email: Some(Email("alice@example.com".into()))
            },
            WithEmail { email: None },
        ]
    );

    // A tuple struct takes the columns by position, a sequence takes them all
    let pairs: Vec<IdName> = db
        .query_as_serde("SELECT id, name FROM users WHERE id = 1", ())
        .unwrap();
    assert_eq!(pairs, vec![IdName(1, "Alice".into())]);
    let all: Vec<Vec<i64>> = db
        .query_as_serde(
            "SELECT id, id * 10 FROM users WHERE id <= 2 ORDER BY id",
            (),
        )
        .unwrap();
    assert_eq!(all, vec![vec![1, 10], vec![2, 20]]);

    // A unit struct or an ignored value can stand for the whole row
    let nothing: Vec<Nothing> = db
        .query_as_serde("SELECT email FROM users WHERE id = 2", ())
        .unwrap();
    assert_eq!(nothing, vec![Nothing]);
    let ignored: Vec<serde::de::IgnoredAny> = db
        .query_as_serde("SELECT id, name FROM users WHERE id <= 2", ())
        .unwrap();
    assert_eq!(ignored.len(), 2);

    // A whole row is a JSON object, and a column can be skipped
    let json: Vec<serde_json::Value> = db
        .query_as_serde("SELECT id, name FROM users WHERE id = 1", ())
        .unwrap();
    assert_eq!(json[0], serde_json::json!({"id": 1, "name": "Alice"}));
    let (_, name): (serde::de::IgnoredAny, String) = db
        .query("SELECT id, name FROM users WHERE id = 1", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .deserialize()
        .unwrap();
    assert_eq!(name, "Alice");
}

#[test]
fn every_scalar_width_reads_from_one_column_or_a_json_document() {
    let db = db();
    macro_rules! reads_42 {
        ($($t:ty),*) => {$(
            let values: Vec<$t> = db.query_as_serde("SELECT 42", ()).unwrap();
            assert_eq!(values, vec![42 as $t]);
        )*};
    }
    reads_42!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, f32, f64);
    let chars: Vec<char> = db.query_as_serde("SELECT 'x'", ()).unwrap();
    assert_eq!(chars, vec!['x']);
    let flags: Vec<bool> = db
        .query_as_serde("SELECT active FROM users ORDER BY id", ())
        .unwrap();
    assert_eq!(flags, vec![true, false]);

    db.execute(
        "INSERT INTO users (id, name, profile) VALUES
            (10, 'Joy', '[1, 2, 3]'), (11, 'Kim', '{\"a\": 1}'), (12, 'Lee', '\"x\"')",
        (),
    )
    .unwrap();
    let bytes: Vec<(Vec<u8>,)> = db
        .query_as_serde("SELECT profile FROM users WHERE id = 10", ())
        .unwrap();
    assert_eq!(bytes, vec![(vec![1, 2, 3],)]);
    let map: Vec<(HashMap<String, i64>,)> = db
        .query_as_serde("SELECT profile FROM users WHERE id = 11", ())
        .unwrap();
    assert_eq!(map[0].0["a"], 1);
    let chars: Vec<char> = db
        .query_as_serde("SELECT profile FROM users WHERE id = 12", ())
        .unwrap();
    assert_eq!(chars, vec!['x']);
}

#[test]
fn a_mismatch_names_the_kind_of_value_it_found() {
    let db = db();
    for (sql, expected) in [
        (
            "SELECT created_at FROM users WHERE id = 1",
            "invalid type: timestamp, expected unit",
        ),
        (
            "SELECT embedding FROM users WHERE id = 1",
            "invalid type: vector, expected unit",
        ),
        (
            "SELECT profile FROM users WHERE id = 1",
            "invalid type: JSON, expected unit",
        ),
        (
            "SELECT active FROM users WHERE id = 1",
            "invalid type: boolean `true`, expected unit",
        ),
        (
            "SELECT score FROM users WHERE id = 1",
            "invalid type: floating point `9.5`, expected unit",
        ),
        (
            "SELECT id - 3 FROM users WHERE id = 1",
            "invalid type: integer `-2`, expected unit",
        ),
    ] {
        match db.query_as_serde::<(), _>(sql, ()) {
            Err(Error::InvalidArgument(m)) => assert!(m.contains(expected), "{}: {}", sql, m),
            other => panic!("expected InvalidArgument for {}, got {:?}", sql, other),
        }
    }
    match db.query_as_serde::<Role, _>("SELECT email FROM users WHERE id = 2", ()) {
        Err(Error::InvalidArgument(m)) => {
            assert!(
                m.contains("invalid type: null, expected enum Role"),
                "{}",
                m
            )
        }
        other => panic!("expected InvalidArgument, got {:?}", other),
    }
    for (sql, expected) in [
        (
            "SELECT score FROM users WHERE id = 1",
            "floating point `9.5`, expected i64",
        ),
        (
            "SELECT name FROM users WHERE id = 1",
            "string \"Alice\", expected i64",
        ),
    ] {
        match db.query_as_serde::<i64, _>(sql, ()) {
            Err(Error::InvalidArgument(m)) => assert!(m.contains(expected), "{}: {}", sql, m),
            other => panic!("expected InvalidArgument for {}, got {:?}", sql, other),
        }
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
