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

//! CAST of text to a number ignores the whitespace around it.

use stoolap::Database;

#[test]
fn test_cast_of_padded_text_to_a_number() {
    let db = Database::open("memory://cast_padded_text").unwrap();
    let row = db
        .query(
            "SELECT CAST(' 7' AS INTEGER), CAST('7 ' AS INTEGER), CAST(' -3' AS INTEGER), CAST(' 7.5 ' AS FLOAT), CAST('x' AS INTEGER)",
            (),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(row.get::<Option<i64>>(0).unwrap(), Some(7));
    assert_eq!(row.get::<Option<i64>>(1).unwrap(), Some(7));
    assert_eq!(row.get::<Option<i64>>(2).unwrap(), Some(-3));
    assert_eq!(row.get::<Option<f64>>(3).unwrap(), Some(7.5));
    assert_eq!(row.get::<Option<i64>>(4).unwrap(), None);
}
