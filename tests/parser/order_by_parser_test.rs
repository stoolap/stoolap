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

//! ORDER BY in Function Parsing Tests
//!
//! Tests parsing of ORDER BY expressions inside function calls like FIRST(), LAST()

use stoolap::parser::parse_sql;

/// Test simple FIRST with ORDER BY
#[test]
fn test_parse_first_with_order_by() {
    let sql = "SELECT FIRST(open ORDER BY time_col) FROM candles";
    let result = parse_sql(sql);
    assert!(
        result.is_ok(),
        "Expected parsing to succeed: {:?}",
        result.err()
    );
}

/// Test FIRST with ORDER BY ASC
#[test]
fn test_parse_first_with_order_by_asc() {
    let sql = "SELECT FIRST(open ORDER BY time_col ASC) FROM candles";
    let result = parse_sql(sql);
    assert!(
        result.is_ok(),
        "Expected parsing to succeed: {:?}",
        result.err()
    );
}

/// Test LAST with ORDER BY DESC
#[test]
fn test_parse_last_with_order_by_desc() {
    let sql = "SELECT LAST(close ORDER BY time_col DESC) FROM candles";
    let result = parse_sql(sql);
    assert!(
        result.is_ok(),
        "Expected parsing to succeed: {:?}",
        result.err()
    );
}

/// Test combined MIN, MAX, FIRST, LAST with ORDER BY
#[test]
fn test_parse_combined_aggregates_with_order_by() {
    let sql = "SELECT FIRST(open ORDER BY time_col), MAX(high), MIN(low), LAST(close ORDER BY time_col), SUM(volume) FROM candles GROUP BY date_col";
    let result = parse_sql(sql);
    assert!(
        result.is_ok(),
        "Expected parsing to succeed: {:?}",
        result.err()
    );
}

/// Test COUNT with ORDER BY in standard SQL syntax
#[test]
fn test_parse_count_with_order_by_standard() {
    let sql = "SELECT COUNT(*) FROM candles ORDER BY time_col";
    let result = parse_sql(sql);
    assert!(
        result.is_ok(),
        "Expected parsing to succeed: {:?}",
        result.err()
    );
}

/// Test TIME_TRUNC with GROUP BY and ordered aggregates
#[test]
fn test_parse_time_trunc_with_ordered_aggregates() {
    let sql = "SELECT TIME_TRUNC('15m', event_time) AS bucket, FIRST(price ORDER BY event_time) AS open, MAX(price) AS high, MIN(price) AS low, LAST(price ORDER BY event_time) AS close, SUM(volume) AS volume FROM trades GROUP BY bucket";
    let result = parse_sql(sql);
    assert!(
        result.is_ok(),
        "Expected parsing to succeed: {:?}",
        result.err()
    );
}

/// Test ORDER BY with multiple columns
#[test]
fn test_parse_order_by_multiple_columns() {
    let sql = "SELECT * FROM test ORDER BY col1, col2 DESC, col3 ASC";
    let result = parse_sql(sql);
    assert!(
        result.is_ok(),
        "Expected parsing to succeed: {:?}",
        result.err()
    );
}

/// Test ORDER BY with expressions
#[test]
fn test_parse_order_by_expression() {
    let sql = "SELECT * FROM test ORDER BY col1 + col2";
    let result = parse_sql(sql);
    assert!(
        result.is_ok(),
        "Expected parsing to succeed: {:?}",
        result.err()
    );
}

/// Test ORDER BY with column alias
#[test]
fn test_parse_order_by_alias() {
    let sql = "SELECT price * quantity AS total FROM orders ORDER BY total";
    let result = parse_sql(sql);
    assert!(
        result.is_ok(),
        "Expected parsing to succeed: {:?}",
        result.err()
    );
}

/// Test ORDER BY with NULLS FIRST/LAST
#[test]
fn test_parse_order_by_nulls_first() {
    let sql = "SELECT * FROM test ORDER BY col1 NULLS FIRST";
    let result = parse_sql(sql);
    assert!(
        result.is_ok(),
        "Expected parsing to succeed: {:?}",
        result.err()
    );
}

/// Test ORDER BY with NULLS LAST
#[test]
fn test_parse_order_by_nulls_last() {
    let sql = "SELECT * FROM test ORDER BY col1 DESC NULLS LAST";
    let result = parse_sql(sql);
    assert!(
        result.is_ok(),
        "Expected parsing to succeed: {:?}",
        result.err()
    );
}
