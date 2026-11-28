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

use stoolap::core::Value;
use stoolap::functions::AggregateOrder;
use stoolap::storage::mvcc::version_store::{
    aggregate_row, column_calls, grouped_aggregates, plain_columns, AggregateInput, AggregateOp,
    GroupBy,
};

#[test]
fn the_public_column_calls_feed_first_and_last_by_their_key() {
    let keys = [GroupBy::column(0)];
    let order = AggregateOrder {
        ascending: true,
        nulls_first: None,
    };
    let calls = [
        (
            AggregateOp::First { key: 2, order },
            AggregateInput::Column(1),
        ),
        (
            AggregateOp::Last { key: 2, order },
            AggregateInput::Column(1),
        ),
        (AggregateOp::Count, AggregateInput::Column(1)),
    ];
    let rows = [
        [Value::Integer(1), Value::Integer(10), Value::Integer(3)],
        [Value::Integer(1), Value::Integer(20), Value::Integer(1)],
        [Value::Integer(1), Value::Integer(30), Value::Integer(2)],
        [Value::Integer(2), Value::Integer(40), Value::Integer(5)],
    ];
    let mut groups = grouped_aggregates(&keys, &calls);
    let columns = plain_columns(&keys).unwrap();
    let mut row_calls = column_calls(&calls).unwrap();
    for row in &rows {
        aggregate_row(&mut groups, &columns, &mut row_calls[..], row);
    }
    let got: Vec<Vec<Value>> = groups
        .into_rows()
        .unwrap()
        .into_iter()
        .map(|(_, row)| row.as_slice().to_vec())
        .collect();
    let want = [[1, 20, 10, 3], [2, 40, 40, 1]].map(|row| row.map(Value::Integer).to_vec());
    assert_eq!(
        got, want,
        "group key, FIRST and LAST by the key column, COUNT"
    );
}
