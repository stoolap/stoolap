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

//! A program a statement holds across its runs is held only once it
//! compiles: a function registered after a run that failed to find it is
//! found by the next run

use stoolap::core::{Result, Value};
use stoolap::functions::registry::global_registry;
use stoolap::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
};
use stoolap::Database;

macro_rules! registered_late {
    ($test:ident, $function:ident, $name:literal, $sql:literal, $expected:expr) => {
        #[derive(Default)]
        struct $function;

        impl ScalarFunction for $function {
            fn name(&self) -> &str {
                $name
            }
            fn info(&self) -> FunctionInfo {
                FunctionInfo::new(
                    $name,
                    FunctionType::Scalar,
                    "returns its argument",
                    FunctionSignature::new(
                        FunctionDataType::Any,
                        vec![FunctionDataType::Any],
                        1,
                        1,
                    ),
                )
            }
            fn evaluate(&self, args: &[Value]) -> Result<Value> {
                Ok(args[0].clone())
            }
            fn clone_box(&self) -> Box<dyn ScalarFunction> {
                Box::new($function)
            }
        }

        #[test]
        fn $test() {
            let db =
                Database::open(concat!("memory://held_program_retry_", stringify!($test))).unwrap();
            db.execute("CREATE TABLE l (id INTEGER PRIMARY KEY)", ())
                .unwrap();
            db.execute("CREATE TABLE r (id INTEGER PRIMARY KEY, w INTEGER)", ())
                .unwrap();
            db.execute("INSERT INTO l VALUES (1), (2)", ()).unwrap();
            db.execute("INSERT INTO r VALUES (1, 10), (2, 20)", ())
                .unwrap();
            let stmt = db.prepare($sql).unwrap();
            let run = || -> Result<Vec<String>> {
                stmt.query(())?
                    .map(|row| Ok(row?.get::<Value>(0)?.to_string()))
                    .collect()
            };
            let missing = run().expect_err("the function is not registered yet");
            assert!(missing.to_string().contains($name), "{missing}");
            global_registry().register_scalar::<$function>();
            assert_eq!(run().unwrap(), $expected, "{}", $sql);
            let queried: Vec<String> = db
                .query($sql, ())
                .unwrap()
                .map(|row| row.unwrap().get::<Value>(0).unwrap().to_string())
                .collect();
            assert_eq!(queried, $expected, "{}", $sql);
        }
    };
}

registered_late!(
    a_grouped_join_feed,
    LateGroupedJoin,
    "HELD_RETRY_GROUPED_JOIN",
    "SELECT SUM(HELD_RETRY_GROUPED_JOIN(r.w)) FROM l JOIN r ON l.id = r.id",
    ["30"]
);

registered_late!(
    a_grouped_feed,
    LateGrouped,
    "HELD_RETRY_GROUPED",
    "SELECT SUM(HELD_RETRY_GROUPED(r.w)) FROM r",
    ["30"]
);

registered_late!(
    a_window_order,
    LateWindowOrder,
    "HELD_RETRY_WINDOW_ORDER",
    "SELECT ROW_NUMBER() OVER (ORDER BY HELD_RETRY_WINDOW_ORDER(r.w) DESC) FROM r ORDER BY r.id",
    ["2", "1"]
);

registered_late!(
    a_window_argument,
    LateWindowArgument,
    "HELD_RETRY_WINDOW_ARGUMENT",
    "SELECT LAG(r.w, HELD_RETRY_WINDOW_ARGUMENT(1), 0) OVER (ORDER BY r.id) FROM r ORDER BY r.id",
    ["0", "10"]
);
