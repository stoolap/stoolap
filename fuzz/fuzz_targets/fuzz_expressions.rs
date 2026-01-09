#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use stoolap::api::Database;

/// Scalar functions to test
#[derive(Debug, Arbitrary)]
enum ScalarFunc {
    // String functions
    Upper,
    Lower,
    Length,
    Trim,
    Ltrim,
    Rtrim,
    Reverse,
    Concat,
    Substring,
    Replace,
    Left,
    Right,
    Lpad,
    Rpad,
    Repeat,

    // Math functions
    Abs,
    Ceil,
    Floor,
    Round,
    Sqrt,
    Power,
    Mod,
    Sign,
    Greatest,
    Least,

    // Null handling
    Coalesce,
    Nullif,
    Ifnull,

    // Type conversion
    Cast,
    Typeof,

    // Date functions
    Now,
    CurrentDate,
    CurrentTimestamp,
    Extract,
    DateTrunc,
}

impl ScalarFunc {
    fn sql(&self, args: &ExprArgs) -> String {
        match self {
            // String functions
            ScalarFunc::Upper => format!("UPPER({})", args.string_arg()),
            ScalarFunc::Lower => format!("LOWER({})", args.string_arg()),
            ScalarFunc::Length => format!("LENGTH({})", args.string_arg()),
            ScalarFunc::Trim => format!("TRIM({})", args.string_arg()),
            ScalarFunc::Ltrim => format!("LTRIM({})", args.string_arg()),
            ScalarFunc::Rtrim => format!("RTRIM({})", args.string_arg()),
            ScalarFunc::Reverse => format!("REVERSE({})", args.string_arg()),
            ScalarFunc::Concat => format!("CONCAT({}, {})", args.string_arg(), args.string_arg2()),
            ScalarFunc::Substring => {
                format!("SUBSTRING({}, {}, {})", args.string_arg(), args.pos(), args.len())
            }
            ScalarFunc::Replace => format!(
                "REPLACE({}, {}, {})",
                args.string_arg(),
                args.string_arg2(),
                args.string_arg()
            ),
            ScalarFunc::Left => format!("LEFT({}, {})", args.string_arg(), args.len()),
            ScalarFunc::Right => format!("RIGHT({}, {})", args.string_arg(), args.len()),
            ScalarFunc::Lpad => format!("LPAD({}, {}, ' ')", args.string_arg(), args.len()),
            ScalarFunc::Rpad => format!("RPAD({}, {}, ' ')", args.string_arg(), args.len()),
            ScalarFunc::Repeat => format!("REPEAT({}, {})", args.string_arg(), args.small_num()),

            // Math functions
            ScalarFunc::Abs => format!("ABS({})", args.num_arg()),
            ScalarFunc::Ceil => format!("CEIL({})", args.float_arg()),
            ScalarFunc::Floor => format!("FLOOR({})", args.float_arg()),
            ScalarFunc::Round => format!("ROUND({}, {})", args.float_arg(), args.small_num()),
            ScalarFunc::Sqrt => format!("SQRT(ABS({}))", args.num_arg()),
            ScalarFunc::Power => format!("POWER({}, {})", args.small_num(), args.small_num()),
            ScalarFunc::Mod => {
                let divisor = args.small_num().max(1);
                format!("MOD({}, {})", args.num_arg(), divisor)
            }
            ScalarFunc::Sign => format!("SIGN({})", args.num_arg()),
            ScalarFunc::Greatest => format!("GREATEST({}, {}, {})", args.num_arg(), args.num2(), args.num3()),
            ScalarFunc::Least => format!("LEAST({}, {}, {})", args.num_arg(), args.num2(), args.num3()),

            // Null handling
            ScalarFunc::Coalesce => {
                format!("COALESCE({}, {}, {})", args.nullable_arg(), args.nullable_arg2(), args.default_val())
            }
            ScalarFunc::Nullif => format!("NULLIF({}, {})", args.num_arg(), args.num2()),
            ScalarFunc::Ifnull => format!("IFNULL({}, {})", args.nullable_arg(), args.default_val()),

            // Type conversion
            ScalarFunc::Cast => {
                let cast_type = match args.type_idx % 4 {
                    0 => "INTEGER",
                    1 => "FLOAT",
                    2 => "TEXT",
                    _ => "BOOLEAN",
                };
                format!("CAST({} AS {})", args.any_arg(), cast_type)
            }
            ScalarFunc::Typeof => format!("TYPEOF({})", args.any_arg()),

            // Date functions
            ScalarFunc::Now => "NOW()".to_string(),
            ScalarFunc::CurrentDate => "CURRENT_DATE".to_string(),
            ScalarFunc::CurrentTimestamp => "CURRENT_TIMESTAMP".to_string(),
            ScalarFunc::Extract => {
                let part = match args.date_part % 6 {
                    0 => "YEAR",
                    1 => "MONTH",
                    2 => "DAY",
                    3 => "HOUR",
                    4 => "MINUTE",
                    _ => "SECOND",
                };
                format!("EXTRACT({} FROM NOW())", part)
            }
            ScalarFunc::DateTrunc => {
                let part = match args.date_part % 4 {
                    0 => "year",
                    1 => "month",
                    2 => "day",
                    _ => "hour",
                };
                format!("DATE_TRUNC('{}', NOW())", part)
            }
        }
    }
}

/// Arguments for expressions
#[derive(Debug, Arbitrary)]
struct ExprArgs {
    str_val: u8,
    str_val2: u8,
    num_val: i16,
    num_val2: i16,
    num_val3: i16,
    float_val: u8,
    pos: u8,
    len: u8,
    type_idx: u8,
    date_part: u8,
    use_column: bool,
    use_null: bool,
}

impl ExprArgs {
    fn string_arg(&self) -> String {
        if self.use_column {
            "name".to_string()
        } else {
            format!("'test_string_{}'", self.str_val)
        }
    }

    fn string_arg2(&self) -> String {
        format!("'str_{}'", self.str_val2)
    }

    fn num_arg(&self) -> String {
        if self.use_column {
            "value".to_string()
        } else {
            self.num_val.to_string()
        }
    }

    fn num2(&self) -> String {
        self.num_val2.to_string()
    }

    fn num3(&self) -> String {
        self.num_val3.to_string()
    }

    fn float_arg(&self) -> String {
        if self.use_column {
            "value".to_string()
        } else {
            format!("{}.{}", self.float_val / 10, self.float_val % 10)
        }
    }

    fn pos(&self) -> u8 {
        self.pos.max(1)
    }

    fn len(&self) -> u8 {
        self.len.saturating_add(1).min(100)
    }

    fn small_num(&self) -> u8 {
        (self.num_val.unsigned_abs() as u8).min(10)
    }

    fn nullable_arg(&self) -> String {
        if self.use_null {
            "NULL".to_string()
        } else {
            self.num_arg()
        }
    }

    fn nullable_arg2(&self) -> String {
        if !self.use_null {
            "NULL".to_string()
        } else {
            self.num2()
        }
    }

    fn default_val(&self) -> String {
        format!("{}", self.num_val3)
    }

    fn any_arg(&self) -> String {
        match self.type_idx % 4 {
            0 => self.num_arg(),
            1 => self.string_arg(),
            2 => self.float_arg(),
            _ => if self.use_null { "true" } else { "false" }.to_string(),
        }
    }
}

/// Arithmetic expressions
#[derive(Debug, Arbitrary)]
enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

impl ArithOp {
    fn sql(&self) -> &'static str {
        match self {
            ArithOp::Add => "+",
            ArithOp::Sub => "-",
            ArithOp::Mul => "*",
            ArithOp::Div => "/",
            ArithOp::Mod => "%",
        }
    }
}

/// Comparison expressions
#[derive(Debug, Arbitrary)]
enum CompOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    IsNull,
    IsNotNull,
    Between,
    In,
    Like,
}

/// Logical expressions
#[derive(Debug, Arbitrary)]
enum LogicalExpr {
    And,
    Or,
    Not,
}

/// Complex expression types
#[derive(Debug, Arbitrary)]
enum ExprType {
    /// Single scalar function call
    ScalarFunction {
        func: ScalarFunc,
        args: ExprArgs,
    },

    /// Nested function calls
    NestedFunction {
        outer: ScalarFunc,
        inner: ScalarFunc,
        args: ExprArgs,
    },

    /// Arithmetic expression
    Arithmetic {
        left: i16,
        op: ArithOp,
        right: i16,
        use_column: bool,
    },

    /// Complex arithmetic with functions
    ArithmeticWithFunc {
        func: ScalarFunc,
        args: ExprArgs,
        op: ArithOp,
        value: i16,
    },

    /// CASE expression
    CaseExpr {
        when_val: i16,
        then_val: i16,
        else_val: i16,
        nested: bool,
    },

    /// Comparison expression
    Comparison {
        left_val: i16,
        op: CompOp,
        right_val: i16,
        use_column: bool,
    },

    /// Logical combination
    Logical {
        op: LogicalExpr,
        val1: i16,
        val2: i16,
    },

    /// String concatenation
    StringConcat {
        parts: Vec<u8>,
    },

    /// JSON expression (if supported)
    JsonExpr {
        key: u8,
    },

    /// Bitwise operations
    Bitwise {
        left: i16,
        right: i16,
        op: u8, // 0=AND, 1=OR, 2=XOR, 3=NOT, 4=LSHIFT, 5=RSHIFT
    },
}

impl ExprType {
    fn to_sql(&self) -> String {
        match self {
            ExprType::ScalarFunction { func, args } => {
                format!("SELECT {} AS result FROM t1 LIMIT 1", func.sql(args))
            }

            ExprType::NestedFunction { outer, inner, args } => {
                // Create a nested function call
                let inner_sql = inner.sql(args);
                let modified_args = ExprArgs {
                    use_column: false,
                    str_val: args.str_val,
                    str_val2: args.str_val2,
                    num_val: args.num_val,
                    num_val2: args.num_val2,
                    num_val3: args.num_val3,
                    float_val: args.float_val,
                    pos: args.pos,
                    len: args.len,
                    type_idx: args.type_idx,
                    date_part: args.date_part,
                    use_null: args.use_null,
                };
                // For nested, wrap the inner result appropriately
                let outer_sql = match outer {
                    ScalarFunc::Length => format!("LENGTH(CAST({} AS TEXT))", inner_sql),
                    ScalarFunc::Abs => format!("ABS(COALESCE({}, 0))", inner_sql),
                    _ => outer.sql(&modified_args),
                };
                format!("SELECT {} AS result FROM t1 LIMIT 1", outer_sql)
            }

            ExprType::Arithmetic {
                left,
                op,
                right,
                use_column,
            } => {
                let l = if *use_column {
                    "value".to_string()
                } else {
                    left.to_string()
                };
                let r = if matches!(op, ArithOp::Div | ArithOp::Mod) && *right == 0 {
                    "1".to_string()
                } else {
                    right.to_string()
                };
                format!("SELECT ({} {} {}) AS result FROM t1 LIMIT 1", l, op.sql(), r)
            }

            ExprType::ArithmeticWithFunc {
                func,
                args,
                op,
                value,
            } => {
                let v = if matches!(op, ArithOp::Div | ArithOp::Mod) && *value == 0 {
                    1
                } else {
                    *value
                };
                format!(
                    "SELECT (COALESCE({}, 0) {} {}) AS result FROM t1 LIMIT 1",
                    func.sql(args),
                    op.sql(),
                    v
                )
            }

            ExprType::CaseExpr {
                when_val,
                then_val,
                else_val,
                nested,
            } => {
                if *nested {
                    format!(
                        "SELECT CASE WHEN id > {} THEN CASE WHEN value > {} THEN {} ELSE {} END ELSE {} END AS result FROM t1",
                        when_val, when_val, then_val, else_val, else_val
                    )
                } else {
                    format!(
                        "SELECT CASE WHEN id > {} THEN {} WHEN id < {} THEN {} ELSE {} END AS result FROM t1",
                        when_val, then_val, when_val.saturating_sub(10), else_val, 0
                    )
                }
            }

            ExprType::Comparison {
                left_val,
                op,
                right_val,
                use_column,
            } => {
                let left = if *use_column {
                    "id".to_string()
                } else {
                    left_val.to_string()
                };

                let expr = match op {
                    CompOp::Eq => format!("{} = {}", left, right_val),
                    CompOp::Ne => format!("{} != {}", left, right_val),
                    CompOp::Lt => format!("{} < {}", left, right_val),
                    CompOp::Le => format!("{} <= {}", left, right_val),
                    CompOp::Gt => format!("{} > {}", left, right_val),
                    CompOp::Ge => format!("{} >= {}", left, right_val),
                    CompOp::IsNull => format!("{} IS NULL", left),
                    CompOp::IsNotNull => format!("{} IS NOT NULL", left),
                    CompOp::Between => {
                        let (lo, hi) = if left_val <= right_val {
                            (left_val, right_val)
                        } else {
                            (right_val, left_val)
                        };
                        format!("id BETWEEN {} AND {}", lo, hi)
                    }
                    CompOp::In => format!("{} IN ({}, {}, {})", left, right_val, right_val + 1, right_val + 2),
                    CompOp::Like => format!("name LIKE 'user_%'"),
                };

                format!("SELECT * FROM t1 WHERE {}", expr)
            }

            ExprType::Logical { op, val1, val2 } => {
                let expr = match op {
                    LogicalExpr::And => format!("id > {} AND value < {}", val1, val2),
                    LogicalExpr::Or => format!("id = {} OR id = {}", val1, val2),
                    LogicalExpr::Not => format!("NOT (id = {})", val1),
                };
                format!("SELECT * FROM t1 WHERE {}", expr)
            }

            ExprType::StringConcat { parts } => {
                let parts_sql: Vec<String> = parts
                    .iter()
                    .take(5)
                    .map(|p| format!("'part_{}'", p))
                    .collect();

                if parts_sql.is_empty() {
                    "SELECT 'empty' AS result".to_string()
                } else {
                    format!("SELECT {} AS result", parts_sql.join(" || "))
                }
            }

            ExprType::JsonExpr { key } => {
                format!(
                    "SELECT JSON_EXTRACT('{{\"key_{}\": \"value_{}\"}}', '$.key_{}') AS result",
                    key, key, key
                )
            }

            ExprType::Bitwise { left, right, op } => {
                let expr = match op % 6 {
                    0 => format!("{} & {}", left, right),
                    1 => format!("{} | {}", left, right),
                    2 => format!("{} ^ {}", left, right),
                    3 => format!("~{}", left),
                    4 => format!("{} << {}", left, (right.unsigned_abs() % 32) as i16),
                    _ => format!("{} >> {}", left, (right.unsigned_abs() % 32) as i16),
                };
                format!("SELECT ({}) AS result", expr)
            }
        }
    }
}

fn setup_database() -> Option<Database> {
    let db = Database::open_in_memory().ok()?;

    db.execute(
        "CREATE TABLE t1 (
            id INTEGER PRIMARY KEY,
            name TEXT,
            value FLOAT,
            active BOOLEAN,
            data TEXT
        )",
        (),
    )
    .ok()?;

    // Insert test data
    for i in 1..=20 {
        let _ = db.execute(
            &format!(
                "INSERT INTO t1 VALUES ({}, 'user_{}', {}.{}, {}, '{}')",
                i,
                i,
                i * 10,
                i % 10,
                i % 2 == 0,
                format!("{{\"key\": {}}}", i)
            ),
            (),
        );
    }

    Some(db)
}

fuzz_target!(|data: &[u8]| {
    let mut unstructured = Unstructured::new(data);

    if let Ok(expr_type) = ExprType::arbitrary(&mut unstructured) {
        let sql = expr_type.to_sql();

        if let Some(db) = setup_database() {
            // Execute the expression - should never panic
            let _ = db.query(&sql, ());
        }
    }
});
