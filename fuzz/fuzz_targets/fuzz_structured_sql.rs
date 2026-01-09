#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use stoolap::api::Database;

/// Column reference for queries
#[derive(Debug, Arbitrary, Clone)]
enum Column {
    Id,
    Name,
    Value,
    Active,
    Category,
    Score,
    Star,
}

impl Column {
    fn name(&self) -> &'static str {
        match self {
            Column::Id => "id",
            Column::Name => "name",
            Column::Value => "value",
            Column::Active => "active",
            Column::Category => "category",
            Column::Score => "score",
            Column::Star => "*",
        }
    }

    fn qualified_name(&self, table: &str) -> String {
        if matches!(self, Column::Star) {
            format!("{}.*", table)
        } else {
            format!("{}.{}", table, self.name())
        }
    }
}

/// Comparison operators
#[derive(Debug, Arbitrary)]
enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CompareOp {
    fn sql(&self) -> &'static str {
        match self {
            CompareOp::Eq => "=",
            CompareOp::Ne => "!=",
            CompareOp::Lt => "<",
            CompareOp::Le => "<=",
            CompareOp::Gt => ">",
            CompareOp::Ge => ">=",
        }
    }
}

/// Logical operators
#[derive(Debug, Arbitrary)]
enum LogicOp {
    And,
    Or,
}

impl LogicOp {
    fn sql(&self) -> &'static str {
        match self {
            LogicOp::And => "AND",
            LogicOp::Or => "OR",
        }
    }
}

/// Literal values
#[derive(Debug, Arbitrary)]
enum Literal {
    Int(i16),
    Float(u8),
    Bool(bool),
    String(u8),
    Null,
}

impl Literal {
    fn sql(&self) -> String {
        match self {
            Literal::Int(i) => i.to_string(),
            Literal::Float(f) => format!("{}.{}", f / 10, f % 10),
            Literal::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            Literal::String(s) => format!("'str_{}'", s),
            Literal::Null => "NULL".to_string(),
        }
    }
}

/// Simple WHERE condition
#[derive(Debug, Arbitrary)]
struct Condition {
    column: Column,
    op: CompareOp,
    value: Literal,
}

impl Condition {
    fn sql(&self, table: Option<&str>) -> String {
        let col = if let Some(t) = table {
            self.column.qualified_name(t)
        } else {
            self.column.name().to_string()
        };
        format!("{} {} {}", col, self.op.sql(), self.value.sql())
    }
}

/// WHERE clause (up to 3 conditions)
#[derive(Debug, Arbitrary)]
struct WhereClause {
    first: Condition,
    rest: Vec<(LogicOp, Condition)>,
}

impl WhereClause {
    fn sql(&self, table: Option<&str>) -> String {
        let mut result = self.first.sql(table);
        for (op, cond) in self.rest.iter().take(2) {
            result.push_str(&format!(" {} {}", op.sql(), cond.sql(table)));
        }
        result
    }
}

/// Aggregate functions
#[derive(Debug, Arbitrary)]
enum AggFunc {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    CountDistinct,
}

impl AggFunc {
    fn sql(&self, col: &Column) -> String {
        match self {
            AggFunc::Count => format!("COUNT({})", col.name()),
            AggFunc::Sum => format!("SUM({})", col.name()),
            AggFunc::Avg => format!("AVG({})", col.name()),
            AggFunc::Min => format!("MIN({})", col.name()),
            AggFunc::Max => format!("MAX({})", col.name()),
            AggFunc::CountDistinct => format!("COUNT(DISTINCT {})", col.name()),
        }
    }
}

/// Window functions
#[derive(Debug, Arbitrary)]
enum WindowFunc {
    RowNumber,
    Rank,
    DenseRank,
    Ntile(u8),
    Lag,
    Lead,
}

impl WindowFunc {
    fn sql(&self, col: &Column, partition: Option<&Column>, order: &Column) -> String {
        let func = match self {
            WindowFunc::RowNumber => "ROW_NUMBER()".to_string(),
            WindowFunc::Rank => "RANK()".to_string(),
            WindowFunc::DenseRank => "DENSE_RANK()".to_string(),
            WindowFunc::Ntile(n) => format!("NTILE({})", (*n).max(1)),
            WindowFunc::Lag => format!("LAG({})", col.name()),
            WindowFunc::Lead => format!("LEAD({})", col.name()),
        };

        let partition_clause = partition
            .map(|p| format!("PARTITION BY {}", p.name()))
            .unwrap_or_default();

        format!(
            "{} OVER ({} ORDER BY {})",
            func,
            partition_clause,
            order.name()
        )
    }
}

/// ORDER BY direction
#[derive(Debug, Arbitrary)]
enum OrderDir {
    Asc,
    Desc,
}

impl OrderDir {
    fn sql(&self) -> &'static str {
        match self {
            OrderDir::Asc => "ASC",
            OrderDir::Desc => "DESC",
        }
    }
}

/// JOIN types
#[derive(Debug, Arbitrary)]
enum JoinType {
    Inner,
    Left,
    Right,
    Cross,
}

impl JoinType {
    fn sql(&self) -> &'static str {
        match self {
            JoinType::Inner => "INNER JOIN",
            JoinType::Left => "LEFT JOIN",
            JoinType::Right => "RIGHT JOIN",
            JoinType::Cross => "CROSS JOIN",
        }
    }
}

/// Subquery types
#[derive(Debug, Arbitrary)]
enum SubqueryType {
    Exists,
    NotExists,
    In,
    NotIn,
    Scalar,
}

/// Query types to generate
#[derive(Debug, Arbitrary)]
enum QueryType {
    /// Simple SELECT
    SimpleSelect {
        columns: Vec<Column>,
        where_clause: Option<WhereClause>,
        order_by: Option<(Column, OrderDir)>,
        limit: Option<u8>,
    },

    /// SELECT with aggregation
    Aggregate {
        group_by: Column,
        agg_func: AggFunc,
        agg_col: Column,
        having: Option<(CompareOp, Literal)>,
    },

    /// SELECT with window function
    Window {
        window_func: WindowFunc,
        col: Column,
        partition: Option<Column>,
        order: Column,
    },

    /// JOIN query
    Join {
        join_type: JoinType,
        where_clause: Option<WhereClause>,
        limit: Option<u8>,
    },

    /// Subquery
    Subquery {
        subquery_type: SubqueryType,
        outer_col: Column,
        inner_col: Column,
    },

    /// CTE query
    Cte {
        cte_where: Option<WhereClause>,
        main_where: Option<WhereClause>,
    },

    /// UNION query
    Union {
        all: bool,
        where1: Option<WhereClause>,
        where2: Option<WhereClause>,
    },

    /// INSERT statement
    Insert {
        id: i16,
        name_suffix: u8,
        value: u8,
        active: bool,
    },

    /// UPDATE statement
    Update {
        set_col: Column,
        set_value: Literal,
        where_clause: Option<WhereClause>,
    },

    /// DELETE statement
    Delete {
        where_clause: Option<WhereClause>,
    },

    /// DISTINCT query
    Distinct {
        columns: Vec<Column>,
        where_clause: Option<WhereClause>,
    },

    /// CASE expression
    Case {
        when_col: Column,
        when_op: CompareOp,
        when_val: Literal,
        then_val: Literal,
        else_val: Literal,
    },

    /// BETWEEN query
    Between {
        col: Column,
        low: i16,
        high: i16,
    },

    /// LIKE query
    Like {
        pattern_type: u8, // 0=prefix%, 1=%suffix, 2=%contains%
    },

    /// COALESCE/NULLIF
    NullHandling {
        use_coalesce: bool,
        col: Column,
        default: Literal,
    },
}

impl QueryType {
    fn to_sql(&self) -> String {
        match self {
            QueryType::SimpleSelect {
                columns,
                where_clause,
                order_by,
                limit,
            } => {
                let cols: Vec<_> = columns.iter().take(4).map(|c| c.name()).collect();
                let cols_str = if cols.is_empty() {
                    "*".to_string()
                } else {
                    cols.join(", ")
                };

                let mut sql = format!("SELECT {} FROM t1", cols_str);

                if let Some(w) = where_clause {
                    sql.push_str(&format!(" WHERE {}", w.sql(None)));
                }

                if let Some((col, dir)) = order_by {
                    sql.push_str(&format!(" ORDER BY {} {}", col.name(), dir.sql()));
                }

                if let Some(l) = limit {
                    sql.push_str(&format!(" LIMIT {}", (*l).max(1)));
                }

                sql
            }

            QueryType::Aggregate {
                group_by,
                agg_func,
                agg_col,
                having,
            } => {
                let mut sql = format!(
                    "SELECT {}, {} FROM t1 GROUP BY {}",
                    group_by.name(),
                    agg_func.sql(agg_col),
                    group_by.name()
                );

                if let Some((op, val)) = having {
                    sql.push_str(&format!(
                        " HAVING {} {} {}",
                        agg_func.sql(agg_col),
                        op.sql(),
                        val.sql()
                    ));
                }

                sql
            }

            QueryType::Window {
                window_func,
                col,
                partition,
                order,
            } => {
                format!(
                    "SELECT id, {}, {} AS win FROM t1",
                    col.name(),
                    window_func.sql(col, partition.as_ref(), order)
                )
            }

            QueryType::Join {
                join_type,
                where_clause,
                limit,
            } => {
                let join_cond = if matches!(join_type, JoinType::Cross) {
                    String::new()
                } else {
                    " ON t1.category = t2.category".to_string()
                };

                let mut sql =
                    format!("SELECT t1.id, t2.name FROM t1 {} t2{}", join_type.sql(), join_cond);

                if let Some(w) = where_clause {
                    sql.push_str(&format!(" WHERE {}", w.sql(Some("t1"))));
                }

                if let Some(l) = limit {
                    sql.push_str(&format!(" LIMIT {}", (*l).max(1)));
                }

                sql
            }

            QueryType::Subquery {
                subquery_type,
                outer_col,
                inner_col,
            } => match subquery_type {
                SubqueryType::Exists => {
                    format!(
                        "SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.{} = t1.{})",
                        inner_col.name(),
                        outer_col.name()
                    )
                }
                SubqueryType::NotExists => {
                    format!(
                        "SELECT * FROM t1 WHERE NOT EXISTS (SELECT 1 FROM t2 WHERE t2.{} = t1.{})",
                        inner_col.name(),
                        outer_col.name()
                    )
                }
                SubqueryType::In => {
                    format!(
                        "SELECT * FROM t1 WHERE {} IN (SELECT {} FROM t2)",
                        outer_col.name(),
                        inner_col.name()
                    )
                }
                SubqueryType::NotIn => {
                    format!(
                        "SELECT * FROM t1 WHERE {} NOT IN (SELECT {} FROM t2 WHERE {} IS NOT NULL)",
                        outer_col.name(),
                        inner_col.name(),
                        inner_col.name()
                    )
                }
                SubqueryType::Scalar => {
                    format!(
                        "SELECT *, (SELECT MAX({}) FROM t2) AS max_val FROM t1",
                        inner_col.name()
                    )
                }
            },

            QueryType::Cte {
                cte_where,
                main_where,
            } => {
                let cte_filter = cte_where
                    .as_ref()
                    .map(|w| format!(" WHERE {}", w.sql(None)))
                    .unwrap_or_default();

                let main_filter = main_where
                    .as_ref()
                    .map(|w| format!(" WHERE {}", w.sql(None)))
                    .unwrap_or_default();

                format!(
                    "WITH cte AS (SELECT * FROM t1{}) SELECT * FROM cte{}",
                    cte_filter, main_filter
                )
            }

            QueryType::Union {
                all,
                where1,
                where2,
            } => {
                let union_type = if *all { "UNION ALL" } else { "UNION" };

                let filter1 = where1
                    .as_ref()
                    .map(|w| format!(" WHERE {}", w.sql(None)))
                    .unwrap_or_default();

                let filter2 = where2
                    .as_ref()
                    .map(|w| format!(" WHERE {}", w.sql(None)))
                    .unwrap_or_default();

                format!(
                    "SELECT id, name FROM t1{} {} SELECT id, name FROM t2{}",
                    filter1, union_type, filter2
                )
            }

            QueryType::Insert {
                id,
                name_suffix,
                value,
                active,
            } => {
                // Use i32 to avoid overflow on i16::MIN
                let safe_id = (*id as i32).abs() % 10000 + 1000;
                format!(
                    "INSERT INTO t1 (id, name, value, active, category, score) VALUES ({}, 'user_{}', {}.{}, {}, 'cat_{}', {})",
                    safe_id,
                    name_suffix,
                    value / 10,
                    value % 10,
                    active,
                    name_suffix % 5,
                    value
                )
            }

            QueryType::Update {
                set_col,
                set_value,
                where_clause,
            } => {
                let mut sql = format!("UPDATE t1 SET {} = {}", set_col.name(), set_value.sql());

                if let Some(w) = where_clause {
                    sql.push_str(&format!(" WHERE {}", w.sql(None)));
                }

                sql
            }

            QueryType::Delete { where_clause } => {
                let mut sql = "DELETE FROM t1".to_string();

                if let Some(w) = where_clause {
                    sql.push_str(&format!(" WHERE {}", w.sql(None)));
                }

                sql
            }

            QueryType::Distinct {
                columns,
                where_clause,
            } => {
                let cols: Vec<_> = columns.iter().take(3).map(|c| c.name()).collect();
                let cols_str = if cols.is_empty() {
                    "id".to_string()
                } else {
                    cols.join(", ")
                };

                let mut sql = format!("SELECT DISTINCT {} FROM t1", cols_str);

                if let Some(w) = where_clause {
                    sql.push_str(&format!(" WHERE {}", w.sql(None)));
                }

                sql
            }

            QueryType::Case {
                when_col,
                when_op,
                when_val,
                then_val,
                else_val,
            } => {
                format!(
                    "SELECT id, CASE WHEN {} {} {} THEN {} ELSE {} END AS result FROM t1",
                    when_col.name(),
                    when_op.sql(),
                    when_val.sql(),
                    then_val.sql(),
                    else_val.sql()
                )
            }

            QueryType::Between { col, low, high } => {
                let (l, h) = if low <= high {
                    (*low, *high)
                } else {
                    (*high, *low)
                };
                format!(
                    "SELECT * FROM t1 WHERE {} BETWEEN {} AND {}",
                    col.name(),
                    l,
                    h
                )
            }

            QueryType::Like { pattern_type } => {
                let pattern = match pattern_type % 3 {
                    0 => "'user_%'",
                    1 => "'%er'",
                    _ => "'%use%'",
                };
                format!("SELECT * FROM t1 WHERE name LIKE {}", pattern)
            }

            QueryType::NullHandling {
                use_coalesce,
                col,
                default,
            } => {
                if *use_coalesce {
                    format!(
                        "SELECT id, COALESCE({}, {}) AS val FROM t1",
                        col.name(),
                        default.sql()
                    )
                } else {
                    format!(
                        "SELECT id, NULLIF({}, {}) AS val FROM t1",
                        col.name(),
                        default.sql()
                    )
                }
            }
        }
    }
}

fn setup_database() -> Option<Database> {
    let db = Database::open_in_memory().ok()?;

    // Create main table
    db.execute(
        "CREATE TABLE t1 (
            id INTEGER PRIMARY KEY,
            name TEXT,
            value FLOAT,
            active BOOLEAN,
            category TEXT,
            score INTEGER
        )",
        (),
    )
    .ok()?;

    // Create second table for JOINs
    db.execute(
        "CREATE TABLE t2 (
            id INTEGER PRIMARY KEY,
            name TEXT,
            value FLOAT,
            active BOOLEAN,
            category TEXT,
            score INTEGER
        )",
        (),
    )
    .ok()?;

    // Insert test data into t1
    for i in 1..=50 {
        let _ = db.execute(
            &format!(
                "INSERT INTO t1 VALUES ({}, 'user_{}', {}.{}, {}, 'cat_{}', {})",
                i,
                i,
                i / 10,
                i % 10,
                i % 2 == 0,
                i % 5,
                i * 10
            ),
            (),
        );
    }

    // Insert test data into t2
    for i in 1..=30 {
        let _ = db.execute(
            &format!(
                "INSERT INTO t2 VALUES ({}, 'other_{}', {}.{}, {}, 'cat_{}', {})",
                i + 100,
                i,
                i / 10,
                i % 10,
                i % 2 == 1,
                i % 5,
                i * 5
            ),
            (),
        );
    }

    // Create indexes
    let _ = db.execute("CREATE INDEX idx_t1_category ON t1(category)", ());
    let _ = db.execute("CREATE INDEX idx_t2_category ON t2(category)", ());
    let _ = db.execute("CREATE INDEX idx_t1_score ON t1(score)", ());

    Some(db)
}

fuzz_target!(|data: &[u8]| {
    let mut unstructured = Unstructured::new(data);

    if let Ok(query_type) = QueryType::arbitrary(&mut unstructured) {
        let sql = query_type.to_sql();

        if let Some(db) = setup_database() {
            // Execute the generated SQL - should never panic
            let _ = db.execute(&sql, ());
            let _ = db.query(&sql, ());
        }
    }
});
