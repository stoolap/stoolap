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

//! How a statement runs, decided once over the statement as the runner bound
//! it: the access that finds its rows and the steps over them

use crate::common::CompactArc;
use crate::core::pattern::Pattern;
use crate::core::{Operator, Schema};
use crate::executor::expression::SharedProgram;
use crate::executor::prepared_select::{PlanInputs, PreparedComparison};
use crate::executor::projection::Projection;
use crate::executor::utils::{extract_literal_value, flip_operator, infix_to_operator};
use crate::parser::ast::{Expression, InfixOperator};
use crate::storage::mvcc::index_plan::{self, IndexPlan, IndexTerm, Term};
use crate::storage::mvcc::VersionStore;
use smallvec::SmallVec;

/// What the planner decided for one statement: its steps, outermost first,
/// over its read of the table. EXPLAIN prints it and nothing else
#[derive(Debug)]
pub struct Plan {
    pub steps: SmallVec<[Step; 4]>,
    pub read: Read,
}

/// A step over the rows of the one below it
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// The LIMIT and OFFSET
    Limit,
    /// The ORDER BY over every row
    Sort,
    Project,
    /// The WHERE's conjuncts the read did not take
    Filter,
}

/// How a plan reads its table; a scan without a choice is the table's to
/// choose, as for a statement whose WHERE asks a block
#[derive(Debug)]
pub enum Read {
    /// The row the primary key equality names
    PkGet(PreparedComparison),
    /// A scan, through the indexes the choice reads
    Scan(Option<ScanChoice>),
    /// The rows in ORDER BY order, or `otherwise` when the table declines
    Ordered {
        read: OrderedRead,
        otherwise: Option<ScanChoice>,
    },
}

impl Read {
    /// The scan the plan chose, the fallback of an ordered read; None for a
    /// key read or a scan the table chooses
    pub fn scan(&self) -> Option<&ScanChoice> {
        match self {
            Self::PkGet(_) => None,
            Self::Scan(scan)
            | Self::Ordered {
                otherwise: scan, ..
            } => scan.as_ref(),
        }
    }

    /// The ordered read, for a plan that reads in ORDER BY order
    pub fn ordered(&self) -> Option<OrderedRead> {
        match self {
            Self::Ordered { read, .. } => Some(*read),
            _ => None,
        }
    }
}

/// What EXPLAIN names a plan's steps and read by, as its statement writes them
pub struct PlanText<'a> {
    pub table: &'a str,
    pub limit: Option<String>,
    pub offset: Option<String>,
    pub sort: Option<String>,
    pub names: &'a [String],
    /// The conjuncts the read takes
    pub pushed: Option<&'a Expression>,
    /// The ORDER BY column and direction of an ordered read
    pub ordered: Option<(&'a str, bool)>,
}

impl Plan {
    /// The plan as EXPLAIN shows it: each step indented under the one it
    /// feeds, the read last
    pub fn explain(&self, text: PlanText<'_>) -> Vec<String> {
        let mut lines = Vec::new();
        for step in &self.steps {
            let line = match step {
                Step::Limit => match (&text.limit, &text.offset) {
                    (None, None) => continue,
                    (Some(limit), None) => format!("Limit {limit}"),
                    (None, Some(offset)) => format!("Offset {offset}"),
                    (Some(limit), Some(offset)) => format!("Limit {limit} offset {offset}"),
                },
                Step::Sort => match &text.sort {
                    Some(keys) => format!("Sort {keys}"),
                    None => continue,
                },
                Step::Project => format!("Project {}", text.names.join(", ")),
                Step::Filter => "Filter".to_string(),
            };
            lines.push(format!("{}{line}", "  ".repeat(lines.len())));
        }
        let read = match (&self.read, text.ordered) {
            (Read::PkGet(key), _) => vec![format!("PkGet {} {}", text.table, key.describe())],
            (Read::Ordered { read, .. }, Some(column)) => {
                vec![read.explain(text.table, column, text.pushed)]
            }
            (read, _) => read
                .scan()
                .map_or_else(Vec::new, |scan| scan.explain(text.table, text.pushed)),
        };
        let depth = "  ".repeat(lines.len());
        lines.extend(read.into_iter().map(|line| format!("{depth}{line}")));
        lines
    }
}

/// A single-table SELECT: its access, the WHERE over the rows it reads, the
/// output and the rows it skips and keeps
#[derive(Debug)]
pub struct SelectPlan {
    pub(crate) plan: Plan,
    pub(crate) filter: Option<SharedProgram>,
    pub(crate) projection: Projection,
    /// Whether the output row is the table row as it is
    pub(crate) whole_row: bool,
    pub(crate) names: CompactArc<Vec<String>>,
    pub(crate) offset: usize,
    pub(crate) limit: Option<usize>,
}

impl SelectPlan {
    /// The plan as EXPLAIN shows it, outermost step first
    pub fn explain(&self, table: &str) -> Vec<String> {
        self.plan.explain(PlanText {
            table,
            limit: self.limit.map(|limit| limit.to_string()),
            offset: (self.offset > 0).then(|| self.offset.to_string()),
            sort: None,
            names: &self.names,
            pushed: None,
            ordered: None,
        })
    }
}

/// How a table's scan finds the rows the conjuncts it takes keep
#[derive(Debug)]
pub enum ScanChoice {
    /// Every row, each one judged
    Seq,
    /// The rows the indexes give for the conjuncts
    Index(IndexPlan),
}

/// The scan for `pushed`, the conjuncts a scan takes of a WHERE, over a
/// table of `schema` whose indexes `store` holds
pub fn scan_choice(
    store: &VersionStore,
    schema: &Schema,
    pushed: Option<&Expression>,
) -> ScanChoice {
    match pushed.and_then(|pushed| index_plan::decide(store, schema, pushed)) {
        Some(plan) => ScanChoice::Index(plan),
        None => ScanChoice::Seq,
    }
}

impl ScanChoice {
    /// The scan as EXPLAIN shows it: its line, then the plan's reads
    /// indented under it
    pub fn explain(&self, table: &str, pushed: Option<&Expression>) -> Vec<String> {
        let mut lines = vec![match (self, pushed) {
            (Self::Index(_), Some(pushed)) => format!("IndexScan {table} {pushed}"),
            (_, Some(pushed)) => format!("SeqScan {table} {pushed}"),
            (_, None) => format!("SeqScan {table}"),
        }];
        if let Self::Index(plan) = self {
            lines.extend(plan.explain().into_iter().map(|read| format!("  {read}")));
        }
        lines
    }
}

/// How the rows an ORDER BY of one column under a LIMIT keeps are read in
/// that order, with no sort of every row
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderedRead {
    /// The column's index, or the primary key, walked in order
    Index,
    /// The primary key walked from the WHERE's lower bound on it
    Keyset,
    /// The table's top-k scan over the column, judging rows by the WHERE
    TopK,
}

/// The ordered read for an ORDER BY of `column` in `ascending` order over a
/// table of `schema`, whose scan takes `pushed`, the WHERE whole when
/// `whole`: the column's index or the key walked for no WHERE; the key from
/// a WHERE of lower bounds on it, ascending with no OFFSET; else the top-k
/// scan. None when the scan leaves part of the WHERE to the statement
pub fn ordered_read(
    store: &VersionStore,
    schema: &Schema,
    (column, ascending): (&str, bool),
    pushed: Option<&Expression>,
    whole: bool,
    offset: bool,
) -> Option<OrderedRead> {
    if !whole {
        return None;
    }
    let keyed = schema
        .pk_column_index()
        .is_some_and(|pk| schema.columns[pk].name.eq_ignore_ascii_case(column));
    let Some(pushed) = pushed else {
        return Some(match keyed || store.get_index_by_column(column).is_some() {
            true => OrderedRead::Index,
            false => OrderedRead::TopK,
        });
    };
    let mut scope = SmallVec::new();
    index_plan::conjuncts(pushed, schema, &mut scope);
    let bounds = scope.iter().all(|conjunct| {
        matches!(
            conjunct.term(schema),
            Term::Compare { column: c, operator: Operator::Gt | Operator::Gte }
                if c.eq_ignore_ascii_case(column)
        )
    });
    Some(match keyed && ascending && !offset && bounds {
        true => OrderedRead::Keyset,
        false => OrderedRead::TopK,
    })
}

impl OrderedRead {
    /// The read as EXPLAIN shows it, in place of the scan
    pub fn explain(
        self,
        table: &str,
        (column, ascending): (&str, bool),
        pushed: Option<&Expression>,
    ) -> String {
        let name = match self {
            Self::Index => "IndexOrder",
            Self::Keyset => "Keyset",
            Self::TopK => "TopK",
        };
        let direction = if ascending { "ASC" } else { "DESC" };
        match pushed {
            Some(pushed) => format!("{name} {table} {column} {direction} {pushed}"),
            None => format!("{name} {table} {column} {direction}"),
        }
    }
}

/// A bound WHERE term as the index rule reads it: a column the builder
/// bound against a literal or a parameter
impl IndexTerm for Expression {
    fn term<'s>(&'s self, schema: &'s Schema) -> Term<'s> {
        let column = |e: &Expression| match e {
            Expression::Column(c) => schema
                .columns
                .get(c.index as usize)
                .map(|c| c.name.as_str()),
            _ => None,
        };
        let value = |e: &Expression| {
            matches!(e, Expression::Parameter(p) if p.index > 0)
                || extract_literal_value(e).is_some()
        };
        match self {
            Expression::Infix(infix) => {
                let operator = match infix.op_type {
                    InfixOperator::And => return Term::And,
                    InfixOperator::Or => return Term::Or,
                    op => infix_to_operator(op),
                };
                let Some(operator) = operator else {
                    return Term::Other;
                };
                for (side, other, operator) in [
                    (&infix.left, &infix.right, operator),
                    (&infix.right, &infix.left, flip_operator(operator)),
                ] {
                    if let Some(column) = column(side).filter(|_| value(other)) {
                        return Term::Compare { column, operator };
                    }
                }
                Term::Other
            }
            Expression::Between(between) if !between.not => {
                match column(&between.expr)
                    .filter(|_| value(&between.lower) && value(&between.upper))
                {
                    Some(column) => Term::Between { column },
                    None => Term::Other,
                }
            }
            Expression::In(list) if !list.not => {
                let listed = match list.right.as_ref() {
                    Expression::List(l) => l.elements.iter().all(value),
                    Expression::ExpressionList(l) => l.expressions.iter().all(value),
                    _ => false,
                };
                match column(&list.left).filter(|_| listed) {
                    Some(column) => Term::In { column },
                    None => Term::Other,
                }
            }
            Expression::Like(like)
                if like.operator.eq_ignore_ascii_case("LIKE") && like.escape.is_none() =>
            {
                let prefixed = match like.pattern.as_ref() {
                    Expression::Parameter(p) => p.index > 0,
                    Expression::StringLiteral(s) => {
                        let prefix = Pattern::literal_prefix(&s.value);
                        !s.value.starts_with('%')
                            && !prefix.is_empty()
                            && Pattern::prefix_range_end(prefix).is_some()
                    }
                    _ => false,
                };
                match column(&like.left).filter(|_| prefixed) {
                    Some(column) => Term::LikePrefix { column },
                    None => Term::Other,
                }
            }
            _ => Term::Other,
        }
    }

    fn each_operand<'s>(&'s self, each: &mut dyn FnMut(&'s Self)) {
        if let Expression::Infix(infix) = self {
            if matches!(infix.op_type, InfixOperator::And | InfixOperator::Or) {
                each(&infix.left);
                each(&infix.right);
            }
        }
    }
}

/// The plan of a statement over a table of `schema` from what its
/// preparation bound, skipping `offset` rows and keeping `limit`; None for a
/// statement the planner does not plan yet
pub fn plan_select(
    inputs: PlanInputs,
    schema: &Schema,
    offset: usize,
    limit: Option<usize>,
) -> Option<SelectPlan> {
    let (key, filter) = inputs.point?;
    let projection = Projection::of(inputs.items);
    let mut steps = SmallVec::new();
    if limit.is_some() || offset > 0 {
        steps.push(Step::Limit);
    }
    steps.push(Step::Project);
    if filter.is_some() {
        steps.push(Step::Filter);
    }
    Some(SelectPlan {
        plan: Plan {
            steps,
            read: Read::PkGet(key),
        },
        filter,
        whole_row: projection.is_columns_in_order(schema.columns.len()),
        projection,
        names: inputs.names,
        offset,
        limit,
    })
}
