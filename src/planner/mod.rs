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

use std::sync::Weak;

use crate::common::{CompactArc, SmartString};
use crate::core::pattern::Pattern;
use crate::core::{Operator, Schema};
use crate::executor::expression::SharedProgram;
use crate::executor::prepared_select::PreparedComparison;
use crate::executor::projection::Projection;
use crate::executor::query_classification::QueryClassification;
use crate::executor::utils::{extract_literal_value, flip_operator, infix_to_operator};
use crate::parser::ast::{Expression, InfixOperator, SelectStatement};
use crate::storage::mvcc::index_plan::{self, IndexPlan, IndexTerm, Term};
use crate::storage::mvcc::version_store::LayoutToken;
use crate::storage::mvcc::VersionStore;
use crate::storage::traits::{ScanAccess, Table};
use smallvec::SmallVec;

pub(crate) mod block;
pub(crate) mod join;

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
    /// The ORDER BY keeping only the rows the LIMIT keeps
    TopN,
    /// The ORDER BY over every row
    Sort,
    /// The first row of each DISTINCT ON key, over the sorted rows
    DistinctOn,
    /// The first row of each output value
    Distinct,
    /// The groups the HAVING keeps
    Having,
    /// The window functions over the rows the read passes, and the SELECT
    /// list over them
    Window,
    Project,
    /// The WHERE's conjuncts the read did not take
    Filter,
}

/// How a plan reads its table; a scan without a choice is the table's to
/// choose, as for a statement whose WHERE asks a block
#[derive(Debug)]
pub enum Read {
    /// The row the primary key equality names
    PkGet(KeyRead),
    /// The row a unique index of one column holds for its equality, asked
    /// the whole WHERE
    UniqueGet(KeyRead),
    /// A scan, through the indexes the choice reads
    Scan(Option<ScanChoice>),
    /// The rows in ORDER BY order. When the table declines the read, its
    /// top-k scan; when that declines too, `otherwise` sorted by `sort`
    Ordered {
        read: OrderedRead,
        otherwise: Option<ScanChoice>,
        sort: Step,
    },
    /// The rows of what the FROM names in place of a table, each judged by
    /// the whole WHERE: its line as EXPLAIN names it
    Source(String),
    /// The distinct values of the column, lowercase, the table keeps in its
    /// index or volumes; `otherwise` when it keeps none
    DistinctValues {
        column: SmartString,
        otherwise: Box<Read>,
    },
    /// The rows a window step reads through an index; `otherwise` when the
    /// table cannot read them so in this run
    Window {
        read: WindowRead,
        otherwise: Box<Read>,
    },
}

/// How a statement's window functions read their table's rows through an
/// index on one column, lowercase
#[derive(Debug, Clone)]
pub enum WindowRead {
    /// Grouped by every window's one PARTITION BY column; `lazy` reads a
    /// partition at a time until the LIMIT is met
    Partitions { column: SmartString, lazy: bool },
    /// In the order of the first window's one ORDER BY column, which holds
    /// no NULL; `limited` reads only the rows the LIMIT keeps
    Ordered {
        column: SmartString,
        ascending: bool,
        limited: bool,
    },
}

impl Read {
    /// The scan the plan chose, the fallback of an ordered read or of a key
    /// read; None for a scan the table chooses
    pub fn scan(&self) -> Option<&ScanChoice> {
        match self {
            Self::PkGet(key) | Self::UniqueGet(key) => key.otherwise.scan(),
            Self::DistinctValues { otherwise, .. } | Self::Window { otherwise, .. } => {
                otherwise.scan()
            }
            Self::Source(_) => None,
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
            Self::PkGet(key) | Self::UniqueGet(key) => key.otherwise.ordered(),
            Self::DistinctValues { otherwise, .. } => otherwise.ordered(),
            _ => None,
        }
    }

    /// The column whose distinct values the plan reads
    pub fn distinct_values(&self) -> Option<&str> {
        match self {
            Self::DistinctValues { column, .. } => Some(column),
            _ => None,
        }
    }

    /// The index read of a plan whose window functions read through one
    pub fn window(&self) -> Option<&WindowRead> {
        match self {
            Self::Window { read, .. } => Some(read),
            _ => None,
        }
    }
}

impl WindowRead {
    /// The read as EXPLAIN shows it over `table`
    fn explain(&self, table: &str) -> String {
        match self {
            Self::Partitions { column, lazy } => match lazy {
                true => format!("PartitionsByIndex {table} ({column}) one at a time to the LIMIT"),
                false => format!("PartitionsByIndex {table} ({column})"),
            },
            Self::Ordered {
                column,
                ascending,
                limited,
            } => {
                let direction = if *ascending { "ASC" } else { "DESC" };
                match limited {
                    true => format!("IndexOrder {table} ({column} {direction}) to the LIMIT"),
                    false => format!("IndexOrder {table} ({column} {direction})"),
                }
            }
        }
    }
}

/// The read of the row a primary key equality names, and what a run keeps
/// of it
#[derive(Debug)]
pub struct KeyRead {
    pub(crate) key: PreparedComparison,
    /// The read of a run whose key names no row id
    pub(crate) otherwise: Box<Read>,
    /// The WHERE's other conjuncts
    pub(crate) filter: Option<SharedProgram>,
    pub(crate) projection: Projection,
    /// Whether the output row is the table row as it is
    pub(crate) whole_row: bool,
    pub(crate) names: CompactArc<Vec<String>>,
    pub(crate) offset: usize,
    pub(crate) limit: Option<usize>,
    pub(crate) table: SmartString,
    pub(crate) schema: CompactArc<Schema>,
    pub(crate) token: LayoutToken,
    /// The table's store, without keeping a dropped table's rows alive
    pub(crate) store: Weak<VersionStore>,
}

/// What EXPLAIN names a plan's steps and read by, as its statement writes them
pub struct PlanText<'a> {
    pub table: &'a str,
    pub limit: Option<String>,
    pub offset: Option<String>,
    /// The ORDER BY keys
    pub keys: Option<String>,
    /// The DISTINCT ON keys
    pub distinct_on: Option<String>,
    /// The HAVING condition
    pub having: Option<String>,
    /// The window functions
    pub windows: Option<String>,
    pub names: &'a [String],
    /// The conjuncts the read takes
    pub pushed: Option<&'a Expression>,
    /// The ORDER BY column and direction of an ordered read
    pub ordered: Option<(&'a str, bool)>,
    /// The WHERE's constant terms, whose run reads no row unless they hold
    pub empty_unless: Option<String>,
    /// The plan of the statement a source's rows come from
    pub body: Vec<String>,
}

impl Plan {
    /// The plan as EXPLAIN shows it: each step indented under the one it
    /// feeds, then the read, a run's alternative reads each after `else`
    pub fn explain(&self, text: PlanText<'_>) -> Vec<String> {
        let mut read = Vec::new();
        if let Some(terms) = &text.empty_unless {
            alternative(&mut read, format!("Empty unless {terms}"));
        }
        read_lines(&self.read, &text, &mut read);
        steps_over(&self.steps, &text, read)
    }
}

/// `steps` as EXPLAIN shows them, each indented under the one it feeds,
/// over `below`, the lines of what the innermost reads
pub fn steps_over(steps: &[Step], text: &PlanText<'_>, below: Vec<String>) -> Vec<String> {
    let mut lines = Vec::new();
    for step in steps {
        let Some(line) = step_line(*step, text) else {
            continue;
        };
        lines.push(format!("{}{line}", "  ".repeat(lines.len())));
    }
    let depth = "  ".repeat(lines.len());
    lines.extend(below.into_iter().map(|line| format!("{depth}{line}")));
    lines
}

/// `line` after the reads before it, as the one a run takes when they decline
fn alternative(read: &mut Vec<String>, line: String) {
    match read.is_empty() {
        true => read.push(line),
        false => read.push(format!("else {line}")),
    }
}

/// `read` as EXPLAIN shows it, after the reads in `lines`
fn read_lines(read: &Read, text: &PlanText<'_>, lines: &mut Vec<String>) {
    let scan = |scan: Option<&ScanChoice>| {
        scan.map_or_else(Vec::new, |scan| scan.explain(text.table, text.pushed))
    };
    match (read, text.ordered) {
        (Read::PkGet(key), _) => {
            alternative(
                lines,
                format!("PkGet {} {}", text.table, key.key.describe()),
            );
            read_lines(&key.otherwise, text, lines);
        }
        (Read::UniqueGet(key), _) => {
            alternative(
                lines,
                format!("UniqueGet {} {}", text.table, key.key.describe()),
            );
            read_lines(&key.otherwise, text, lines);
        }
        (Read::DistinctValues { column, otherwise }, _) => {
            alternative(lines, format!("DistinctValues {} ({column})", text.table));
            read_lines(otherwise, text, lines);
        }
        (Read::Window { read, otherwise }, _) => {
            alternative(lines, read.explain(text.table));
            read_lines(otherwise, text, lines);
        }
        (
            Read::Ordered {
                read: ordered,
                otherwise,
                sort,
            },
            Some(column),
        ) => {
            alternative(lines, ordered.explain(text.table, column, text.pushed));
            if *ordered != OrderedRead::TopK {
                let top = OrderedRead::TopK.explain(text.table, column, text.pushed);
                alternative(lines, top);
            }
            if let Some(line) = step_line(*sort, text) {
                alternative(lines, line);
            }
            lines.extend(
                scan(otherwise.as_ref())
                    .into_iter()
                    .map(|l| format!("  {l}")),
            );
        }
        (Read::Source(source), _) => {
            let line = match text.pushed {
                Some(pushed) => format!("{source} {pushed}"),
                None => source.clone(),
            };
            alternative(lines, line);
            lines.extend(text.body.iter().map(|l| format!("  {l}")));
        }
        (other, _) => {
            for (at, line) in scan(other.scan()).into_iter().enumerate() {
                match at {
                    0 => alternative(lines, line),
                    _ => lines.push(line),
                }
            }
        }
    }
}

/// `step` as EXPLAIN shows it; None for a step the statement leaves empty
fn step_line(step: Step, text: &PlanText<'_>) -> Option<String> {
    Some(match step {
        Step::Limit => match (&text.limit, &text.offset) {
            (None, None) => return None,
            (Some(limit), None) => format!("Limit {limit}"),
            (None, Some(offset)) => format!("Offset {offset}"),
            (Some(limit), Some(offset)) => format!("Limit {limit} offset {offset}"),
        },
        Step::TopN => format!("TopN {}", text.keys.as_deref()?),
        Step::Sort => format!("Sort {}", text.keys.as_deref()?),
        Step::DistinctOn => format!("DistinctOn {}", text.distinct_on.as_deref()?),
        Step::Distinct => "Distinct".to_string(),
        Step::Having => format!("Having {}", text.having.as_deref()?),
        Step::Window => format!("Window {}", text.windows.as_deref()?),
        Step::Project => format!("Project {}", text.names.join(", ")),
        Step::Filter => "Filter".to_string(),
    })
}

/// How a table's scan finds the rows the conjuncts it takes keep
#[derive(Debug)]
pub enum ScanChoice {
    /// Every row, each one judged
    Seq,
    /// The rows the indexes of `store` give for the conjuncts
    Index(IndexPlan, Weak<VersionStore>),
}

/// The column of `schema` a DISTINCT statement reads the distinct values of
/// instead of its rows: one plain column selected, and no DISTINCT ON, WHERE,
/// grouping, aggregate or window
pub(crate) fn distinct_values_column(
    stmt: &SelectStatement,
    classification: &QueryClassification,
    schema: &Schema,
) -> Option<SmartString> {
    if !stmt.distinct
        || !stmt.distinct_on.is_empty()
        || classification.has_where
        || classification.has_group_by
        || classification.has_having
        || classification.has_aggregation
        || classification.has_window_functions
    {
        return None;
    }
    let [item] = stmt.columns.as_slice() else {
        return None;
    };
    let item = match item {
        Expression::Aliased(aliased) => &*aliased.expression,
        item => item,
    };
    let column = match item {
        Expression::Identifier(id) => &id.value_lower,
        Expression::QualifiedIdentifier(qid) => &qid.name.value_lower,
        _ => return None,
    };
    schema
        .has_column(column)
        .then(|| SmartString::from(column.as_str()))
}

/// The scan for `pushed`, the conjuncts a scan takes of a WHERE, over a
/// table of `schema` whose indexes `store` holds
pub fn scan_choice(
    store: &std::sync::Arc<VersionStore>,
    schema: &Schema,
    pushed: Option<&Expression>,
) -> ScanChoice {
    match pushed.and_then(|pushed| index_plan::decide(store, schema, pushed, false)) {
        Some(plan) => ScanChoice::Index(plan, std::sync::Arc::downgrade(store)),
        None => ScanChoice::Seq,
    }
}

impl ScanChoice {
    /// The access a scan of `table` reads by; None when the indexes were
    /// chosen over another store than the one `table` reads, which leaves
    /// the choice to the run
    pub(crate) fn access(&self, table: &dyn Table) -> Option<ScanAccess<'_>> {
        match self {
            Self::Seq => Some(ScanAccess::Seq),
            Self::Index(plan, store) => table.reads_store(store).then_some(ScanAccess::Index(plan)),
        }
    }

    /// The scan as EXPLAIN shows it: its line, then the plan's reads
    /// indented under it
    pub fn explain(&self, table: &str, pushed: Option<&Expression>) -> Vec<String> {
        let mut lines = vec![match (self, pushed) {
            (Self::Index(..), Some(pushed)) => format!("IndexScan {table} {pushed}"),
            (_, Some(pushed)) => format!("SeqScan {table} {pushed}"),
            (_, None) => format!("SeqScan {table}"),
        }];
        if let Self::Index(plan, _) = self {
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
