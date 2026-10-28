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

//! Which indexes read a WHERE: one rule, for a statement's plan and for a
//! scan nobody planned. A plan holds no values; a run reads them from the
//! condition it scans by

use std::sync::Arc;

use smallvec::SmallVec;

use crate::core::{DataType, IndexType, Operator, Schema};
use crate::storage::expression::{BetweenExpr, Expression, InListExpr};
use crate::storage::traits::Index;

use super::VersionStore;

/// A WHERE term as the index rule reads it
#[derive(Clone, Copy)]
pub enum Term<'a> {
    And,
    Or,
    /// `column operator value`
    Compare {
        column: &'a str,
        operator: Operator,
    },
    /// `column BETWEEN low AND high`
    Between {
        column: &'a str,
    },
    /// `column IN (values)`
    In {
        column: &'a str,
    },
    /// `column LIKE pattern`, its pattern a prefix or bound per run
    LikePrefix {
        column: &'a str,
    },
    Other,
}

/// A WHERE tree the index rule reads: the bound statement's at preparation,
/// the scan's condition otherwise
pub trait IndexTerm {
    fn term<'s>(&'s self, schema: &'s Schema) -> Term<'s>;
    /// Each operand of an AND or OR term
    fn each_operand<'s>(&'s self, each: &mut dyn FnMut(&'s Self));
}

impl<'e> IndexTerm for dyn Expression + 'e {
    fn term<'s>(&'s self, _schema: &'s Schema) -> Term<'s> {
        if self.get_and_operands().is_some() {
            return Term::And;
        }
        if self.get_or_operands().is_some() {
            return Term::Or;
        }
        if let Some((column, operator, _)) = self.get_comparison_info() {
            return Term::Compare { column, operator };
        }
        let any = self.as_any();
        if let Some(between) = any.downcast_ref::<BetweenExpr>() {
            return match (between.is_negated(), between.get_column_name()) {
                (false, Some(column)) => Term::Between { column },
                _ => Term::Other,
            };
        }
        if let Some(list) = any.downcast_ref::<InListExpr>() {
            return match (list.is_not(), list.get_column_name()) {
                (false, Some(column)) => Term::In { column },
                _ => Term::Other,
            };
        }
        match self.get_like_prefix_info() {
            Some((column, prefix, false))
                if crate::core::pattern::Pattern::prefix_range_end(&prefix).is_some() =>
            {
                Term::LikePrefix { column }
            }
            _ => Term::Other,
        }
    }

    fn each_operand<'s>(&'s self, each: &mut dyn FnMut(&'s Self)) {
        let operands = self.get_and_operands().or_else(|| self.get_or_operands());
        for operand in operands.into_iter().flatten() {
            each(operand.as_ref());
        }
    }
}

/// How an index reads one column's conjuncts
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Read {
    Equal,
    In,
    LikePrefix,
    Range,
}

/// The indexes a WHERE's rows are found by
pub enum IndexPlan {
    /// The row a primary key equality names
    Key,
    /// The rows a primary key IN list's members name, in member order
    Keys,
    /// One index reading one column's conjuncts
    Read { index: Arc<dyn Index>, read: Read },
    /// A multi-column index: equalities on its first `equal` columns, and
    /// bounds on the next one when `ranged`
    Prefix {
        index: Arc<dyn Index>,
        equal: usize,
        ranged: bool,
    },
    /// The rows every read names
    Intersect(Vec<IndexPlan>),
    /// The rows any read names, one read for each operand of an OR
    Union(Vec<IndexPlan>),
}

impl std::fmt::Debug for IndexPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.explain().join("; "))
    }
}

impl IndexPlan {
    /// The plan as EXPLAIN shows it, one line per read, nested reads
    /// indented under their set operation
    pub fn explain(&self) -> Vec<String> {
        let mut lines = Vec::new();
        self.explain_into(0, &mut lines);
        lines
    }

    fn explain_into(&self, depth: usize, lines: &mut Vec<String>) {
        let indent = "  ".repeat(depth);
        match self {
            Self::Key => lines.push(format!("{indent}PkKey")),
            Self::Keys => lines.push(format!("{indent}PkKeys")),
            Self::Read { index, read } => lines.push(format!(
                "{indent}Index {} ({} {})",
                index.name(),
                index.column_names()[0],
                match read {
                    Read::Equal => "=",
                    Read::In => "IN",
                    Read::LikePrefix => "LIKE prefix",
                    Read::Range => "range",
                }
            )),
            Self::Prefix {
                index,
                equal,
                ranged,
            } => {
                let columns = index.column_names();
                let mut keys: Vec<String> =
                    columns[..*equal].iter().map(|c| format!("{c} =")).collect();
                if *ranged {
                    keys.push(format!("{} range", columns[*equal]));
                }
                lines.push(format!(
                    "{indent}CompositeIndex {} ({})",
                    index.name(),
                    keys.join(", ")
                ));
            }
            Self::Intersect(reads) | Self::Union(reads) => {
                let name = match self {
                    Self::Intersect(_) => "Intersect",
                    _ => "Union",
                };
                lines.push(format!("{indent}{name}"));
                for read in reads {
                    read.explain_into(depth + 1, lines);
                }
            }
        }
    }
}

/// The conjuncts of `term`, AND terms opened
pub fn conjuncts<'s, T: IndexTerm + ?Sized>(
    term: &'s T,
    schema: &'s Schema,
    out: &mut SmallVec<[&'s T; 8]>,
) {
    match term.term(schema) {
        Term::And => term.each_operand(&mut |operand| conjuncts(operand, schema, out)),
        _ => out.push(term),
    }
}

/// The operands of an OR term, nested OR terms opened
pub fn disjuncts<'s, T: IndexTerm + ?Sized>(
    term: &'s T,
    schema: &'s Schema,
    out: &mut SmallVec<[&'s T; 4]>,
) {
    match term.term(schema) {
        Term::Or => term.each_operand(&mut |operand| disjuncts(operand, schema, out)),
        _ => out.push(term),
    }
}

/// What a scope's conjuncts ask of one column
struct Asked<'a> {
    column: &'a str,
    equal: bool,
    lower: bool,
    upper: bool,
    listed: bool,
    prefixed: bool,
}

/// The indexes that find the rows of `where_` on a table of `schema`, None
/// when no index answers it. A PK equality names its row and a PK list its
/// members' rows; a multi-column
/// index reads equalities on its leading columns and a bounded range on the
/// next; each other column is read by its own index, an equality before a
/// list, a LIKE prefix and a range. A truth value equality is left to the
/// scan. An OR is read when each operand is
pub fn decide<T: IndexTerm + ?Sized>(
    store: &VersionStore,
    schema: &Schema,
    where_: &T,
) -> Option<IndexPlan> {
    let mut scope = SmallVec::new();
    conjuncts(where_, schema, &mut scope);
    decide_scope(store, schema, &scope)
}

fn decide_scope<T: IndexTerm + ?Sized>(
    store: &VersionStore,
    schema: &Schema,
    scope: &[&T],
) -> Option<IndexPlan> {
    let pk = schema
        .pk_column_index()
        .map(|index| schema.columns[index].name.as_str());
    let mut asked: SmallVec<[Asked<'_>; 4]> = SmallVec::new();
    let mut ors: SmallVec<[&T; 2]> = SmallVec::new();
    for &conjunct in scope {
        let term = conjunct.term(schema);
        let (column, operator) = match term {
            Term::Compare { column, operator } => (column, Some(operator)),
            Term::Between { column } | Term::In { column } | Term::LikePrefix { column } => {
                (column, None)
            }
            Term::Or => {
                ors.push(conjunct);
                continue;
            }
            Term::And | Term::Other => continue,
        };
        if operator == Some(Operator::Eq) && pk.is_some_and(|pk| pk.eq_ignore_ascii_case(column)) {
            return Some(IndexPlan::Key);
        }
        let at = match asked
            .iter()
            .position(|a| a.column.eq_ignore_ascii_case(column))
        {
            Some(at) => at,
            None => {
                asked.push(Asked {
                    column,
                    equal: false,
                    lower: false,
                    upper: false,
                    listed: false,
                    prefixed: false,
                });
                asked.len() - 1
            }
        };
        let a = &mut asked[at];
        match (term, operator) {
            (_, Some(Operator::Eq)) => a.equal = true,
            (_, Some(Operator::Gt | Operator::Gte)) => a.lower = true,
            (_, Some(Operator::Lt | Operator::Lte)) => a.upper = true,
            (Term::Between { .. }, _) => (a.lower, a.upper) = (true, true),
            (Term::In { .. }, _) => a.listed = true,
            (Term::LikePrefix { .. }, _) => a.prefixed = true,
            _ => {}
        }
    }
    if asked
        .iter()
        .any(|a| a.listed && pk.is_some_and(|pk| pk.eq_ignore_ascii_case(a.column)))
    {
        return Some(IndexPlan::Keys);
    }

    let mut reads: SmallVec<[IndexPlan; 2]> = SmallVec::new();
    let equal: SmallVec<[&str; 4]> = asked.iter().filter(|a| a.equal).map(|a| a.column).collect();
    // Two columns at least: two equalities, or one and another's range
    let pair = equal.len() >= 2 || (equal.len() == 1 && asked.iter().any(|a| a.upper && !a.equal));
    let prefix = pair.then(|| composite(store, &asked, &equal, 2)).flatten();
    let covered = prefix.as_ref().map_or(&[][..], |(index, (equal, ranged))| {
        &index.column_names()[..equal + *ranged as usize]
    });
    for a in &asked {
        if covered.iter().any(|c| c.eq_ignore_ascii_case(a.column)) {
            continue;
        }
        let Some(index) = store.get_index_by_column(a.column) else {
            // A column only a multi-column index leads is read by it
            if let Some((index, (1, ranged))) = composite(store, &asked, &[a.column], 1) {
                reads.push(IndexPlan::Prefix {
                    index,
                    equal: 1,
                    ranged,
                });
            }
            continue;
        };
        let truth = index.data_types() == [DataType::Boolean];
        let read = if a.equal && !truth {
            Read::Equal
        } else if a.listed {
            Read::In
        } else if a.prefixed && index.data_types() == [DataType::Text] {
            Read::LikePrefix
        } else if (a.lower || a.upper)
            && !matches!(index.index_type(), IndexType::Hash | IndexType::Bitmap)
        {
            Read::Range
        } else {
            continue;
        };
        reads.push(IndexPlan::Read { index, read });
    }
    if let Some((index, (equal, ranged))) = prefix {
        reads.insert(
            0,
            IndexPlan::Prefix {
                index,
                equal,
                ranged,
            },
        );
    }
    for or in ors {
        let mut operands = SmallVec::new();
        disjuncts(or, schema, &mut operands);
        let union: Option<Vec<IndexPlan>> = operands
            .iter()
            .map(|&operand| {
                let mut scope = SmallVec::new();
                conjuncts(operand, schema, &mut scope);
                decide_scope(store, schema, &scope)
            })
            .collect();
        if let Some(union) = union {
            reads.push(IndexPlan::Union(union));
        }
    }
    match reads.len() {
        0 => None,
        1 => reads.pop(),
        _ => Some(IndexPlan::Intersect(reads.into_vec())),
    }
}

/// The multi-column index `equal`'s columns lead, with how many it reads by
/// equality and whether it reads the next column's range, when it reads at
/// least `least` columns. A range needs an upper bound, and the next column
/// must be the index's last: a bound shorter than the keys cuts them wrongly
fn composite(
    store: &VersionStore,
    asked: &[Asked<'_>],
    equal: &[&str],
    least: usize,
) -> Option<(Arc<dyn Index>, (usize, bool))> {
    if equal.is_empty() {
        return None;
    }
    let (index, matched) = store.get_multi_column_index(equal)?;
    let columns = index.column_names();
    let ranged = matched + 1 == columns.len()
        && asked
            .iter()
            .any(|a| a.upper && a.column.eq_ignore_ascii_case(&columns[matched]));
    (matched + ranged as usize >= least).then_some((index, (matched, ranged)))
}
