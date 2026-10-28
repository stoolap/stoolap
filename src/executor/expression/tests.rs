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

// Tests for the Compiled Expression VM

use std::sync::Arc;

use crate::common::SmartString;

use crate::common::CompactArc;

use super::compiler::{CompileContext, ExprCompiler};
use super::ops::Op;
use super::program::Program;
use super::vm::{ExecuteContext, ExprVM};
use crate::core::pattern::Pattern;
use crate::core::{Value, ValueSet};
use crate::parser::ast::Expression;
use crate::Row;

#[test]
fn test_simple_load_and_compare() {
    let mut vm = ExprVM::new();

    // col[0] > 5
    let program = Program::new(vec![
        Op::LoadColumn(0),
        Op::LoadConst(Value::Integer(5)),
        Op::Gt,
        Op::Return,
    ]);

    // True case
    let row = Row::from_values(vec![Value::Integer(10)]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));

    // False case
    let row = Row::from_values(vec![Value::Integer(3)]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(false));
}

#[test]
fn test_null_comparison() {
    let mut vm = ExprVM::new();

    // col[0] > 5 with NULL col
    let program = Program::new(vec![
        Op::LoadColumn(0),
        Op::LoadConst(Value::Integer(5)),
        Op::Gt,
        Op::Return,
    ]);

    let row = Row::from_values(vec![Value::Null(crate::core::DataType::Integer)]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert!(result.is_null());
}

#[test]
fn test_and_short_circuit() {
    let mut vm = ExprVM::new();

    // col[0] > 5 AND col[1] < 10
    let program = Program::new(vec![
        Op::LoadColumn(0),
        Op::LoadConst(Value::Integer(5)),
        Op::Gt,
        Op::And(10), // Jump to position 10 if false
        Op::LoadColumn(1),
        Op::LoadConst(Value::Integer(10)),
        Op::Lt,
        Op::AndFinalize,
        Op::Return,
        Op::Nop,                              // Position 9
        Op::LoadConst(Value::Boolean(false)), // Position 10
        Op::Return,
    ]);

    // Both true
    let row = Row::from_values(vec![Value::Integer(10), Value::Integer(5)]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));

    // First false (short circuit)
    let row = Row::from_values(vec![Value::Integer(3), Value::Integer(5)]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(false));

    // First true, second false
    let row = Row::from_values(vec![Value::Integer(10), Value::Integer(15)]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(false));
}

#[test]
fn test_or_short_circuit() {
    let mut vm = ExprVM::new();

    // col[0] < 5 OR col[1] > 10
    let program = Program::new(vec![
        Op::LoadColumn(0),
        Op::LoadConst(Value::Integer(5)),
        Op::Lt,
        Op::Or(10), // Jump to position 10 if true
        Op::LoadColumn(1),
        Op::LoadConst(Value::Integer(10)),
        Op::Gt,
        Op::OrFinalize,
        Op::Return,
        Op::Nop,                             // Position 9
        Op::LoadConst(Value::Boolean(true)), // Position 10
        Op::Return,
    ]);

    // First true (short circuit)
    let row = Row::from_values(vec![Value::Integer(3), Value::Integer(5)]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));

    // First false, second true
    let row = Row::from_values(vec![Value::Integer(10), Value::Integer(15)]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));

    // Both false
    let row = Row::from_values(vec![Value::Integer(10), Value::Integer(5)]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(false));
}

#[test]
fn test_arithmetic() {
    let mut vm = ExprVM::new();

    // col[0] + col[1] * 2
    let program = Program::new(vec![
        Op::LoadColumn(0),
        Op::LoadColumn(1),
        Op::LoadConst(Value::Integer(2)),
        Op::Mul,
        Op::Add,
        Op::Return,
    ]);

    let row = Row::from_values(vec![Value::Integer(5), Value::Integer(3)]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Integer(11)); // 5 + 3*2 = 11
}

#[test]
fn test_in_set() {
    let mut vm = ExprVM::new();

    let set: ValueSet = [Value::Integer(1), Value::Integer(2), Value::Integer(3)]
        .into_iter()
        .collect();

    let program = Program::new(vec![
        Op::LoadColumn(0),
        Op::InSet(CompactArc::new(crate::core::predicate::PredicateSet::new(
            CompactArc::new(set),
            false,
        ))),
        Op::Return,
    ]);

    // In set
    let row = Row::from_values(vec![Value::Integer(2)]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));

    // Not in set
    let row = Row::from_values(vec![Value::Integer(5)]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(false));
}

#[test]
fn test_between() {
    let mut vm = ExprVM::new();

    // col[0] BETWEEN 5 AND 10
    let program = Program::new(vec![
        Op::LoadColumn(0),
        Op::LoadConst(Value::Integer(5)),
        Op::LoadConst(Value::Integer(10)),
        Op::Between,
        Op::Return,
    ]);

    // In range
    let row = Row::from_values(vec![Value::Integer(7)]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));

    // Below range
    let row = Row::from_values(vec![Value::Integer(3)]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(false));

    // Above range
    let row = Row::from_values(vec![Value::Integer(15)]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(false));
}

#[test]
fn test_like_pattern() {
    let mut vm = ExprVM::new();

    // col[0] LIKE 'test%'
    let pattern = Pattern::like("test%", false);
    let program = Program::new(vec![
        Op::LoadColumn(0),
        Op::Like(Arc::new(pattern)),
        Op::Return,
    ]);

    // Match
    let row = Row::from_values(vec![Value::Text(SmartString::from("testing"))]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));

    // No match
    let row = Row::from_values(vec![Value::Text(SmartString::from("other"))]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(false));
}

#[test]
fn test_is_null() {
    let mut vm = ExprVM::new();

    // col[0] IS NULL
    let program = Program::new(vec![Op::LoadColumn(0), Op::IsNull, Op::Return]);

    // Is null
    let row = Row::from_values(vec![Value::Null(crate::core::DataType::Integer)]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));

    // Not null
    let row = Row::from_values(vec![Value::Integer(5)]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(false));
}

#[test]
fn test_coalesce() {
    let mut vm = ExprVM::new();

    // COALESCE(col[0], col[1], 'default')
    let program = Program::new(vec![
        Op::LoadColumn(0),
        Op::LoadColumn(1),
        Op::LoadConst(Value::Text(SmartString::from("default"))),
        Op::Coalesce(3),
        Op::Return,
    ]);

    // First non-null
    let row = Row::from_values(vec![
        Value::Text(SmartString::from("first")),
        Value::Text(SmartString::from("second")),
    ]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Text(SmartString::from("first")));

    // Second non-null
    let row = Row::from_values(vec![
        Value::Null(crate::core::DataType::Text),
        Value::Text(SmartString::from("second")),
    ]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Text(SmartString::from("second")));

    // Default
    let row = Row::from_values(vec![
        Value::Null(crate::core::DataType::Text),
        Value::Null(crate::core::DataType::Text),
    ]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Text(SmartString::from("default")));
}

#[test]
fn test_join_context() {
    let mut vm = ExprVM::new();

    // row1.col[0] = row2.col[0]
    let program = Program::new(vec![
        Op::LoadColumn(0),  // From first row
        Op::LoadColumn2(0), // From second row
        Op::Eq,
        Op::Return,
    ]);

    let row1 = Row::from_values(vec![Value::Integer(5)]);
    let row2 = Row::from_values(vec![Value::Integer(5)]);
    let ctx = ExecuteContext::for_join(&row1, &row2);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));

    let row1 = Row::from_values(vec![Value::Integer(5)]);
    let row2 = Row::from_values(vec![Value::Integer(10)]);
    let ctx = ExecuteContext::for_join(&row1, &row2);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(false));
}

#[test]
fn test_parameters() {
    let mut vm = ExprVM::new();

    // col[0] = $1
    let program = Program::new(vec![
        Op::LoadColumn(0),
        Op::LoadParam(0),
        Op::Eq,
        Op::Return,
    ]);

    let row = Row::from_values(vec![Value::Integer(42)]);
    let params = vec![Value::Integer(42)];
    let ctx = ExecuteContext::new(&row).with_params(&params);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));

    let params = vec![Value::Integer(100)];
    let ctx = ExecuteContext::new(&row).with_params(&params);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(false));
}

#[test]
fn test_execute_bool() {
    let mut vm = ExprVM::new();

    let program = Program::new(vec![
        Op::LoadColumn(0),
        Op::LoadConst(Value::Integer(5)),
        Op::Gt,
        Op::Return,
    ]);

    // True
    let row = Row::from_values(vec![Value::Integer(10)]);
    let ctx = ExecuteContext::new(&row);
    assert!(vm.execute_bool(&program, &ctx));

    // False
    let row = Row::from_values(vec![Value::Integer(3)]);
    let ctx = ExecuteContext::new(&row);
    assert!(!vm.execute_bool(&program, &ctx));

    // NULL -> false
    let row = Row::from_values(vec![Value::Null(crate::core::DataType::Integer)]);
    let ctx = ExecuteContext::new(&row);
    assert!(!vm.execute_bool(&program, &ctx));
}

// ============================================================================
// Compiler Tests
// ============================================================================

#[test]
fn test_compiler_simple_expression() {
    use crate::parser::ast::*;
    use crate::parser::token::{Position, Token, TokenType};

    let columns = vec!["a".to_string(), "b".to_string()];
    let ctx = CompileContext::with_global_registry(&columns);
    let compiler = ExprCompiler::new(&ctx);

    fn make_token() -> Token {
        Token {
            token_type: TokenType::Integer,
            literal: "1".into(),
            position: Position {
                offset: 0,
                line: 1,
                column: 1,
            },
            quoted: false,
            keyword: None,
        }
    }

    // a > 5
    let expr = Expression::Infix(InfixExpression {
        token: make_token(),
        left: Box::new(Expression::Identifier(Identifier::new(
            make_token(),
            "a".to_string(),
        ))),
        operator: ">".into(),
        op_type: InfixOperator::GreaterThan,
        right: Box::new(Expression::IntegerLiteral(IntegerLiteral {
            token: make_token(),
            value: 5,
        })),
    });

    let program = compiler.compile(&expr).unwrap();

    // Execute
    let mut vm = ExprVM::new();
    let row = Row::from_values(vec![Value::Integer(10), Value::Integer(20)]);
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));
}

#[test]
fn test_compiled_pattern_prefix() {
    let pattern = Pattern::like("test%", false);
    assert!(pattern.matches("testing"));
    assert!(pattern.matches("test"));
    assert!(!pattern.matches("atest"));
}

#[test]
fn test_compiled_pattern_suffix() {
    let pattern = Pattern::like("%test", false);
    assert!(pattern.matches("mytest"));
    assert!(pattern.matches("test"));
    assert!(!pattern.matches("testa"));
}

#[test]
fn test_compiled_pattern_contains() {
    let pattern = Pattern::like("%test%", false);
    assert!(pattern.matches("mytesting"));
    assert!(pattern.matches("test"));
    assert!(pattern.matches("atest"));
    assert!(!pattern.matches("other"));
}

#[test]
fn test_compiled_pattern_case_insensitive() {
    let pattern = Pattern::like("TEST%", true);
    assert!(pattern.matches("testing"));
    assert!(pattern.matches("TESTING"));
    assert!(pattern.matches("TeStInG"));
}

/// The WHERE of `SELECT * FROM t WHERE {predicate}`
fn where_of(predicate: &str) -> Expression {
    let statements =
        crate::parser::parse_sql(&format!("SELECT * FROM t WHERE {predicate}")).unwrap();
    match &statements[0] {
        crate::parser::ast::Statement::Select(select) => {
            select.where_clause.as_deref().unwrap().clone()
        }
        other => panic!("not a SELECT: {other:?}"),
    }
}

/// Mark the one subquery block `expr` holds as the prepared block `id`
fn prepare_block(expr: &mut Expression, id: u16) {
    let block = match expr {
        Expression::Exists(exists) => &mut exists.subquery,
        Expression::AllAny(all_any) => &mut all_any.subquery,
        Expression::In(in_expr) => match &mut *in_expr.right {
            Expression::ScalarSubquery(sq) => &mut sq.subquery,
            other => panic!("no block on the right: {other:?}"),
        },
        Expression::Prefix(prefix) => return prepare_block(&mut prefix.right, id),
        other => panic!("no block: {other:?}"),
    };
    block.prepared = Some(id);
}

#[test]
fn test_a_prepared_block_compiles_to_its_id() {
    let columns = vec!["x".to_string()];
    for (predicate, expected) in [
        ("x IN (SELECT y FROM u)", vec!["ExecInSubquery(4)"]),
        (
            "x NOT IN (SELECT y FROM u)",
            vec!["ExecInSubquery(4)", "Not"],
        ),
        ("EXISTS (SELECT 1 FROM u)", vec!["ExecExists(4)"]),
        (
            "x > ALL (SELECT y FROM u)",
            vec!["ExecQuantified(4, Gt, true)"],
        ),
        (
            "x <= ANY (SELECT y FROM u)",
            vec!["ExecQuantified(4, Le, false)"],
        ),
    ] {
        let mut expr = where_of(predicate);
        prepare_block(&mut expr, 4);
        let ctx = CompileContext::with_global_registry(&columns).with_prepared_subqueries();
        let program = ExprCompiler::new(&ctx).compile(&expr).unwrap();
        let ops: Vec<String> = program.ops().iter().map(|op| format!("{op:?}")).collect();
        let copied = ExprCompiler::new(&ctx).compile(&expr.clone()).unwrap();
        let copied: Vec<String> = copied.ops().iter().map(|op| format!("{op:?}")).collect();
        assert_eq!(
            copied, ops,
            "{predicate}: a copy of the mark is the same block"
        );
        let at = ops
            .iter()
            .position(|op| *op == expected[0])
            .unwrap_or_else(|| panic!("{predicate}: no {} in {ops:?}", expected[0]));
        assert_eq!(
            &ops[at..at + expected.len()],
            expected.as_slice(),
            "{predicate}"
        );
    }
}

#[test]
fn test_a_block_outside_a_prepared_plan_does_not_compile() {
    let columns = vec!["x".to_string()];
    for predicate in [
        "x IN (SELECT y FROM u)",
        "EXISTS (SELECT 1 FROM u)",
        "x > ALL (SELECT y FROM u)",
        "x = (SELECT y FROM u)",
    ] {
        let expr = where_of(predicate);
        let ctx = CompileContext::with_global_registry(&columns);
        assert!(
            ExprCompiler::new(&ctx).compile(&expr).is_err(),
            "{predicate}: without ids no subquery op is built"
        );
        let ctx = CompileContext::with_global_registry(&columns).with_prepared_subqueries();
        assert!(
            ExprCompiler::new(&ctx).compile(&expr).is_err(),
            "{predicate}: a block the plan did not prepare does not compile"
        );
    }
}

#[test]
fn a_bound_program_reads_its_columns_by_index_and_no_name() {
    use crate::functions::registry::global_registry;
    use crate::parser::ast::{ColumnRef, Identifier};
    use crate::parser::token::{Position, Token, TokenType};

    let ident = |name: &str| {
        Identifier::new(
            Token::new(TokenType::Identifier, name, Position::default()),
            name,
        )
    };
    let column = |index: u32| {
        Expression::Column(ColumnRef {
            index,
            name: ident("a"),
        })
    };
    let bound = CompileContext::bound(global_registry());
    let row = Row::from_values(
        (0..=u16::MAX)
            .map(|i| Value::Integer(i64::from(i)))
            .collect(),
    );
    for index in [1, u32::from(u16::MAX)] {
        let program = ExprCompiler::new(&bound).compile(&column(index)).unwrap();
        let read = ExprVM::new().execute(&program, &ExecuteContext::new(&row));
        assert_eq!(
            read.unwrap(),
            Value::Integer(i64::from(index)),
            "column {index}"
        );
    }
    assert!(
        ExprCompiler::new(&bound)
            .compile(&Expression::Identifier(ident("a")))
            .is_err(),
        "a bound program looks no name up"
    );
    // A joined index past u16 still picks its row, then the column within it
    let pair = CompileContext::bound_pair(global_registry(), 40_000);
    let program = ExprCompiler::new(&pair).compile(&column(69_999)).unwrap();
    assert!(
        matches!(program.ops().first(), Some(Op::LoadColumn2(29_999))),
        "{:?}",
        program.ops()
    );
    assert!(
        ExprCompiler::new(&bound).compile(&column(1 << 16)).is_err(),
        "a column past what a row holds"
    );
    let named = CompileContext::with_global_registry(&["a".to_string()]);
    assert!(
        ExprCompiler::new(&named).compile(&column(0)).is_err(),
        "a bound column compiles only in a bound program"
    );
}
