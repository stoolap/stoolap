// Copyright 2025 Stoolap Contributors
use ahash::AHashSet;

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

use super::compiler::{CompileContext, ExprCompiler};
use super::ops::{CompiledPattern, Op};
use super::program::Program;
use super::vm::{ExecuteContext, ExprVM};
use crate::core::Value;

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
    let row = vec![Value::Integer(10)];
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));

    // False case
    let row = vec![Value::Integer(3)];
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

    let row = vec![Value::Null(crate::core::DataType::Integer)];
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
    let row = vec![Value::Integer(10), Value::Integer(5)];
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));

    // First false (short circuit)
    let row = vec![Value::Integer(3), Value::Integer(5)];
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(false));

    // First true, second false
    let row = vec![Value::Integer(10), Value::Integer(15)];
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
    let row = vec![Value::Integer(3), Value::Integer(5)];
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));

    // First false, second true
    let row = vec![Value::Integer(10), Value::Integer(15)];
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));

    // Both false
    let row = vec![Value::Integer(10), Value::Integer(5)];
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

    let row = vec![Value::Integer(5), Value::Integer(3)];
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Integer(11)); // 5 + 3*2 = 11
}

#[test]
fn test_in_set() {
    let mut vm = ExprVM::new();

    let set: AHashSet<Value> = [Value::Integer(1), Value::Integer(2), Value::Integer(3)]
        .into_iter()
        .collect();

    let program = Program::new(vec![
        Op::LoadColumn(0),
        Op::InSet(Arc::new(set), false),
        Op::Return,
    ]);

    // In set
    let row = vec![Value::Integer(2)];
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));

    // Not in set
    let row = vec![Value::Integer(5)];
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
    let row = vec![Value::Integer(7)];
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));

    // Below range
    let row = vec![Value::Integer(3)];
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(false));

    // Above range
    let row = vec![Value::Integer(15)];
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(false));
}

#[test]
fn test_like_pattern() {
    let mut vm = ExprVM::new();

    // col[0] LIKE 'test%'
    let pattern = CompiledPattern::compile("test%", false);
    let program = Program::new(vec![
        Op::LoadColumn(0),
        Op::Like(Arc::new(pattern), false),
        Op::Return,
    ]);

    // Match
    let row = vec![Value::Text(Arc::from("testing"))];
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));

    // No match
    let row = vec![Value::Text(Arc::from("other"))];
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
    let row = vec![Value::Null(crate::core::DataType::Integer)];
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));

    // Not null
    let row = vec![Value::Integer(5)];
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
        Op::LoadConst(Value::Text(Arc::from("default"))),
        Op::Coalesce(3),
        Op::Return,
    ]);

    // First non-null
    let row = vec![
        Value::Text(Arc::from("first")),
        Value::Text(Arc::from("second")),
    ];
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Text(Arc::from("first")));

    // Second non-null
    let row = vec![
        Value::Null(crate::core::DataType::Text),
        Value::Text(Arc::from("second")),
    ];
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Text(Arc::from("second")));

    // Default
    let row = vec![
        Value::Null(crate::core::DataType::Text),
        Value::Null(crate::core::DataType::Text),
    ];
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Text(Arc::from("default")));
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

    let row1 = vec![Value::Integer(5)];
    let row2 = vec![Value::Integer(5)];
    let ctx = ExecuteContext::for_join(&row1, &row2);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));

    let row1 = vec![Value::Integer(5)];
    let row2 = vec![Value::Integer(10)];
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

    let row = vec![Value::Integer(42)];
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
    let row = vec![Value::Integer(10)];
    let ctx = ExecuteContext::new(&row);
    assert!(vm.execute_bool(&program, &ctx));

    // False
    let row = vec![Value::Integer(3)];
    let ctx = ExecuteContext::new(&row);
    assert!(!vm.execute_bool(&program, &ctx));

    // NULL -> false
    let row = vec![Value::Null(crate::core::DataType::Integer)];
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
            literal: "1".to_string(),
            position: Position {
                offset: 0,
                line: 1,
                column: 1,
            },
            error: None,
        }
    }

    // a > 5
    let expr = Expression::Infix(InfixExpression {
        token: make_token(),
        left: Box::new(Expression::Identifier(Identifier::new(
            make_token(),
            "a".to_string(),
        ))),
        operator: ">".to_string(),
        op_type: InfixOperator::GreaterThan,
        right: Box::new(Expression::IntegerLiteral(IntegerLiteral {
            token: make_token(),
            value: 5,
        })),
    });

    let program = compiler.compile(&expr).unwrap();

    // Execute
    let mut vm = ExprVM::new();
    let row = vec![Value::Integer(10), Value::Integer(20)];
    let ctx = ExecuteContext::new(&row);
    let result = vm.execute(&program, &ctx).unwrap();
    assert_eq!(result, Value::Boolean(true));
}

#[test]
fn test_compiled_pattern_prefix() {
    let pattern = CompiledPattern::compile("test%", false);
    assert!(pattern.matches("testing", false));
    assert!(pattern.matches("test", false));
    assert!(!pattern.matches("atest", false));
}

#[test]
fn test_compiled_pattern_suffix() {
    let pattern = CompiledPattern::compile("%test", false);
    assert!(pattern.matches("mytest", false));
    assert!(pattern.matches("test", false));
    assert!(!pattern.matches("testa", false));
}

#[test]
fn test_compiled_pattern_contains() {
    let pattern = CompiledPattern::compile("%test%", false);
    assert!(pattern.matches("mytesting", false));
    assert!(pattern.matches("test", false));
    assert!(pattern.matches("atest", false));
    assert!(!pattern.matches("other", false));
}

#[test]
fn test_compiled_pattern_case_insensitive() {
    let pattern = CompiledPattern::compile("TEST%", true);
    assert!(pattern.matches("testing", true));
    assert!(pattern.matches("TESTING", true));
    assert!(pattern.matches("TeStInG", true));
}
