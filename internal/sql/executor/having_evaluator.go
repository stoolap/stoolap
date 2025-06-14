/*
Copyright 2025 Stoolap Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package executor

import (
	"fmt"
	"strings"

	"github.com/stoolap/stoolap/internal/parser"
	"github.com/stoolap/stoolap/internal/storage"
)

// EvaluateHavingExpression evaluates a HAVING clause expression where aggregate functions
// are treated as column references to already-computed aggregate values
func (e *Evaluator) EvaluateHavingExpression(expr parser.Expression, row map[string]storage.ColumnValue) (storage.ColumnValue, error) {
	if expr == nil {
		return storage.NewBooleanValue(true), nil
	}

	switch expr := expr.(type) {
	case *parser.FunctionCall:
		// In HAVING context, aggregate functions are column references
		if IsAggregateFunction(expr.Function) {
			// Build the column name that matches how aggregates are named
			var columnName string
			if len(expr.Arguments) == 1 {
				if ident, ok := expr.Arguments[0].(*parser.Identifier); ok {
					if ident.Value == "*" {
						columnName = fmt.Sprintf("%s(*)", strings.ToUpper(expr.Function))
					} else {
						columnName = fmt.Sprintf("%s(%s)", strings.ToUpper(expr.Function), ident.Value)
					}
				} else {
					// For other expressions, use the string representation
					columnName = fmt.Sprintf("%s(%s)", strings.ToUpper(expr.Function), expr.Arguments[0].String())
				}
			} else if len(expr.Arguments) == 0 {
				// Functions without arguments
				columnName = fmt.Sprintf("%s()", strings.ToUpper(expr.Function))
			}

			// Look up the column value directly
			if val, ok := row[columnName]; ok {
				return val, nil
			}

			// Try lowercase function name
			columnNameLower := fmt.Sprintf("%s(", strings.ToLower(expr.Function))
			for colName, val := range row {
				if strings.HasPrefix(colName, columnNameLower) {
					return val, nil
				}
			}

			// Try with aliases - check if any alias maps to this aggregate function
			if e.columnAliases != nil {
				// First try direct lookup
				if alias, ok := e.columnAliases[columnName]; ok {
					if val, ok := row[alias]; ok {
						return val, nil
					}
				}

				// Then check if any column in the row has an alias that maps to our aggregate
				for colName, val := range row {
					// Check if this column name (possibly an alias) maps back to our aggregate
					for aliasName, aliasTarget := range e.columnAliases {
						if colName == aliasName && aliasTarget == columnName {
							return val, nil
						}
					}
				}
			}

			// If none of the above worked, check if there's only one column and it's likely our aggregate
			// This handles cases like "SELECT SUM(value) as total" where we're looking for SUM(value)
			// but the only column is "total"
			if len(row) == 1 {
				for _, val := range row {
					return val, nil
				}
			}

			return storage.StaticNullUnknown, fmt.Errorf("aggregate column %s not found in result", columnName)
		}

		// Non-aggregate functions are evaluated normally
		return e.Evaluate(expr)

	case *parser.InfixExpression:
		// Create a new expression with resolved aggregate functions
		newLeft := expr.Left
		newRight := expr.Right

		// If left is an aggregate function, replace it with an identifier
		if leftFunc, ok := expr.Left.(*parser.FunctionCall); ok && IsAggregateFunction(leftFunc.Function) {
			var columnName string
			if len(leftFunc.Arguments) == 1 {
				if ident, ok := leftFunc.Arguments[0].(*parser.Identifier); ok {
					if ident.Value == "*" {
						columnName = fmt.Sprintf("%s(*)", strings.ToUpper(leftFunc.Function))
					} else {
						columnName = fmt.Sprintf("%s(%s)", strings.ToUpper(leftFunc.Function), ident.Value)
					}
				}
			}
			newLeft = &parser.Identifier{Value: columnName}
		}

		// If right is an aggregate function, replace it with an identifier
		if rightFunc, ok := expr.Right.(*parser.FunctionCall); ok && IsAggregateFunction(rightFunc.Function) {
			var columnName string
			if len(rightFunc.Arguments) == 1 {
				if ident, ok := rightFunc.Arguments[0].(*parser.Identifier); ok {
					if ident.Value == "*" {
						columnName = fmt.Sprintf("%s(*)", strings.ToUpper(rightFunc.Function))
					} else {
						columnName = fmt.Sprintf("%s(%s)", strings.ToUpper(rightFunc.Function), ident.Value)
					}
				}
			}
			newRight = &parser.Identifier{Value: columnName}
		}

		// Create a new infix expression and evaluate it
		newExpr := &parser.InfixExpression{
			Left:     newLeft,
			Operator: expr.Operator,
			Right:    newRight,
		}

		e.WithRow(row)
		return e.Evaluate(newExpr)

	case *parser.PrefixExpression:
		// Evaluate the operand
		rightVal, err := e.EvaluateHavingExpression(expr.Right, row)
		if err != nil {
			return storage.StaticNullUnknown, err
		}
		defer storage.PutPooledColumnValue(rightVal)

		// Apply the prefix operator inline
		if expr.Operator == "NOT" {
			// Handle NOT operator
			if rightVal.IsNull() {
				return storage.StaticNullBoolean, nil
			}
			b, ok := rightVal.AsBoolean()
			if !ok {
				return storage.StaticNullBoolean, fmt.Errorf("NOT operator requires boolean operand")
			}
			return storage.NewBooleanValue(!b), nil
		}
		return storage.StaticNullUnknown, fmt.Errorf("unsupported prefix operator: %s", expr.Operator)

	default:
		// For all other expressions, use the regular evaluator
		// This includes literals, identifiers, etc.
		e.WithRow(row)
		return e.Evaluate(expr)
	}
}

// EvaluateHavingClause evaluates a HAVING clause and returns true if the row matches
func (e *Evaluator) EvaluateHavingClause(expr parser.Expression, row map[string]storage.ColumnValue) (bool, error) {
	result, err := e.EvaluateHavingExpression(expr, row)
	if err != nil {
		return false, err
	}
	defer storage.PutPooledColumnValue(result)

	// Convert to boolean
	if result.IsNull() {
		return false, nil
	}

	b, ok := result.AsBoolean()
	if !ok {
		return false, fmt.Errorf("HAVING clause must evaluate to boolean")
	}

	return b, nil
}
