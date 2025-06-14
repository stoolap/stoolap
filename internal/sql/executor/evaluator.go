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
// package executor provides SQL execution functionality
package executor

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/stoolap/stoolap/internal/common"
	"github.com/stoolap/stoolap/internal/functions/contract"
	"github.com/stoolap/stoolap/internal/parser"
	"github.com/stoolap/stoolap/internal/storage"
)

// Evaluator evaluates SQL expressions
type Evaluator struct {
	ctx              context.Context
	functionRegistry contract.FunctionRegistry
	// Row data for column references, storing ColumnValue directly
	row map[string]storage.ColumnValue
	// Mapping from table alias to actual table name
	tableAliases map[string]string
	// Mapping from column alias to original column name
	columnAliases map[string]string
	// For parameter binding
	ps *parameter
}

// NewEvaluator creates a new expression evaluator
func NewEvaluator(ctx context.Context, registry contract.FunctionRegistry) *Evaluator {
	var ps *parameter
	if p, ok := ctx.Value(psContextKey).(*parameter); ok {
		ps = p
	}

	return &Evaluator{
		ctx:              ctx,
		functionRegistry: registry,
		row:              common.GetColumnValueMap(16), // Start with small map from pool
		tableAliases:     make(map[string]string),
		columnAliases:    make(map[string]string),
		ps:               ps,
	}
}

// WithRow sets the current row data for column references
func (e *Evaluator) WithRow(row map[string]storage.ColumnValue) *Evaluator {
	e.row = row
	return e
}

// WithStorageRow sets the current row data using storage.Row
// This optimized version avoids unnecessary type conversions
func (e *Evaluator) WithStorageRow(row storage.Row, columnMap map[string]int) *Evaluator {
	// Get a map from the pool based on column count
	result := common.GetColumnValueMap(len(columnMap))

	for colName, colIdx := range columnMap {
		if colIdx >= 0 && colIdx < len(row) {
			// Store the raw storage.ColumnValue directly in the map
			result[colName] = row[colIdx]
		}
	}

	e.row = result
	return e
}

// WithColumnAliases sets column aliases for resolving alias references
func (e *Evaluator) WithColumnAliases(aliases map[string]string) *Evaluator {
	e.columnAliases = aliases
	return e
}

// WithTableAliases sets the table aliases mapping
func (e *Evaluator) WithTableAliases(aliases map[string]string) *Evaluator {
	e.tableAliases = aliases
	return e
}

// EvaluateWithRow evaluates an expression with a simple row context
func (e *Evaluator) EvaluateWithRow(expr parser.Expression, row map[string]interface{}) (storage.ColumnValue, error) {
	// Convert row to storage.ColumnValue format
	columnValueRow := common.GetColumnValueMap(len(row))
	defer common.PutColumnValueMap(columnValueRow, len(row))

	for col, val := range row {
		columnValueRow[col] = storage.GetPooledColumnValue(val)
	}

	// Set the row and evaluate
	oldRow := e.row
	e.row = columnValueRow
	defer func() { e.row = oldRow }()

	return e.Evaluate(expr)
}

// Evaluate evaluates an expression and returns its storage.ColumnValue
func (e *Evaluator) Evaluate(expr parser.Expression) (storage.ColumnValue, error) {
	if expr == nil {
		return storage.StaticNullUnknown, nil
	}

	switch expr := expr.(type) {
	case *parser.IntegerLiteral:
		return storage.GetPooledIntegerValue(expr.Value), nil

	case *parser.FloatLiteral:
		return storage.GetPooledFloatValue(expr.Value), nil

	case *parser.StringLiteral:
		// Handle special types based on TypeHint
		switch expr.TypeHint {
		case "TIMESTAMP", "DATE", "TIME":
			// Try various timestamp formats
			timestamp, err := storage.ParseTimestamp(expr.Value)
			if err != nil {
				return nil, err
			}
			return storage.GetPooledTimestampValue(timestamp), nil
		case "JSON":
			// For now, treat JSON as string
			return storage.GetPooledJSONValue(expr.Value), nil
		default:
			return storage.GetPooledStringValue(expr.Value), nil
		}

	case *parser.BooleanLiteral:
		return storage.NewBooleanValue(expr.Value), nil

	case *parser.NullLiteral:
		return storage.StaticNullUnknown, nil

	case *parser.IntervalLiteral:
		// Convert interval to duration in nanoseconds
		var duration int64
		switch expr.Unit {
		case "second":
			duration = expr.Quantity * 1e9
		case "minute":
			duration = expr.Quantity * 60 * 1e9
		case "hour":
			duration = expr.Quantity * 3600 * 1e9
		case "day":
			duration = expr.Quantity * 86400 * 1e9
		case "week":
			duration = expr.Quantity * 7 * 86400 * 1e9
		case "month":
			// Approximate: 30 days per month
			duration = expr.Quantity * 30 * 86400 * 1e9
		case "year":
			// Approximate: 365 days per year
			duration = expr.Quantity * 365 * 86400 * 1e9
		default:
			return storage.StaticNullUnknown, fmt.Errorf("unsupported interval unit: %s", expr.Unit)
		}
		// Store as integer (nanoseconds)
		return storage.GetPooledIntegerValue(duration), nil

	case *parser.CastExpression:
		// Handle CAST expressions by evaluating the expression first
		exprValue, err := e.Evaluate(expr.Expr)
		defer storage.PutPooledColumnValue(exprValue)
		if err != nil {
			return storage.StaticNullUnknown, err
		}

		// Special handling for NULL values
		if exprValue.IsNull() {
			switch strings.ToUpper(expr.TypeName) {
			case "INTEGER", "INT":
				return storage.StaticNullInteger, nil
			case "FLOAT", "REAL", "DOUBLE":
				return storage.StaticNullFloat, nil
			case "TEXT", "STRING", "VARCHAR":
				return storage.StaticNullString, nil
			case "BOOLEAN", "BOOL":
				return storage.StaticNullBoolean, nil
			case "TIMESTAMP", "DATE", "TIME":
				return storage.StaticNullTimestamp, nil
			case "JSON":
				return storage.StaticNullJSON, nil
			default:
				return storage.StaticNullUnknown, nil
			}
		}

		// Perform the cast based on the target type
		switch strings.ToUpper(expr.TypeName) {
		case "INTEGER", "INT":
			// Cast to integer
			if exprValue.Type() == storage.INTEGER {
				// Already an integer, return as is
				return exprValue, nil
			} else if exprValue.Type() == storage.FLOAT {
				// Convert float to integer
				if f, ok := exprValue.AsFloat64(); ok {
					return storage.GetPooledIntegerValue(int64(f)), nil
				}
			} else if exprValue.Type() == storage.TEXT {
				// Try to convert string to integer
				if s, ok := exprValue.AsString(); ok {
					if i, err := strconv.ParseInt(s, 10, 64); err == nil {
						return storage.GetPooledIntegerValue(i), nil
					}
				}
			} else if exprValue.Type() == storage.BOOLEAN {
				// Convert boolean to integer (true=1, false=0)
				if b, ok := exprValue.AsBoolean(); ok {
					if b {
						return storage.GetPooledIntegerValue(1), nil
					}
					return storage.GetPooledIntegerValue(0), nil
				}
			}
			return storage.StaticNullInteger, fmt.Errorf("cannot cast to INTEGER")

		case "FLOAT", "REAL", "DOUBLE":
			// Cast to float
			if exprValue.Type() == storage.FLOAT {
				// Already a float, return as is
				return exprValue, nil
			} else if exprValue.Type() == storage.INTEGER {
				// Convert integer to float
				if i, ok := exprValue.AsInt64(); ok {
					return storage.GetPooledFloatValue(float64(i)), nil
				}
			} else if exprValue.Type() == storage.TEXT {
				// Try to convert string to float
				if s, ok := exprValue.AsString(); ok {
					if f, err := strconv.ParseFloat(s, 64); err == nil {
						return storage.GetPooledFloatValue(f), nil
					}
				}
			} else if exprValue.Type() == storage.BOOLEAN {
				// Convert boolean to float (true=1.0, false=0.0)
				if b, ok := exprValue.AsBoolean(); ok {
					if b {
						return storage.GetPooledFloatValue(1.0), nil
					}
					return storage.GetPooledFloatValue(0.0), nil
				}
			}
			return storage.StaticNullFloat, fmt.Errorf("cannot cast to FLOAT")

		case "TEXT", "STRING", "VARCHAR":
			// Cast to string
			if exprValue.Type() == storage.TEXT {
				// Already a string, return as is
				return exprValue, nil
			} else if s, ok := exprValue.AsString(); ok {
				return storage.GetPooledStringValue(s), nil
			}
			return storage.StaticNullString, fmt.Errorf("cannot cast to TEXT")

		case "BOOLEAN", "BOOL":
			// Cast to boolean
			if exprValue.Type() == storage.BOOLEAN {
				// Already a boolean, return as is
				return exprValue, nil
			} else if exprValue.Type() == storage.INTEGER {
				// Convert integer to boolean (0=false, non-0=true)
				if i, ok := exprValue.AsInt64(); ok {
					return storage.NewBooleanValue(i != 0), nil
				}
			} else if exprValue.Type() == storage.FLOAT {
				// Convert float to boolean (0.0=false, non-0.0=true)
				if f, ok := exprValue.AsFloat64(); ok {
					return storage.NewBooleanValue(f != 0.0), nil
				}
			} else if exprValue.Type() == storage.TEXT {
				// Convert string to boolean
				if s, ok := exprValue.AsString(); ok {
					s = strings.ToLower(s)
					if s == "true" || s == "t" || s == "yes" || s == "y" || s == "1" {
						return storage.NewBooleanValue(true), nil
					} else if s == "false" || s == "f" || s == "no" || s == "n" || s == "0" {
						return storage.NewBooleanValue(false), nil
					}
				}
			}
			return storage.StaticNullBoolean, fmt.Errorf("cannot cast to BOOLEAN")

		case "TIMESTAMP", "DATE", "TIME":
			// Cast to timestamp - parse from string or use existing timestamp value
			if exprValue.Type() == storage.TEXT {
				tsStr, _ := exprValue.AsString()
				ts, err := storage.ParseTimestamp(tsStr)
				if err != nil {
					return storage.StaticNullTimestamp, fmt.Errorf("cannot cast '%s' to TIMESTAMP: %v", tsStr, err)
				}
				return storage.GetPooledTimestampValue(ts), nil
			} else if exprValue.Type() == storage.TIMESTAMP {
				t, ok := exprValue.AsTimestamp()
				if ok {
					return storage.GetPooledTimestampValue(t), nil
				}
			}
			return storage.StaticNullTimestamp, fmt.Errorf("cannot cast to TIMESTAMP")

		case "JSON":
			// Cast to JSON - parse from string or use existing JSON value
			if exprValue.Type() == storage.TEXT {
				jsonStr, _ := exprValue.AsString()
				// Check if the string is valid JSON
				var jsonObj interface{}
				if err := json.Unmarshal([]byte(jsonStr), &jsonObj); err != nil {
					return storage.StaticNullJSON, fmt.Errorf("cannot cast '%s' to JSON: %v", jsonStr, err)
				}
				// Return as valid JSON
				return storage.GetPooledJSONValue(jsonStr), nil
			} else if exprValue.Type() == storage.JSON {
				// Already a JSON value
				return exprValue, nil
			}
			return storage.StaticNullJSON, fmt.Errorf("cannot cast to JSON")

		default:
			return storage.StaticNullUnknown, fmt.Errorf("unsupported cast type: %s", expr.TypeName)
		}

	case *parser.Parameter:
		if e.ps != nil {
			// Get the parameter value
			nm := e.ps.GetValue(expr)

			// If the value is already a storage.ColumnValue, use it directly
			if cv, ok := nm.Value.(storage.ColumnValue); ok {
				return cv, nil
			}

			// Otherwise convert it to a storage.ColumnValue
			return storage.GetPooledColumnValue(nm.Value), nil
		}

		return storage.StaticNullUnknown, errors.New("cannot evaluate parameter in expressions")

	case *parser.Identifier:
		// Handle column references
		columnName := expr.Value

		// Handle the special * case
		if columnName == "*" {
			return storage.StaticNullUnknown, errors.New("cannot evaluate * in expressions")
		}

		if val, exists := e.row[columnName]; exists {
			// Return the ColumnValue directly - no conversion needed
			return val, nil
		}

		// If not found directly, check if this is an alias and resolve to the original column name
		if originalCol, exists := e.columnAliases[columnName]; exists {
			// Try several ways to find the original column value

			// 1. Try direct match on original column name
			if val, exists := e.row[originalCol]; exists {
				return val, nil
			}

			// 2. Try case-insensitive match on original column name
			for rowCol, rowVal := range e.row {
				if strings.EqualFold(rowCol, originalCol) && rowVal != nil {
					return rowVal, nil
				}
			}

			// 3. Try variants of original column name if it's an expression
			if exprParts := strings.Split(originalCol, "("); len(exprParts) > 1 {
				// Might be a function expression - try simplified form
				simpleName := strings.TrimSpace(exprParts[0])
				for rowCol, rowVal := range e.row {
					if strings.HasPrefix(rowCol, simpleName) && rowVal != nil {
						return rowVal, nil
					}
				}
			}
		}

		// Try case-insensitive lookup on the original column name as a fallback
		for rowCol, rowVal := range e.row {
			if strings.EqualFold(rowCol, columnName) {
				return rowVal, nil
			}
		}

		// If we have aliases, check if any alias is a case-insensitive match
		if len(e.columnAliases) > 0 {
			for alias, origCol := range e.columnAliases {
				if strings.EqualFold(alias, columnName) {
					// Try multiple variants of the original column name
					// 1. Direct match
					if val, exists := e.row[origCol]; exists {
						return val, nil
					}

					// 2. Case-insensitive match
					for rowCol, rowVal := range e.row {
						if strings.EqualFold(rowCol, origCol) && rowVal != nil {
							return rowVal, nil
						}
					}
				}
			}
		}

		return storage.StaticNullUnknown, fmt.Errorf("column '%s' not found in row", columnName)

	case *parser.QualifiedIdentifier:
		// Handle table.column references
		tableName := expr.Qualifier.Value
		columnName := expr.Name.Value

		// If we have table aliases, resolve the actual table name
		if actualTable, exists := e.tableAliases[tableName]; exists {
			tableName = actualTable
		}

		// Check if the column name is an alias and resolve to the original column name
		if originalCol, exists := e.columnAliases[columnName]; exists {
			columnName = originalCol
		}

		// Look for qualified column in the row (e.g., "table.column")
		qualifiedName := tableName + "." + columnName
		if val, exists := e.row[qualifiedName]; exists {
			return val, nil
		}

		// As a fallback, try just the column name
		if val, exists := e.row[columnName]; exists {
			return val, nil
		}

		// Additional fallback for CTE aliases - try without any table prefix
		// This handles cases where CTEs are accessed with table aliases
		for colName, colVal := range e.row {
			// Check if this column matches what we're looking for
			if colName == columnName {
				return colVal, nil
			}
		}

		return storage.StaticNullUnknown, fmt.Errorf("column '%s.%s' not found in row", tableName, columnName)

	case *parser.PrefixExpression:
		// Evaluate the right expression
		right, err := e.Evaluate(expr.Right)
		defer storage.PutPooledColumnValue(right)
		if err != nil {
			return storage.StaticNullUnknown, err
		}

		// Apply the prefix operator
		switch expr.Operator {
		case "-":
			// Numeric negation
			if right.Type() == storage.INTEGER {
				if i, ok := right.AsInt64(); ok {
					return storage.GetPooledIntegerValue(-i), nil
				}
			} else if right.Type() == storage.FLOAT {
				if f, ok := right.AsFloat64(); ok {
					return storage.GetPooledFloatValue(-f), nil
				}
			}
			return storage.StaticNullUnknown, fmt.Errorf("cannot apply - operator to type %v", right.Type())

		case "NOT", "!":
			// Special handling for NOT with complex expressions
			switch rightExpr := expr.Right.(type) {
			case *parser.InfixExpression:
				if rightExpr.Operator == "IN" {
					// For NOT IN, evaluate the IN expression and negate the result
					val, err := e.Evaluate(rightExpr)
					defer storage.PutPooledColumnValue(val)
					if err != nil {
						return storage.StaticNullUnknown, err
					}

					// If result is NULL, NOT NULL is still NULL
					if val.IsNull() {
						return storage.StaticNullUnknown, nil
					}

					// If boolean, negate it
					if val.Type() == storage.BOOLEAN {
						if b, ok := val.AsBoolean(); ok {
							return storage.NewBooleanValue(!b), nil
						}
					}
				} else if rightExpr.Operator == "AND" {
					// Special handling for NOT (a AND b) case
					// We need to check if this is part of "NOT in_stock AND price > 100"
					// In this case, we should really only negate the first part

					// Try to detect if the first operand is a boolean identifier (like in_stock)
					if _, ok := rightExpr.Left.(*parser.Identifier); ok {
						// Evaluate just the left side with NOT
						negatedLeft := &parser.PrefixExpression{
							Operator: "NOT",
							Right:    rightExpr.Left,
						}
						leftVal, err := e.Evaluate(negatedLeft)
						defer storage.PutPooledColumnValue(leftVal)
						if err != nil {
							return storage.StaticNullUnknown, err
						}

						// Then evaluate the right side as is
						rightVal, err := e.Evaluate(rightExpr.Right)
						defer storage.PutPooledColumnValue(rightVal)
						if err != nil {
							return storage.StaticNullUnknown, err
						}

						// Combine them with AND logic
						if leftVal.Type() == storage.BOOLEAN && rightVal.Type() == storage.BOOLEAN {
							leftBool, leftOk := leftVal.AsBoolean()
							rightBool, rightOk := rightVal.AsBoolean()
							if leftOk && rightOk && leftBool && rightBool {
								return storage.NewBooleanValue(true), nil
							}
							return storage.NewBooleanValue(false), nil
						}
					}

					// Regular evaluation for other cases
					val, err := e.Evaluate(rightExpr)
					defer storage.PutPooledColumnValue(val)
					if err != nil {
						return storage.StaticNullUnknown, err
					}

					// If boolean, negate it
					if val.Type() == storage.BOOLEAN {
						if b, ok := val.AsBoolean(); ok {
							return storage.NewBooleanValue(!b), nil
						}
					}
				}
			}

			// Regular logical negation for other expressions
			if right.Type() == storage.TEXT {
				// Try to convert string to boolean
				if s, ok := right.AsString(); ok {
					s = strings.ToLower(s)
					if s == "true" || s == "t" || s == "yes" || s == "y" || s == "1" {
						return storage.NewBooleanValue(false), nil
					} else if s == "false" || s == "f" || s == "no" || s == "n" || s == "0" {
						return storage.NewBooleanValue(true), nil
					}
				}
			}

			if b, ok := right.AsBoolean(); ok {
				return storage.NewBooleanValue(!b), nil
			}
			return storage.StaticNullUnknown, fmt.Errorf("cannot negate non-boolean value")

		default:
			return storage.StaticNullUnknown, fmt.Errorf("unknown prefix operator: %s", expr.Operator)
		}

	case *parser.InfixExpression:
		// For arithmetic operators, use the specialized EvaluateArithmeticExpression method
		if expr.Operator == "+" || expr.Operator == "-" || expr.Operator == "*" || expr.Operator == "/" || expr.Operator == "%" || expr.Operator == "||" {
			return e.EvaluateArithmeticExpression(expr.Left, expr.Right, expr.Operator)
		}

		// For other operators, continue with normal evaluation
		left, err := e.Evaluate(expr.Left)
		defer storage.PutPooledColumnValue(left)
		if err != nil {
			return storage.StaticNullUnknown, err
		}

		// Short-circuit evaluation for logical operators
		if expr.Operator == "AND" {
			// If left is false, the result is false regardless of right
			if left.Type() == storage.BOOLEAN {
				if b, ok := left.AsBoolean(); ok && !b {
					return storage.NewBooleanValue(false), nil
				}
			}
		} else if expr.Operator == "OR" {
			// If left is true, the result is true regardless of right
			if left.Type() == storage.BOOLEAN {
				if b, ok := left.AsBoolean(); ok && b {
					return storage.NewBooleanValue(true), nil
				}
			}
		}

		// Evaluate the right side
		right, err := e.Evaluate(expr.Right)
		defer storage.PutPooledColumnValue(right)
		if err != nil {
			return storage.StaticNullUnknown, err
		}

		// Apply the operator
		switch expr.Operator {
		// Comparison operators
		case "=", "==":
			// Special case for NULL comparisons
			if left.IsNull() || right.IsNull() {
				// NULL = NULL is true, NULL = non-NULL is false
				if left.IsNull() && right.IsNull() {
					return storage.NewBooleanValue(true), nil
				}
				return storage.NewBooleanValue(false), nil
			}

			// Try to compare values
			cmp, err := left.Compare(right)
			if err != nil {
				return storage.StaticNullUnknown, err
			}
			if cmp == 0 {
				return storage.NewBooleanValue(true), nil
			}
			return storage.NewBooleanValue(false), nil

		case "<>", "!=":
			// Special case for NULL comparisons
			if left.IsNull() || right.IsNull() {
				// NULL != NULL is false, NULL != non-NULL is true
				if left.IsNull() && right.IsNull() {
					return storage.NewBooleanValue(false), nil
				}
				return storage.NewBooleanValue(true), nil
			}

			// Try to compare values
			cmp, err := left.Compare(right)
			if err != nil {
				return storage.StaticNullUnknown, err
			}
			if cmp != 0 {
				return storage.NewBooleanValue(true), nil
			}
			return storage.NewBooleanValue(false), nil

		case "<":
			// NULL comparisons always yield false for < operator
			if left.IsNull() || right.IsNull() {
				return storage.NewBooleanValue(false), nil
			}

			// Try to compare values
			cmp, err := left.Compare(right)
			if err != nil {
				return storage.StaticNullUnknown, err
			}
			if cmp < 0 {
				return storage.NewBooleanValue(true), nil
			}
			return storage.NewBooleanValue(false), nil

		case ">":
			// NULL comparisons always yield false for > operator
			if left.IsNull() || right.IsNull() {
				return storage.NewBooleanValue(false), nil
			}

			// Try to compare values
			cmp, err := left.Compare(right)
			if err != nil {
				return storage.StaticNullUnknown, err
			}
			if cmp > 0 {
				return storage.NewBooleanValue(true), nil
			}
			return storage.NewBooleanValue(false), nil

		case "<=":
			// NULL comparisons always yield false for <= operator
			if left.IsNull() || right.IsNull() {
				return storage.NewBooleanValue(false), nil
			}

			// Try to compare values
			cmp, err := left.Compare(right)
			if err != nil {
				return storage.StaticNullUnknown, err
			}
			if cmp <= 0 {
				return storage.NewBooleanValue(true), nil
			}
			return storage.NewBooleanValue(false), nil

		case ">=":
			// NULL comparisons always yield false for >= operator
			if left.IsNull() || right.IsNull() {
				return storage.NewBooleanValue(false), nil
			}

			// Try to compare values
			cmp, err := left.Compare(right)
			if err != nil {
				return storage.StaticNullUnknown, err
			}
			if cmp >= 0 {
				return storage.NewBooleanValue(true), nil
			}
			return storage.NewBooleanValue(false), nil

		// Logical operators
		case "AND":
			// For NULL, NULL AND anything is NULL
			if left.IsNull() || right.IsNull() {
				return storage.StaticNullUnknown, nil
			}

			// Convert to boolean
			leftBool, leftOk := left.AsBoolean()
			rightBool, rightOk := right.AsBoolean()

			if leftOk && rightOk {
				return storage.NewBooleanValue(leftBool && rightBool), nil
			}
			return storage.StaticNullUnknown, fmt.Errorf("cannot perform AND with non-boolean values")

		case "OR":
			// For NULL, NULL OR true is true, NULL OR false is NULL
			if left.IsNull() && !right.IsNull() {
				if b, ok := right.AsBoolean(); ok && b {
					return storage.NewBooleanValue(true), nil
				}
				return storage.StaticNullUnknown, nil
			} else if right.IsNull() && !left.IsNull() {
				if b, ok := left.AsBoolean(); ok && b {
					return storage.NewBooleanValue(true), nil
				}
				return storage.StaticNullUnknown, nil
			} else if left.IsNull() || right.IsNull() {
				return storage.StaticNullUnknown, nil
			}

			// Convert to boolean
			leftBool, leftOk := left.AsBoolean()
			rightBool, rightOk := right.AsBoolean()

			if leftOk && rightOk {
				return storage.NewBooleanValue(leftBool || rightBool), nil
			}
			return storage.StaticNullUnknown, fmt.Errorf("cannot perform OR with non-boolean values")

		case "LIKE":
			// LIKE is string comparison with wildcards
			// For NULL operands, LIKE returns NULL
			if left.IsNull() || right.IsNull() {
				return storage.StaticNullUnknown, nil
			}

			// Convert to string
			leftStr, leftOk := left.AsString()
			pattern, rightOk := right.AsString()

			if !leftOk || !rightOk {
				return storage.StaticNullUnknown, fmt.Errorf("cannot perform LIKE with non-string values")
			}

			// Convert SQL LIKE pattern to regex pattern
			// In SQL LIKE, % is a wildcard meaning any sequence of characters
			// _ is a wildcard meaning any single character
			// First, escape any regex special characters
			regexPattern := regexp.QuoteMeta(pattern)

			// Then replace LIKE wildcards with regex equivalents
			// In SQL LIKE, % is a wildcard meaning any sequence of characters
			// We need to unescape the % first if it was escaped by QuoteMeta
			regexPattern = strings.ReplaceAll(regexPattern, `\%`, "%")
			regexPattern = strings.ReplaceAll(regexPattern, `%`, ".*")

			// In SQL LIKE, _ is a wildcard meaning any single character
			// We need to unescape the _ first if it was escaped by QuoteMeta
			regexPattern = strings.ReplaceAll(regexPattern, `\_`, "_")
			regexPattern = strings.ReplaceAll(regexPattern, `_`, ".")

			// Add anchors for exact match
			regexPattern = "^" + regexPattern + "$"

			matched, err := regexp.MatchString(regexPattern, leftStr)
			if err != nil {
				return storage.StaticNullUnknown, err
			}

			return storage.NewBooleanValue(matched), nil

		// IS NULL and IS NOT NULL
		case "IS":
			// We only support IS NULL and IS NOT NULL
			if !right.IsNull() {
				return storage.StaticNullUnknown, errors.New("IS operator can only be used with NULL")
			}
			return storage.NewBooleanValue(left.IsNull()), nil

		case "IS NOT":
			// We only support IS NOT NULL
			if !right.IsNull() {
				return storage.StaticNullUnknown, errors.New("IS NOT operator can only be used with NULL")
			}
			return storage.NewBooleanValue(!left.IsNull()), nil

		case "IN":
			// IN operator checks if a value is in a list
			// If left is NULL, result is NULL
			if left.IsNull() {
				return storage.StaticNullUnknown, nil
			}

			// Check if right is an expression list (IN list of values)
			if exprList, ok := expr.Right.(*parser.ExpressionList); ok {
				// Check each element in the list
				for _, elemExpr := range exprList.Expressions {
					elemVal, err := e.Evaluate(elemExpr)
					defer storage.PutPooledColumnValue(elemVal)
					if err != nil {
						continue // Skip errors in list elements
					}

					// For NULL in the list, we ignore it - no match
					if elemVal.IsNull() {
						continue
					}

					// Compare with left value
					cmp, err := left.Compare(elemVal)
					if err == nil && cmp == 0 {
						// Found a match
						return storage.NewBooleanValue(true), nil
					}
				}

				// No match found
				return storage.NewBooleanValue(false), nil
			}

			// For other types of right operand, return error
			return storage.StaticNullUnknown, fmt.Errorf("right side of IN operator must be an expression list")

		default:
			return storage.StaticNullUnknown, fmt.Errorf("unknown infix operator: %s", expr.Operator)
		}

	case *parser.FunctionCall:
		// Evaluate function arguments
		args := make([]storage.ColumnValue, len(expr.Arguments))
		for i, arg := range expr.Arguments {
			// Skip evaluating star argument for COUNT(*)
			if arg.String() == "*" && strings.ToUpper(expr.Function) == "COUNT" {
				args[i] = storage.GetPooledStringValue("*")
				continue
			}

			// Evaluate the argument
			val, err := e.Evaluate(arg)
			defer storage.PutPooledColumnValue(val)
			if err != nil {
				return storage.StaticNullUnknown, err
			}
			args[i] = val
		}

		// Special case for COUNT(*) in WHERE clause (usually won't work but let's make it possible)
		if strings.ToUpper(expr.Function) == "COUNT" && len(args) == 1 {
			if s, ok := args[0].AsString(); ok && s == "*" {
				// We can't actually compute an aggregate within a WHERE clause properly
				return storage.GetPooledIntegerValue(1), nil
			}
		}

		// Look up the function in the registry
		if e.functionRegistry == nil {
			return storage.StaticNullUnknown, errors.New("function registry not set")
		}

		// Get a scalar function from the registry by name
		fn := e.functionRegistry.GetScalarFunction(expr.Function)
		if fn == nil {
			// Check for aggregate function used directly (without GROUP BY)
			if IsAggregateFunction(expr.Function) {
				return storage.StaticNullUnknown, fmt.Errorf("aggregate function %s used without GROUP BY", expr.Function)
			}
			return storage.StaticNullUnknown, fmt.Errorf("function not found: %s", expr.Function)
		}

		// Extract argument values for the function call
		argValues := make([]interface{}, len(args))
		for i, arg := range args {
			argValues[i] = arg.AsInterface()
		}

		// Call the function
		result, err := fn.Evaluate(argValues...)
		if err != nil {
			return storage.StaticNullUnknown, err
		}

		// If result is already a ColumnValue, return it directly
		if cv, ok := result.(storage.ColumnValue); ok {
			return cv, nil
		}

		val := storage.GetPooledColumnValue(result)

		// Otherwise convert to ColumnValue
		return val, nil

	case *parser.CaseExpression:
		// Evaluate the case value if present
		var caseVal storage.ColumnValue
		var err error
		if expr.Value != nil {
			caseVal, err = e.Evaluate(expr.Value)
			defer storage.PutPooledColumnValue(caseVal)
			if err != nil {
				return storage.StaticNullUnknown, err
			}
		}

		// Evaluate each WHEN clause in order
		for _, when := range expr.WhenClauses {
			// Evaluate the condition
			var condResult bool

			if expr.Value != nil {
				// Simple CASE: Compare case value with each WHEN expression
				whenVal, err := e.Evaluate(when.Condition)
				defer storage.PutPooledColumnValue(whenVal)
				if err != nil {
					return storage.StaticNullUnknown, err
				}

				// Compare the values
				cmp, err := caseVal.Compare(whenVal)
				if err != nil {
					// If comparison fails, treat it as not equal
					condResult = false
				} else {
					condResult = cmp == 0
				}
			} else {
				// Searched CASE: Evaluate each WHEN condition as a boolean
				condVal, err := e.Evaluate(when.Condition)
				defer storage.PutPooledColumnValue(condVal)
				if err != nil {
					return storage.StaticNullUnknown, err
				}

				// Convert to boolean
				if b, ok := condVal.AsBoolean(); ok {
					condResult = b
				} else {
					condResult = false
				}
			}

			// If condition is true, evaluate and return the THEN result
			if condResult {
				return e.Evaluate(when.ThenResult)
			}
		}

		// No matching WHEN clause, evaluate ELSE expression if present
		if expr.ElseValue != nil {
			return e.Evaluate(expr.ElseValue)
		}

		// No ELSE expression, return NULL
		return storage.StaticNullUnknown, nil

	case *parser.DistinctExpression:
		// This is for DISTINCT in a function call like COUNT(DISTINCT column)
		// For the purpose of evaluating an individual row, we can just evaluate the expression
		return e.Evaluate(expr.Expr)

	case *parser.InExpression:
		// Evaluate the left expression
		leftVal, err := e.Evaluate(expr.Left)
		defer storage.PutPooledColumnValue(leftVal)
		if err != nil {
			return storage.StaticNullUnknown, err
		}

		// For NULL, IN returns NULL (not false!)
		if leftVal.IsNull() {
			return storage.StaticNullUnknown, nil
		}

		// Evaluate the right expression depending on its type
		switch rightExpr := expr.Right.(type) {
		case *parser.ExpressionList:
			// IN list of values
			for _, elemExpr := range rightExpr.Expressions {
				elemVal, err := e.Evaluate(elemExpr)
				defer storage.PutPooledColumnValue(elemVal)
				if err != nil {
					continue // Skip errors in list elements
				}

				// For NULL in the list, we ignore it
				if elemVal.IsNull() {
					continue
				}

				// Compare with left value
				cmp, err := leftVal.Compare(elemVal)
				if err == nil && cmp == 0 {
					// Found a match
					if expr.Not {
						return storage.NewBooleanValue(false), nil
					} else {
						return storage.NewBooleanValue(true), nil
					}
				}
			}

			// No match found
			if expr.Not {
				return storage.NewBooleanValue(true), nil
			} else {
				return storage.NewBooleanValue(false), nil
			}

		case *parser.ScalarSubquery:
			// IN subquery - this is not fully implemented as we can't evaluate subqueries here
			return storage.StaticNullUnknown, errors.New("IN subquery not supported in WHERE clause evaluation")

		default:
			return storage.StaticNullUnknown, fmt.Errorf("unsupported right operand for IN: %T", expr.Right)
		}

	case *parser.BetweenExpression:
		// Evaluate the expression and bounds
		exprValue, err := e.Evaluate(expr.Expr)
		defer storage.PutPooledColumnValue(exprValue)
		if err != nil {
			return storage.StaticNullUnknown, err
		}

		// If expression is NULL, result is NULL
		if exprValue.IsNull() {
			return storage.StaticNullUnknown, nil
		}

		lowerValue, err := e.Evaluate(expr.Lower)
		defer storage.PutPooledColumnValue(lowerValue)
		if err != nil {
			return storage.StaticNullUnknown, err
		}

		upperValue, err := e.Evaluate(expr.Upper)
		defer storage.PutPooledColumnValue(upperValue)
		if err != nil {
			return storage.StaticNullUnknown, err
		}

		// If any bound is NULL, result is NULL
		if lowerValue.IsNull() || upperValue.IsNull() {
			return storage.StaticNullUnknown, nil
		}

		// Compare expression with lower bound
		lowerCmp, err := exprValue.Compare(lowerValue)
		if err != nil {
			return storage.StaticNullUnknown, err
		}

		// Compare expression with upper bound
		upperCmp, err := exprValue.Compare(upperValue)
		if err != nil {
			return storage.StaticNullUnknown, err
		}

		// For BETWEEN, the value must be >= lower AND <= upper
		result := lowerCmp >= 0 && upperCmp <= 0

		// For NOT BETWEEN, negate the result
		if expr.Not {
			result = !result
		}

		return storage.NewBooleanValue(result), nil

	case *parser.AliasedExpression:
		// For evaluation purposes, we just evaluate the underlying expression
		return e.Evaluate(expr.Expression)

	case *parser.ScalarSubquery:
		// Scalar subqueries need to be evaluated by the executor
		// This is a placeholder - the actual evaluation should be done by processWhereSubqueries
		// before the expression reaches the evaluator
		return storage.StaticNullUnknown, fmt.Errorf("scalar subquery should have been processed before evaluation")

	default:
		return storage.StaticNullUnknown, fmt.Errorf("unsupported expression type: %T", expr)
	}
}

// determineCommonType determines the common type for arithmetic operations
func determineCommonType(left, right storage.ColumnValue) (storage.DataType, error) {
	// Special cases for NULL values
	if left.IsNull() || right.IsNull() {
		return storage.NULL, fmt.Errorf("cannot perform arithmetic on NULL values")
	}

	// If either is TEXT, result is TEXT (for + and || operators)
	if left.Type() == storage.TEXT || right.Type() == storage.TEXT {
		return storage.TEXT, nil
	}

	// If either is FLOAT, result is FLOAT
	if left.Type() == storage.FLOAT || right.Type() == storage.FLOAT {
		return storage.FLOAT, nil
	}

	// If both are INTEGER, result is INTEGER
	if left.Type() == storage.INTEGER && right.Type() == storage.INTEGER {
		return storage.INTEGER, nil
	}

	// For any other valid numeric types, default to FLOAT
	if isNumericType(left.Type()) && isNumericType(right.Type()) {
		return storage.FLOAT, nil
	}

	// Incompatible types
	return storage.NULL, fmt.Errorf("incompatible types for arithmetic: %v and %v", left.Type(), right.Type())
}

// isNumericType checks if the data type is numeric (INTEGER, FLOAT, or can be converted to numeric)
func isNumericType(dt storage.DataType) bool {
	return dt == storage.INTEGER || dt == storage.FLOAT || dt == storage.BOOLEAN
}

// convertToType converts a value to the specified target type
func convertToType(val storage.ColumnValue, targetType storage.DataType) (interface{}, bool) {
	switch targetType {
	case storage.INTEGER:
		return val.AsInt64()
	case storage.FLOAT:
		return val.AsFloat64()
	case storage.TEXT:
		return val.AsString()
	case storage.BOOLEAN:
		return val.AsBoolean()
	default:
		return nil, false
	}
}

// EvaluateArithmeticExpression evaluates an arithmetic expression (binary operation)
func (e *Evaluator) EvaluateArithmeticExpression(left, right parser.Expression, operator string) (storage.ColumnValue, error) {
	// Evaluate the left operand
	leftVal, err := e.Evaluate(left)
	defer storage.PutPooledColumnValue(leftVal)
	if err != nil {
		return storage.StaticNullUnknown, err
	}

	// Evaluate the right operand
	rightVal, err := e.Evaluate(right)
	defer storage.PutPooledColumnValue(rightVal)
	if err != nil {
		return storage.StaticNullUnknown, err
	}

	// Special fast path for string concatenation (|| operator)
	if operator == "||" {
		leftStr, leftOk := leftVal.AsString()
		rightStr, rightOk := rightVal.AsString()
		if leftOk && rightOk {
			return storage.GetPooledStringValue(leftStr + rightStr), nil
		}
		return storage.StaticNullUnknown, fmt.Errorf("cannot concatenate non-string values")
	}

	// For + operator with a text operand, treat as string concatenation
	if operator == "+" && (leftVal.Type() == storage.TEXT || rightVal.Type() == storage.TEXT) {
		leftStr, leftOk := leftVal.AsString()
		rightStr, rightOk := rightVal.AsString()
		if leftOk && rightOk {
			return storage.GetPooledStringValue(leftStr + rightStr), nil
		}
	}

	// Handle timestamp arithmetic (timestamp +/- interval)
	if leftVal.Type() == storage.TIMESTAMP && rightVal.Type() == storage.INTEGER {
		if operator == "+" || operator == "-" {
			timestamp, ok := leftVal.AsTimestamp()
			if !ok {
				return storage.StaticNullUnknown, fmt.Errorf("invalid timestamp value")
			}

			// Right value is interval in nanoseconds
			intervalNanos, ok := rightVal.AsInt64()
			if !ok {
				return storage.StaticNullUnknown, fmt.Errorf("invalid interval value")
			}

			var result time.Time
			if operator == "+" {
				result = timestamp.Add(time.Duration(intervalNanos))
			} else {
				result = timestamp.Add(-time.Duration(intervalNanos))
			}

			return storage.GetPooledTimestampValue(result), nil
		}
	}

	// Determine the common type for the operation
	commonType, err := determineCommonType(leftVal, rightVal)
	if err != nil {
		return storage.StaticNullUnknown, err
	}

	// Handle the % operator separately as it only works with integers
	if operator == "%" {
		leftInt, leftOk := leftVal.AsInt64()
		rightInt, rightOk := rightVal.AsInt64()
		if leftOk && rightOk {
			if rightInt == 0 {
				return storage.StaticNullUnknown, errors.New("modulo by zero")
			}
			return storage.GetPooledIntegerValue(leftInt % rightInt), nil
		}
		return storage.StaticNullUnknown, fmt.Errorf("modulo operation requires integer operands")
	}

	// Convert operands to the common type
	leftConverted, leftOk := convertToType(leftVal, commonType)
	rightConverted, rightOk := convertToType(rightVal, commonType)

	if !leftOk || !rightOk {
		return storage.StaticNullUnknown, fmt.Errorf("cannot convert operands to compatible types")
	}

	// Perform the operation based on the common type
	switch commonType {
	case storage.INTEGER:
		leftInt := leftConverted.(int64)
		rightInt := rightConverted.(int64)

		switch operator {
		case "+":
			return storage.GetPooledIntegerValue(leftInt + rightInt), nil
		case "-":
			return storage.GetPooledIntegerValue(leftInt - rightInt), nil
		case "*":
			return storage.GetPooledIntegerValue(leftInt * rightInt), nil
		case "/":
			// Division with integers should return float
			if rightInt == 0 {
				return storage.StaticNullUnknown, errors.New("division by zero")
			}
			return storage.GetPooledFloatValue(float64(leftInt) / float64(rightInt)), nil
		}

	case storage.FLOAT:
		leftFloat := leftConverted.(float64)
		rightFloat := rightConverted.(float64)

		switch operator {
		case "+":
			return storage.GetPooledFloatValue(leftFloat + rightFloat), nil
		case "-":
			return storage.GetPooledFloatValue(leftFloat - rightFloat), nil
		case "*":
			return storage.GetPooledFloatValue(leftFloat * rightFloat), nil
		case "/":
			if rightFloat == 0 {
				return storage.StaticNullUnknown, errors.New("division by zero")
			}
			return storage.GetPooledFloatValue(leftFloat / rightFloat), nil
		}

	case storage.TEXT:
		// Only + operator makes sense for text
		if operator == "+" {
			leftStr := leftConverted.(string)
			rightStr := rightConverted.(string)
			return storage.GetPooledStringValue(leftStr + rightStr), nil
		}
	}

	return storage.StaticNullUnknown, fmt.Errorf("unsupported operation %s for types %v and %v", operator, leftVal.Type(), rightVal.Type())
}

// EvaluateWhereClause evaluates a WHERE clause for a specific row
func (e *Evaluator) EvaluateWhereClause(whereExpr parser.Expression, row map[string]storage.ColumnValue) (bool, error) {
	if whereExpr == nil {
		// No WHERE clause means all rows match
		return true, nil
	}

	// Set the current row for evaluation
	e.row = row

	// Special handling for HAVING clause with aggregate functions
	// This is specifically to handle expressions like COUNT(*) > 2
	if infix, ok := whereExpr.(*parser.InfixExpression); ok &&
		(infix.Operator == ">" || infix.Operator == "<" || infix.Operator == "=" ||
			infix.Operator == ">=" || infix.Operator == "<=") {

		// Check if left side is an aggregate function
		if fnCall, ok := infix.Left.(*parser.FunctionCall); ok {
			if strings.ToUpper(fnCall.Function) == "COUNT" && len(fnCall.Arguments) == 1 {
				// Handle COUNT(*) in HAVING clause
				if ident, ok := fnCall.Arguments[0].(*parser.Identifier); ok && ident.Value == "*" {
					// Look for the aggregate result directly in the row
					countKey := "COUNT(*)"
					var countVal storage.ColumnValue
					var countFound bool

					// Try various formats for the count column
					for k, v := range row {
						if k == countKey || k == "count(*)" ||
							strings.HasPrefix(k, "COUNT(") || strings.HasPrefix(k, "count(") {
							countVal = v
							countFound = true
							break
						}
					}

					if countFound {
						// Convert the count to numeric
						var count int64
						if i, ok := countVal.AsInt64(); ok {
							count = i
						} else if f, ok := countVal.AsFloat64(); ok {
							count = int64(f)
						} else if s, ok := countVal.AsString(); ok {
							// Try to parse as string
							if intVal, err := strconv.ParseInt(s, 10, 64); err == nil {
								count = intVal
							}
						}

						// Get the right value (comparison operand)
						rightVal, err := e.Evaluate(infix.Right)
						defer storage.PutPooledColumnValue(rightVal)
						if err != nil {
							return false, err
						}

						// Convert right value to numeric
						var rightNum int64
						if rightVal.Type() == storage.INTEGER {
							if i, ok := rightVal.AsInt64(); ok {
								rightNum = i
							}
						} else if rightVal.Type() == storage.FLOAT {
							if f, ok := rightVal.AsFloat64(); ok {
								rightNum = int64(f)
							}
						} else {
							// Not a number, can't compare
							return false, fmt.Errorf("right side of comparison is not a number")
						}

						// Do the comparison directly
						switch infix.Operator {
						case ">":
							return count > rightNum, nil
						case "<":
							return count < rightNum, nil
						case "=", "==":
							return count == rightNum, nil
						case ">=":
							return count >= rightNum, nil
						case "<=":
							return count <= rightNum, nil
						}
					}
				}
			}
		}
	}

	// Special handling for direct comparison with boolean literals
	if infix, ok := whereExpr.(*parser.InfixExpression); ok && (infix.Operator == "=" || infix.Operator == "==") {
		// Check for qualified identifier (table.column) or regular identifier
		var columnName string
		var isQualified bool
		var tableName string

		if qualIdent, ok := infix.Left.(*parser.QualifiedIdentifier); ok {
			isQualified = true
			tableName = qualIdent.Qualifier.Value
			columnName = qualIdent.Name.Value
		} else if ident, ok := infix.Left.(*parser.Identifier); ok {
			isQualified = false
			columnName = ident.Value
		} else {
			// Not an identifier we can optimize
			goto regularEvaluation
		}

		// Check if right side is boolean literal
		if boolLit, ok := infix.Right.(*parser.BooleanLiteral); ok {
			expectedBool := boolLit.Value

			// Try different column name variations based on whether it's qualified
			var lookupKeys []string

			if isQualified {
				// For qualified identifiers like "p.in_stock", try various formats
				fullQualifiedName := tableName + "." + columnName
				lookupKeys = []string{
					fullQualifiedName,                      // "p.in_stock"
					strings.ToLower(fullQualifiedName),     // "p.in_stock" (lowercase)
					columnName,                             // "in_stock" (just the column part)
					strings.ToLower(columnName),            // "in_stock" (lowercase column part)
					tableName + "." + columnName + "_bool", // "p.in_stock_bool"
					columnName + "_bool",                   // "in_stock_bool"
				}
			} else {
				// For simple identifiers, try these formats
				lookupKeys = []string{
					columnName,                  // Direct name
					strings.ToLower(columnName), // Lowercase name
					columnName + "_bool",        // Name with _bool suffix
				}
			}

			// Check all possible key variations
			for _, key := range lookupKeys {
				if val, ok := row[key]; ok {
					// Try as direct boolean
					if b, ok := val.AsBoolean(); ok {
						return b == expectedBool, nil
					}

					// Try as numeric (1/0)
					if i, ok := val.AsInt64(); ok {
						return (i != 0) == expectedBool, nil
					}

					// Try as string representation
					if s, ok := val.AsString(); ok {
						strLower := strings.ToLower(s)
						actualBool := strLower == "true" || strLower == "t" || strLower == "yes" || strLower == "y" || strLower == "1"
						return actualBool == expectedBool, nil
					}
				}
			}

			// If we have true comparison, check for any non-null value too
			if expectedBool && !isQualified {
				if val, ok := row[columnName]; ok && !val.IsNull() {
					// The column exists and isn't null, could be implicitly true
					return true, nil
				}
			}
		}
	}

regularEvaluation:

	// Evaluate the expression
	result, err := e.Evaluate(whereExpr)
	if err != nil {
		return false, err
	}

	// If the result is NULL, the row doesn't match (NULL in WHERE is treated as false)
	if result.IsNull() {
		return false, nil
	}

	// Convert to boolean
	boolResult, ok := result.AsBoolean()
	if !ok {
		return false, fmt.Errorf("where clause did not evaluate to a boolean")
	}

	return boolResult, nil
}

// EvaluateWhereClauseWithStorageRow evaluates a WHERE clause for a specific storage.Row
func (e *Evaluator) EvaluateWhereClauseWithStorageRow(whereExpr parser.Expression, row storage.Row, columnMap map[string]int) (bool, error) {
	if whereExpr == nil {
		// No WHERE clause means all rows match
		return true, nil
	}

	// Special handling for HAVING clause with aggregate functions
	// This is specifically to handle expressions like COUNT(*) > 2
	if infix, ok := whereExpr.(*parser.InfixExpression); ok &&
		(infix.Operator == ">" || infix.Operator == "<" || infix.Operator == "=" ||
			infix.Operator == ">=" || infix.Operator == "<=") {

		// Check if left side is an aggregate function
		if fnCall, ok := infix.Left.(*parser.FunctionCall); ok {
			if strings.ToUpper(fnCall.Function) == "COUNT" && len(fnCall.Arguments) == 1 {
				// Handle COUNT(*) in HAVING clause
				if ident, ok := fnCall.Arguments[0].(*parser.Identifier); ok && ident.Value == "*" {
					// Look for the aggregate result directly in the row
					countKey := "COUNT(*)"
					var countVal storage.ColumnValue
					var countFound bool

					// Try various formats for the count column
					for k, v := range columnMap {
						if k == countKey || k == "count(*)" ||
							strings.HasPrefix(k, "COUNT(") || strings.HasPrefix(k, "count(") {
							countVal = row[v]
							countFound = true
							break
						}
					}

					if countFound {
						// Convert the count to numeric
						var count int64
						if i, ok := countVal.AsInt64(); ok {
							count = i
						} else if f, ok := countVal.AsFloat64(); ok {
							count = int64(f)
						} else if s, ok := countVal.AsString(); ok {
							// Try to parse as string
							if intVal, err := strconv.ParseInt(s, 10, 64); err == nil {
								count = intVal
							}
						}

						// Get the right value (comparison operand)
						rightVal, err := e.Evaluate(infix.Right)
						defer storage.PutPooledColumnValue(rightVal)
						if err != nil {
							return false, err
						}

						// Convert right value to numeric
						var rightNum int64
						if rightVal.Type() == storage.INTEGER {
							if i, ok := rightVal.AsInt64(); ok {
								rightNum = i
							}
						} else if rightVal.Type() == storage.FLOAT {
							if f, ok := rightVal.AsFloat64(); ok {
								rightNum = int64(f)
							}
						} else {
							// Not a number, can't compare
							return false, fmt.Errorf("right side of comparison is not a number")
						}

						// Do the comparison directly
						switch infix.Operator {
						case ">":
							return count > rightNum, nil
						case "<":
							return count < rightNum, nil
						case "=", "==":
							return count == rightNum, nil
						case ">=":
							return count >= rightNum, nil
						case "<=":
							return count <= rightNum, nil
						}
					}
				}
			}
		}
	}

	// Set the current row for evaluation
	e.WithStorageRow(row, columnMap)
	defer common.PutColumnValueMap(e.row, len(e.row))

	// Evaluate the expression
	result, err := e.Evaluate(whereExpr)
	if err != nil {
		return false, err
	}

	// If the result is NULL, the row doesn't match (NULL in WHERE is treated as false)
	if result.IsNull() {
		return false, nil
	}

	// Convert to boolean
	boolResult, ok := result.AsBoolean()
	if !ok {
		return false, fmt.Errorf("where clause did not evaluate to a boolean")
	}

	return boolResult, nil
}
