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
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"

	"github.com/stoolap/stoolap/internal/functions/contract"
	"github.com/stoolap/stoolap/internal/parser"
	"github.com/stoolap/stoolap/internal/storage"
	"github.com/stoolap/stoolap/internal/storage/expression"
	"github.com/stoolap/stoolap/internal/storage/mvcc"
)

var (
	interfaceSlicePool = &sync.Pool{
		New: func() interface{} {
			p := make([]interface{}, 0, 16)
			return &p // Pre-allocate with reasonable capacity
		},
	}

	stringSlicePool = &sync.Pool{
		New: func() interface{} {
			p := make([]string, 0, 16)
			return &p // Pre-allocate with reasonable capacity
		},
	}

	// Pool for function argument slices to avoid allocations
	argSlicePool = &sync.Pool{
		New: func() interface{} {
			p := make([]interface{}, 0, 8)
			return &p // Pre-allocate with reasonable capacity for function args
		},
	}
)

// evaluateExpressionFast is a lightweight expression evaluator optimized for DML operations
// It avoids the allocations of the full Evaluator but supports all expression types for DML
func evaluateExpressionFast(ctx context.Context, expr parser.Expression, registry contract.FunctionRegistry, ps *parameter, row map[string]interface{}) (interface{}, error) {
	switch e := expr.(type) {
	case *parser.Parameter:
		if ps != nil {
			nm := ps.GetValue(e)
			return nm.Value, nil
		}
		return nil, nil
	case *parser.IntegerLiteral:
		return e.Value, nil
	case *parser.FloatLiteral:
		return e.Value, nil
	case *parser.StringLiteral:
		return e.Value, nil
	case *parser.BooleanLiteral:
		return e.Value, nil
	case *parser.NullLiteral:
		return nil, nil
	case *parser.FunctionCall:
		// Handle function calls with minimal allocations
		if registry == nil {
			return nil, errors.New("function registry not available")
		}

		// Get function from registry
		fn := registry.GetScalarFunction(e.Function)
		if fn == nil {
			return nil, fmt.Errorf("function not found: %s", e.Function)
		}

		// Handle zero-argument functions quickly
		if len(e.Arguments) == 0 {
			result, err := fn.Evaluate()
			return result, err
		}

		// Use pooled slice for arguments to avoid allocation
		cp := argSlicePool.Get().(*[]interface{})
		args := *cp
		if cap(args) < len(e.Arguments) {
			args = make([]interface{}, len(e.Arguments))
		} else {
			args = args[:len(e.Arguments)]
		}

		defer func() {
			*cp = args[:0]
			argSlicePool.Put(cp)
		}()

		// Evaluate arguments recursively
		for i, arg := range e.Arguments {
			val, err := evaluateExpressionFast(ctx, arg, registry, ps, row)
			if err != nil {
				return nil, fmt.Errorf("error evaluating argument %d for function %s: %w", i+1, e.Function, err)
			}
			args[i] = val
		}

		// Call the function
		result, err := fn.Evaluate(args...)
		if err != nil {
			return nil, fmt.Errorf("error calling function %s: %w", e.Function, err)
		}

		return result, nil
	case *parser.Identifier:
		// Handle column references for UPDATE SET col = other_col
		if row != nil {
			if val, exists := row[e.Value]; exists {
				return val, nil
			}
		}
		return nil, fmt.Errorf("column not found: %s", e.Value)
	case *parser.InfixExpression:
		// Handle arithmetic and comparison operations
		left, err := evaluateExpressionFast(ctx, e.Left, registry, ps, row)
		if err != nil {
			return nil, err
		}
		right, err := evaluateExpressionFast(ctx, e.Right, registry, ps, row)
		if err != nil {
			return nil, err
		}

		switch e.Operator {
		case "+":
			return evaluateArithmetic(left, right, "+")
		case "-":
			return evaluateArithmetic(left, right, "-")
		case "*":
			return evaluateArithmetic(left, right, "*")
		case "/":
			return evaluateArithmetic(left, right, "/")
		case "||":
			// String concatenation
			leftStr := fmt.Sprintf("%v", left)
			rightStr := fmt.Sprintf("%v", right)
			return leftStr + rightStr, nil
		case "=":
			return compareValues(left, right), nil
		case "!=", "<>":
			return !compareValues(left, right), nil
		case ">":
			return compareNumbers(left, right, ">")
		case ">=":
			return compareNumbers(left, right, ">=")
		case "<":
			return compareNumbers(left, right, "<")
		case "<=":
			return compareNumbers(left, right, "<=")
		default:
			return nil, fmt.Errorf("unsupported infix operator: %s", e.Operator)
		}
	case *parser.CastExpression:
		// Handle type casting using the CAST function from registry
		if registry == nil {
			return nil, errors.New("function registry not available for CAST")
		}

		fn := registry.GetScalarFunction("CAST")
		if fn == nil {
			return nil, fmt.Errorf("CAST function not found in registry")
		}

		// Evaluate the expression to cast
		val, err := evaluateExpressionFast(ctx, e.Expr, registry, ps, row)
		if err != nil {
			return nil, err
		}

		// Call CAST function with value and type
		result, err := fn.Evaluate(val, e.TypeName)
		if err != nil {
			return nil, fmt.Errorf("error calling CAST function: %w", err)
		}

		return result, nil
	case *parser.CaseExpression:
		// Handle CASE expressions
		return evaluateCase(ctx, e, registry, ps, row)
	default:
		return nil, fmt.Errorf("unsupported expression type: %T", expr)
	}
}

// evaluateArithmetic performs arithmetic operations on two values
func evaluateArithmetic(left, right interface{}, op string) (interface{}, error) {
	// Handle NULL values
	if left == nil || right == nil {
		return nil, nil
	}

	// Convert to numbers for arithmetic
	leftNum, leftIsNum := convertToNumber(left)
	rightNum, rightIsNum := convertToNumber(right)

	if !leftIsNum || !rightIsNum {
		return nil, fmt.Errorf("arithmetic operation %s requires numeric values", op)
	}

	switch op {
	case "+":
		return leftNum + rightNum, nil
	case "-":
		return leftNum - rightNum, nil
	case "*":
		return leftNum * rightNum, nil
	case "/":
		if rightNum == 0 {
			return nil, fmt.Errorf("division by zero")
		}
		return leftNum / rightNum, nil
	default:
		return nil, fmt.Errorf("unsupported arithmetic operator: %s", op)
	}
}

// convertToNumber converts a value to float64 for arithmetic operations
func convertToNumber(val interface{}) (float64, bool) {
	switch v := val.(type) {
	case int64:
		return float64(v), true
	case float64:
		return v, true
	case int:
		return float64(v), true
	case float32:
		return float64(v), true
	default:
		return 0, false
	}
}

// evaluateCase handles CASE expressions
func evaluateCase(ctx context.Context, caseExpr *parser.CaseExpression, registry contract.FunctionRegistry, ps *parameter, row map[string]interface{}) (interface{}, error) {
	// Evaluate the case value if present (simple CASE)
	var caseVal interface{}
	var err error
	if caseExpr.Value != nil {
		caseVal, err = evaluateExpressionFast(ctx, caseExpr.Value, registry, ps, row)
		if err != nil {
			return nil, err
		}
	}

	// Evaluate each WHEN clause
	for _, when := range caseExpr.WhenClauses {
		var condResult bool

		if caseExpr.Value != nil {
			// Simple CASE: Compare case value with WHEN expression
			whenVal, err := evaluateExpressionFast(ctx, when.Condition, registry, ps, row)
			if err != nil {
				return nil, err
			}
			condResult = compareValues(caseVal, whenVal)
		} else {
			// Searched CASE: Evaluate WHEN condition as boolean
			condVal, err := evaluateExpressionFast(ctx, when.Condition, registry, ps, row)
			if err != nil {
				return nil, err
			}
			condResult = isTruthy(condVal)
		}

		if condResult {
			// Return the THEN value
			return evaluateExpressionFast(ctx, when.ThenResult, registry, ps, row)
		}
	}

	// No WHEN clause matched, return ELSE value if present
	if caseExpr.ElseValue != nil {
		return evaluateExpressionFast(ctx, caseExpr.ElseValue, registry, ps, row)
	}

	// No ELSE clause, return NULL
	return nil, nil
}

// compareValues compares two values for equality
func compareValues(left, right interface{}) bool {
	if left == nil && right == nil {
		return true
	}
	if left == nil || right == nil {
		return false
	}
	return fmt.Sprintf("%v", left) == fmt.Sprintf("%v", right)
}

// compareNumbers compares two values numerically
func compareNumbers(left, right interface{}, op string) (interface{}, error) {
	// Handle NULL values
	if left == nil || right == nil {
		return false, nil
	}

	// Convert to numbers for comparison
	leftNum, leftIsNum := convertToNumber(left)
	rightNum, rightIsNum := convertToNumber(right)

	if !leftIsNum || !rightIsNum {
		return false, fmt.Errorf("comparison operation %s requires numeric values", op)
	}

	switch op {
	case ">":
		return leftNum > rightNum, nil
	case ">=":
		return leftNum >= rightNum, nil
	case "<":
		return leftNum < rightNum, nil
	case "<=":
		return leftNum <= rightNum, nil
	default:
		return false, fmt.Errorf("unsupported comparison operator: %s", op)
	}
}

// isTruthy determines if a value is considered true
func isTruthy(val interface{}) bool {
	if val == nil {
		return false
	}
	switch v := val.(type) {
	case bool:
		return v
	case int64:
		return v != 0
	case float64:
		return v != 0
	case string:
		return v != "" && v != "0" && strings.ToLower(v) != "false"
	default:
		return true
	}
}

// executeInsertWithContext executes an INSERT statement
// Returns the number of rows affected and the last insert ID (for auto-increment columns)
func (e *Executor) executeInsertWithContext(ctx context.Context, tx storage.Transaction, stmt *parser.InsertStatement) (int64, int64, error) {
	// Extract table name from the table expression
	var tableName string

	// Handle different types of table expressions
	if stmt.TableName != nil {
		tableName = stmt.TableName.Value
	} else {
		return 0, 0, fmt.Errorf("missing table name in INSERT statement")
	}

	// Check if the table exists
	exists, err := e.engine.TableExists(tableName)
	if err != nil {
		return 0, 0, err
	}

	if !exists {
		return 0, 0, storage.ErrTableNotFound
	}

	// Get the table schema
	schema, err := e.engine.GetTableSchema(tableName)
	if err != nil {
		return 0, 0, err
	}

	cp := stringSlicePool.Get().(*[]string)
	columnNames := *cp
	if cap(columnNames) < len(stmt.Columns) {
		// Resize the slice if necessary
		columnNames = make([]string, len(stmt.Columns))
	} else {
		columnNames = columnNames[:len(stmt.Columns)]
	}

	defer func() {
		*cp = columnNames[:0]   // Reset the slice in the pool
		stringSlicePool.Put(cp) // Return to pool
	}()

	// Create a map of column names to indices for O(1) lookup
	columnMap := make(map[string]int, len(schema.Columns))
	for j, col := range schema.Columns {
		// Store both exact case and lowercase for case-insensitive lookup
		columnMap[col.Name] = j
		columnMap[strings.ToLower(col.Name)] = j
	}

	// Extract column names
	for i, col := range stmt.Columns {
		columnNames[i] = col.Value
	}

	// If no columns were specified, use all columns
	if len(columnNames) == 0 {
		if cap(columnNames) < len(schema.Columns) {
			// Resize the slice if necessary
			columnNames = make([]string, len(schema.Columns))
		} else {
			columnNames = columnNames[:len(schema.Columns)]
		}

		for i, col := range schema.Columns {
			columnNames[i] = col.Name
		}
	}

	// Process the values - support multi-row insert
	if len(stmt.Values) == 0 {
		return 0, 0, fmt.Errorf("no values provided for INSERT")
	}

	// Get the table to insert rows
	table, err := tx.GetTable(tableName)
	if err != nil {
		return 0, 0, err
	}

	// Extract parameter object once outside the loop
	var ps *parameter
	if pCtx := ctx.Value(psContextKey); pCtx != nil {
		ps, _ = pCtx.(*parameter)
	}

	// Optimize for batch insertion if we have multiple rows
	if len(stmt.Values) > 1 {
		// Check if table supports batch operations (MVCCTable)
		mvccTable, isMVCC := table.(*mvcc.MVCCTable)

		if isMVCC {
			// Pre-allocate all rows for batch processing
			rows := make([]storage.Row, 0, len(stmt.Values))

			// Process each row of values
			for rowIndex, rowExprs := range stmt.Values {
				if len(rowExprs) == 0 {
					return int64(rowIndex), 0, fmt.Errorf("empty values in row %d", rowIndex+1)
				}

				// Check if column count matches value count
				if len(columnNames) != len(rowExprs) {
					return int64(rowIndex), 0, fmt.Errorf("column count (%d) does not match value count (%d) in row %d",
						len(columnNames), len(rowExprs), rowIndex+1)
				}

				// Extract values from the expressions for this row
				cp := interfaceSlicePool.Get().(*[]interface{})
				columnValues := *cp
				if cap(columnValues) < len(rowExprs) {
					// Resize the slice if necessary
					columnValues = make([]interface{}, len(rowExprs))
				} else {
					columnValues = columnValues[:len(rowExprs)]
				}

				defer func() {
					*cp = columnValues[:0]     // Reset the slice in the pool
					interfaceSlicePool.Put(cp) // Return to pool
				}()

				for i, expr := range rowExprs {
					// Use fast evaluator for all expressions including function calls
					val, err := evaluateExpressionFast(ctx, expr, e.functionRegistry, ps, nil)
					if err != nil {
						return int64(rowIndex), 0, fmt.Errorf("error evaluating expression in row %d: %w", rowIndex+1, err)
					}
					columnValues[i] = val
				}

				// Create a row with the values in the correct order
				row := make(storage.Row, len(schema.Columns))

				// Initialize null columns
				for i := range row {
					// Set explicit NULL values for columns not included in the INSERT
					if !slices.Contains(columnNames, schema.Columns[i].Name) {
						row[i] = storage.ValueToColumnValue(nil, schema.Columns[i].Type)
					}
				}

				// Set the values for the specified columns
				for i, colName := range columnNames {
					// O(1) lookup
					colIndex, exists := columnMap[colName]
					if !exists {
						// Try lowercase version for case-insensitive match
						colIndex, exists = columnMap[strings.ToLower(colName)]
						if !exists {
							return int64(rowIndex), 0, fmt.Errorf("column not found: %s", colName)
						}
					}
					// Convert value to proper column value
					colType := schema.Columns[colIndex].Type
					row[colIndex] = storage.ValueToColumnValue(columnValues[i], colType)
				}

				// Add row to batch
				rows = append(rows, row)
			}

			// Insert all rows in a single batch operation
			if err := mvccTable.InsertBatch(rows); err != nil {
				return 0, 0, err
			}

			// Get the current auto-increment value after the batch insert
			lastInsertID := mvccTable.GetCurrentAutoIncrementValue()

			return int64(len(rows)), lastInsertID, nil
		}
	}

	// Fallback to single-row insertion for non-batch cases
	var totalRowsInserted int64
	var totalRowsUpdated int64

	for rowIndex, rowExprs := range stmt.Values {
		if len(rowExprs) == 0 {
			return totalRowsInserted + totalRowsUpdated, 0, fmt.Errorf("empty values in row %d", rowIndex+1)
		}

		cp := interfaceSlicePool.Get().(*[]interface{})
		columnValues := *cp
		if cap(columnValues) < len(rowExprs) {
			// Resize the slice if necessary
			columnValues = make([]interface{}, len(rowExprs))
		} else {
			columnValues = columnValues[:len(rowExprs)]
		}

		defer func() {
			*cp = columnValues[:0]     // Reset the slice in the pool
			interfaceSlicePool.Put(cp) // Return to pool
		}()

		for i, expr := range rowExprs {
			// Use fast evaluator for all expressions including function calls
			val, err := evaluateExpressionFast(ctx, expr, e.functionRegistry, ps, nil)
			if err != nil {
				return totalRowsInserted + totalRowsUpdated, 0, fmt.Errorf("error evaluating expression: %w", err)
			}
			columnValues[i] = val
		}

		// Check if the number of columns match the number of values
		if len(columnNames) != len(columnValues) {
			return totalRowsInserted + totalRowsUpdated, 0, fmt.Errorf("column count (%d) does not match value count (%d)",
				len(columnNames), len(columnValues))
		}

		// Create a row with the values in the correct order
		row := make(storage.Row, len(schema.Columns))

		// Initialize null columns
		for i := range row {
			// Set explicit NULL values
			if !slices.Contains(columnNames, schema.Columns[i].Name) {
				row[i] = storage.ValueToColumnValue(nil, schema.Columns[i].Type)
			}
		}

		// Set the values for the specified columns
		for i, colName := range columnNames {
			// O(1) lookup instead of O(n) scan
			colIndex, exists := columnMap[colName]
			if !exists {
				// Try lowercase version for case-insensitive match
				colIndex, exists = columnMap[strings.ToLower(colName)]
				if !exists {
					return totalRowsInserted + totalRowsUpdated, 0, fmt.Errorf("column not found: %s", colName)
				}
			}
			// Convert raw value to proper column value
			colType := schema.Columns[colIndex].Type
			row[colIndex] = storage.ValueToColumnValue(columnValues[i], colType)
		}

		// Try to insert the row
		err = table.Insert(row)
		if err != nil {
			// If we have ON DUPLICATE KEY UPDATE clause and we get a unique constraint violation
			if stmt.OnDuplicate {
				var pkErr *storage.ErrPrimaryKeyConstraint
				var uniqueErr *storage.ErrUniqueConstraint

				if errors.As(err, &pkErr) || errors.As(err, &uniqueErr) {
					// Create a search expression to find the duplicate row
					var searchExpr storage.Expression

					// Case 1: Primary key constraint violation - we know exactly which row ID to find
					if errors.As(err, &pkErr) {
						// Find primary key column to create the search expression
						for _, col := range schema.Columns {
							if col.PrimaryKey {
								searchExpr = expression.NewSimpleExpression(col.Name, storage.EQ, pkErr.RowID)
								searchExpr.PrepareForSchema(schema)
								break
							}
						}
					}

					// Case 2: Unique constraint violation - we have column name and value
					if searchExpr == nil && errors.As(err, &uniqueErr) {
						searchExpr = expression.NewSimpleExpression(uniqueErr.Column, storage.EQ, uniqueErr.Value.AsInterface())
						searchExpr.PrepareForSchema(schema)
					}

					// If we found a valid search expression, find and update the row
					if searchExpr != nil {
						// Get column indices for all columns
						colIndices := make([]int, len(schema.Columns))
						for i := range colIndices {
							colIndices[i] = i
						}

						// Scan for the duplicate row
						scanner, scanErr := table.Scan(colIndices, searchExpr)
						if scanErr == nil {
							defer scanner.Close()

							// If we found a row
							if scanner.Next() {
								// Create updater function for applying the ON DUPLICATE KEY UPDATE
								updaterFn := func(oldRow storage.Row) (storage.Row, bool) {
									// Create a new row as a copy of the old row
									newRow := make(storage.Row, len(oldRow))
									copy(newRow, oldRow)

									// Create a map of current row values for expression evaluation
									rowMap := make(map[string]interface{})
									for j, col := range schema.Columns {
										if j < len(oldRow) {
											// Convert ColumnValue to interface{}
											rowMap[col.Name] = oldRow[j].AsInterface()
										}
									}

									// Apply the updates
									for i, updateColumn := range stmt.UpdateColumns {
										colName := updateColumn.Value
										expr := stmt.UpdateExpressions[i]

										// Find the column index
										colIndex := -1
										for j, col := range schema.Columns {
											if col.Name == colName {
												colIndex = j
												break
											}
										}

										if colIndex != -1 {
											// Use fast evaluator for all expressions including function calls
											updateValue, err := evaluateExpressionFast(ctx, expr, e.functionRegistry, ps, rowMap)
											if err != nil {
												// If evaluation fails, skip this update
												continue
											}

											// Set the new value
											colType := schema.Columns[colIndex].Type
											newRow[colIndex] = storage.ValueToColumnValue(updateValue, colType)
										}
									}

									return newRow, false
								}

								// Update the row using the Update API
								_, updateErr := table.Update(searchExpr, updaterFn)
								if updateErr == nil {
									// MySQL reports 2 for ON DUPLICATE KEY UPDATE
									totalRowsUpdated += 2
									continue
								}
							}
						}
					}

					// If we couldn't handle the duplicate key properly, return the original error
					return totalRowsInserted + totalRowsUpdated, 0, err
				}
			}
			// For any other error, return it without ON DUPLICATE KEY handling
			return totalRowsInserted + totalRowsUpdated, 0, err
		}

		// Normal insert success
		totalRowsInserted++
	}

	// Get the current auto-increment value after all inserts
	var lastInsertID int64 = 0
	if len(stmt.Values) > 0 {
		if mvccTable, ok := table.(*mvcc.MVCCTable); ok {
			lastInsertID = mvccTable.GetCurrentAutoIncrementValue()
		}
	}

	return totalRowsInserted + totalRowsUpdated, lastInsertID, nil
}

// executeUpdate executes an UPDATE statement
func (e *Executor) executeUpdateWithContext(ctx context.Context, tx storage.Transaction, stmt *parser.UpdateStatement) (int64, error) {
	// Extract table name from the table expression
	var tableName string

	// Handle different types of table expressions
	if stmt.TableName != nil {
		tableName = stmt.TableName.Value
	} else {
		return 0, fmt.Errorf("missing table name in UPDATE statement")
	}

	// Check if the table exists
	exists, err := e.engine.TableExists(tableName)
	if err != nil {
		return 0, err
	}
	if !exists {
		return 0, storage.ErrTableNotFound
	}

	// Get the table
	table, err := tx.GetTable(tableName)
	if err != nil {
		return 0, err
	}

	// Get the schema to know the column types
	schema, err := e.engine.GetTableSchema(tableName)
	if err != nil {
		return 0, err
	}

	// Create a map of column names to indices and types for O(1) lookup
	columnMap := make(map[string]struct {
		Index int
		Type  storage.DataType
	}, len(schema.Columns))

	for i, col := range schema.Columns {
		// Store both exact case and lowercase for case-insensitive lookup
		columnMap[col.Name] = struct {
			Index int
			Type  storage.DataType
		}{
			Index: i,
			Type:  col.Type,
		}
		columnMap[strings.ToLower(col.Name)] = struct {
			Index int
			Type  storage.DataType
		}{
			Index: i,
			Type:  col.Type,
		}
	}

	// Extract parameter object once outside the loop
	var ps *parameter
	if pCtx := ctx.Value(psContextKey); pCtx != nil {
		ps, _ = pCtx.(*parameter)
	}

	// Create a setter function that updates the values
	setter := func(row storage.Row) (storage.Row, bool) {
		// Create a copy of the row to avoid races when modifying in place
		updatedRow := make(storage.Row, len(row))
		copy(updatedRow, row)

		// Create row context for column references in expressions
		rowContext := make(map[string]interface{})
		for i, col := range schema.Columns {
			if i < len(updatedRow) && updatedRow[i] != nil {
				rowContext[col.Name] = updatedRow[i].AsInterface()
			}
		}

		// Apply updates
		for colName, expr := range stmt.Updates {
			// Find the column using O(1) map lookup instead of O(n) scan
			colInfo, exists := columnMap[colName]
			if !exists {
				// Try lowercase version for case-insensitive match
				colInfo, exists = columnMap[strings.ToLower(colName)]
				if !exists {
					continue // Skip unknown column
				}
			}

			colIndex := colInfo.Index
			colType := colInfo.Type

			// Use fast evaluator with current row context for column references
			value, err := evaluateExpressionFast(ctx, expr, e.functionRegistry, ps, rowContext)
			if err != nil {
				continue // Skip expressions that fail to evaluate
			}

			// Convert to column value and update
			updatedRow[colIndex] = storage.ValueToColumnValue(value, colType)
		}

		return updatedRow, true
	}

	// Create a storage-level expression from the SQL WHERE clause
	var updateExpr storage.Expression
	if stmt.Where != nil {
		// First, process any subqueries in the WHERE clause
		processedWhere, err := e.processWhereSubqueries(ctx, tx, stmt.Where)
		if err != nil {
			return 0, fmt.Errorf("error processing subqueries in WHERE clause: %w", err)
		}

		// Check if the processed WHERE is a boolean literal
		if boolLit, ok := processedWhere.(*parser.BooleanLiteral); ok {
			// Handle constant boolean WHERE clauses
			if !boolLit.Value {
				// WHERE false - update no rows
				return 0, nil
			}
			// WHERE true - update all rows (set updateExpr to nil)
			updateExpr = nil
		} else {
			// Convert the SQL WHERE expression to a storage-level expression
			updateExpr = createWhereExpression(ctx, processedWhere, e.functionRegistry)
			if updateExpr == nil {
				return 0, fmt.Errorf("unsupported WHERE clause in UPDATE statement: %T", processedWhere)
			}
			updateExpr.PrepareForSchema(schema)
		}
	} else {
		// If no WHERE clause, update all rows - use a simple expression that always returns true
		updateExpr = nil
	}

	// Execute a single UPDATE operation with the WHERE expression
	// and our setter function that applies column updates.
	// The Update method in the storage layer will handle the expression evaluation
	// and visibility concerns, making this much more efficient for large tables.
	count, err := table.Update(updateExpr, setter)
	if err != nil {
		return 0, fmt.Errorf("error executing UPDATE: %w", err)
	}
	return int64(count), nil
}

// executeDeleteWithContext executes a DELETE statement
func (e *Executor) executeDeleteWithContext(ctx context.Context, tx storage.Transaction, stmt *parser.DeleteStatement) (int64, error) {
	// Extract table name from the table expression
	var tableName string

	// Handle different types of table expressions
	if stmt.TableName != nil {
		tableName = stmt.TableName.Value
	} else {
		return 0, fmt.Errorf("missing table name in DELETE statement")
	}

	// Check if the table exists
	exists, err := e.engine.TableExists(tableName)
	if err != nil {
		return 0, err
	}
	if !exists {
		return 0, storage.ErrTableNotFound
	}

	// Get the table
	table, err := tx.GetTable(tableName)
	if err != nil {
		return 0, err
	}

	// Get the schema to know the column types
	schema, err := e.engine.GetTableSchema(tableName)
	if err != nil {
		return 0, err
	}

	// Create a WHERE expression directly from the SQL WHERE clause
	var deleteExpr storage.Expression
	if stmt.Where != nil {
		// First, process any subqueries in the WHERE clause
		processedWhere, err := e.processWhereSubqueries(ctx, tx, stmt.Where)
		if err != nil {
			return 0, fmt.Errorf("error processing subqueries in WHERE clause: %w", err)
		}

		// Check if the processed WHERE is a boolean literal
		if boolLit, ok := processedWhere.(*parser.BooleanLiteral); ok {
			// Handle constant boolean WHERE clauses
			if !boolLit.Value {
				// WHERE false - delete no rows
				return 0, nil
			}
			// WHERE true - delete all rows (set deleteExpr to nil)
			deleteExpr = nil
		} else {
			// Convert the SQL WHERE expression to a storage-level expression
			deleteExpr = createWhereExpression(ctx, processedWhere, e.functionRegistry)
			if deleteExpr == nil {
				return 0, fmt.Errorf("unsupported WHERE clause in DELETE statement: %T", processedWhere)
			}
			deleteExpr.PrepareForSchema(schema)
		}
	} else {
		// If no WHERE clause, delete all rows - use a simple expression that always returns true
		deleteExpr = nil
	}

	// Execute the deletion directly using the WHERE expression at the storage layer
	// This avoids materializing rows in memory and is much more efficient
	rowsDeleted, err := table.Delete(deleteExpr)
	if err != nil {
		return 0, fmt.Errorf("error executing DELETE: %w", err)
	}

	return int64(rowsDeleted), nil
}

// createWhereExpression creates a storage.Expression from a parser.Expression
func createWhereExpression(ctx context.Context, expr parser.Expression, registry contract.FunctionRegistry) storage.Expression {
	return CreateWhereExpression(ctx, expr, registry)
}

// CreateWhereExpression converts a parser expression to a storage expression
func CreateWhereExpression(ctx context.Context, expr parser.Expression, registry contract.FunctionRegistry) storage.Expression {
	if expr == nil {
		return nil
	}

	// Debug: print what we're converting
	// fmt.Printf("DEBUG createWhereExpression: converting %T: %s\n", expr, expr.String())

	// Handle NOT expressions (implemented as PrefixExpression with operator "NOT")
	if prefixExpr, ok := expr.(*parser.PrefixExpression); ok && prefixExpr.Operator == "NOT" {
		// Recursively process the inner expression
		innerExpr := createWhereExpression(ctx, prefixExpr.Right, registry)

		// If we successfully created a storage expression, wrap it in a NOT
		if innerExpr != nil {
			return expression.NewNotExpression(innerExpr)
		}
		return nil
	}

	// For binary comparison operations, we can optimize by creating direct storage expressions
	if binaryExpr, ok := expr.(*parser.InfixExpression); ok {
		// Special handling for CAST expressions in comparisons
		if castExpr, ok := binaryExpr.Left.(*parser.CastExpression); ok {
			// Handle CAST(column AS type) operator value
			if ident, ok := castExpr.Expr.(*parser.Identifier); ok {
				// Extract the column name
				colName := ident.Value

				// Get the target data type
				var targetType storage.DataType
				switch strings.ToUpper(castExpr.TypeName) {
				case "INTEGER", "INT":
					targetType = storage.INTEGER
				case "FLOAT", "REAL", "DOUBLE":
					targetType = storage.FLOAT
				case "TEXT", "STRING", "VARCHAR":
					targetType = storage.TEXT
				case "BOOLEAN", "BOOL":
					targetType = storage.BOOLEAN
				case "TIMESTAMP":
					targetType = storage.TIMESTAMP
				case "JSON":
					targetType = storage.JSON
				default:
					return nil
				}

				// Create a cast expression
				castStorageExpr := expression.NewCastExpression(colName, targetType)

				// The value to compare to
				var value interface{}
				if isLiteral(binaryExpr.Right) {
					value = getLiteralValue(ctx, binaryExpr.Right)
				} else {
					// If not a simple literal, delegate to default handling
					return nil
				}

				// Get the operator
				var operator storage.Operator
				switch binaryExpr.Operator {
				case ">":
					operator = storage.GT
				case ">=":
					operator = storage.GTE
				case "<":
					operator = storage.LT
				case "<=":
					operator = storage.LTE
				case "=":
					operator = storage.EQ
				case "!=", "<>":
					operator = storage.NE
				default:
					return nil
				}

				// Create a compound expression
				compoundExpr := &expression.CompoundExpression{
					CastExpr: castStorageExpr,
					Operator: operator,
					Value:    value,
				}
				return compoundExpr
			}
		}
		if binaryExpr.Operator == ">" && isColumnAndLiteral(ctx, binaryExpr) {
			colName, value := extractColumnAndValue(ctx, binaryExpr)
			return expression.NewSimpleExpression(colName, storage.GT, value)
		} else if binaryExpr.Operator == ">=" && isColumnAndLiteral(ctx, binaryExpr) {
			colName, value := extractColumnAndValue(ctx, binaryExpr)
			return expression.NewSimpleExpression(colName, storage.GTE, value)
		} else if binaryExpr.Operator == "<" && isColumnAndLiteral(ctx, binaryExpr) {
			colName, value := extractColumnAndValue(ctx, binaryExpr)
			return expression.NewSimpleExpression(colName, storage.LT, value)
		} else if binaryExpr.Operator == "<=" && isColumnAndLiteral(ctx, binaryExpr) {
			colName, value := extractColumnAndValue(ctx, binaryExpr)
			return expression.NewSimpleExpression(colName, storage.LTE, value)
		} else if binaryExpr.Operator == "=" && isColumnAndLiteral(ctx, binaryExpr) {
			colName, value := extractColumnAndValue(ctx, binaryExpr)
			return expression.NewSimpleExpression(colName, storage.EQ, value)
		} else if binaryExpr.Operator == "!=" && isColumnAndLiteral(ctx, binaryExpr) {
			colName, value := extractColumnAndValue(ctx, binaryExpr)
			return expression.NewSimpleExpression(colName, storage.NE, value)
		} else if binaryExpr.Operator == "BETWEEN" {
			// Extract column name
			colName := ""
			if col, ok := binaryExpr.Left.(*parser.Identifier); ok {
				colName = col.Value
			} else {
				return nil
			}

			// Extract the min and max values from the AND expression
			if andExpr, ok := binaryExpr.Right.(*parser.InfixExpression); ok && andExpr.Operator == "AND" {
				lowerValue := getLiteralValue(ctx, andExpr.Left)
				upperValue := getLiteralValue(ctx, andExpr.Right)

				// Create a BetweenExpression that handles both bounds
				if colName != "" && lowerValue != nil && upperValue != nil {
					betweenExpr := &expression.BetweenExpression{
						Column:     colName,
						LowerBound: lowerValue,
						UpperBound: upperValue,
						Inclusive:  true, // BETWEEN is inclusive on both ends
					}

					return betweenExpr
				}
			}

			// If we can't properly extract the BETWEEN values, return nil
			return nil
		} else if binaryExpr.Operator == "AND" {
			// Check for range pattern first, like "column > value1 AND column <= value2"
			if isRangePattern(binaryExpr) {
				column, minValue, maxValue, includeMin, includeMax := extractRangeValues(ctx, binaryExpr)
				if column != "" && (minValue != nil || maxValue != nil) {
					// Create a BetweenExpression with custom inclusivity
					return expression.NewRangeExpression(column, minValue, maxValue, includeMin, includeMax)
				}
			}

			// For regular AND expressions, optimize both sides
			leftExpr := createWhereExpression(ctx, binaryExpr.Left, registry)
			rightExpr := createWhereExpression(ctx, binaryExpr.Right, registry)

			// If both sides could be optimized, create an AND expression
			if leftExpr != nil && rightExpr != nil {
				return &expression.AndExpression{
					Expressions: []storage.Expression{leftExpr, rightExpr},
				}
			}
		} else if binaryExpr.Operator == "OR" {
			// Special case optimization for "column = true OR column = false" which is equivalent to "column IS NOT NULL"
			if isBooleanEqualityPair(ctx, binaryExpr) {
				colName, isBoolean := extractBooleanOrColumn(ctx, binaryExpr)
				if colName != "" && isBoolean {
					// Directly create IS NOT NULL expression instead of OR
					return expression.NewIsNotNullExpression(colName)
				}
			}

			// For OR expressions, optimize both sides
			leftExpr := createWhereExpression(ctx, binaryExpr.Left, registry)
			rightExpr := createWhereExpression(ctx, binaryExpr.Right, registry)

			// If both sides could be optimized, create an OR expression
			if leftExpr != nil && rightExpr != nil {
				return &expression.OrExpression{
					Expressions: []storage.Expression{leftExpr, rightExpr},
				}
			}
		}
	}

	// Special handling for BETWEEN expressions
	if betweenExpr, ok := expr.(*parser.BetweenExpression); ok {
		// Extract column name
		var columnName string
		if col, ok := betweenExpr.Expr.(*parser.Identifier); ok {
			columnName = col.Value
		} else {
			return nil
		}

		// Extract lower and upper bounds
		lowerValue := getLiteralValue(ctx, betweenExpr.Lower)
		upperValue := getLiteralValue(ctx, betweenExpr.Upper)

		if lowerValue != nil && upperValue != nil {
			// Create a BetweenExpression directly from the parser expression
			betweenStorageExpr := &expression.BetweenExpression{
				Column:     columnName,
				LowerBound: lowerValue,
				UpperBound: upperValue,
				Inclusive:  true, // BETWEEN is always inclusive
			}

			return betweenStorageExpr
		}

		return nil
	}

	// Special handling for CAST expressions
	if castExpr, ok := expr.(*parser.CastExpression); ok {
		// Handle CAST(column AS type)
		if ident, ok := castExpr.Expr.(*parser.Identifier); ok {
			// Extract the target data type
			var targetType storage.DataType
			switch strings.ToUpper(castExpr.TypeName) {
			case "INTEGER", "INT":
				targetType = storage.INTEGER
			case "FLOAT", "REAL", "DOUBLE":
				targetType = storage.FLOAT
			case "TEXT", "STRING", "VARCHAR":
				targetType = storage.TEXT
			case "BOOLEAN", "BOOL":
				targetType = storage.BOOLEAN
			case "TIMESTAMP":
				targetType = storage.TIMESTAMP
			case "JSON":
				targetType = storage.JSON
			default:
				// Unsupported type, fall back to default handling
				return nil
			}

			// Create a CastExpression for the storage layer
			return expression.NewCastExpression(ident.Value, targetType)
		}
	}

	// Special handling for IS NULL and IS NOT NULL expressions
	if infix, ok := expr.(*parser.InfixExpression); ok {
		if infix.Operator == "IS" || infix.Operator == "IS NOT" {
			if colExpr, ok := infix.Left.(*parser.Identifier); ok {
				if _, ok := infix.Right.(*parser.NullLiteral); ok {
					if infix.Operator == "IS" {
						// IS NULL expression
						return expression.NewIsNullExpression(colExpr.Value)
					} else {
						// IS NOT NULL expression
						return expression.NewIsNotNullExpression(colExpr.Value)
					}
				}
			}
		}
	}

	// Special handling for IN expressions
	if inExpr, ok := expr.(*parser.InExpression); ok {
		// Extract column name from the left side
		var columnName string
		if col, ok := inExpr.Left.(*parser.Identifier); ok {
			columnName = col.Value
		} else {
			// Fall back to string representation if not a simple identifier
			columnName = inExpr.Left.String()
		}

		// Check if right side is an expression list
		if exprList, ok := inExpr.Right.(*parser.ExpressionList); ok {
			// Extract values from the expression list
			values := make([]interface{}, 0, len(exprList.Expressions))
			for _, e := range exprList.Expressions {
				switch v := e.(type) {
				case *parser.StringLiteral:
					values = append(values, v.Value)
				case *parser.IntegerLiteral:
					values = append(values, v.Value)
				case *parser.FloatLiteral:
					values = append(values, v.Value)
				case *parser.BooleanLiteral:
					values = append(values, v.Value)
				case *parser.Parameter:
					// Handle parameter binding if available
					if ps := getParameterFromContext(ctx); ps != nil {
						param := ps.GetValue(v)
						values = append(values, param.Value)
					}
				case *parser.NullLiteral:
					values = append(values, nil)
				}
			}

			// Create a direct in-list expression with the Not flag
			return &expression.InListExpression{
				Column: columnName,
				Values: values,
				Not:    inExpr.Not, // Pass the Not flag from the parser InExpression
			}
		}
	}

	return nil
}

// isColumnAndLiteral checks if an expression is a comparison between a column and a literal value
func isColumnAndLiteral(ctx context.Context, expr *parser.InfixExpression) bool {
	_ = ctx // Unused context parameter

	// Check if left is a column and right is a literal
	if _, isColLeft := expr.Left.(*parser.Identifier); isColLeft {
		return isLiteral(expr.Right)
	}

	// Check if right is a column and left is a literal
	if _, isColRight := expr.Right.(*parser.Identifier); isColRight {
		return isLiteral(expr.Left)
	}

	return false
}

// isRangePattern detects if an AND expression represents a range query pattern
// like "column > value1 AND column <= value2"
func isRangePattern(expr *parser.InfixExpression) bool {
	// Must be an AND expression
	if expr.Operator != "AND" {
		return false
	}

	// Check if both sides are infix expressions (comparisons)
	leftExpr, leftOk := expr.Left.(*parser.InfixExpression)
	rightExpr, rightOk := expr.Right.(*parser.InfixExpression)

	if !leftOk || !rightOk {
		return false
	}

	// Both must be comparing a column with a literal
	if !isColumnAndLiteral(context.Background(), leftExpr) || !isColumnAndLiteral(context.Background(), rightExpr) {
		return false
	}

	// Extract column names from both expressions
	leftCol, rightCol := extractColumnName(leftExpr), extractColumnName(rightExpr)

	// Both comparisons must be on the same column
	if leftCol == "" || rightCol == "" || leftCol != rightCol {
		return false
	}

	// Check if we have range operators
	leftOp := leftExpr.Operator
	rightOp := rightExpr.Operator

	// Either (left is lower bound AND right is upper bound) OR (left is upper bound AND right is lower bound)
	isLeftLower := leftOp == ">" || leftOp == ">="
	isRightUpper := rightOp == "<" || rightOp == "<="

	isRightLower := rightOp == ">" || rightOp == ">="
	isLeftUpper := leftOp == "<" || leftOp == "<="

	return (isLeftLower && isRightUpper) || (isRightLower && isLeftUpper)
}

// extractColumnName gets the column name from a comparison expression
func extractColumnName(expr *parser.InfixExpression) string {
	// Check if left side is column
	if col, ok := expr.Left.(*parser.Identifier); ok {
		return col.Value
	}

	// Check if right side is column
	if col, ok := expr.Right.(*parser.Identifier); ok {
		return col.Value
	}

	return ""
}

// extractRangeValues extracts all values needed for a range expression
func extractRangeValues(ctx context.Context, expr *parser.InfixExpression) (
	column string, minValue, maxValue interface{}, includeMin, includeMax bool) {

	// Get the left and right comparison expressions
	leftExpr, _ := expr.Left.(*parser.InfixExpression)
	rightExpr, _ := expr.Right.(*parser.InfixExpression)

	// Extract column name (either side will have the same column)
	column = extractColumnName(leftExpr)

	// Figure out which one is the lower bound and which is the upper bound
	leftOp := leftExpr.Operator
	rightOp := rightExpr.Operator

	isLeftLower := leftOp == ">" || leftOp == ">="
	isRightUpper := rightOp == "<" || rightOp == "<="

	if isLeftLower && isRightUpper {
		// Left is minimum bound, right is maximum bound
		if extractColumnFromLeft(leftExpr) {
			minValue = getLiteralValue(ctx, leftExpr.Right)
		} else {
			minValue = getLiteralValue(ctx, leftExpr.Left)
		}

		if extractColumnFromLeft(rightExpr) {
			maxValue = getLiteralValue(ctx, rightExpr.Right)
		} else {
			maxValue = getLiteralValue(ctx, rightExpr.Left)
		}

		includeMin = leftOp == ">="
		includeMax = rightOp == "<="
	} else {
		// Right is minimum bound, left is maximum bound
		if extractColumnFromLeft(rightExpr) {
			minValue = getLiteralValue(ctx, rightExpr.Right)
		} else {
			minValue = getLiteralValue(ctx, rightExpr.Left)
		}

		if extractColumnFromLeft(leftExpr) {
			maxValue = getLiteralValue(ctx, leftExpr.Right)
		} else {
			maxValue = getLiteralValue(ctx, leftExpr.Left)
		}

		includeMin = rightOp == ">="
		includeMax = leftOp == "<="
	}

	return
}

// extractColumnFromLeft returns true if the column is on the left side of the expression
func extractColumnFromLeft(expr *parser.InfixExpression) bool {
	_, ok := expr.Left.(*parser.Identifier)
	return ok
}

// isLiteral checks if an expression is a literal value
func isLiteral(expr parser.Expression) bool {
	switch expr.(type) {
	case *parser.IntegerLiteral, *parser.FloatLiteral, *parser.StringLiteral, *parser.BooleanLiteral, *parser.NullLiteral, *parser.Parameter:
		return true
	default:
		return false
	}
}

// isBooleanEqualityPair checks if an expression is like "column = true OR column = false"
func isBooleanEqualityPair(ctx context.Context, expr *parser.InfixExpression) bool {
	if expr.Operator != "OR" {
		return false
	}

	// Check if both sides are equality comparisons
	leftInfix, leftOk := expr.Left.(*parser.InfixExpression)
	rightInfix, rightOk := expr.Right.(*parser.InfixExpression)

	if !leftOk || !rightOk || leftInfix.Operator != "=" || rightInfix.Operator != "=" {
		return false
	}

	// Check if both sides reference the same column
	leftCol := extractColumnName(leftInfix)
	rightCol := extractColumnName(rightInfix)

	if leftCol == "" || rightCol == "" || leftCol != rightCol {
		return false
	}

	// Check if the values are true and false boolean literals
	leftValue := getLiteralValue(ctx, leftInfix.Right)
	rightValue := getLiteralValue(ctx, rightInfix.Right)

	leftBool, leftIsBool := leftValue.(bool)
	rightBool, rightIsBool := rightValue.(bool)

	// True when one side is true and one side is false
	return leftIsBool && rightIsBool && leftBool != rightBool
}

// extractBooleanOrColumn extracts the column name from a "column = true OR column = false" expression
// and returns whether it appears to be a boolean comparison
func extractBooleanOrColumn(ctx context.Context, expr *parser.InfixExpression) (string, bool) {
	if expr.Operator != "OR" {
		return "", false
	}

	// Extract the column name from the left side
	leftInfix, leftOk := expr.Left.(*parser.InfixExpression)
	if !leftOk || leftInfix.Operator != "=" {
		return "", false
	}

	colName := extractColumnName(leftInfix)
	if colName == "" {
		return "", false
	}

	// Verify we have boolean literals on both sides
	leftValue := getLiteralValue(ctx, leftInfix.Right)
	rightInfix, rightOk := expr.Right.(*parser.InfixExpression)

	if !rightOk || rightInfix.Operator != "=" {
		return "", false
	}

	rightValue := getLiteralValue(ctx, rightInfix.Right)

	leftBool, leftIsBool := leftValue.(bool)
	rightBool, rightIsBool := rightValue.(bool)

	// Return true only if we have a boolean column compared with true OR false
	return colName, leftIsBool && rightIsBool && leftBool != rightBool
}

// extractColumnAndValue extracts the column name and literal value from a binary expression
func extractColumnAndValue(ctx context.Context, expr *parser.InfixExpression) (string, interface{}) {
	// Check if left is column and right is literal
	if colExpr, isColLeft := expr.Left.(*parser.Identifier); isColLeft && isLiteral(expr.Right) {
		return colExpr.Value, getLiteralValue(ctx, expr.Right)
	}

	// Check if right is column and left is literal
	if colExpr, isColRight := expr.Right.(*parser.Identifier); isColRight && isLiteral(expr.Left) {
		// For reversed comparisons, we need to adjust the operator
		// e.g., "5 > id" should be interpreted as "id < 5"
		return colExpr.Value, getLiteralValue(ctx, expr.Left)
	}

	// This should never happen if isColumnAndLiteral was called first
	return "", nil
}

// Core function to extract parameter from context - do this once per query processing
func getParameterFromContext(ctx context.Context) *parameter {
	if pCtx := ctx.Value(psContextKey); pCtx != nil {
		if ps, ok := pCtx.(*parameter); ok {
			return ps
		}
	}
	return nil
}

// getLiteralValue extracts the value from a literal expression
// Pass in the pre-extracted parameter object to avoid context lookups
func getLiteralValue(ctx context.Context, expr parser.Expression) interface{} {
	switch e := expr.(type) {
	case *parser.IntegerLiteral:
		return e.Value
	case *parser.FloatLiteral:
		return e.Value
	case *parser.StringLiteral:
		return e.Value
	case *parser.BooleanLiteral:
		return e.Value
	case *parser.NullLiteral:
		return nil
	case *parser.Parameter:
		// Extract parameter just once if we need it
		ps := getParameterFromContext(ctx)
		if ps != nil {
			// Get the parameter value
			nm := ps.GetValue(e)
			return nm.Value
		}
		return nil
	default:
		return nil
	}
}
