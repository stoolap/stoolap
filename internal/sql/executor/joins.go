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
	"database/sql"
	"fmt"
	"strings"

	"github.com/stoolap/stoolap/internal/parser"
	"github.com/stoolap/stoolap/internal/storage"
)

// ExecuteJoin executes a JOIN operation between two table sources
func (e *Executor) ExecuteJoin(ctx context.Context, joinSource *parser.JoinTableSource,
	evaluator *Evaluator, params map[string]interface{}) (storage.Result, error) {
	return e.ExecuteJoinWithRegistry(ctx, joinSource, evaluator, params)
}

// ExecuteJoinWithRegistry executes a JOIN operation with CTE registry
func (e *Executor) ExecuteJoinWithRegistry(ctx context.Context, joinSource *parser.JoinTableSource,
	evaluator *Evaluator, params map[string]interface{}) (storage.Result, error) {

	// Execute the left side of the join
	leftSource := joinSource.Left
	if leftSource == nil {
		return nil, fmt.Errorf("left side of JOIN is not a valid table source")
	}
	leftResult, leftAlias, err := e.executeTableSourceWithRegistry(ctx, leftSource, evaluator, params)
	if err != nil {
		return nil, fmt.Errorf("error executing left side of JOIN: %w", err)
	}

	// Execute the right side of the join
	rightSource := joinSource.Right
	if rightSource == nil {
		return nil, fmt.Errorf("right side of JOIN is not a valid table source")
	}
	rightResult, rightAlias, err := e.executeTableSourceWithRegistry(ctx, rightSource, evaluator, params)
	if err != nil {
		leftResult.Close()
		return nil, fmt.Errorf("error executing right side of JOIN: %w", err)
	}

	// Determine the join type
	joinType := strings.ToUpper(joinSource.JoinType)
	if joinType == "" {
		joinType = "INNER" // Default to INNER JOIN
	}

	// Validate join type
	switch joinType {
	case "INNER", "LEFT", "RIGHT", "FULL", "CROSS":
		// Valid join types
	default:
		leftResult.Close()
		rightResult.Close()
		return nil, fmt.Errorf("unsupported join type: %s", joinType)
	}

	// For CROSS JOIN, use nil condition to match all rows
	var joinCond parser.Expression
	if joinType != "CROSS" {
		joinCond = joinSource.Condition

		// Ensure there is a join condition for non-CROSS joins
		if joinCond == nil {
			leftResult.Close()
			rightResult.Close()
			return nil, fmt.Errorf("%s JOIN requires a join condition", joinType)
		}
	}

	// Create the join result using the original implementation
	// TODO: Fix streaming join implementation and re-enable
	joinResult := NewJoinResult(
		leftResult,
		rightResult,
		joinType,
		joinCond,
		evaluator,
		leftAlias,
		rightAlias,
	)

	return joinResult, nil
}

// executeTableSourceWithRegistry executes a table source with CTE registry
func (e *Executor) executeTableSourceWithRegistry(ctx context.Context, tableSource parser.TableSource,
	evaluator *Evaluator, params map[string]interface{}) (storage.Result, string, error) {

	switch source := tableSource.(type) {
	case *parser.SimpleTableSource:
		// Simple table source (table name)
		tableName := source.Name.Value

		// Check if this is a CTE reference
		if cteResult, isCTE := e.resolveCTETable(ctx, nil, tableName); isCTE {
			// Use the table alias if provided, otherwise use the CTE name
			tableAlias := tableName
			if source.Alias != nil {
				tableAlias = source.Alias.Value
			}

			return cteResult, tableAlias, nil
		}

		// Not a CTE, proceed with regular table
		// Start a transaction to get the table
		tx, err := e.engine.BeginTx(ctx, sql.LevelReadCommitted)
		if err != nil {
			return nil, "", fmt.Errorf("error beginning transaction: %w", err)
		}
		defer tx.Rollback() // Rollback if not committed

		table, err := tx.GetTable(tableName)
		if err != nil {
			return nil, "", fmt.Errorf("error getting table %s: %w", tableName, err)
		}

		// Get columns to scan
		columns := make([]string, 0)
		for _, col := range table.Schema().Columns {
			columns = append(columns, col.Name)
		}

		// Create a result from the table
		result, err := tx.Select(tableName, columns, nil)
		if err != nil {
			return nil, "", fmt.Errorf("error creating scanner for table %s: %w", tableName, err)
		}

		// Use the table alias if provided, otherwise use the table name
		tableAlias := tableName
		if source.Alias != nil {
			tableAlias = source.Alias.Value
		}

		return result, tableAlias, nil

	case *parser.SubqueryTableSource:
		// Subquery table source
		// We need to handle subqueries differently since we don't have a full ExecuteSelect implementation
		// For now, just return an error indicating this is not implemented yet
		return nil, "", fmt.Errorf("subqueries in JOIN are not fully implemented yet")

	case *parser.JoinTableSource:
		// Nested JOIN
		joinResult, err := e.ExecuteJoinWithRegistry(ctx, source, evaluator, params)
		if err != nil {
			return nil, "", fmt.Errorf("error executing nested JOIN: %w", err)
		}

		// No obvious alias for a nested join, use empty string
		return joinResult, "", nil

	case *parser.CTEReference:
		// Common Table Expression reference
		// This would need to be implemented to support WITH clauses
		return nil, "", fmt.Errorf("CTE references in JOINs are not yet supported")

	default:
		return nil, "", fmt.Errorf("unsupported table source type: %T", tableSource)
	}
}

// ExecuteJoinQuery executes a SELECT query that contains JOINs
func (e *Executor) ExecuteJoinQuery(ctx context.Context, stmt *parser.SelectStatement,
	evaluator *Evaluator, params map[string]interface{}) (storage.Result, error) {
	return e.ExecuteJoinQueryWithRegistry(ctx, stmt, evaluator, params)
}

// ExecuteJoinQueryWithRegistry executes a SELECT query with JOINs and CTE registry
func (e *Executor) ExecuteJoinQueryWithRegistry(ctx context.Context, stmt *parser.SelectStatement,
	evaluator *Evaluator, params map[string]interface{}) (storage.Result, error) {

	// Get the joined result
	joinSource, ok := stmt.TableExpr.(*parser.JoinTableSource)
	if !ok {
		return nil, fmt.Errorf("FROM clause is not a JOIN")
	}

	joinResult, err := e.ExecuteJoinWithRegistry(ctx, joinSource, evaluator, params)
	if err != nil {
		return nil, err
	}

	// Apply WHERE clause if present
	if stmt.Where != nil {
		joinResult = &FilteredResult{
			result:    joinResult,
			whereExpr: stmt.Where,
			evaluator: evaluator,
		}
	}

	// Apply column projection based on SELECT clause
	if len(stmt.Columns) > 0 && !isAllColumns(stmt.Columns) {
		joinResult, err = applyColumnProjection(joinResult, stmt.Columns)
		if err != nil {
			joinResult.Close()
			return nil, err
		}
	}

	// Apply ORDER BY, LIMIT, OFFSET if specified
	if stmt.OrderBy != nil || stmt.Limit != nil || stmt.Offset != nil {
		// Extract values needed for ORDER BY, LIMIT, OFFSET
		var limit, offset int64 = -1, 0

		// Extract LIMIT value
		if stmt.Limit != nil {
			if limitLit, ok := stmt.Limit.(*parser.IntegerLiteral); ok {
				limit = limitLit.Value
			} else {
				// For more complex expressions, default to -1 (no limit)
				limit = -1
			}
		}

		// Extract OFFSET value
		if stmt.Offset != nil {
			if offsetLit, ok := stmt.Offset.(*parser.IntegerLiteral); ok {
				offset = offsetLit.Value
			} else {
				// For more complex expressions, default to 0 (no offset)
				offset = 0
			}
		}

		// Apply ORDER BY if specified
		if len(stmt.OrderBy) > 0 {
			joinResult = &OrderedResult{
				baseResult: joinResult,
				orderBy:    stmt.OrderBy,
			}
		}

		// Apply LIMIT and OFFSET if specified
		if limit >= 0 || offset > 0 {
			joinResult = &LimitedResult{
				baseResult: joinResult,
				limit:      limit,
				offset:     offset,
			}
		}
	}

	return joinResult, nil
}

// isAllColumns checks if the columns clause contains just "*"
func isAllColumns(columns []parser.Expression) bool {
	if len(columns) != 1 {
		return false
	}

	if ident, ok := columns[0].(*parser.Identifier); ok && ident.Value == "*" {
		return true
	}

	return false
}

// ProjectedColumnsResult wraps a result to project specific columns
type ProjectedColumnsResult struct {
	baseResult       storage.Result
	columns          []parser.Expression
	evaluator        *Evaluator
	projectedColumns []string
	currentRow       []storage.ColumnValue
}

// applyColumnProjection applies a column projection to a result
func applyColumnProjection(result storage.Result, columns []parser.Expression) (storage.Result, error) {
	evaluator := NewEvaluator(context.Background(), nil)

	// Build the list of projected column names
	projectedColumns := make([]string, len(columns))

	for i, col := range columns {
		switch expr := col.(type) {
		case *parser.Identifier:
			projectedColumns[i] = expr.Value
		case *parser.QualifiedIdentifier:
			// Use just the column name without the table qualifier for the result
			projectedColumns[i] = expr.Name.Value
		case *parser.AliasedExpression:
			projectedColumns[i] = expr.Alias.Value
		default:
			// For other expressions, generate a column name
			projectedColumns[i] = fmt.Sprintf("column%d", i+1)
		}
	}

	return &ProjectedColumnsResult{
		baseResult:       result,
		columns:          columns,
		evaluator:        evaluator,
		projectedColumns: projectedColumns,
		currentRow:       make([]storage.ColumnValue, len(columns)),
	}, nil
}

// Next advances to the next row
func (r *ProjectedColumnsResult) Next() bool {
	if !r.baseResult.Next() {
		return false
	}

	// Get the base row and columns
	baseRow := r.baseResult.Row()
	baseColumns := r.baseResult.Columns()

	// Create a row map for the evaluator
	rowMap := make(map[string]storage.ColumnValue)
	for i, col := range baseColumns {
		if i < len(baseRow) {
			rowMap[col] = baseRow[i]
		}
	}

	// Evaluate each projected column
	for i, col := range r.columns {
		var value storage.ColumnValue

		switch expr := col.(type) {
		case *parser.Identifier:
			// Simple column reference - check both with and without table prefix
			if val, ok := rowMap[expr.Value]; ok {
				value = val
			} else {
				// Try to find the column by checking all columns
				found := false
				for colName, colVal := range rowMap {
					// Check if column name matches after removing table prefix
					parts := strings.Split(colName, ".")
					if len(parts) > 1 && parts[len(parts)-1] == expr.Value {
						value = colVal
						found = true
						break
					}
				}
				if !found {
					// Use a NULL TEXT value since we're projecting column names
					value = storage.NewNullValue(storage.TEXT)
				}
			}

		case *parser.QualifiedIdentifier:
			// Table-qualified column reference
			fullName := fmt.Sprintf("%s.%s", expr.Qualifier.Value, expr.Name.Value)
			if val, ok := rowMap[fullName]; ok {
				value = val
			} else {
				value = storage.NewNullValue(storage.TEXT)
			}

		case *parser.AliasedExpression:
			// Handle aliased expressions recursively
			// Set the row map on the evaluator before evaluating
			r.evaluator.WithRow(rowMap)
			baseValue, err := r.evaluator.Evaluate(expr.Expression)
			if err != nil {
				value = storage.NewNullValue(storage.TEXT)
			} else {
				// Check if the evaluated value is a NULL ColumnValue
				if baseValue != nil && baseValue.IsNull() {
					value = baseValue // Use the NULL value directly
				} else {
					value = storage.ValueToColumnValue(baseValue, storage.TEXT)
				}
			}

		default:
			// For other expressions, use the evaluator
			// Set the row map on the evaluator before evaluating
			r.evaluator.WithRow(rowMap)
			evaluatedValue, err := r.evaluator.Evaluate(expr)
			if err != nil {
				value = storage.NewNullValue(storage.TEXT)
			} else {
				value = evaluatedValue
			}
		}

		r.currentRow[i] = value
	}

	return true
}

// Row returns the current row
func (r *ProjectedColumnsResult) Row() storage.Row {
	return r.currentRow
}

// Columns returns the column names
func (r *ProjectedColumnsResult) Columns() []string {
	return r.projectedColumns
}

// Close closes the result
func (r *ProjectedColumnsResult) Close() error {
	return r.baseResult.Close()
}

// Context returns the context
func (r *ProjectedColumnsResult) Context() context.Context {
	return r.baseResult.Context()
}

// Scan scans the current row into dest
func (r *ProjectedColumnsResult) Scan(dest ...interface{}) error {
	if len(dest) != len(r.currentRow) {
		return fmt.Errorf("scan: expected %d destination arguments, got %d", len(r.currentRow), len(dest))
	}

	for i, val := range r.currentRow {
		if err := storage.ScanColumnValueToDestination(val, dest[i]); err != nil {
			return err
		}
	}

	return nil
}

// RowsAffected returns 0 for SELECT queries
func (r *ProjectedColumnsResult) RowsAffected() int64 {
	return r.baseResult.RowsAffected()
}

// LastInsertID returns 0 for SELECT queries
func (r *ProjectedColumnsResult) LastInsertID() int64 {
	return r.baseResult.LastInsertID()
}

// WithAliases applies aliases to the result
func (r *ProjectedColumnsResult) WithAliases(aliases map[string]string) storage.Result {
	r.baseResult = r.baseResult.WithAliases(aliases)
	return r
}
