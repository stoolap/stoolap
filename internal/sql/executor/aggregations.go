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
	"fmt"
	"strings"
	"sync"

	"github.com/stoolap/stoolap/internal/functions/aggregate"
	"github.com/stoolap/stoolap/internal/functions/contract"
	"github.com/stoolap/stoolap/internal/parser"
	"github.com/stoolap/stoolap/internal/storage"
)

// Object pool for distinct value maps used in aggregations
var distinctMapPool = &sync.Pool{
	New: func() interface{} {
		return make(map[interface{}]bool, 64) // Pre-allocate with reasonable capacity
	},
}

// executeSelectWithAggregation executes a SELECT statement with aggregation (GROUP BY)
// executeSelectWithJoinsAndAggregation handles SELECT with JOINs and GROUP BY
func (e *Executor) executeSelectWithJoinsAndAggregation(ctx context.Context, tx storage.Transaction, stmt *parser.SelectStatement) (storage.Result, error) {
	// First execute the JOIN to get the joined result
	joinResult, err := e.executeSelectWithJoins(ctx, tx, &parser.SelectStatement{
		TableExpr: stmt.TableExpr,
		Where:     stmt.Where,
		Columns:   []parser.Expression{&parser.Identifier{Value: "*"}}, // Select all columns for aggregation
	})
	if err != nil {
		return nil, fmt.Errorf("error executing join: %w", err)
	}
	defer joinResult.Close()

	// Now perform aggregation on the joined result
	return e.executeAggregationOnResult(ctx, joinResult, stmt)
}

// executeAggregationOnResult performs aggregation on an existing result set
func (e *Executor) executeAggregationOnResult(ctx context.Context, result storage.Result, stmt *parser.SelectStatement) (storage.Result, error) {
	// First, we need to materialize the result if it's not already materialized
	// This is necessary because we'll need to iterate through it multiple times
	materializedRows := make([]map[string]storage.ColumnValue, 0)
	columns := result.Columns()

	// Build a column index map for faster lookups
	colIndexMap := make(map[string]int)
	// Also build a map for unqualified column names to handle JOIN results
	unqualifiedColMap := make(map[string]string)
	for i, col := range columns {
		colIndexMap[col] = i
		// Also add lowercase version for case-insensitive lookups
		colIndexMap[strings.ToLower(col)] = i

		// If this is a qualified column name (e.g., "c.id"), also map the unqualified name
		if dotIndex := strings.LastIndex(col, "."); dotIndex > 0 {
			unqualifiedName := col[dotIndex+1:]
			// Only add if there's no conflict (i.e., the unqualified name is unique)
			if _, exists := unqualifiedColMap[unqualifiedName]; !exists {
				unqualifiedColMap[unqualifiedName] = col
			}
		}
	}

	// Materialize all rows from the result
	for result.Next() {
		row := result.Row()
		if row == nil {
			continue
		}

		// Convert the row to a map for easier access
		rowMap := make(map[string]storage.ColumnValue, len(columns))
		for i, col := range columns {
			if i < len(row) {
				rowMap[col] = row[i]
			}
		}
		materializedRows = append(materializedRows, rowMap)
	}

	// Close the original result
	result.Close()

	// Now we have all rows materialized, perform aggregation
	// Identify aggregate and grouping columns
	aggregations := make([]*SqlFunction, 0)
	groupByColumns := make([]string, 0)

	// Create an alias map to track column aliases
	aliases := make(map[string]string)

	// First, look for aliases in the SELECT list
	selectAliases := make(map[string]string)
	for _, colExpr := range stmt.Columns {
		if aliasedExpr, ok := colExpr.(*parser.AliasedExpression); ok {
			// Add the alias mapping
			exprStr := aliasedExpr.Expression.String()
			selectAliases[aliasedExpr.Alias.Value] = exprStr
		}
	}

	// Process the GROUP BY clause
	if stmt.GroupBy != nil {
		for _, groupExpr := range stmt.GroupBy {
			switch expr := groupExpr.(type) {
			case *parser.Identifier:
				groupCol := expr.Value
				groupByColumns = append(groupByColumns, groupCol)

				// Check if this is an alias from the SELECT list
				if exprStr, isAlias := selectAliases[groupCol]; isAlias {
					// Also add the aliased expression to the group columns for evaluation
					if !strings.Contains(exprStr, groupCol) {
						aliases[groupCol] = exprStr
						aliases[exprStr] = groupCol
					}
				}

			case *parser.QualifiedIdentifier:
				// For table.column format, we need to keep the full qualified name
				// to match against the JOIN result columns
				qualifiedName := expr.Qualifier.Value + "." + expr.Name.Value
				groupByColumns = append(groupByColumns, qualifiedName)

			case *parser.InfixExpression:
				// For expressions like id / 10, use the string representation
				exprStr := expr.String()
				groupByColumns = append(groupByColumns, exprStr)

			case *parser.FunctionCall:
				// For function calls like TIME_TRUNC('1h', sale_time)
				exprStr := expr.String()
				groupByColumns = append(groupByColumns, exprStr)

			default:
				return nil, fmt.Errorf("unsupported GROUP BY expression type: %T", expr)
			}
		}
	}

	// Process the select list columns to identify aggregations
	for _, colExpr := range stmt.Columns {
		switch expr := colExpr.(type) {
		case *parser.FunctionCall:
			// Check if the function is an aggregate function
			funcName := expr.Function
			if IsAggregateFunction(funcName) {
				// Create an aggregate function descriptor
				var columnName string
				var isDistinct bool

				// Check if the function call has the DISTINCT flag set
				isDistinct = expr.IsDistinct

				if len(expr.Arguments) == 1 {
					// Extract column name from the argument
					switch argExpr := expr.Arguments[0].(type) {
					case *parser.Identifier:
						columnName = argExpr.Value
					case *parser.QualifiedIdentifier:
						// For qualified identifiers in aggregates, use the full name
						columnName = argExpr.Qualifier.Value + "." + argExpr.Name.Value
					}
				}

				// Check for COUNT(*) special case
				if funcName == "COUNT" && columnName == "*" {
					aggregations = append(aggregations, &SqlFunction{
						Name:       "COUNT",
						Column:     "*",
						IsDistinct: isDistinct,
						OrderBy:    expr.OrderBy,
					})
				} else {
					aggregations = append(aggregations, &SqlFunction{
						Name:       funcName,
						Column:     columnName,
						IsDistinct: isDistinct,
						OrderBy:    expr.OrderBy,
					})
				}
			}

		case *parser.AliasedExpression:
			// Handle aliased expressions
			innerExpr := expr.Expression
			alias := expr.Alias.Value

			if funcCall, ok := innerExpr.(*parser.FunctionCall); ok {
				funcName := funcCall.Function
				if IsAggregateFunction(funcName) {
					// Create an aggregate function descriptor
					var columnName string
					var isDistinct bool

					// Check if the function call has the DISTINCT flag set
					isDistinct = funcCall.IsDistinct

					// Extract column name from arguments
					if len(funcCall.Arguments) == 1 {
						switch argExpr := funcCall.Arguments[0].(type) {
						case *parser.Identifier:
							columnName = argExpr.Value
						case *parser.QualifiedIdentifier:
							columnName = argExpr.Name.Value
						}
					}

					// Check for COUNT(*) special case
					if funcName == "COUNT" && columnName == "*" {
						aggregations = append(aggregations, &SqlFunction{
							Name:       "COUNT",
							Column:     "*",
							Alias:      alias,
							IsDistinct: isDistinct,
							OrderBy:    funcCall.OrderBy,
						})
					} else {
						aggregations = append(aggregations, &SqlFunction{
							Name:       funcName,
							Column:     columnName,
							Alias:      alias,
							IsDistinct: isDistinct,
							OrderBy:    funcCall.OrderBy,
						})
					}

					// Add alias mapping for this function
					if alias != "" {
						funcColName := fmt.Sprintf("%s(%s)", funcName, columnName)
						aliases[alias] = funcColName
					}
				}
			}
		}
	}

	// If we have no GROUP BY but have aggregation functions, it's a global aggregation
	if stmt.GroupBy == nil && len(aggregations) > 0 {
		// Special case for global aggregation
		result := e.executeGlobalAggregationOnMaterialized(ctx, materializedRows, columns, aggregations, unqualifiedColMap)

		// If we have a HAVING clause, apply it to the aggregated result
		if stmt.Having != nil {
			// Create a new evaluator for the HAVING clause with alias support
			evaluator := NewEvaluator(ctx, e.functionRegistry)
			evaluator.WithColumnAliases(aliases)

			// Use HavingFilteredResult for HAVING clauses
			result = &HavingFilteredResult{
				result:     result,
				havingExpr: stmt.Having,
				evaluator:  evaluator,
			}
		}

		// Apply ORDER BY, LIMIT, OFFSET if specified
		if stmt.OrderBy != nil || stmt.Limit != nil || stmt.Offset != nil {
			var err error
			result, err = applyOrderByLimitOffset(ctx, result, stmt)
			if err != nil {
				return nil, err
			}
		}

		return result, nil
	}

	// For GROUP BY aggregation, use the existing AggregateResult type
	// Create a memory result that wraps our materialized rows
	memoryResult := &MaterializedResult{
		columns: columns,
		rows:    materializedRows,
		current: -1,
	}

	// Pass the unqualified column map to AggregateResult for proper column resolution
	// We need to embed it in the aliases map with a special prefix
	enhancedAliases := make(map[string]string)
	for k, v := range aliases {
		enhancedAliases[k] = v
	}
	// Add unqualified mappings with a special prefix
	for unqual, qual := range unqualifiedColMap {
		enhancedAliases["__unqual_"+unqual] = qual
	}

	// Perform the aggregation
	var aggregateResult storage.Result = &AggregateResult{
		baseResult:     memoryResult,
		functions:      aggregations,
		groupByColumns: groupByColumns,
		aliases:        enhancedAliases,
	}

	// If we have a HAVING clause, apply it to the aggregated result
	if stmt.Having != nil {
		// Create a new evaluator for the HAVING clause with alias support
		evaluator := NewEvaluator(ctx, e.functionRegistry)
		evaluator.WithColumnAliases(aliases)

		// Use HavingFilteredResult for HAVING clauses
		aggregateResult = &HavingFilteredResult{
			result:     aggregateResult,
			havingExpr: stmt.Having,
			evaluator:  evaluator,
		}
	}

	// Apply ORDER BY, LIMIT, OFFSET if specified
	if stmt.OrderBy != nil || stmt.Limit != nil || stmt.Offset != nil {
		var err error
		aggregateResult, err = applyOrderByLimitOffset(ctx, aggregateResult, stmt)
		if err != nil {
			return nil, err
		}
	}

	return aggregateResult, nil
}

// executeGlobalAggregationOnMaterialized performs global aggregation on materialized rows
func (e *Executor) executeGlobalAggregationOnMaterialized(ctx context.Context, rows []map[string]storage.ColumnValue, columns []string, aggregations []*SqlFunction, unqualifiedColMap map[string]string) storage.Result {
	// Create result rows - one row for global aggregation
	resultRows := make([][]interface{}, 0, 1)

	// Compute each aggregation function
	resultValues := make([]interface{}, len(aggregations))

	// Initialize aggregation state and create new function instances
	aggFuncInstances := make([]contract.AggregateFunction, len(aggregations))
	for i, agg := range aggregations {
		if agg.Name == "COUNT" {
			resultValues[i] = int64(0)
		} else {
			// Create new instances for each aggregate function to avoid concurrency issues
			var aggFunc contract.AggregateFunction
			switch strings.ToUpper(agg.Name) {
			case "SUM":
				aggFunc = aggregate.NewSumFunction()
			case "AVG":
				aggFunc = aggregate.NewAvgFunction()
			case "MIN":
				aggFunc = aggregate.NewMinFunction()
			case "MAX":
				aggFunc = aggregate.NewMaxFunction()
			case "FIRST":
				aggFunc = aggregate.NewFirstFunction()
			case "LAST":
				aggFunc = aggregate.NewLastFunction()
			default:
				// For unknown functions, try the registry
				if e.functionRegistry != nil {
					aggFunc = e.functionRegistry.GetAggregateFunction(strings.ToUpper(agg.Name))
					if aggFunc != nil {
						aggFunc.Reset()
					}
				}
			}
			aggFuncInstances[i] = aggFunc
		}
	}

	// Process the rows and accumulate aggregation results
	for _, row := range rows {
		// Process each aggregation
		for i, agg := range aggregations {
			if agg.Name == "COUNT" {
				// For COUNT(*), just increment
				if agg.Column == "*" {
					// Always increment for COUNT(*)
					resultValues[i] = resultValues[i].(int64) + 1
				} else if agg.IsDistinct {
					// For COUNT DISTINCT, use a map to track distinct values
					var distinctMap map[interface{}]bool

					if distinctMapVal, ok := resultValues[i].(map[interface{}]bool); ok {
						// Map already exists, use it
						distinctMap = distinctMapVal
					} else {
						// Get a map from the pool
						distinctMap = distinctMapPool.Get().(map[interface{}]bool)
						// Clear the map if it's not empty
						clear(distinctMap)

						resultValues[i] = distinctMap
					}

					// Find the column value
					resolvedCol := resolveColumnName(agg.Column, columns, unqualifiedColMap)
					if val, ok := row[resolvedCol]; ok && val != nil {
						distinctMap[val.AsInterface()] = true
						resultValues[i] = distinctMap
					}
				} else {
					// Regular COUNT - increment for non-NULL values
					resolvedCol := resolveColumnName(agg.Column, columns, unqualifiedColMap)
					if val, ok := row[resolvedCol]; ok && val != nil {
						resultValues[i] = resultValues[i].(int64) + 1
					}
				}
			} else {
				// For other aggregate functions, use the stored instance
				if aggFuncInstances[i] != nil {
					// Find the column value
					resolvedCol := resolveColumnName(agg.Column, columns, unqualifiedColMap)
					if val, ok := row[resolvedCol]; ok && val != nil {
						aggFuncInstances[i].Accumulate(val.AsInterface(), agg.IsDistinct)
					}
				}
			}
		}
	}

	// Finalize aggregation results
	for i, agg := range aggregations {
		if agg.Name == "COUNT" && agg.IsDistinct {
			// Convert COUNT DISTINCT map to count
			if distinctMap, ok := resultValues[i].(map[interface{}]bool); ok {
				// Store the count
				resultValues[i] = int64(len(distinctMap))

				// Clear and return the map to the pool
				clear(distinctMap)
				distinctMapPool.Put(distinctMap)
			}
		} else if agg.Name != "COUNT" {
			// For other aggregate functions, get the final result from stored instance
			if aggFuncInstances[i] != nil {
				resultValues[i] = aggFuncInstances[i].Result()
			}
		}
	}

	// Create result columns
	resultColumns := make([]string, len(aggregations))
	for i, agg := range aggregations {
		if agg.Alias != "" {
			// Use the provided alias
			resultColumns[i] = agg.Alias
		} else {
			// Generate a column name based on the function
			if agg.Column == "*" {
				resultColumns[i] = fmt.Sprintf("%s(*)", agg.Name)
			} else {
				resultColumns[i] = fmt.Sprintf("%s(%s)", agg.Name, agg.Column)
			}
		}
	}

	// Add the single result row
	resultRows = append(resultRows, resultValues)

	// Create and return the result
	return &ExecResult{
		columns:  resultColumns,
		rows:     resultRows,
		isMemory: true,
	}
}

func (e *Executor) executeSelectWithAggregation(ctx context.Context, tx storage.Transaction, stmt *parser.SelectStatement) (storage.Result, error) {
	// Extract table name from the table expression
	var tableName string

	// Handle different types of table expressions
	switch tableExpr := stmt.TableExpr.(type) {
	case *parser.Identifier:
		// Simple table name
		tableName = tableExpr.Value
	case *parser.SimpleTableSource:
		// Table with optional alias
		tableName = tableExpr.Name.Value
	case *parser.JoinTableSource:
		// For JOINs with aggregation, we need to handle them differently
		// This is a complex case that requires fetching data from multiple tables
		return e.executeSelectWithJoinsAndAggregation(ctx, tx, stmt)
	default:
		return nil, fmt.Errorf("unsupported table expression type: %T", stmt.TableExpr)
	}

	// First check if this is a CTE
	if cteResult, isCTE := e.resolveCTETable(ctx, tx, tableName); isCTE {
		// For CTEs with aggregation, we need to handle them specially
		return e.executeAggregationOnResult(ctx, cteResult, stmt)
	}

	// Check if the table exists
	exists, err := e.engine.TableExists(tableName)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, storage.ErrTableNotFound
	}

	// Identify aggregate and grouping columns
	aggregations := make([]*SqlFunction, 0)
	groupByColumns := make([]string, 0)
	projectColumns := make([]string, 0)

	// Create an alias map to track column aliases
	aliases := make(map[string]string)

	// First, look for aliases in the SELECT list
	selectAliases := make(map[string]string)
	for _, colExpr := range stmt.Columns {
		if aliasedExpr, ok := colExpr.(*parser.AliasedExpression); ok {
			// Add the alias mapping
			exprStr := aliasedExpr.Expression.String()
			selectAliases[aliasedExpr.Alias.Value] = exprStr
		}
	}

	// Process the GROUP BY clause
	if stmt.GroupBy != nil {
		for _, groupExpr := range stmt.GroupBy {
			switch expr := groupExpr.(type) {
			case *parser.Identifier:
				groupCol := expr.Value
				groupByColumns = append(groupByColumns, groupCol)
				projectColumns = append(projectColumns, groupCol)

				// Check if this is an alias from the SELECT list
				if exprStr, isAlias := selectAliases[groupCol]; isAlias {
					// Also add the aliased expression to the group columns for evaluation
					if !strings.Contains(exprStr, groupCol) {
						aliases[groupCol] = exprStr
						aliases[exprStr] = groupCol
					}
				}

			case *parser.QualifiedIdentifier:
				// For table.column format, we need to keep the full qualified name
				// to match against the JOIN result columns
				qualifiedName := expr.Qualifier.Value + "." + expr.Name.Value
				groupByColumns = append(groupByColumns, qualifiedName)
				projectColumns = append(projectColumns, qualifiedName)

			case *parser.InfixExpression:
				// For expressions like id / 10, use the string representation
				exprStr := expr.String()
				groupByColumns = append(groupByColumns, exprStr)
				projectColumns = append(projectColumns, exprStr)

			case *parser.FunctionCall:
				// For function calls like TIME_TRUNC('1h', sale_time)
				exprStr := expr.String()

				// Add function expression to the group by columns
				groupByColumns = append(groupByColumns, exprStr)

				// For function calls, we need to add the function arguments that are column references
				// to the projection list so they're available for evaluation
				for _, arg := range expr.Arguments {
					if ident, ok := arg.(*parser.Identifier); ok && ident.Value != "*" {
						// Add column reference to projection list if not already there
						found := false
						for _, col := range projectColumns {
							if col == ident.Value {
								found = true
								break
							}
						}
						if !found {
							projectColumns = append(projectColumns, ident.Value)
						}
					}
				}

			default:
				return nil, fmt.Errorf("unsupported GROUP BY expression type: %T", expr)
			}
		}
	}

	// Process the select list columns
	for _, colExpr := range stmt.Columns {
		switch expr := colExpr.(type) {
		case *parser.Identifier:
			// If it's a wildcard (*), we need to get all columns from schema
			if expr.Value == "*" {
				schema, err := e.engine.GetTableSchema(tableName)
				if err != nil {
					return nil, err
				}
				for _, col := range schema.Columns {
					// For wildcard with GROUP BY, columns must appear in GROUP BY
					if stmt.GroupBy != nil {
						found := false
						for _, groupCol := range groupByColumns {
							if groupCol == col.Name {
								found = true
								break
							}
						}
						if !found {
							continue // Skip columns not in GROUP BY
						}
					}
					projectColumns = append(projectColumns, col.Name)
				}
			} else {
				// Non-wildcard column
				if stmt.GroupBy == nil {
					// Without GROUP BY, non-aggregated columns are not allowed
					// However, we'll add them to the list and handle it during execution
					projectColumns = append(projectColumns, expr.Value)
				} else {
					// With GROUP BY, non-aggregated columns must appear in GROUP BY
					found := false
					for _, groupCol := range groupByColumns {
						if groupCol == expr.Value {
							found = true
							break
						}
					}
					if found {
						projectColumns = append(projectColumns, expr.Value)
					} else {
						return nil, fmt.Errorf("non-aggregated column must appear in GROUP BY: %s", expr.Value)
					}
				}
			}

		case *parser.QualifiedIdentifier:
			// Handle table.column format
			colName := expr.Name.Value
			if stmt.GroupBy == nil {
				// Without GROUP BY, non-aggregated columns are not allowed
				// However, we'll add them to the list and handle it during execution
				projectColumns = append(projectColumns, colName)
			} else {
				// With GROUP BY, non-aggregated columns must appear in GROUP BY
				found := false
				for _, groupCol := range groupByColumns {
					if groupCol == colName {
						found = true
						break
					}
				}
				if found {
					projectColumns = append(projectColumns, colName)
				} else {
					return nil, fmt.Errorf("non-aggregated column must appear in GROUP BY: %s", colName)
				}
			}

		case *parser.FunctionCall:
			// Check if the function is an aggregate function
			funcName := expr.Function
			if IsAggregateFunction(funcName) {
				// Create an aggregate function descriptor
				var columnName string
				var isDistinct bool

				// Check if the function call has the DISTINCT flag set
				isDistinct = expr.IsDistinct

				if len(expr.Arguments) == 1 {
					// Extract column name from the argument
					switch argExpr := expr.Arguments[0].(type) {
					case *parser.Identifier:
						columnName = argExpr.Value
					case *parser.QualifiedIdentifier:
						// For qualified identifiers in aggregates, use the full name
						columnName = argExpr.Qualifier.Value + "." + argExpr.Name.Value
					case *parser.DistinctExpression:
						// Handle older parsers that might use DistinctExpression
						isDistinct = true
						switch e := argExpr.Expr.(type) {
						case *parser.Identifier:
							columnName = e.Value
						case *parser.QualifiedIdentifier:
							columnName = e.Name.Value
						}
					}
				}

				// Check for COUNT(*) special case
				if funcName == "COUNT" && columnName == "*" {
					// Special handling for COUNT(*)
					// Special handling for COUNT(*)
					aggregations = append(aggregations, &SqlFunction{
						Name:       "COUNT",
						Column:     "*",
						IsDistinct: isDistinct,
						OrderBy:    expr.OrderBy,
					})
				} else {
					// Other aggregate functions
					sqlFunc := &SqlFunction{
						Name:       funcName,
						Column:     columnName,
						IsDistinct: isDistinct,
						OrderBy:    expr.OrderBy,
					}

					// Extract ORDER BY columns for projections if present
					if len(expr.OrderBy) > 0 {
						// For ordered functions like FIRST and LAST, add order columns to projections
						for _, orderExpr := range expr.OrderBy {
							if ident, ok := orderExpr.Expression.(*parser.Identifier); ok {
								orderColName := ident.Value
								// Add order column to projections if not already there
								found := false
								for _, col := range projectColumns {
									if strings.EqualFold(col, orderColName) {
										found = true
										break
									}
								}
								if !found {
									projectColumns = append(projectColumns, orderColName)
								}
							}
						}
					}

					aggregations = append(aggregations, sqlFunc)

					// Add the column to the projection list if it's not already there
					// This ensures the column will be available for aggregation
					if columnName != "" && columnName != "*" {
						found := false
						for _, col := range projectColumns {
							if strings.EqualFold(col, columnName) {
								found = true
								break
							}
						}
						if !found {
							projectColumns = append(projectColumns, columnName)
						}
					}
				}
			} else {
				// Handle non-aggregate functions in SELECT list
				// For functions like TIME_TRUNC, they need to be in GROUP BY
				if stmt.GroupBy != nil {
					exprStr := expr.String()

					// Check if this function appears in GROUP BY
					found := false
					for _, groupCol := range groupByColumns {
						if groupCol == exprStr {
							found = true
							break
						}
					}

					if found {
						// Function is in GROUP BY, so it's allowed
						// Add column arguments to projection list
						for _, arg := range expr.Arguments {
							if ident, ok := arg.(*parser.Identifier); ok && ident.Value != "*" {
								// Add column reference if not already there
								colFound := false
								for _, col := range projectColumns {
									if col == ident.Value {
										colFound = true
										break
									}
								}
								if !colFound {
									projectColumns = append(projectColumns, ident.Value)
								}
							}
						}
					} else {
						// Function is not in GROUP BY, error
						return nil, fmt.Errorf("non-aggregate function %s must appear in GROUP BY", funcName)
					}
				} else {
					// Without GROUP BY, non-aggregate functions generally aren't allowed
					return nil, fmt.Errorf("non-aggregate function in SELECT with aggregation: %s", funcName)
				}
			}

		case *parser.AliasedExpression:
			// Handle aliased expressions
			innerExpr := expr.Expression
			alias := expr.Alias.Value

			switch innerExprTyped := innerExpr.(type) {
			case *parser.FunctionCall:
				// Check if the function is an aggregate function
				funcName := innerExprTyped.Function
				if IsAggregateFunction(funcName) {
					// Create an aggregate function descriptor
					var columnName string
					var isDistinct bool

					// Check if the function call has the DISTINCT flag set
					isDistinct = innerExprTyped.IsDistinct

					// Extract column name from arguments
					if len(innerExprTyped.Arguments) == 1 {
						switch argExpr := innerExprTyped.Arguments[0].(type) {
						case *parser.Identifier:
							columnName = argExpr.Value
						case *parser.QualifiedIdentifier:
							columnName = argExpr.Name.Value
						case *parser.DistinctExpression:
							// Handle older parsers that might use DistinctExpression
							isDistinct = true
							switch e := argExpr.Expr.(type) {
							case *parser.Identifier:
								columnName = e.Value
							case *parser.QualifiedIdentifier:
								columnName = e.Name.Value
							}
						}
					}

					// Check for COUNT(*) special case
					if funcName == "COUNT" && columnName == "*" {
						// Special handling for COUNT(*)
						aggregations = append(aggregations, &SqlFunction{
							Name:       "COUNT",
							Column:     "*",
							Alias:      alias,
							IsDistinct: isDistinct,
							OrderBy:    innerExprTyped.OrderBy,
						})
					} else {
						// Other aggregate functions
						sqlFunc := &SqlFunction{
							Name:       funcName,
							Column:     columnName,
							Alias:      alias,
							IsDistinct: isDistinct,
							OrderBy:    innerExprTyped.OrderBy,
						}

						// Extract ORDER BY columns for projections if present
						if len(innerExprTyped.OrderBy) > 0 {
							// For ordered functions like FIRST and LAST, add order columns to projections
							for _, orderExpr := range innerExprTyped.OrderBy {
								if ident, ok := orderExpr.Expression.(*parser.Identifier); ok {
									orderColName := ident.Value
									// Add order column to projections if not already there
									found := false
									for _, col := range projectColumns {
										if strings.EqualFold(col, orderColName) {
											found = true
											break
										}
									}
									if !found {
										projectColumns = append(projectColumns, orderColName)
									}
								}
							}
						}

						aggregations = append(aggregations, sqlFunc)

						// Add the column to the projection list if it's not already there
						// This ensures the column will be available for aggregation
						if columnName != "" && columnName != "*" {
							found := false
							for _, col := range projectColumns {
								if strings.EqualFold(col, columnName) {
									found = true
									break
								}
							}
							if !found {
								projectColumns = append(projectColumns, columnName)
							}
						}
					}

					// Add alias mapping for this function
					// This is important for HAVING clauses that refer to aggregated values by alias
					// For example: SELECT COUNT(*) AS cnt FROM table HAVING cnt > 5
					if alias != "" {
						funcColName := fmt.Sprintf("%s(%s)", funcName, columnName)
						aliases[alias] = funcColName
					}
				} else {
					// Handle non-aggregate function with alias
					if stmt.GroupBy != nil {
						// With GROUP BY, check if the function is in GROUP BY list
						exprStr := innerExprTyped.String()

						found := false
						for _, groupCol := range groupByColumns {
							if groupCol == exprStr {
								found = true
								break
							}
						}

						// Also check if the alias is in GROUP BY
						for _, groupCol := range groupByColumns {
							if groupCol == alias {
								found = true
								break
							}
						}

						if found {
							// Function is in GROUP BY or its alias is, so it's allowed
							// Add column arguments to projection list
							for _, arg := range innerExprTyped.Arguments {
								if ident, ok := arg.(*parser.Identifier); ok && ident.Value != "*" {
									// Add column reference if not already there
									colFound := false
									for _, col := range projectColumns {
										if col == ident.Value {
											colFound = true
											break
										}
									}
									if !colFound {
										projectColumns = append(projectColumns, ident.Value)
									}
								}
							}

							// Add alias mapping
							aliases[alias] = exprStr
							aliases[exprStr] = alias
						} else {
							// Not in GROUP BY, error
							return nil, fmt.Errorf("non-aggregate function %s must appear in GROUP BY", funcName)
						}
					} else {
						// Without GROUP BY, non-aggregate functions aren't allowed
						return nil, fmt.Errorf("non-aggregate function in SELECT with aggregation: %s", funcName)
					}
				}

			case *parser.Identifier:
				// Aliased column
				colName := innerExprTyped.Value
				if stmt.GroupBy == nil {
					// Without GROUP BY, non-aggregated columns are not allowed
					return nil, fmt.Errorf("non-aggregated column must appear in GROUP BY: %s", colName)
				} else {
					// With GROUP BY, non-aggregated columns must appear in GROUP BY
					found := false
					for _, groupCol := range groupByColumns {
						if groupCol == colName {
							found = true
							break
						}
					}
					if found {
						projectColumns = append(projectColumns, colName)
						// Add the alias mapping
						aliases[alias] = colName
						aliases[colName] = alias
					} else {
						return nil, fmt.Errorf("non-aggregated column must appear in GROUP BY: %s", colName)
					}
				}

			case *parser.InfixExpression:
				// Handle infix expressions (like id / 10 for decade)
				colName := innerExprTyped.String()

				// With GROUP BY, this expression must appear in GROUP BY
				found := false

				// First check: exact match in GROUP BY
				for _, groupExpr := range stmt.GroupBy {
					groupExprStr := groupExpr.String()
					if groupExprStr == innerExprTyped.String() {
						found = true
						break
					}
				}

				// Second check: if the alias itself is used in GROUP BY
				for _, groupCol := range groupByColumns {
					if groupCol == alias {
						found = true
						break
					}
				}

				// When a GROUP BY references an alias (column_expr AS alias),
				// we need to map the alias back to the original expression
				// This is a standard SQL feature - GROUP BY alias is valid SQL
				// We set 'found' to true because GROUP BY decade is valid when decade is an alias

				if found {
					// Add the infix expression to the project columns
					// For now, we'll use the full string representation as the column name
					projectColumns = append(projectColumns, colName)

					// Add mappings to support aliased columns in expressions
					// 1. Map the alias to the expression for the evaluator
					aliases[alias] = colName

					// 2. Map the expression to the alias for result handling
					// This is useful when the expression appears in GROUP BY or other contexts
					aliases[colName] = alias
				} else {
					return nil, fmt.Errorf("non-aggregated expression must appear in GROUP BY: %s", colName)
				}

			default:
				return nil, fmt.Errorf("unsupported expression in aliased SELECT item: %T", innerExpr)
			}

		default:
			return nil, fmt.Errorf("unsupported expression in SELECT with aggregation: %T", colExpr)
		}
	}

	// Apply WHERE clause if present
	var whereExpr parser.Expression
	if stmt.Where != nil {
		// Process any subqueries in the WHERE clause first
		processedWhere, err := e.processWhereSubqueries(ctx, tx, stmt.Where)
		if err != nil {
			return nil, fmt.Errorf("error processing subqueries in WHERE clause: %w", err)
		}
		whereExpr = processedWhere
	}

	// If we have no GROUP BY but have aggregation functions, it's a global aggregation
	if stmt.GroupBy == nil && len(aggregations) > 0 {
		// Special case for COUNT(*) and other global aggregates
		result, err := e.executeGlobalAggregation(ctx, tx, tableName, aggregations, whereExpr)
		if err != nil {
			return nil, err
		}

		// If we have a HAVING clause, apply it to the aggregated result
		if stmt.Having != nil {
			// Create a new evaluator for the HAVING clause with alias support
			evaluator := NewEvaluator(ctx, e.functionRegistry)

			// Create a complete alias map that includes:
			// 1. All the original aliases
			// 2. Special handling for aggregate functions
			// 3. Special handling for expressions like "id / 10"
			havingAliases := make(map[string]string)

			// Copy all existing aliases
			for k, v := range aliases {
				havingAliases[k] = v
			}

			// Add aliases for aggregation functions
			for _, fn := range aggregations {
				if fn.Alias != "" {
					// For functions with explicit aliases
					colName := fmt.Sprintf("%s(%s)", fn.Name, fn.Column)
					havingAliases[fn.Alias] = colName
				}
			}

			// Apply the complete alias map to the evaluator
			evaluator.WithColumnAliases(havingAliases)

			// Before wrapping, ensure the AggregateResult aliases are set
			if aggResult, ok := result.(*AggregateResult); ok {
				aggResult.aliases = havingAliases
			}

			// Use HavingFilteredResult for HAVING clauses
			result = &HavingFilteredResult{
				result:     result,
				havingExpr: stmt.Having,
				evaluator:  evaluator,
			}
		}

		// Apply ORDER BY, LIMIT, OFFSET if specified
		if stmt.OrderBy != nil || stmt.Limit != nil || stmt.Offset != nil {
			var err error
			result, err = applyOrderByLimitOffset(ctx, result, stmt)
			if err != nil {
				return nil, err
			}
		}

		return result, nil
	}

	// Get the base result - all rows that match the WHERE condition
	var baseResult storage.Result

	// First, try to convert WHERE to a storage expression
	if whereExpr != nil {
		whereColumns := getColumnsFromWhereClause(whereExpr)

		// Make sure we're fetching all columns needed by the WHERE clause
		for _, whereCol := range whereColumns {
			found := false
			for _, col := range projectColumns {
				if strings.EqualFold(col, whereCol) {
					found = true
					break
				}
			}
			if !found {
				projectColumns = append(projectColumns, whereCol)
			}
		}
	}

	// Execute the base query
	baseResult, err = tx.Select(tableName, projectColumns, nil)
	if err != nil {
		return nil, err
	}

	// If we have a WHERE clause, apply filtering
	if whereExpr != nil {
		// Create an evaluator with support for column aliases
		evaluator := NewEvaluator(ctx, e.functionRegistry)
		if len(aliases) > 0 {
			evaluator.WithColumnAliases(aliases)
		}

		baseResult = &FilteredResult{
			result:    baseResult,
			whereExpr: whereExpr,
			evaluator: evaluator,
		}
	}

	// Perform the aggregation
	var result storage.Result = &AggregateResult{
		baseResult:     baseResult,
		functions:      aggregations,
		groupByColumns: groupByColumns,
		aliases:        aliases,
	}

	// If we have a HAVING clause, apply it to the aggregated result
	if stmt.Having != nil {
		// Create a new evaluator for the HAVING clause with alias support
		evaluator := NewEvaluator(ctx, e.functionRegistry)
		evaluator.WithColumnAliases(aliases)

		// Use HavingFilteredResult for HAVING clauses
		result = &HavingFilteredResult{
			result:     result,
			havingExpr: stmt.Having,
			evaluator:  evaluator,
		}
	}

	// Apply ORDER BY, LIMIT, OFFSET if specified
	if stmt.OrderBy != nil || stmt.Limit != nil || stmt.Offset != nil {
		var err error
		result, err = applyOrderByLimitOffset(ctx, result, stmt)
		if err != nil {
			return nil, err
		}
	}

	return result, nil
}

// executeGlobalAggregation handles aggregation without GROUP BY (global aggregation)
func (e *Executor) executeGlobalAggregation(ctx context.Context, tx storage.Transaction, tableName string,
	aggregations []*SqlFunction, whereExpr parser.Expression) (storage.Result, error) {

	// Check which columns we need to retrieve
	neededColumns := make([]string, 0)

	// For COUNT(*), we don't need any specific columns
	needsAllRows := false
	for _, agg := range aggregations {
		if agg.Name == "COUNT" && agg.Column == "*" {
			needsAllRows = true
		} else if agg.Column != "" && agg.Column != "*" {
			neededColumns = append(neededColumns, agg.Column)
		}
	}

	// If we don't need specific columns, just use the first column (or the key column)
	if len(neededColumns) == 0 && needsAllRows {
		// Get the schema to find a suitable column
		schema, err := e.engine.GetTableSchema(tableName)
		if err != nil {
			return nil, err
		}

		// Use the first column
		if len(schema.Columns) > 0 {
			neededColumns = append(neededColumns, schema.Columns[0].Name)
		} else {
			return nil, fmt.Errorf("table %s has no columns", tableName)
		}
	}

	// Get the base result
	baseResult, err := tx.Select(tableName, neededColumns, nil)
	if err != nil {
		return nil, err
	}

	// If we have a WHERE clause, we need to ensure we have all columns referenced in WHERE
	if whereExpr != nil {
		// Extract columns referenced in WHERE clause
		whereColumns := extractColumnsFromExpression(whereExpr)

		// Add WHERE columns to neededColumns if not already present
		for _, col := range whereColumns {
			found := false
			for _, nc := range neededColumns {
				if nc == col {
					found = true
					break
				}
			}
			if !found {
				neededColumns = append(neededColumns, col)
			}
		}

		// Re-fetch with all needed columns
		baseResult.Close()
		baseResult, err = tx.Select(tableName, neededColumns, nil)
		if err != nil {
			return nil, err
		}

		baseResult = &FilteredResult{
			result:    baseResult,
			whereExpr: whereExpr,
			evaluator: NewEvaluator(ctx, e.functionRegistry),
		}
	}

	// Perform the aggregation
	resultRows := make([][]interface{}, 0, 1) // One row for global aggregation

	// Compute each aggregation function
	resultValues := make([]interface{}, len(aggregations))

	// Initialize aggregation state and create new function instances
	aggFuncInstances := make([]contract.AggregateFunction, len(aggregations))
	for i, agg := range aggregations {
		if agg.Name == "COUNT" {
			resultValues[i] = int64(0)
		} else {
			// Create new instances for each aggregate function to avoid concurrency issues
			var aggFunc contract.AggregateFunction
			switch strings.ToUpper(agg.Name) {
			case "SUM":
				aggFunc = aggregate.NewSumFunction()
			case "AVG":
				aggFunc = aggregate.NewAvgFunction()
			case "MIN":
				aggFunc = aggregate.NewMinFunction()
			case "MAX":
				aggFunc = aggregate.NewMaxFunction()
			case "FIRST":
				aggFunc = aggregate.NewFirstFunction()
			case "LAST":
				aggFunc = aggregate.NewLastFunction()
			default:
				// For unknown functions, try the registry (though this is not ideal for concurrency)
				if e.functionRegistry != nil {
					aggFunc = e.functionRegistry.GetAggregateFunction(strings.ToUpper(agg.Name))
					if aggFunc != nil {
						aggFunc.Reset()
					}
				}
			}
			aggFuncInstances[i] = aggFunc
		}
	}

	// Process the rows and accumulate aggregation results
	for baseResult.Next() {
		// Use Row() instead of Scan() to avoid allocations
		row := baseResult.Row()
		if row == nil {
			continue
		}

		// Process each aggregation
		for i, agg := range aggregations {
			if agg.Name == "COUNT" {
				// For COUNT(*), just increment
				if agg.Column == "*" {
					// Always increment for COUNT(*)
					resultValues[i] = resultValues[i].(int64) + 1
				} else if agg.IsDistinct {
					// For COUNT DISTINCT, use a map to track distinct values
					// Get or initialize the map
					var distinctMap map[interface{}]bool

					if distinctMapVal, ok := resultValues[i].(map[interface{}]bool); ok {
						// Map already exists, use it
						distinctMap = distinctMapVal
					} else {
						// Get a map from the pool
						distinctMap = distinctMapPool.Get().(map[interface{}]bool)
						// Clear the map if it's not empty
						clear(distinctMap)

						resultValues[i] = distinctMap
					}

					// Find the column value
					colIndex := -1
					for j, colName := range neededColumns {
						if colName == agg.Column {
							colIndex = j
							break
						}
					}

					// If found, add to distinct set
					if colIndex >= 0 && colIndex < len(row) && row[colIndex] != nil {
						distinctMap[row[colIndex].AsInterface()] = true
						resultValues[i] = distinctMap
					}
				} else {
					// Regular COUNT - increment for non-NULL values
					colIndex := -1
					for j, colName := range neededColumns {
						if colName == agg.Column {
							colIndex = j
							break
						}
					}

					if colIndex >= 0 && colIndex < len(row) && row[colIndex] != nil {
						resultValues[i] = resultValues[i].(int64) + 1
					}
				}
			} else {
				// For other aggregate functions, use the stored instance
				if aggFuncInstances[i] != nil {
					// Find the column index based on name
					colIndex := -1
					for j, colName := range neededColumns {
						if colName == agg.Column {
							colIndex = j
							break
						}
					}

					// If column found, accumulate the value
					if colIndex >= 0 && colIndex < len(row) {
						aggFuncInstances[i].Accumulate(row[colIndex].AsInterface(), agg.IsDistinct)
					}
				}
			}
		}
	}

	// Finalize aggregation results
	for i, agg := range aggregations {
		if agg.Name == "COUNT" && agg.IsDistinct {
			// Convert COUNT DISTINCT map to count
			if distinctMap, ok := resultValues[i].(map[interface{}]bool); ok {
				// Store the count
				resultValues[i] = int64(len(distinctMap))

				// Clear and return the map to the pool
				clear(distinctMap)

				distinctMapPool.Put(distinctMap)
			}
		} else if agg.Name != "COUNT" {
			// For other aggregate functions, get the final result from stored instance
			if aggFuncInstances[i] != nil {
				resultValues[i] = aggFuncInstances[i].Result()
			}
		}
	}

	// Create result columns
	resultColumns := make([]string, len(aggregations))
	for i, agg := range aggregations {
		if agg.Alias != "" {
			// Use the provided alias
			resultColumns[i] = agg.Alias
		} else {
			// Generate a column name based on the function
			if agg.Column == "*" {
				resultColumns[i] = fmt.Sprintf("%s(*)", agg.Name)
			} else {
				resultColumns[i] = fmt.Sprintf("%s(%s)", agg.Name, agg.Column)
			}
		}
	}

	// Add the single result row
	resultRows = append(resultRows, resultValues)

	baseResult.Close() // Close the base result

	// Create and return the result
	return &ExecResult{
		columns:  resultColumns,
		rows:     resultRows,
		isMemory: true,
	}, nil
}

// extractColumnsFromExpression extracts column names referenced in an expression
func extractColumnsFromExpression(expr parser.Expression) []string {
	columns := make([]string, 0)

	switch e := expr.(type) {
	case *parser.Identifier:
		columns = append(columns, e.Value)
	case *parser.InfixExpression:
		columns = append(columns, extractColumnsFromExpression(e.Left)...)
		columns = append(columns, extractColumnsFromExpression(e.Right)...)
	case *parser.PrefixExpression:
		columns = append(columns, extractColumnsFromExpression(e.Right)...)
	case *parser.BetweenExpression:
		columns = append(columns, extractColumnsFromExpression(e.Expr)...)
		columns = append(columns, extractColumnsFromExpression(e.Lower)...)
		columns = append(columns, extractColumnsFromExpression(e.Upper)...)
	case *parser.InExpression:
		columns = append(columns, extractColumnsFromExpression(e.Left)...)
		// Don't recurse into right side if it's a subquery
		if _, ok := e.Right.(*parser.ScalarSubquery); !ok {
			columns = append(columns, extractColumnsFromExpression(e.Right)...)
		}
	case *parser.FunctionCall:
		for _, arg := range e.Arguments {
			columns = append(columns, extractColumnsFromExpression(arg)...)
		}
		// Skip literals and other non-column expressions
	}

	return columns
}

// SqlFunction defined in result_helpers.go

// resolveColumnName attempts to find the actual column name in the result set
// It handles both qualified (table.column) and unqualified (column) names
func resolveColumnName(colName string, columns []string, unqualifiedColMap map[string]string) string {
	// First, check if the exact column name exists
	for _, col := range columns {
		if col == colName {
			return col
		}
	}

	// If not found, check if it's an unqualified name that maps to a qualified name
	if qualified, exists := unqualifiedColMap[colName]; exists {
		return qualified
	}

	// If still not found, check with case-insensitive matching
	lowerColName := strings.ToLower(colName)
	for _, col := range columns {
		if strings.ToLower(col) == lowerColName {
			return col
		}
	}

	// Check unqualified map with case-insensitive matching
	for unqual, qual := range unqualifiedColMap {
		if strings.ToLower(unqual) == lowerColName {
			return qual
		}
	}

	// Return the original name if not found
	return colName
}
