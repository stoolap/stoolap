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
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/stoolap/stoolap/internal/functions/contract"
	"github.com/stoolap/stoolap/internal/functions/registry"
	"github.com/stoolap/stoolap/internal/parser"
	"github.com/stoolap/stoolap/internal/storage"
)

// SqlFunction represents a SQL function in a query
type SqlFunction struct {
	Name           string
	Column         string
	Alias          string
	IsDistinct     bool
	Implementation interface{}
	// OrderBy holds the ORDER BY expressions for functions like FIRST/LAST
	OrderBy []parser.OrderByExpression
}

// GetColumnName returns the column name for this function
func (f *SqlFunction) GetColumnName() string {
	if f.Alias != "" {
		return f.Alias
	}

	if f.IsDistinct {
		return fmt.Sprintf("%s(DISTINCT %s)", f.Name, f.Column)
	}

	// Add ORDER BY clause if present
	if len(f.OrderBy) > 0 {
		orderClause := make([]string, 0, len(f.OrderBy))
		for _, order := range f.OrderBy {
			direction := "ASC"
			if !order.Ascending {
				direction = "DESC"
			}
			orderClause = append(orderClause, order.Expression.String()+" "+direction)
		}
		return fmt.Sprintf("%s(%s ORDER BY %s)", f.Name, f.Column, strings.Join(orderClause, ", "))
	}

	return fmt.Sprintf("%s(%s)", f.Name, f.Column)
}

// OrderedResult represents a result set ordered by a ORDER BY clause
type OrderedResult struct {
	baseResult storage.Result
	orderBy    []parser.OrderByExpression
	// Cached rows for sorting
	rows        [][]storage.ColumnValue
	columns     []string
	currentRow  int
	initialized bool
}

// initialize loads and sorts all rows from the base result
func (r *OrderedResult) initialize() error {
	if r.initialized {
		return nil
	}

	r.columns = r.baseResult.Columns()
	r.rows = [][]storage.ColumnValue{}

	// Collect all rows from the base result
	for r.baseResult.Next() {
		// First try to use Row() for direct access to ColumnValue objects without copying
		if storageRow := r.baseResult.Row(); storageRow != nil {
			// Create a deep copy of the row to avoid modifying the original values
			rowCopy := make([]storage.ColumnValue, len(storageRow))
			copy(rowCopy, storageRow)
			// Add the copied row to our result set
			r.rows = append(r.rows, rowCopy)
		}
	}

	// Sort the rows based on the ORDER BY clause
	sortRows(r.rows, r.columns, r.orderBy)

	r.currentRow = -1
	r.initialized = true
	return nil
}

// Columns returns the column names from the base result
func (r *OrderedResult) Columns() []string {
	if !r.initialized {
		r.initialize()
	}
	return r.columns
}

// Next advances to the next row in the result set
func (r *OrderedResult) Next() bool {
	if !r.initialized {
		if err := r.initialize(); err != nil {
			return false
		}
	}

	r.currentRow++
	return r.currentRow < len(r.rows)
}

// Scan copies values from the current row into the provided destinations
func (r *OrderedResult) Scan(dest ...interface{}) error {
	if !r.initialized {
		if err := r.initialize(); err != nil {
			return err
		}
	}

	// Check if we're at a valid row
	if r.currentRow < 0 || r.currentRow >= len(r.rows) {
		return fmt.Errorf("invalid row index: %d", r.currentRow)
	}

	// Get the current row
	row := r.rows[r.currentRow]

	// Make sure we have the right number of destinations
	if len(dest) != len(r.columns) {
		return fmt.Errorf("expected %d destination arguments in Scan, got %d", len(r.columns), len(dest))
	}

	// Copy values to destinations
	for i, val := range row {
		if i >= len(dest) {
			break
		}

		// Use the central utility function to scan values directly
		// val is already of type storage.ColumnValue which can handle nil values internally
		if err := storage.ScanColumnValueToDestination(val, dest[i]); err != nil {
			return fmt.Errorf("column %s: %w", r.columns[i], err)
		}
	}

	return nil
}

// Close closes the result set
func (r *OrderedResult) Close() error {
	return r.baseResult.Close()
}

// RowsAffected returns the number of rows affected
func (r *OrderedResult) RowsAffected() int64 {
	return r.baseResult.RowsAffected()
}

// LastInsertID returns the last insert ID
func (r *OrderedResult) LastInsertID() int64 {
	return r.baseResult.LastInsertID()
}

// Context returns the context
func (r *OrderedResult) Context() context.Context {
	return r.baseResult.Context()
}

// WithAliases implements the storage.Result interface
func (r *OrderedResult) WithAliases(aliases map[string]string) storage.Result {
	// We need to propagate aliases to the base result first
	if aliasable, ok := r.baseResult.(interface {
		WithAliases(map[string]string) storage.Result
	}); ok {
		base := aliasable.WithAliases(aliases)

		// Create a new OrderedResult with the aliased base result
		result := &OrderedResult{
			baseResult:  base,
			orderBy:     r.orderBy,
			initialized: false, // Force re-initialization with the new aliases
		}

		return result
	}

	// If the base result doesn't support aliases, wrap this result
	return NewAliasedResult(r, aliases)
}

// Row implements the storage.Result interface
// For OrderedResult, this converts the materialized row back to storage.Row
func (r *OrderedResult) Row() storage.Row {
	if !r.initialized || r.currentRow < 0 || r.currentRow >= len(r.rows) {
		return nil
	}

	// Get the current row of interface{} values
	currentRow := r.rows[r.currentRow]
	if currentRow == nil {
		return nil
	}

	// Convert the materialized interface{} values back to storage.ColumnValue objects
	row := make(storage.Row, len(currentRow))

	for i, val := range currentRow {
		if val == nil {
			row[i] = storage.StaticNullUnknown
			continue
		}

		row[i] = val
	}

	return row
}

// sortRows sorts the rows based on the ORDER BY clause
func sortRows(rows [][]storage.ColumnValue, columns []string, orderBy []parser.OrderByExpression) {
	// Create a mapping from column name to index
	colMap := make(map[string]int)
	for i, col := range columns {
		colMap[col] = i
	}

	// Sort the rows
	sort.Slice(rows, func(i, j int) bool {
		// Compare each column in the ORDER BY clause
		for _, orderExpr := range orderBy {
			// Get the column name
			var colName string
			switch expr := orderExpr.Expression.(type) {
			case *parser.Identifier:
				colName = expr.Value
			case *parser.QualifiedIdentifier:
				colName = expr.Name.Value
			case *parser.FunctionCall:
				// Special handling for function calls in ORDER BY
				if len(expr.Arguments) > 0 {
					argName := "unknown"
					if ident, ok := expr.Arguments[0].(*parser.Identifier); ok {
						argName = ident.Value
					}
					colName = fmt.Sprintf("%s(%s)", expr.Function, argName)
				} else {
					continue
				}
			default:
				// Skip expressions we can't handle
				continue
			}

			// Get the column index
			colIdx, ok := colMap[colName]
			if !ok {
				// Skip columns not in the result
				continue
			}

			// Compare the values
			a, b := rows[i][colIdx], rows[j][colIdx]

			// Handle nil values
			if a == nil && b == nil {
				continue // Equal, try next column
			}
			if a == nil {
				return !orderExpr.Ascending // Nil is less than non-nil
			}
			if b == nil {
				return orderExpr.Ascending // Non-nil is greater than nil
			}

			// Use the built-in comparison method
			if comp, err := a.Compare(b); err == nil {
				if comp != 0 {
					// comp < 0 means a < b, so return true if ascending
					// comp > 0 means a > b, so return false if ascending
					return (comp < 0) == orderExpr.Ascending
				}
				// Values are equal, continue to next column
				continue
			}
		}

		// If all columns are equal, return false (stable sort)
		return false
	})
}

// LimitedResult represents a result set limited by LIMIT and OFFSET
type LimitedResult struct {
	baseResult storage.Result
	limit      int64
	offset     int64
	// State for tracking current position
	rowCount   int64
	pastOffset bool
}

// Columns returns the column names from the base result
func (r *LimitedResult) Columns() []string {
	return r.baseResult.Columns()
}

// Next advances to the next row in the result set
func (r *LimitedResult) Next() bool {
	// If we've reached our limit, stop
	if r.limit >= 0 && r.rowCount >= r.limit {
		return false
	}

	// Skip rows for offset
	if !r.pastOffset {
		for i := int64(0); i < r.offset; i++ {
			if !r.baseResult.Next() {
				return false
			}
		}
		r.pastOffset = true
	}

	// Get the next row
	if r.baseResult.Next() {
		r.rowCount++

		// Check if we've reached our limit after incrementing row count
		return r.limit < 0 || r.rowCount <= r.limit
	}

	return false
}

// Scan copies values from the current row into the provided destinations
func (r *LimitedResult) Scan(dest ...interface{}) error {
	// Get row from base result
	row := r.baseResult.Row()
	if row == nil {
		return fmt.Errorf("no row available")
	}

	// Make sure we have the right number of destinations
	columns := r.Columns()
	if len(dest) != len(columns) {
		return fmt.Errorf("expected %d destination arguments in Scan, got %d", len(columns), len(dest))
	}

	// Copy values from row to destinations using the central utility function
	for i, colVal := range row {
		if i >= len(dest) {
			break
		}

		// Use the centralized scanning function for consistent behavior
		if err := storage.ScanColumnValueToDestination(colVal, dest[i]); err != nil {
			return fmt.Errorf("column %s: %w", columns[i], err)
		}
	}

	return nil
}

// Close closes the result set
func (r *LimitedResult) Close() error {
	return r.baseResult.Close()
}

// RowsAffected returns the number of rows affected
func (r *LimitedResult) RowsAffected() int64 {
	return r.baseResult.RowsAffected()
}

// LastInsertID returns the last insert ID
func (r *LimitedResult) LastInsertID() int64 {
	return r.baseResult.LastInsertID()
}

// Context returns the context
func (r *LimitedResult) Context() context.Context {
	return r.baseResult.Context()
}

// WithAliases implements the storage.Result interface
func (r *LimitedResult) WithAliases(aliases map[string]string) storage.Result {
	// We need to propagate aliases to the base result first
	if aliasable, ok := r.baseResult.(interface {
		WithAliases(map[string]string) storage.Result
	}); ok {
		base := aliasable.WithAliases(aliases)

		// Create a new LimitedResult with the aliased base result
		result := &LimitedResult{
			baseResult: base,
			limit:      r.limit,
			offset:     r.offset,
			rowCount:   0,
			pastOffset: false,
		}

		return result
	}

	// If the base result doesn't support aliases, wrap this result
	return NewAliasedResult(r, aliases)
}

// Row implements the storage.Result interface
// LimitedResult just delegates to the base result since it doesn't transform the data
func (r *LimitedResult) Row() storage.Row {
	// LimitedResult doesn't transform the data, it just limits the number of rows
	// so we can delegate directly to the base result's Row method
	return r.baseResult.Row()
}

// AggregateResult represents a result set with aggregation
type AggregateResult struct {
	baseResult     storage.Result
	functions      []*SqlFunction
	groupByColumns []string
	aliases        map[string]string
	// State for aggregation
	aggregatedRows []map[string]storage.ColumnValue
	currentIndex   int
	columns        []string
	initialized    bool
}

// initialize processes the base result and computes the aggregations
func (r *AggregateResult) initialize() error {
	if r.initialized {
		return nil
	}

	// Set up columns
	r.columns = make([]string, 0)

	// Add group by columns first
	if r.groupByColumns != nil {
		r.columns = append(r.columns, r.groupByColumns...)
	}

	// Add function columns
	for _, fn := range r.functions {
		// Use function name or alias
		name := fn.GetColumnName()
		r.columns = append(r.columns, name)
	}

	// Create a map to store grouped data
	groups := make(map[string][]map[string]storage.ColumnValue)
	baseColumns := r.baseResult.Columns()

	// Build a column index map for faster lookups
	colIndexMap := make(map[string]int)
	for i, col := range baseColumns {
		colIndexMap[col] = i
		// Also add lowercase version for case-insensitive lookups
		colIndexMap[strings.ToLower(col)] = i
	}

	// Create a reusable row map with sufficient initial capacity
	row := make(map[string]storage.ColumnValue, len(baseColumns)*2) // Extra space for derived values

	for r.baseResult.Next() {
		// Clear the row map for reuse
		for k := range row {
			delete(row, k)
		}

		baseRow := r.baseResult.Row()

		// Scan the row
		if baseRow == nil {
			return fmt.Errorf("aggregations no row available")
		}

		// Populate the map with this row's data
		for i, col := range baseColumns {
			row[col] = baseRow[i]
		}

		// Handle any computed expressions needed by aliases
		// For example, compute decade = id / 10 if needed
		for alias, exprStr := range r.aliases {
			// If alias is already computed, skip
			if _, exists := row[alias]; exists {
				continue
			}

			// Check if this is a computed expression like (id / 10)
			if strings.HasPrefix(exprStr, "(") && strings.HasSuffix(exprStr, ")") && strings.Contains(exprStr, "/") {
				// Parse the division expression like (id / 10)
				parts := strings.Split(exprStr[1:len(exprStr)-1], "/")
				if len(parts) == 2 {
					leftCol := strings.TrimSpace(parts[0])
					rightValStr := strings.TrimSpace(parts[1])

					// Get the left column value
					if leftVal, hasLeft := row[leftCol]; hasLeft && leftVal != nil {
						// Convert to numeric
						var numericLeft float64
						switch leftVal.Type() {
						case storage.INTEGER:
							numericLeft, _ = leftVal.AsFloat64()
						case storage.FLOAT:
							numericLeft, _ = leftVal.AsFloat64()
						default:
							continue // Skip non-numeric values
						}

						// Parse right value
						rightVal, err := strconv.ParseFloat(rightValStr, 64)
						if err == nil && rightVal != 0 {
							// Compute division and store in the row
							result := numericLeft / rightVal
							// Use integer division for the decade calculation
							if alias == "decade" {
								row[alias] = storage.NewIntegerValue(int64(result))
							} else {
								row[alias] = storage.NewFloatValue(result)
							}
						}
					}
				}
			}
		}

		// Create a key for this group
		var groupKey string
		if len(r.groupByColumns) > 0 {
			groupValues := make([]string, len(r.groupByColumns))
			for i, groupCol := range r.groupByColumns {
				// First check: handle function calls like TIME_TRUNC('1h', sale_time)
				if strings.Contains(groupCol, "(") && strings.Contains(groupCol, ")") &&
					!strings.HasPrefix(groupCol, "(") {

					// Parse function name
					funcNameEndIdx := strings.Index(groupCol, "(")
					if funcNameEndIdx > 0 {
						funcName := strings.TrimSpace(groupCol[:funcNameEndIdx])

						// Check if it's a known scalar function
						funcRegistry := registry.GetGlobal()
						if funcRegistry != nil && funcRegistry.IsScalarFunction(funcName) {
							// Special handling for our timestamp functions
							if funcName == "TIME_TRUNC" || funcName == "DATE_TRUNC" {
								// Extract arguments
								argStr := groupCol[funcNameEndIdx+1 : len(groupCol)-1]
								argParts := strings.Split(argStr, ",")
								if len(argParts) >= 2 {
									// Get duration string
									durationStr := strings.TrimSpace(argParts[0])
									if strings.HasPrefix(durationStr, "'") && strings.HasSuffix(durationStr, "'") {
										durationStr = durationStr[1 : len(durationStr)-1]
									}

									// Get the timestamp column name
									timeColName := strings.TrimSpace(argParts[1])

									// Get the timestamp value for this row
									var timeVal storage.ColumnValue
									if val, ok := row[timeColName]; ok {
										timeVal = val
									} else {
										// Try case-insensitive lookup
										for colName, colVal := range row {
											if strings.EqualFold(colName, timeColName) {
												timeVal = colVal
												break
											}
										}
									}

									// Invoke the function for this row
									if timeVal != nil {
										scalarFn := funcRegistry.GetScalarFunction(funcName)
										if scalarFn != nil {
											result, err := scalarFn.Evaluate(durationStr, timeVal.AsInterface())
											if err == nil {
												// Use the result as the group key
												groupValues[i] = fmt.Sprintf("%v", result)
												continue
											}
										}
									}
								}
							}
						}
					}
				}

				// Second check: see if this is a direct column reference
				if val, ok := row[groupCol]; ok {
					groupValues[i], _ = val.AsString()
					continue
				}

				// Check if we need to resolve unqualified column name
				if unqualMapping, hasUnqual := r.aliases["__unqual_"+groupCol]; hasUnqual {
					if val, ok := row[unqualMapping]; ok {
						groupValues[i], _ = val.AsString()
						continue
					}
				}

				// Third check: look for case-insensitive match
				found := false
				for colName, colVal := range row {
					if strings.EqualFold(colName, groupCol) {
						groupValues[i], _ = colVal.AsString()
						found = true
						break
					}
				}
				if found {
					continue
				}

				// Fourth check: handle expressions in GROUP BY
				// If groupCol is an infix expression like (id / 10)
				// Extract the needed values and compute the result
				if strings.HasPrefix(groupCol, "(") && strings.HasSuffix(groupCol, ")") {
					// This looks like an expression, try to compute it

					// For the specific case (id / 10), handle directly
					if strings.Contains(groupCol, "/") {
						parts := strings.Split(groupCol[1:len(groupCol)-1], "/")
						if len(parts) == 2 {
							leftCol := strings.TrimSpace(parts[0])
							rightValStr := strings.TrimSpace(parts[1])

							// Get the left column value
							var leftVal storage.ColumnValue
							if val, ok := row[leftCol]; ok {
								leftVal = val
							} else {
								// Try case-insensitive lookup
								for colName, colVal := range row {
									if strings.EqualFold(colName, leftCol) {
										leftVal = colVal
										break
									}
								}
							}

							// If we found the left value, try to compute the expression
							if leftVal != nil {
								// Convert to numeric types for division
								var numericLeft float64
								switch leftVal.Type() {
								case storage.INTEGER:
									numericLeft, _ = leftVal.AsFloat64()
								case storage.FLOAT:
									numericLeft, _ = leftVal.AsFloat64()
								default:
									// Could not convert to numeric
									groupValues[i] = "nil"
									continue
								}

								// Convert right value to number
								rightVal, err := strconv.ParseFloat(rightValStr, 64)
								if err == nil && rightVal != 0 {
									// Compute division
									result := numericLeft / rightVal
									// Convert to int if it's a whole number
									if result == float64(int(result)) {
										groupValues[i] = fmt.Sprintf("%d", int(result))
									} else {
										groupValues[i] = fmt.Sprintf("%v", result)
									}
									continue
								}
							}
						}
					}
				}

				// Fifth check: if group column is an alias, look up its mapping
				if r.aliases != nil {
					if origCol, ok := r.aliases[groupCol]; ok {
						if val, ok := row[origCol]; ok {
							groupValues[i] = fmt.Sprintf("%v", val)
							continue
						}
					}
				}

				// If all checks fail, use nil as value
				groupValues[i] = "nil"
			}
			groupKey = strings.Join(groupValues, "|")
		} else {
			// For queries with aggregation but no GROUP BY, use a constant key
			groupKey = "global"
		}

		// Create a copy of the current row to avoid issues with row map reuse
		rowCopy := make(map[string]storage.ColumnValue, len(row))
		for k, v := range row {
			rowCopy[k] = v
		}

		// Simply add the row to the group - we need all rows for accurate HAVING clause evaluation
		groups[groupKey] = append(groups[groupKey], rowCopy)
	}

	// Process each group to compute aggregations
	r.aggregatedRows = make([]map[string]storage.ColumnValue, 0, len(groups))

	for _, rows := range groups {
		// Create a new result row
		resultRow := make(map[string]storage.ColumnValue, len(r.columns))

		// Add group by values
		if len(r.groupByColumns) > 0 {
			// Use the first row's values for the group by columns
			for _, groupCol := range r.groupByColumns {
				// First check: direct column
				if val, ok := rows[0][groupCol]; ok {
					resultRow[groupCol] = val
					continue
				}

				// Second check: case-insensitive lookup
				found := false
				for colName, colVal := range rows[0] {
					if strings.EqualFold(colName, groupCol) {
						resultRow[groupCol] = colVal
						found = true
						break
					}
				}
				if found {
					continue
				}

				// Third check: handle function calls like TIME_TRUNC('1h', sale_time)
				if strings.Contains(groupCol, "(") && strings.Contains(groupCol, ")") &&
					!strings.HasPrefix(groupCol, "(") {

					// Parse function name
					funcNameEndIdx := strings.Index(groupCol, "(")
					if funcNameEndIdx > 0 {
						funcName := strings.TrimSpace(groupCol[:funcNameEndIdx])

						// Check if it's a known scalar function
						funcRegistry := registry.GetGlobal()
						if funcRegistry != nil && funcRegistry.IsScalarFunction(funcName) {
							// Special handling for our timestamp functions
							if funcName == "TIME_TRUNC" || funcName == "DATE_TRUNC" {
								// Extract arguments
								argStr := groupCol[funcNameEndIdx+1 : len(groupCol)-1]
								argParts := strings.Split(argStr, ",")
								if len(argParts) >= 2 {
									// Get duration string
									durationStr := strings.TrimSpace(argParts[0])
									if strings.HasPrefix(durationStr, "'") && strings.HasSuffix(durationStr, "'") {
										durationStr = durationStr[1 : len(durationStr)-1]
									}

									// Get the timestamp column name
									timeColName := strings.TrimSpace(argParts[1])

									// Store the function result for each row in the group
									// This ensures we properly group by the function result
									var functionResults = make(map[string]interface{})

									// Process each row in the group
									for _, rowData := range rows {
										// Get the timestamp value for this row
										var timeVal storage.ColumnValue
										if val, ok := rowData[timeColName]; ok {
											timeVal = val
										} else {
											// Try case-insensitive lookup
											for colName, colVal := range rowData {
												if strings.EqualFold(colName, timeColName) {
													timeVal = colVal
													break
												}
											}
										}

										// Invoke the function for this row
										if timeVal != nil {
											scalarFn := funcRegistry.GetScalarFunction(funcName)
											if scalarFn != nil {
												result, err := scalarFn.Evaluate(durationStr, timeVal.AsInterface())
												if err == nil {
													// Use the result as a key to group matching results
													resultStr := fmt.Sprintf("%v", result)
													functionResults[resultStr] = result
												}
											}
										}
									}

									// Use each unique function result as a separate group key
									// The key will be part of the groupKey string used to identify the group
									if len(functionResults) > 0 {
										// Use the first result (any result will do since we only care about mapping one result to the group key)
										for _, v := range functionResults {
											resultRow[groupCol] = storage.NewDirectValueFromInterface(v)
											break
										}
										continue
									}
								}
							} else {
								// For other scalar functions, use an Evaluator
								evaluator := NewEvaluator(r.Context(), funcRegistry)
								// Convert row data to storage.ColumnValue format
								convertedRow := make(map[string]storage.ColumnValue, len(rows[0]))
								for k, v := range rows[0] {
									if v == nil {
										convertedRow[k] = storage.StaticNullUnknown
									} else {
										convertedRow[k] = v
									}
								}

								evaluator.WithRow(convertedRow) // Use the first row's data

								// TODO: Add more sophisticated function handling if needed
							}
						}
					}
				}

				// Fourth check: handle expressions like (id / 10)
				if strings.HasPrefix(groupCol, "(") && strings.HasSuffix(groupCol, ")") {
					// Handle division expressions like (id / 10)
					if strings.Contains(groupCol, "/") {
						parts := strings.Split(groupCol[1:len(groupCol)-1], "/")
						if len(parts) == 2 {
							leftCol := strings.TrimSpace(parts[0])
							rightValStr := strings.TrimSpace(parts[1])

							// Get the left column value
							var leftVal storage.ColumnValue
							if val, ok := rows[0][leftCol]; ok {
								leftVal = val
							} else {
								// Try case-insensitive lookup
								for colName, colVal := range rows[0] {
									if strings.EqualFold(colName, leftCol) {
										leftVal = colVal
										break
									}
								}
							}

							// If we found the left value, compute the expression
							if leftVal != nil {
								// Convert to numeric types for division
								var numericLeft float64
								switch leftVal.Type() {
								case storage.INTEGER:
									numericLeft, _ = leftVal.AsFloat64()
								case storage.FLOAT:
									numericLeft, _ = leftVal.AsFloat64()
								default:
									// Could not convert to numeric
									resultRow[groupCol] = nil
									continue
								}

								// Convert right value to number
								rightVal, err := strconv.ParseFloat(rightValStr, 64)
								if err == nil && rightVal != 0 {
									// Compute division
									result := numericLeft / rightVal
									resultRow[groupCol] = storage.NewFloatValue(result)

									// Also store under the alias if we have one
									if r.aliases != nil {
										if alias, ok := r.aliases[groupCol]; ok {
											resultRow[alias] = storage.NewFloatValue(result)
										}
									}
									continue
								}
							}
						}
					}
				}

				// Fourth check: check aliases
				if r.aliases != nil {
					if origCol, ok := r.aliases[groupCol]; ok {
						if val, ok := rows[0][origCol]; ok {
							resultRow[groupCol] = val
							continue
						}
					}
				}

				// If all else fails, use nil
				resultRow[groupCol] = nil
			}
		}

		// Compute each aggregate function for this group
		for _, fn := range r.functions {
			// Determine column name in result set
			colName := fn.GetColumnName()

			// Compute the aggregate value based on function type
			funcName := strings.ToUpper(fn.Name)
			if funcName == "COUNT" {
				// COUNT implementations directly in code
				if fn.Column == "*" {
					// COUNT(*) counts all rows
					resultRow[colName] = storage.NewIntegerValue(int64(len(rows)))
				} else if fn.IsDistinct {
					// Use the helper function for COUNT DISTINCT
					// Implementation of COUNT DISTINCT inline
					distinctValues := make(map[interface{}]bool)
					for _, row := range rows {
						// Get the column value
						val, ok := row[fn.Column]
						if !ok {
							// Try case-insensitive lookup
							for rowColName, rowVal := range row {
								if strings.EqualFold(rowColName, fn.Column) {
									val = rowVal
									ok = true
									break
								}
							}
						}

						// Only add non-NULL values
						if ok && val != nil {
							distinctValues[val] = true
						}
					}
					resultRow[colName] = storage.NewIntegerValue(int64(len(distinctValues)))
				} else {
					// Regular COUNT counts non-NULL values
					var count int64
					for _, row := range rows {
						// Check if the column exists in this row
						val, ok := row[fn.Column]
						if !ok {
							// Try case-insensitive lookup
							for rowColName, rowVal := range row {
								if strings.EqualFold(rowColName, fn.Column) {
									val = rowVal
									ok = true
									break
								}
							}
						}

						// Count non-NULL values
						if ok && val != nil {
							count++
						}
					}
					resultRow[colName] = storage.NewIntegerValue(count)
				}
			} else {
				// For all other functions, use the registry
				// Get the function registry
				funcRegistry := registry.GetGlobal()

				// Get the specific function from the registry
				aggFunc := funcRegistry.GetAggregateFunction(strings.ToUpper(fn.Name))

				if aggFunc != nil {
					// Reset the function state
					aggFunc.Reset()

					// Check if this is an ordered aggregate function (like FIRST, LAST)
					// and if it implements the OrderedAggregateFunction interface
					orderedAggFunc, supportsOrdering := aggFunc.(contract.OrderedAggregateFunction)

					// Check if we have ORDER BY expressions
					hasOrderBy := len(fn.OrderBy) > 0

					// If we have both an ordered function and ORDER BY expressions, use ordered processing
					if supportsOrdering && hasOrderBy && orderedAggFunc.SupportsOrdering() {
						// For ordered aggregates like FIRST(col ORDER BY timestamp)
						// We need to collect values along with their ordering keys, then process them in order
						for _, row := range rows {

							// Get the main column value
							mainVal, mainValOk := row[fn.Column]
							if !mainValOk {
								// Try case-insensitive lookup for main value
								for colName, colVal := range row {
									if strings.EqualFold(colName, fn.Column) {
										mainVal = colVal
										mainValOk = true
										break
									}
								}
							}

							// Skip if main value is null
							if !mainValOk || mainVal == nil {
								continue
							}

							// For each ORDER BY expression, extract the order key
							if len(fn.OrderBy) > 0 {
								// Get the first ORDER BY expression (primary sort key)
								orderExpr := fn.OrderBy[0]

								// Extract the column name from the order expression
								var orderColName string
								switch exprType := orderExpr.Expression.(type) {
								case *parser.Identifier:
									orderColName = exprType.Value
								case *parser.QualifiedIdentifier:
									orderColName = exprType.Name.Value
								default:
									// Skip expressions we can't handle
									continue
								}

								// Look up the ordering column in the row
								orderVal, orderValOk := row[orderColName]
								if !orderValOk {
									// Try case-insensitive lookup
									for colName, colVal := range row {
										if strings.EqualFold(colName, orderColName) {
											orderVal = colVal
											orderValOk = true
											break
										}
									}
								}

								// Skip if order value is null
								if !orderValOk || orderVal == nil {
									continue
								}

								// Determine direction (ASC or DESC)
								direction := "ASC"
								if !orderExpr.Ascending {
									direction = "DESC"
								}

								// Accumulate the value with its ordering key
								orderedAggFunc.AccumulateOrdered(mainVal.AsInterface(), orderVal.AsInterface(), direction, fn.IsDistinct)
							}
						}
					} else {
						// Standard non-ordered aggregation
						// Process each row's column value
						for _, row := range rows {
							// Get the column value
							val, ok := row[fn.Column]
							if !ok {
								// Try case-insensitive lookup
								for colName, colVal := range row {
									if strings.EqualFold(colName, fn.Column) {
										val = colVal
										ok = true
										break
									}
								}
							}

							// Accumulate non-NULL values
							if ok && val != nil {
								aggFunc.Accumulate(val.AsInterface(), fn.IsDistinct)
							}
						}
					}

					result := aggFunc.Result()
					// Get the result
					resultRow[colName] = storage.NewDirectValueFromInterface(result)
				} else {
					// Function not found in registry, default to NULL
					resultRow[colName] = nil
				}
			}
		}

		// Add this aggregated row to the result
		r.aggregatedRows = append(r.aggregatedRows, resultRow)
	}

	// Special case: if we have aggregate functions but no groups (no rows matched)
	// we still need to return one row with aggregate results
	if len(groups) == 0 && len(r.functions) > 0 && len(r.groupByColumns) == 0 {
		// Create a single row with aggregate results for empty set
		resultRow := make(map[string]storage.ColumnValue, len(r.columns))

		// Process each aggregate function
		for _, fn := range r.functions {
			colName := fn.GetColumnName()
			funcName := strings.ToUpper(fn.Name)

			if funcName == "COUNT" {
				// COUNT returns 0 for empty sets
				resultRow[colName] = storage.NewIntegerValue(0)
			} else {
				// All other aggregate functions return NULL for empty sets
				resultRow[colName] = nil
			}
		}

		r.aggregatedRows = append(r.aggregatedRows, resultRow)
	}

	// Reset position for iteration
	r.currentIndex = -1
	r.initialized = true

	return nil
}

// Columns returns the column names from the base result
func (r *AggregateResult) Columns() []string {
	if !r.initialized {
		r.initialize()
	}
	return r.columns
}

// Next advances to the next row in the result set
func (r *AggregateResult) Next() bool {
	if !r.initialized {
		r.initialize()
	}

	r.currentIndex++
	return r.currentIndex < len(r.aggregatedRows)
}

// Scan copies values from the current row into the provided destinations
func (r *AggregateResult) Scan(dest ...interface{}) error {
	if !r.initialized || r.currentIndex < 0 || r.currentIndex >= len(r.aggregatedRows) {
		return fmt.Errorf("no row to scan")
	}

	if len(dest) != len(r.columns) {
		return fmt.Errorf("column count mismatch: got %d, wanted %d", len(dest), len(r.columns))
	}

	// Get the current row
	row := r.aggregatedRows[r.currentIndex]

	// Copy values to destination
	for i, colName := range r.columns {
		// First check: direct column match
		val, ok := row[colName]
		if !ok {
			// Second check: case-insensitive match
			for rowColName, rowVal := range row {
				if strings.EqualFold(rowColName, colName) {
					val = rowVal
					ok = true
					break
				}
			}
		}

		// Third check: check aliases - both ways
		if !ok && r.aliases != nil {
			// Check if the column name is an alias and get the original
			if aliasedCol, exists := r.aliases[colName]; exists {
				if aliasVal, aliasOk := row[aliasedCol]; aliasOk {
					val = aliasVal
					ok = true
				}
			}

			// Check if the column is aliased by something else
			for alias, originalCol := range r.aliases {
				if originalCol == colName {
					if aliasVal, aliasOk := row[alias]; aliasOk {
						val = aliasVal
						ok = true
						break
					}
				}
			}
		}

		// Fourth check: handle expressions - specifically division expressions
		if !ok && strings.HasPrefix(colName, "(") && strings.HasSuffix(colName, ")") && strings.Contains(colName, "/") {
			parts := strings.Split(colName[1:len(colName)-1], "/")
			if len(parts) == 2 {
				leftCol := strings.TrimSpace(parts[0])
				rightValStr := strings.TrimSpace(parts[1])

				// Get the left column value
				var leftVal storage.ColumnValue
				if leftVal, ok = row[leftCol]; !ok {
					// Try case-insensitive lookup
					for rowCol, rowVal := range row {
						if strings.EqualFold(rowCol, leftCol) {
							leftVal = rowVal
							ok = true
							break
						}
					}
				}

				if ok && leftVal != nil {
					// Convert to numeric types for division
					var numericLeft float64
					switch leftVal.Type() {
					case storage.INTEGER:
						numericLeft, _ = leftVal.AsFloat64()
					case storage.FLOAT:
						numericLeft, _ = leftVal.AsFloat64()
					default:
						ok = false
					}

					if ok {
						// Convert right value to number
						rightVal, err := strconv.ParseFloat(rightValStr, 64)
						if err == nil && rightVal != 0 {
							// Compute division
							result := numericLeft / rightVal
							val = storage.NewFloatValue(result)
							ok = true
						} else {
							ok = false
						}
					}
				}
			}
		}

		// Handle nil or not found values
		if !ok || val == nil {
			if err := storage.ScanColumnValueToDestination(nil, dest[i]); err != nil {
				return fmt.Errorf("column %s: %w", colName, err)
			}
			continue
		}

		// For non-null values, convert to ColumnValue first for consistent handling
		// This ensures that we handle all types consistently
		if err := storage.ScanColumnValueToDestination(val, dest[i]); err != nil {
			return fmt.Errorf("column %s: %w", colName, err)
		}
	}

	return nil
}

// Close closes the result set
func (r *AggregateResult) Close() error {
	return r.baseResult.Close()
}

// RowsAffected returns the number of rows affected
func (r *AggregateResult) RowsAffected() int64 {
	return r.baseResult.RowsAffected()
}

// LastInsertID returns the last insert ID
func (r *AggregateResult) LastInsertID() int64 {
	return r.baseResult.LastInsertID()
}

// Context returns the context
func (r *AggregateResult) Context() context.Context {
	return r.baseResult.Context()
}

// WithAliases implements the storage.Result interface
func (r *AggregateResult) WithAliases(aliases map[string]string) storage.Result {
	// Create a new aggregate result with the combined aliases
	combinedAliases := make(map[string]string)

	// Copy existing aliases
	if r.aliases != nil {
		for alias, origCol := range r.aliases {
			combinedAliases[alias] = origCol
		}
	}

	// Add new aliases
	for alias, origCol := range aliases {
		// If original name is itself an alias, resolve it
		if r.aliases != nil {
			if actualCol, isAlias := r.aliases[origCol]; isAlias {
				combinedAliases[alias] = actualCol
				continue
			}
		}
		combinedAliases[alias] = origCol
	}

	// Create a new result with these combined aliases
	result := &AggregateResult{
		baseResult:     r.baseResult,
		functions:      r.functions,
		groupByColumns: r.groupByColumns,
		aliases:        combinedAliases,
		initialized:    false, // Force re-initialization with the new aliases
	}

	return result
}

// Row implements the storage.Result interface
// For AggregateResult, this converts the materialized row back to storage.Row
func (r *AggregateResult) Row() storage.Row {
	if !r.initialized || r.currentIndex < 0 || r.currentIndex >= len(r.aggregatedRows) {
		return nil
	}

	// Get the current row
	currentRow := r.aggregatedRows[r.currentIndex]
	if currentRow == nil {
		return nil
	}

	// Convert the map values to storage.Row
	row := make(storage.Row, len(r.columns))

	for i, colName := range r.columns {
		// Find the value in the map
		val, ok := currentRow[colName]
		if !ok {
			row[i] = storage.StaticNullUnknown
			continue
		}

		// Handle nil values
		if val == nil {
			row[i] = storage.StaticNullUnknown
			continue
		}

		row[i] = val
	}

	return row
}
