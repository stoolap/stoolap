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

	"github.com/stoolap/stoolap/internal/parser"
	"github.com/stoolap/stoolap/internal/storage"
)

// CTERegistry manages CTE execution and results
type CTERegistry struct {
	materialized map[string]storage.Result
	schemas      map[string][]string
}

// NewCTERegistry creates a new CTE registry
func NewCTERegistry() *CTERegistry {
	return &CTERegistry{
		materialized: make(map[string]storage.Result),
		schemas:      make(map[string][]string),
	}
}

// Execute executes all CTEs in a WITH clause
func (r *CTERegistry) Execute(ctx context.Context, e *Executor, tx storage.Transaction, withClause *parser.WithClause) error {
	for _, cte := range withClause.CTEs {
		// Execute the CTE query
		result, err := e.executeCTEQuery(ctx, tx, cte.Query, r)
		if err != nil {
			return fmt.Errorf("failed to execute CTE '%s': %w", cte.Name.Value, err)
		}

		// Materialize the result so it can be accessed multiple times
		columns := result.Columns()
		rows := make([]map[string]storage.ColumnValue, 0)

		for result.Next() {
			row := result.Row()
			if row != nil {
				rowMap := make(map[string]storage.ColumnValue, len(columns))
				for i, col := range columns {
					if i < len(row) {
						rowMap[col] = row[i]
					}
				}
				rows = append(rows, rowMap)
			}
		}
		result.Close()

		// Create a materialized result
		materializedResult := &MaterializedResult{
			columns: columns,
			rows:    rows,
			current: -1,
		}

		// Handle column aliases if specified
		if len(cte.ColumnNames) > 0 {
			aliasedColumns := make([]string, len(cte.ColumnNames))
			for i, alias := range cte.ColumnNames {
				aliasedColumns[i] = alias.Value
			}

			// Update row maps to use aliased column names
			aliasedRows := make([]map[string]storage.ColumnValue, len(rows))
			for i, row := range rows {
				aliasedRow := make(map[string]storage.ColumnValue)
				for j, aliasCol := range aliasedColumns {
					if j < len(columns) {
						originalCol := columns[j]
						if val, ok := row[originalCol]; ok {
							aliasedRow[aliasCol] = val
						}
					}
				}
				aliasedRows[i] = aliasedRow
			}

			materializedResult = &MaterializedResult{
				columns: aliasedColumns,
				rows:    aliasedRows,
				current: -1,
			}
		}

		// Store the materialized result
		r.materialized[cte.Name.Value] = materializedResult
		r.schemas[cte.Name.Value] = materializedResult.columns
	}
	return nil
}

// GetCTE retrieves a CTE result
func (r *CTERegistry) GetCTE(name string) (storage.Result, bool) {
	result, exists := r.materialized[name]
	if !exists {
		return nil, false
	}

	// Return a fresh copy of the materialized result so it can be read from the beginning
	if mr, ok := result.(*MaterializedResult); ok {
		return &MaterializedResult{
			columns: mr.columns,
			rows:    mr.rows,
			current: -1,
		}, true
	}

	return result, exists
}

// executeCTEQuery executes a single CTE query
func (e *Executor) executeCTEQuery(ctx context.Context, tx storage.Transaction, stmt *parser.SelectStatement, registry *CTERegistry) (storage.Result, error) {
	// Store current registry
	oldRegistry := e.cteRegistry
	e.cteRegistry = registry
	defer func() {
		e.cteRegistry = oldRegistry
	}()

	// Execute using the standard query path
	return e.executeSelectWithContext(ctx, tx, stmt)
}

// CTETableSource implements a table source backed by a CTE
type CTETableSource struct {
	name   string
	result storage.Result
}

// GetTable returns the CTE as a table-like result
func (c *CTETableSource) GetTable(name string) (storage.Result, error) {
	if name != c.name {
		return nil, fmt.Errorf("CTE '%s' not found", name)
	}
	return c.result, nil
}

// TableExists checks if the CTE exists
func (c *CTETableSource) TableExists(name string) bool {
	return name == c.name
}

// resolveCTETable checks if a table reference is a CTE and returns appropriate result
func (e *Executor) resolveCTETable(ctx context.Context, tx storage.Transaction, tableName string) (storage.Result, bool) {
	if e.cteRegistry == nil {
		return nil, false
	}

	result, exists := e.cteRegistry.GetCTE(tableName)
	return result, exists
}

// processCTESelect processes a SELECT statement that references a CTE
func (e *Executor) processCTESelect(ctx context.Context, tx storage.Transaction, cteResult storage.Result, stmt *parser.SelectStatement) (storage.Result, error) {
	// Apply WHERE clause if present (BEFORE aggregation)
	if stmt.Where != nil {
		// Process any subqueries in the WHERE clause
		processedWhere, err := e.processWhereSubqueries(ctx, tx, stmt.Where)
		if err != nil {
			return nil, fmt.Errorf("error processing subqueries in WHERE clause: %w", err)
		}

		// Create an evaluator and apply the WHERE filter
		evaluator := NewEvaluator(ctx, e.functionRegistry)
		cteResult = &FilteredResult{
			result:    cteResult,
			whereExpr: processedWhere,
			evaluator: evaluator,
		}
	}

	// Check if the query has aggregations
	hasAggregation := false
	for _, col := range stmt.Columns {
		if ContainsAggregateFunction(col) {
			hasAggregation = true
			break
		}
	}

	// If we have aggregation or GROUP BY, handle it specially
	if hasAggregation || stmt.GroupBy != nil {
		return e.executeAggregationOnResult(ctx, cteResult, stmt)
	}

	// Handle column selection
	if len(stmt.Columns) > 0 && !(len(stmt.Columns) == 1 && IsAsterisk(stmt.Columns[0])) {
		// We need to project specific columns
		columnNames := make([]string, len(stmt.Columns))
		for i, col := range stmt.Columns {
			switch expr := col.(type) {
			case *parser.Identifier:
				columnNames[i] = expr.Value
			case *parser.AliasedExpression:
				if alias := expr.Alias; alias != nil {
					columnNames[i] = alias.Value
				} else {
					// Fallback to expression string
					columnNames[i] = expr.Expression.String()
				}
			default:
				// For other expressions, use a generic name
				columnNames[i] = fmt.Sprintf("column%d", i+1)
			}
		}

		// Use ProjectedResult to select only the requested columns
		cteResult = NewProjectedResult(cteResult, columnNames, stmt.Columns, e.functionRegistry)
	}

	// Apply ORDER BY, LIMIT, OFFSET if needed
	if stmt.OrderBy != nil || stmt.Limit != nil || stmt.Offset != nil {
		var err error
		cteResult, err = applyOrderByLimitOffset(ctx, cteResult, stmt)
		if err != nil {
			return nil, err
		}
	}

	return cteResult, nil
}
