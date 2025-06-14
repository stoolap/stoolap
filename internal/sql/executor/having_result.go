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

// HavingFilteredResult is a specialized result that filters aggregated rows based on a HAVING clause
type HavingFilteredResult struct {
	result     storage.Result
	havingExpr parser.Expression
	evaluator  *Evaluator

	// Current state
	currentRow   storage.Row
	currentValid bool
}

// Columns returns the column names from the underlying result
func (h *HavingFilteredResult) Columns() []string {
	return h.result.Columns()
}

// Next advances to the next row that matches the HAVING clause
func (h *HavingFilteredResult) Next() bool {
	for h.result.Next() {
		// Get the current row as a map
		row := h.result.Row()
		if row == nil {
			continue
		}

		// Build column map
		columns := h.result.Columns()
		rowMap := make(map[string]storage.ColumnValue, len(columns))
		for i, col := range columns {
			if i < len(row) {
				rowMap[col] = row[i]

				// Also add reverse alias mapping for HAVING clause
				// If this column has an alias that maps to an aggregate expression,
				// add the aggregate expression as a key too
				if h.evaluator.columnAliases != nil {
					// Check if this column (e.g., "total") maps to an aggregate (e.g., "SUM(amount)")
					if aggExpr, ok := h.evaluator.columnAliases[col]; ok {
						rowMap[aggExpr] = row[i]
					}
				}
			}
		}

		// Evaluate HAVING clause
		match, err := h.evaluator.EvaluateHavingClause(h.havingExpr, rowMap)
		if err != nil {
			// Skip rows that cause evaluation errors
			continue
		}

		if match {
			h.currentRow = row
			h.currentValid = true
			return true
		}
	}

	h.currentValid = false
	return false
}

// Scan copies column values to the provided destinations
func (h *HavingFilteredResult) Scan(dest ...interface{}) error {
	if !h.currentValid {
		return fmt.Errorf("no current row")
	}
	return h.result.Scan(dest...)
}

// Row returns the current row
func (h *HavingFilteredResult) Row() storage.Row {
	if !h.currentValid {
		return nil
	}
	return h.currentRow
}

// Close closes the result
func (h *HavingFilteredResult) Close() error {
	return h.result.Close()
}

// RowsAffected returns the number of rows affected
func (h *HavingFilteredResult) RowsAffected() int64 {
	return h.result.RowsAffected()
}

// LastInsertID returns the last insert ID
func (h *HavingFilteredResult) LastInsertID() int64 {
	return h.result.LastInsertID()
}

// Context returns the context
func (h *HavingFilteredResult) Context() context.Context {
	return h.result.Context()
}

// WithAliases returns a new result with column aliases
func (h *HavingFilteredResult) WithAliases(aliases map[string]string) storage.Result {
	h.result = h.result.WithAliases(aliases)
	return h
}
