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

import "github.com/stoolap/stoolap/internal/storage"

// CTEResult wraps a Result to act as a table source
type CTEResult struct {
	name    string
	columns []string
	rows    [][]interface{}
	types   []storage.DataType
}

// NewCTEResult creates a new CTE result from an existing result
func NewCTEResult(name string, result storage.Result) *CTEResult {
	columns := result.Columns()

	// Infer types from first row
	types := make([]storage.DataType, len(columns))
	// Default to TEXT for all columns
	for i := range types {
		types[i] = storage.TEXT
	}

	// TODO: Collect rows from result
	rows := [][]interface{}{}

	return &CTEResult{
		name:    name,
		columns: columns,
		rows:    rows,
		types:   types,
	}
}

// Columns returns the column names
func (c *CTEResult) Columns() []string {
	return c.columns
}

// Next returns the next row
func (c *CTEResult) Next() bool {
	// This is handled by the iterator
	return false
}

// Scan returns the current row values
func (c *CTEResult) Scan(dest ...interface{}) error {
	// This is handled by the iterator
	return nil
}

// Close closes the result
func (c *CTEResult) Close() error {
	// Nothing to close for in-memory result
	return nil
}

// Rows returns all rows
func (c *CTEResult) Rows() [][]interface{} {
	return c.rows
}

// RowsAffected returns 0 for CTE results
func (c *CTEResult) RowsAffected() int64 {
	return 0
}

// LastInsertID returns 0 for CTE results
func (c *CTEResult) LastInsertID() int64 {
	return 0
}

// GetColumnIndex returns the index of a column by name
func (c *CTEResult) GetColumnIndex(name string) (int, bool) {
	for i, col := range c.columns {
		if col == name {
			return i, true
		}
	}
	return -1, false
}

// GetValueType returns the type of a column
func (c *CTEResult) GetValueType(index int) storage.DataType {
	if index >= 0 && index < len(c.types) {
		return c.types[index]
	}
	return storage.TEXT
}
