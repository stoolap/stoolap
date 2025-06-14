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
	"reflect"
	"strconv"

	"github.com/stoolap/stoolap/internal/storage"
	"github.com/stoolap/stoolap/internal/storage/mvcc"
)

// MaterializedResult wraps materialized rows as a Result
type MaterializedResult struct {
	columns []string
	rows    []map[string]storage.ColumnValue
	current int
}

// Columns returns the column names
func (r *MaterializedResult) Columns() []string {
	return r.columns
}

// Next advances to the next row
func (r *MaterializedResult) Next() bool {
	r.current++
	return r.current < len(r.rows)
}

// Scan copies column values to the provided destinations
func (r *MaterializedResult) Scan(dest ...interface{}) error {
	if r.current < 0 || r.current >= len(r.rows) {
		return fmt.Errorf("no current row")
	}

	row := r.rows[r.current]
	if len(dest) != len(r.columns) {
		return fmt.Errorf("scan column count mismatch: %d != %d", len(dest), len(r.columns))
	}

	for i, col := range r.columns {
		if val, ok := row[col]; ok {
			if err := mvcc.ScanValue(val, dest[i]); err != nil {
				return err
			}
		} else {
			// Column not found, scan NULL
			if err := mvcc.ScanNull(dest[i]); err != nil {
				return err
			}
		}
	}

	return nil
}

// Row returns the current row as storage.Row
func (r *MaterializedResult) Row() storage.Row {
	if r.current < 0 || r.current >= len(r.rows) {
		return nil
	}

	row := r.rows[r.current]
	result := make([]storage.ColumnValue, len(r.columns))
	for i, col := range r.columns {
		if val, ok := row[col]; ok {
			result[i] = val
		} else {
			result[i] = storage.NewDirectValueFromInterface(nil)
		}
	}
	return result
}

// Close closes the result
func (r *MaterializedResult) Close() error {
	return nil
}

// RowsAffected returns 0 for SELECT
func (r *MaterializedResult) RowsAffected() int64 {
	return 0
}

// LastInsertID returns 0 for SELECT
func (r *MaterializedResult) LastInsertID() int64 {
	return 0
}

// Context returns the context
func (r *MaterializedResult) Context() context.Context {
	return context.Background()
}

// WithAliases returns a new result with column aliases
func (r *MaterializedResult) WithAliases(aliases map[string]string) storage.Result {
	// For now, just return self - aliases are handled elsewhere
	return r
}

// ExecResult represents an execution result
type ExecResult struct {
	rowsAffected int64
	lastInsertID int64
	ctx          context.Context

	// For memory result support
	columns    []string
	rows       [][]interface{}
	currentRow int
	isMemory   bool
}

// IsMemory returns true if this is an in-memory result (for testing)
func (r *ExecResult) IsMemory() bool {
	return r.isMemory
}

// GetRows returns the raw rows (for testing)
func (r *ExecResult) GetRows() [][]interface{} {
	return r.rows
}

// NewExecMemoryResult creates a new in-memory result set
func NewExecMemoryResult(columns []string, rows [][]interface{}, ctx context.Context) *ExecResult {
	return &ExecResult{
		columns:    columns,
		rows:       rows,
		ctx:        ctx,
		currentRow: 0,
		isMemory:   true,
	}
}

// Columns returns the column names in the result
func (r *ExecResult) Columns() []string {
	if r.isMemory {
		return r.columns
	}
	return []string{}
}

// Next moves the cursor to the next row
func (r *ExecResult) Next() bool {
	if r.isMemory {
		r.currentRow++
		return r.currentRow <= len(r.rows)
	}
	return false
}

// Scan scans the current row into the specified variables
func (r *ExecResult) Scan(dest ...interface{}) error {
	if !r.isMemory {
		return errors.New("not a query result")
	}

	if r.currentRow <= 0 || r.currentRow > len(r.rows) {
		return errors.New("no row to scan")
	}

	row := r.rows[r.currentRow-1]
	if len(dest) != len(row) {
		return fmt.Errorf("expected %d destination arguments in Scan, got %d", len(row), len(dest))
	}

	for i, val := range row {
		// Get the pointer to the destination
		destPtr := dest[i]

		// Use the centralized converter if value is a ColumnValue
		if colVal, ok := val.(storage.ColumnValue); ok {
			if err := storage.ScanColumnValueToDestination(colVal, destPtr); err != nil {
				return fmt.Errorf("error scanning column %d: %w", i, err)
			}
			continue
		}

		// For non-ColumnValue types, use the copyValue helper
		if err := copyValue(val, destPtr); err != nil {
			return fmt.Errorf("error scanning column %d: %w", i, err)
		}
	}

	return nil
}

// Close closes the result set
func (r *ExecResult) Close() error {
	return nil
}

// RowsAffected returns the number of rows affected
func (r *ExecResult) RowsAffected() int64 {
	return r.rowsAffected
}

// LastInsertID returns the last insert ID
func (r *ExecResult) LastInsertID() int64 {
	return r.lastInsertID
}

// Context returns the result's context
func (r *ExecResult) Context() context.Context {
	if r.ctx != nil {
		return r.ctx
	}
	return context.Background()
}

// WithAliases sets column aliases for this result
func (r *ExecResult) WithAliases(aliases map[string]string) storage.Result {
	if !r.isMemory {
		// If this isn't a memory result, just return self as there are no columns to alias
		return r
	}

	// For memory results, use NewAliasedResult from storage/v3
	// First, we need to convert this to a storage.Result
	return NewAliasedResult(r, aliases)
}

// Row implements the storage.Result interface
// For ExecResult, this returns direct column values when available
func (r *ExecResult) Row() storage.Row {
	if !r.isMemory || r.currentRow <= 0 || r.currentRow > len(r.rows) {
		// No row available
		return nil
	}

	// Get the current row
	row := r.rows[r.currentRow-1]

	// Check if we already have storage.ColumnValue objects
	result := make(storage.Row, len(row))

	// Convert each value to a storage.ColumnValue if necessary
	for i, val := range row {
		// Check if it's already a ColumnValue
		if cv, ok := val.(storage.ColumnValue); ok {
			result[i] = cv
		} else {
			// Convert based on type
			result[i] = storage.NewDirectValueFromInterface(val)
		}
	}

	return result
}

// copyValue copies a value to a destination pointer
func copyValue(src interface{}, destPtr interface{}) error {
	// Check if the destination is nil
	if destPtr == nil {
		return errors.New("destination pointer is nil")
	}

	// Get the reflect values
	destVal := reflect.ValueOf(destPtr)
	if destVal.Kind() != reflect.Ptr {
		return errors.New("destination is not a pointer")
	}

	// Get the value that the pointer points to
	elemVal := destVal.Elem()

	// Handle nil source
	if src == nil {
		// For nil, just zero out the destination
		elemVal.Set(reflect.Zero(elemVal.Type()))
		return nil
	}

	// Get the source value
	srcVal := reflect.ValueOf(src)

	// Try to convert the source type to the destination type
	if elemVal.Kind() == srcVal.Kind() {
		// Direct assignment for same types
		elemVal.Set(srcVal)
		return nil
	}

	// Handle type conversions
	switch elemVal.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		// Convert to int64
		var intVal int64
		switch v := src.(type) {
		case int:
			intVal = int64(v)
		case int8:
			intVal = int64(v)
		case int16:
			intVal = int64(v)
		case int32:
			intVal = int64(v)
		case int64:
			intVal = v
		case uint:
			intVal = int64(v)
		case uint8:
			intVal = int64(v)
		case uint16:
			intVal = int64(v)
		case uint32:
			intVal = int64(v)
		case uint64:
			intVal = int64(v)
		case float32:
			intVal = int64(v)
		case float64:
			intVal = int64(v)
		case string:
			// Try to parse the string as an int
			i, err := strconv.ParseInt(v, 10, 64)
			if err != nil {
				return fmt.Errorf("cannot convert string %q to int: %v", v, err)
			}
			intVal = i
		default:
			return fmt.Errorf("cannot convert %T to int", src)
		}
		elemVal.SetInt(intVal)

	case reflect.Float32, reflect.Float64:
		// Convert to float64
		var floatVal float64
		switch v := src.(type) {
		case int:
			floatVal = float64(v)
		case int8:
			floatVal = float64(v)
		case int16:
			floatVal = float64(v)
		case int32:
			floatVal = float64(v)
		case int64:
			floatVal = float64(v)
		case uint:
			floatVal = float64(v)
		case uint8:
			floatVal = float64(v)
		case uint16:
			floatVal = float64(v)
		case uint32:
			floatVal = float64(v)
		case uint64:
			floatVal = float64(v)
		case float32:
			floatVal = float64(v)
		case float64:
			floatVal = v
		case string:
			// Try to parse the string as a float
			f, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return fmt.Errorf("cannot convert string %q to float: %v", v, err)
			}
			floatVal = f
		default:
			return fmt.Errorf("cannot convert %T to float", src)
		}
		elemVal.SetFloat(floatVal)

	case reflect.String:
		// Convert to string
		var strVal string
		switch v := src.(type) {
		case string:
			strVal = v
		case []byte:
			strVal = string(v)
		default:
			// Use fmt.Sprint for everything else
			strVal = fmt.Sprint(v)
		}
		elemVal.SetString(strVal)

	case reflect.Bool:
		// Convert to bool
		var boolVal bool
		switch v := src.(type) {
		case bool:
			boolVal = v
		case int:
			boolVal = v != 0
		case int8:
			boolVal = v != 0
		case int16:
			boolVal = v != 0
		case int32:
			boolVal = v != 0
		case int64:
			boolVal = v != 0
		case uint:
			boolVal = v != 0
		case uint8:
			boolVal = v != 0
		case uint16:
			boolVal = v != 0
		case uint32:
			boolVal = v != 0
		case uint64:
			boolVal = v != 0
		case string:
			// Parse string as bool
			b, err := strconv.ParseBool(v)
			if err != nil {
				return fmt.Errorf("cannot convert string %q to bool: %v", v, err)
			}
			boolVal = b
		default:
			return fmt.Errorf("cannot convert %T to bool", src)
		}
		elemVal.SetBool(boolVal)

	default:
		// Try a generic assignment
		if srcVal.Type().ConvertibleTo(elemVal.Type()) {
			elemVal.Set(srcVal.Convert(elemVal.Type()))
			return nil
		}

		return fmt.Errorf("unsupported type conversion from %T to %s", src, elemVal.Type())
	}

	return nil
}
