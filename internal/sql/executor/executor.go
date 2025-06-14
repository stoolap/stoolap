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
	"database/sql/driver"
	"errors"
	"fmt"
	"strings"

	"github.com/stoolap/stoolap/internal/common"
	"github.com/stoolap/stoolap/internal/functions/contract"
	"github.com/stoolap/stoolap/internal/functions/registry"
	"github.com/stoolap/stoolap/internal/parser"
	"github.com/stoolap/stoolap/internal/sql/executor/vectorized"
	"github.com/stoolap/stoolap/internal/storage"
)

// Executor executes SQL statements
type Executor struct {
	engine           storage.Engine
	functionRegistry contract.FunctionRegistry
	queryCache       *QueryCache

	vectorizedMode bool // Whether to use vectorized execution when appropriate

	// Connection-level default isolation level
	defaultIsolationLevel storage.IsolationLevel

	// CTE registry for the current query execution
	cteRegistry *CTERegistry
}

// parameterContextKey is the context key for parameter
var psContextKey = interface{}("sql:parameter")

// NewExecutor creates a new SQL executor
func NewExecutor(engine storage.Engine) *Executor {
	// Get the global function registry
	registry := registry.GetGlobal()

	// Create a query cache with a default size of 200 entries
	// This is a reasonable size for most applications, but can be made configurable
	queryCache := NewQueryCache(1000)

	return &Executor{
		engine:                engine,
		functionRegistry:      registry,
		queryCache:            queryCache,
		vectorizedMode:        false,                 // Disabled by default, can be enabled with EnableVectorizedMode
		defaultIsolationLevel: storage.ReadCommitted, // Default to READ COMMITTED
	}
}

// EnableVectorizedMode enables vectorized execution for appropriate query types
func (e *Executor) EnableVectorizedMode() {
	e.vectorizedMode = true
}

// GetDefaultIsolationLevel returns the connection's default isolation level
func (e *Executor) GetDefaultIsolationLevel() storage.IsolationLevel {
	return e.defaultIsolationLevel
}

// DisableVectorizedMode disables vectorized execution
func (e *Executor) DisableVectorizedMode() {
	e.vectorizedMode = false
}

// IsVectorizedModeEnabled returns whether vectorized execution is enabled
func (e *Executor) IsVectorizedModeEnabled() bool {
	return e.vectorizedMode
}

// GetVectorizedStatus returns a string indicating the current vectorized mode status
// This is useful for debugging and testing
func (e *Executor) GetVectorizedStatus() string {
	if e.vectorizedMode {
		return "Vectorized mode is enabled"
	}
	return "Vectorized mode is disabled"
}

// executeSet handles SET statements for session variables
func (e *Executor) executeSet(stmt *parser.SetStatement) error {
	// We only support a few specific session variables at this time
	varName := strings.ToUpper(stmt.Name.Value)

	switch varName {
	case "ISOLATIONLEVEL":
		// Handle isolation level setting
		// Get the value as string
		value, ok := stmt.Value.(*parser.StringLiteral)
		if !ok {
			return fmt.Errorf("isolation level must be a string value")
		}

		level := strings.ToUpper(value.Value)

		// Map to the appropriate isolation level
		var isolationLevel storage.IsolationLevel
		switch level {
		case "READ COMMITTED":
			isolationLevel = storage.ReadCommitted
		case "SNAPSHOT", "REPEATABLE READ":
			isolationLevel = storage.SnapshotIsolation
		default:
			return fmt.Errorf("unsupported isolation level: %s", level)
		}

		// Set the isolation level for this connection/executor only
		e.defaultIsolationLevel = isolationLevel
		return nil
	case "VECTORIZED":
		// Extract the value
		var boolValue bool

		switch value := stmt.Value.(type) {
		case *parser.BooleanLiteral:
			boolValue = value.Value
		case *parser.IntegerLiteral:
			boolValue = value.Value != 0
		case *parser.Identifier:
			upperValue := strings.ToUpper(value.Value)
			if upperValue == "ON" || upperValue == "TRUE" {
				boolValue = true
			} else if upperValue == "OFF" || upperValue == "FALSE" {
				boolValue = false
			} else {
				return fmt.Errorf("invalid value for VECTORIZED: %s (expected TRUE/FALSE, ON/OFF, or 1/0)", value.Value)
			}
		case *parser.StringLiteral:
			upperValue := strings.ToUpper(value.Value)
			if upperValue == "ON" || upperValue == "TRUE" || upperValue == "1" {
				boolValue = true
			} else if upperValue == "OFF" || upperValue == "FALSE" || upperValue == "0" {
				boolValue = false
			} else {
				return fmt.Errorf("invalid value for VECTORIZED: %s (expected TRUE/FALSE, ON/OFF, or 1/0)", value.Value)
			}
		default:
			return fmt.Errorf("invalid value type for VECTORIZED: %T (expected boolean, integer, or string)", stmt.Value)
		}

		// Set the vectorized mode based on the extracted value
		if boolValue {
			e.EnableVectorizedMode()
		} else {
			e.DisableVectorizedMode()
		}

		return nil
	default:
		return fmt.Errorf("unknown session variable: %s", stmt.Name.Value)
	}
}

// executeShowTables returns a list of all tables in the database
func (e *Executor) executeShowTables(ctx context.Context, tx storage.Transaction) (storage.Result, error) {
	// Get list of tables from the transaction
	tables, err := tx.ListTables()
	if err != nil {
		return nil, err
	}

	// Create a result with one column named "Tables"
	columns := []string{"Tables"}

	// Convert table names to rows
	rows := make([][]interface{}, len(tables))
	for i, tableName := range tables {
		rows[i] = []interface{}{tableName}
	}

	// Return as a result object
	return &ExecResult{
		columns:    columns,
		rows:       rows,
		ctx:        ctx,
		currentRow: 0,
		isMemory:   true,
	}, nil
}

// executeShowCreateTable returns the CREATE TABLE statement for a table
func (e *Executor) executeShowCreateTable(ctx context.Context, tx storage.Transaction, stmt *parser.ShowCreateTableStatement) (storage.Result, error) {
	tableName := stmt.TableName.Value

	// Get table schema
	table, err := tx.GetTable(tableName)
	if err != nil {
		return nil, err
	}

	// Get CREATE TABLE statement as string
	createTableStmt, err := e.generateCreateTableStatement(table)
	if err != nil {
		return nil, err
	}

	// Create a result with two columns: "Table" and "Create Table"
	columns := []string{"Table", "Create Table"}

	// Create a single row with table name and CREATE TABLE statement
	rows := [][]interface{}{{tableName, createTableStmt}}

	// Return as a result object
	return &ExecResult{
		columns:  columns,
		rows:     rows,
		ctx:      ctx,
		isMemory: true,
	}, nil
}

// executeShowIndexes returns information about indexes on a table
func (e *Executor) executeShowIndexes(ctx context.Context, tx storage.Transaction, stmt *parser.ShowIndexesStatement) (storage.Result, error) {
	tableName := stmt.TableName.Value

	// Get table
	_, err := tx.GetTable(tableName)
	if err != nil {
		return nil, err
	}

	// Create columns for the result
	columns := []string{"Table", "Index Name", "Column Name(s)", "Type", "Unique"}

	// We'll build the result rows directly by getting all indexes from the engine
	rows := make([][]interface{}, 0)

	// Get all indexes directly from the engine, bypassing the index name mapping
	allIndexes, err := e.engine.GetAllIndexes(tableName)
	if err != nil {
		// Fall back to the older ListTableIndexes method if GetAllIndexes isn't implemented
		// but modify our approach to handle its limitations
		indexes, err := e.engine.ListTableIndexes(tableName)
		if err != nil {
			return nil, err
		}

		// Process the index names to extract more information
		for indexName, colName := range indexes {
			// Try to get the actual index to get complete information
			index, err := e.engine.GetIndex(tableName, indexName)
			if err == nil && index != nil {
				// Use the real index properties
				indexType := string(index.IndexType())
				isUnique := index.IsUnique()

				// Get all column names for this index
				columnNames := index.ColumnNames()
				allColumnsStr := strings.Join(columnNames, ", ")

				// Add a single row for the index with all columns joined
				rows = append(rows, []interface{}{
					tableName,
					indexName,
					allColumnsStr,
					indexType,
					isUnique,
				})
			} else {
				// Fallback to name-based heuristics if we can't get the actual index
				indexType := "BTREE" // Default type
				isUnique := false

				// Check for columnar indexes - they are usually prefixed with "columnar_"
				if strings.HasPrefix(indexName, "columnar_") {
					indexType = "COLUMNAR"
				} else if strings.HasPrefix(indexName, "unique_") {
					// Unique indexes often have a "unique_" prefix
					isUnique = true
					if strings.HasPrefix(indexName, "unique_columnar_") {
						indexType = "COLUMNAR"
					}
				}

				// Add a single row for this index
				rows = append(rows, []interface{}{
					tableName,
					indexName,
					colName,
					indexType,
					isUnique,
				})
			}
		}
	} else {
		// Process all indexes from GetAllIndexes
		for _, index := range allIndexes {
			indexName := index.Name()
			indexType := string(index.IndexType())
			isUnique := index.IsUnique()

			// Get all column names for this index
			columnNames := index.ColumnNames()

			// Create a comma-separated string of all column names
			allColumnsStr := strings.Join(columnNames, ", ")

			// Add a single row for this index with all columns joined
			rows = append(rows, []interface{}{
				tableName,
				indexName,
				allColumnsStr,
				indexType,
				isUnique,
			})
		}
	}

	// Return as a result object
	return &ExecResult{
		columns:  columns,
		rows:     rows,
		ctx:      ctx,
		isMemory: true,
	}, nil
}

// generateCreateTableStatement generates a CREATE TABLE statement for a table
func (e *Executor) generateCreateTableStatement(table storage.Table) (string, error) {
	var sb strings.Builder

	sb.WriteString("CREATE TABLE ")
	sb.WriteString(table.Name())
	sb.WriteString(" (")

	schema := table.Schema()

	// Add columns
	for i, col := range schema.Columns {
		if i > 0 {
			sb.WriteString(", ")
		}

		// Column name
		sb.WriteString("")
		sb.WriteString(col.Name)
		sb.WriteString(" ")

		// Column type
		sb.WriteString(e.getDataTypeString(col.Type))

		// Constraints
		if col.PrimaryKey {
			sb.WriteString(" PRIMARY KEY")
		}
		if !col.Nullable && !col.PrimaryKey {
			sb.WriteString(" NOT NULL")
		}
	}

	sb.WriteString(")")

	return sb.String(), nil
}

// getDataTypeString converts a storage data type to a SQL type string
func (e *Executor) getDataTypeString(dataType storage.DataType) string {
	switch dataType {
	case storage.TypeInteger:
		return "INTEGER"
	case storage.TypeFloat:
		return "FLOAT"
	case storage.TypeBoolean:
		return "BOOLEAN"
	case storage.TypeString:
		return "TEXT"
	case storage.TypeTimestamp:
		return "TIMESTAMP"
	case storage.TypeJSON:
		return "JSON"
	case storage.TypeNull:
		return "NULL"
	default:
		return "UNKNOWN"
	}
}

// shouldUseVectorizedExecution determines if vectorized execution should be used for a query
// This is a key decision point that integrates with the vectorized execution engine
func (e *Executor) shouldUseVectorizedExecution(stmt parser.Statement) bool {
	// If vectorized mode is disabled, never use it
	if !e.vectorizedMode {
		return false
	}

	// Only use vectorized execution for SELECT statements for now
	selectStmt, isSelect := stmt.(*parser.SelectStatement)
	if !isSelect {
		return false
	}

	// For simple point queries or very small result sets, row-based execution is faster
	// Check for optimization hints like estimated row count or query complexity

	// Future optimization: Add size-based heuristics
	// - For queries with small result sets (<100 rows), row-based execution is often faster
	// - For queries with large result sets (>10,000 rows), vectorized execution has clear benefits

	// Check if query has complex computations that benefit from vectorized execution
	hasComplexComputation := false

	// Check for arithmetic expressions in SELECT list
	// These are ideal candidates for SIMD optimization
	for _, col := range selectStmt.Columns {
		if isMathExpression(col) {
			hasComplexComputation = true
			break
		}
	}

	// Check for complex expressions in WHERE clause
	// Filtering with complex conditions can also benefit from vectorization
	if selectStmt.Where != nil {
		if isMathExpression(selectStmt.Where) {
			hasComplexComputation = true
		}
	}

	// Future enhancements:
	// 1. Consider ORDER BY clauses with complex expressions
	// 2. Analyze JOIN conditions for vectorization potential
	// 3. Check for window functions that could benefit from vectorization
	// 4. Examine aggregate functions with math expressions

	return hasComplexComputation
}

// isMathExpression checks if an expression contains arithmetic operations
func isMathExpression(expr parser.Expression) bool {
	switch e := expr.(type) {
	case *parser.InfixExpression:
		// Check for arithmetic operators
		op := e.Operator
		if op == "+" || op == "-" || op == "*" || op == "/" || op == "%" {
			return true
		}

		// Recursively check both sides
		return isMathExpression(e.Left) || isMathExpression(e.Right)

	case *parser.FunctionCall:
		// Common math functions
		name := strings.ToLower(e.Function)
		mathFunctions := []string{"abs", "round", "floor", "ceiling", "sqrt", "power", "mod"}
		for _, fn := range mathFunctions {
			if name == fn {
				return true
			}
		}

		// Check arguments
		for _, arg := range e.Arguments {
			if isMathExpression(arg) {
				return true
			}
		}

	case *parser.AliasedExpression:
		// Check the underlying expression
		return isMathExpression(e.Expression)
	}

	return false
}

// executePragma handles PRAGMA statements for database configuration
func (e *Executor) executePragma(stmt *parser.PragmaStatement) (storage.Result, error) {
	// Get the pragma name
	pragmaName := strings.ToUpper(stmt.Name.Value)

	// Get the current configuration
	config := e.engine.GetConfig()

	// Structure to hold the result
	var result *ExecResult

	// Handle different PRAGMA settings
	switch pragmaName {
	case "VERSION":
		if stmt.Value == nil {
			// This is a read operation, return current value
			result = &ExecResult{
				columns:    []string{"version"},
				rows:       [][]interface{}{{common.VersionString}},
				isMemory:   true,
				ctx:        context.Background(),
				currentRow: 0,
			}
		}
	case "SNAPSHOT_INTERVAL":
		// Handle snapshot_interval setting
		if stmt.Value == nil {
			// This is a read operation, return current value
			result = &ExecResult{
				columns:    []string{"snapshot_interval"},
				rows:       [][]interface{}{{config.Persistence.SnapshotInterval}},
				isMemory:   true,
				ctx:        context.Background(),
				currentRow: 0,
			}
		} else {
			// This is a write operation, set new value
			var newInterval int
			switch value := stmt.Value.(type) {
			case *parser.IntegerLiteral:
				newInterval = int(value.Value)
			case *parser.FloatLiteral:
				newInterval = int(value.Value)
			default:
				return nil, fmt.Errorf("snapshot_interval value must be an integer")
			}

			// Validate the value
			if newInterval <= 0 {
				return nil, fmt.Errorf("snapshot_interval must be positive")
			}

			// Update the config
			config.Persistence.SnapshotInterval = newInterval

			// Apply the configuration change
			if err := e.engine.UpdateConfig(config); err != nil {
				return nil, fmt.Errorf("failed to update snapshot_interval: %v", err)
			}

			// Return the new value
			result = &ExecResult{
				columns:    []string{"snapshot_interval"},
				rows:       [][]interface{}{{newInterval}},
				isMemory:   true,
				ctx:        context.Background(),
				currentRow: 0,
			}
		}
	case "KEEP_SNAPSHOTS":
		// Handle keep_snapshots setting
		if stmt.Value == nil {
			// This is a read operation, return current value
			result = &ExecResult{
				columns:    []string{"keep_snapshots"},
				rows:       [][]interface{}{{config.Persistence.KeepSnapshots}},
				isMemory:   true,
				ctx:        context.Background(),
				currentRow: 0,
			}
		} else {
			// This is a write operation, set new value
			var newValue int
			switch value := stmt.Value.(type) {
			case *parser.IntegerLiteral:
				newValue = int(value.Value)
			case *parser.FloatLiteral:
				newValue = int(value.Value)
			default:
				return nil, fmt.Errorf("keep_snapshots value must be an integer")
			}

			// Validate the value
			if newValue <= 0 {
				return nil, fmt.Errorf("keep_snapshots must be positive")
			}

			// Update the config
			config.Persistence.KeepSnapshots = newValue

			// Apply the configuration change
			if err := e.engine.UpdateConfig(config); err != nil {
				return nil, fmt.Errorf("failed to update keep_snapshots: %v", err)
			}

			// Return the new value
			result = &ExecResult{
				columns:    []string{"keep_snapshots"},
				rows:       [][]interface{}{{newValue}},
				isMemory:   true,
				ctx:        context.Background(),
				currentRow: 0,
			}
		}
	case "SYNC_MODE":
		// Handle sync_mode setting
		if stmt.Value == nil {
			// This is a read operation, return current value
			result = &ExecResult{
				columns:    []string{"sync_mode"},
				rows:       [][]interface{}{{config.Persistence.SyncMode}},
				isMemory:   true,
				ctx:        context.Background(),
				currentRow: 0,
			}
		} else {
			// This is a write operation, set new value
			var newValue int
			switch value := stmt.Value.(type) {
			case *parser.IntegerLiteral:
				newValue = int(value.Value)
			case *parser.FloatLiteral:
				newValue = int(value.Value)
			default:
				return nil, fmt.Errorf("sync_mode value must be an integer (0=None, 1=Normal, 2=Full)")
			}

			// Validate the value
			if newValue < 0 || newValue > 2 {
				return nil, fmt.Errorf("sync_mode must be 0 (None), 1 (Normal), or 2 (Full)")
			}

			// Update the config
			config.Persistence.SyncMode = newValue

			// Apply the configuration change
			if err := e.engine.UpdateConfig(config); err != nil {
				return nil, fmt.Errorf("failed to update sync_mode: %v", err)
			}

			// Return the new value
			result = &ExecResult{
				columns:    []string{"sync_mode"},
				rows:       [][]interface{}{{newValue}},
				isMemory:   true,
				ctx:        context.Background(),
				currentRow: 0,
			}
		}
	case "WAL_FLUSH_TRIGGER":
		// Handle wal_flush_trigger setting
		if stmt.Value == nil {
			// This is a read operation, return current value
			result = &ExecResult{
				columns:    []string{"wal_flush_trigger"},
				rows:       [][]interface{}{{config.Persistence.WALFlushTrigger}},
				isMemory:   true,
				ctx:        context.Background(),
				currentRow: 0,
			}
		} else {
			// This is a write operation, set new value
			var newValue int
			switch value := stmt.Value.(type) {
			case *parser.IntegerLiteral:
				newValue = int(value.Value)
			case *parser.FloatLiteral:
				newValue = int(value.Value)
			default:
				return nil, fmt.Errorf("wal_flush_trigger value must be an integer")
			}

			// Validate the value
			if newValue <= 0 {
				return nil, fmt.Errorf("wal_flush_trigger must be positive")
			}

			// Update the config
			config.Persistence.WALFlushTrigger = newValue

			// Apply the configuration change
			if err := e.engine.UpdateConfig(config); err != nil {
				return nil, fmt.Errorf("failed to update wal_flush_trigger: %v", err)
			}

			// Return the new value
			result = &ExecResult{
				columns:    []string{"wal_flush_trigger"},
				rows:       [][]interface{}{{newValue}},
				isMemory:   true,
				ctx:        context.Background(),
				currentRow: 0,
			}
		}
	case "SNAPSHOT":
		// Handle SNAPSHOT command - creates a manual snapshot
		if stmt.Value != nil {
			return nil, fmt.Errorf("PRAGMA SNAPSHOT does not accept values")
		}

		// Create a snapshot
		if err := e.engine.CreateSnapshot(); err != nil {
			return nil, fmt.Errorf("failed to create snapshot: %v", err)
		}

		// Return success result
		result = &ExecResult{
			columns:    []string{"result"},
			rows:       [][]interface{}{{"Snapshot created successfully"}},
			isMemory:   true,
			ctx:        context.Background(),
			currentRow: 0,
		}
	default:
		return nil, fmt.Errorf("unknown PRAGMA setting: %s", pragmaName)
	}

	return result, nil
}

// Execute executes a SQL statement
func (e *Executor) Execute(ctx context.Context, tx storage.Transaction, query string) (storage.Result, error) {
	// Execute without parameters
	result, err := e.ExecuteWithParams(ctx, tx, query, nil)
	return result, err
}

// executeWithVectorizedEngine attempts to execute a statement using the vectorized engine
// This is the integration point where the main executor delegates to the vectorized engine
func (e *Executor) executeWithVectorizedEngine(ctx context.Context, tx storage.Transaction, stmt *parser.SelectStatement) (storage.Result, error) {
	// Sanity check - ensure vectorized mode is enabled
	if !e.vectorizedMode {
		return nil, fmt.Errorf("vectorized mode is not enabled - this is likely a bug in the query engine")
	}

	// Create vectorized engine instance with the function registry
	// This allows the vectorized engine to evaluate functions in a vectorized manner
	vectorEngine := vectorized.NewEngine(e.functionRegistry)

	// Try to execute with vectorized engine
	// The execution flow:
	// 1. vectorEngine extracts data from the storage layer
	// 2. Converts row-based data to columnar format (in Batch objects)
	// 3. Applies vectorized operations using optimized SIMD functions
	// 4. Converts results back to row-based format for the SQL interface
	result, err := vectorEngine.ExecuteQuery(ctx, tx, stmt)
	if err != nil {
		// If the vectorized engine fails, fall back to the regular execution
		// Potential failures could be from unsupported operations or data types
		return nil, err
	}

	// Return the result to the caller
	// The result is a VectorizedResult object that implements the storage.Result interface
	return result, nil
}

// getStatementFromCache tries to get a statement from the cache, or parses and adds it if not found
// It properly clones any cached statement to prevent modifications to the cached copy
func (e *Executor) getStatementFromCache(query string, params []driver.NamedValue) (parser.Statement, error) {
	// Try to get the cached plan
	cachedPlan := e.queryCache.Get(query)

	if cachedPlan != nil {
		// Check if the correct number of parameters are provided
		if cachedPlan.HasParams && len(params) < cachedPlan.ParamCount {
			return nil, fmt.Errorf("not enough parameters: query requires %d parameters but only %d were provided",
				cachedPlan.ParamCount, len(params))
		}

		return cachedPlan.Statement, nil
	}

	l := parser.NewLexer(query)
	p := parser.NewParser(l)

	program := p.ParseProgram()

	if len(p.Errors()) > 0 {
		// Use the new formatted error message for better readability
		return nil, errors.New(p.FormatErrors())
	}

	if len(program.Statements) == 0 {
		return nil, errors.New("no statements found in query")
	}

	stmt := program.Statements[0]

	// Count parameter placeholders
	paramCount := p.GetParameterCount()

	if paramCount != len(params) {
		return nil, fmt.Errorf("parameter count mismatch: query requires %d parameters but got %d",
			paramCount, len(params))
	}

	// Cache the parsed statement
	hasParams := len(params) > 0
	e.queryCache.Put(query, stmt, hasParams, paramCount)

	return stmt, nil
}

// ExecuteWithParams executes a SQL statement with parameters
func (e *Executor) ExecuteWithParams(ctx context.Context, tx storage.Transaction, query string, params []driver.NamedValue) (storage.Result, error) {
	var shouldAutoCommit bool
	var err error

	if tx == nil {
		// Begin a transaction with the executor's default isolation level
		sqlLevel := sql.LevelReadCommitted
		if e.defaultIsolationLevel == storage.SnapshotIsolation {
			sqlLevel = sql.LevelSnapshot
		}

		tx, err = e.engine.BeginTx(ctx, sqlLevel)
		if err != nil {
			return nil, err
		}
		shouldAutoCommit = true
	}

	var commitErr error
	defer func() {
		if shouldAutoCommit {
			if err == nil {
				// Only commit if there was no error
				commitErr = tx.Commit()
				if commitErr != nil {
					// If commit fails, roll back
					_ = tx.Rollback()
				}
			} else {
				// If there was an error, roll back
				_ = tx.Rollback()
			}
		}
	}()

	// Get statement from cache or parse it
	stmt, err := e.getStatementFromCache(query, params)
	if err != nil {
		return nil, err
	}

	// Bind parameters if provided
	if len(params) > 0 {
		ps, err := newParameter(params)
		if err != nil {
			return nil, err
		}
		ctx = context.WithValue(ctx, psContextKey, ps)
	}

	// Execute based on statement type
	var result storage.Result
	switch s := stmt.(type) {
	case *parser.CreateTableStatement:
		err = e.executeCreateTable(tx, s)
		if err == nil {
			result = &ExecResult{
				rowsAffected: 0,
				lastInsertID: 0,
				ctx:          ctx,
			}
		}
	case *parser.DropTableStatement:
		err = e.executeDropTable(tx, s)
		if err == nil {
			result = &ExecResult{
				rowsAffected: 0,
				lastInsertID: 0,
				ctx:          ctx,
			}
		}
	case *parser.CreateIndexStatement:
		err = e.executeCreateIndex(tx, s)
		if err == nil {
			result = &ExecResult{
				rowsAffected: 0,
				lastInsertID: 0,
				ctx:          ctx,
			}
		}
	case *parser.DropIndexStatement:
		err = e.executeDropIndex(tx, s)
		if err == nil {
			result = &ExecResult{
				rowsAffected: 0,
				lastInsertID: 0,
				ctx:          ctx,
			}
		}
	case *parser.CreateColumnarIndexStatement:
		err = e.executeCreateColumnarIndex(tx, s)
		if err == nil {
			result = &ExecResult{
				rowsAffected: 0,
				lastInsertID: 0,
				ctx:          ctx,
			}
		}
	case *parser.DropColumnarIndexStatement:
		err = e.executeDropColumnarIndex(tx, s)
		if err == nil {
			result = &ExecResult{
				rowsAffected: 0,
				lastInsertID: 0,
				ctx:          ctx,
			}
		}
	case *parser.AlterTableStatement:
		err = e.executeAlterTable(tx, s)
		if err == nil {
			result = &ExecResult{
				rowsAffected: 0,
				lastInsertID: 0,
				ctx:          ctx,
			}
		}
	case *parser.InsertStatement:
		var rowsAffected, lastInsertID int64
		rowsAffected, lastInsertID, err = e.executeInsertWithContext(ctx, tx, s)
		if err == nil {
			result = &ExecResult{
				rowsAffected: rowsAffected,
				lastInsertID: lastInsertID,
				ctx:          ctx,
			}
		}
	case *parser.UpdateStatement:
		var rowsAffected int64
		rowsAffected, err = e.executeUpdateWithContext(ctx, tx, s)
		if err == nil {
			result = &ExecResult{
				rowsAffected: rowsAffected,
				lastInsertID: 0,
				ctx:          ctx,
			}
		}
	case *parser.DeleteStatement:
		var rowsAffected int64
		rowsAffected, err = e.executeDeleteWithContext(ctx, tx, s)
		if err == nil {
			result = &ExecResult{
				rowsAffected: rowsAffected,
				lastInsertID: 0,
				ctx:          ctx,
			}
		}
	case *parser.SelectStatement:
		result, err = e.executeSelectWithContext(ctx, tx, s)
	case *parser.BeginStatement:
		// Begin is a no-op because we already started a transaction
		// However, if an isolation level was specified, we should set it
		var isolationLevel storage.IsolationLevel
		if s.IsolationLevel != "" {
			// Map the SQL isolation level string to our storage.IsolationLevel
			switch strings.ToUpper(s.IsolationLevel) {
			case "READ COMMITTED":
				isolationLevel = storage.ReadCommitted
			case "REPEATABLE READ", "SNAPSHOT":
				isolationLevel = storage.SnapshotIsolation
			default:
				err = fmt.Errorf("unsupported isolation level: %s", s.IsolationLevel)
				return nil, err
			}

			// Set the isolation level at the engine level
			if err = tx.SetIsolationLevel(isolationLevel); err != nil {
				return nil, fmt.Errorf("failed to set isolation level: %v", err)
			}
		}

		result = &ExecResult{
			rowsAffected: 0,
			lastInsertID: 0,
			ctx:          ctx,
		}
	case *parser.CommitStatement:
		err = tx.Commit()
		if err == nil {
			shouldAutoCommit = false
			result = &ExecResult{
				rowsAffected: 0,
				lastInsertID: 0,
				ctx:          ctx,
			}
		}
	case *parser.RollbackStatement:
		err = tx.Rollback()
		if err == nil {
			shouldAutoCommit = false // We've handled the rollback
			result = &ExecResult{
				rowsAffected: 0,
				lastInsertID: 0,
				ctx:          ctx,
			}
		}
	case *parser.SetStatement:
		err = e.executeSet(stmt.(*parser.SetStatement))
		if err == nil {
			// SET statements are handled immediately and don't need an explicit commit
			// But we can simply auto-commit than transaction can close
			result = &ExecResult{
				rowsAffected: 0,
				lastInsertID: 0,
				ctx:          ctx,
			}
		}
	case *parser.PragmaStatement:
		// PRAGMA statements are handled differently as they're related to database settings
		// rather than data operations, and they don't need a transaction
		result, err = e.executePragma(s)
		if err == nil {
			// We've handled the PRAGMA command, no transaction-specific operations needed
			shouldAutoCommit = true
		}
	case *parser.ShowTablesStatement:
		// Execute SHOW TABLES statement
		result, err = e.executeShowTables(ctx, tx)
	case *parser.ShowCreateTableStatement:
		// Execute SHOW CREATE TABLE statement
		result, err = e.executeShowCreateTable(ctx, tx, s)
	case *parser.ShowIndexesStatement:
		// Execute SHOW INDEXES statement
		result, err = e.executeShowIndexes(ctx, tx, s)
	default:
		err = fmt.Errorf("unsupported statement type: %T", stmt)
	}

	if err != nil {
		return nil, err
	}

	return result, nil
}
