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

	"github.com/stoolap/stoolap/internal/functions/contract"
	"github.com/stoolap/stoolap/internal/parser"
	"github.com/stoolap/stoolap/internal/storage"
)

// ProjectedResult wraps a result to handle projection of literals and expressions
// This is used when the SELECT list contains literals like SELECT 1, 'hello', etc.
type ProjectedResult struct {
	baseResult  storage.Result
	columns     []string                  // The expected column names
	expressions []parser.Expression       // The original SELECT expressions
	registry    contract.FunctionRegistry // For evaluating expressions
	currentRow  storage.Row               // Current projected row
	baseColumns []string                  // Columns from the base result
	evaluator   *Evaluator                // For evaluating expressions
}

// NewProjectedResult creates a new ProjectedResult
func NewProjectedResult(baseResult storage.Result, columns []string, expressions []parser.Expression, registry contract.FunctionRegistry) storage.Result {
	return &ProjectedResult{
		baseResult:  baseResult,
		columns:     columns,
		expressions: expressions,
		registry:    registry,
		baseColumns: baseResult.Columns(),
		evaluator:   NewEvaluator(context.Background(), registry),
	}
}

// Columns returns the column names including projected literals
func (p *ProjectedResult) Columns() []string {
	return p.columns
}

// Next advances to the next row
func (p *ProjectedResult) Next() bool {
	if !p.baseResult.Next() {
		return false
	}

	// Build the projected row
	p.currentRow = make(storage.Row, len(p.columns))
	baseRow := p.baseResult.Row()

	// Create a map of base column values for expression evaluation
	rowData := make(map[string]storage.ColumnValue)
	for i, colName := range p.baseColumns {
		if i < len(baseRow) {
			rowData[colName] = baseRow[i]
		}
	}

	// Set the row data for the evaluator
	p.evaluator.WithRow(rowData)

	// Process each expression in the SELECT list
	for i, expr := range p.expressions {
		switch e := expr.(type) {
		case *parser.Identifier:
			// This is a column reference - find it in the base row
			found := false
			for j, baseCol := range p.baseColumns {
				if baseCol == e.Value {
					if j < len(baseRow) {
						p.currentRow[i] = baseRow[j]
						found = true
						break
					}
				}
			}
			if !found {
				p.currentRow[i] = storage.StaticNullUnknown
			}

		case *parser.IntegerLiteral:
			p.currentRow[i] = storage.NewIntegerValue(e.Value)

		case *parser.FloatLiteral:
			p.currentRow[i] = storage.NewFloatValue(e.Value)

		case *parser.StringLiteral:
			p.currentRow[i] = storage.NewStringValue(e.Value)

		case *parser.BooleanLiteral:
			p.currentRow[i] = storage.NewBooleanValue(e.Value)

		case *parser.NullLiteral:
			p.currentRow[i] = storage.StaticNullUnknown

		case *parser.AliasedExpression:
			// Evaluate the inner expression
			switch inner := e.Expression.(type) {
			case *parser.IntegerLiteral:
				p.currentRow[i] = storage.NewIntegerValue(inner.Value)
			case *parser.FloatLiteral:
				p.currentRow[i] = storage.NewFloatValue(inner.Value)
			case *parser.StringLiteral:
				p.currentRow[i] = storage.NewStringValue(inner.Value)
			case *parser.BooleanLiteral:
				p.currentRow[i] = storage.NewBooleanValue(inner.Value)
			case *parser.NullLiteral:
				p.currentRow[i] = storage.StaticNullUnknown
			case *parser.Identifier:
				// This is an aliased column reference
				found := false
				for j, baseCol := range p.baseColumns {
					if baseCol == inner.Value {
						if j < len(baseRow) {
							p.currentRow[i] = baseRow[j]
							found = true
							break
						}
					}
				}
				if !found {
					p.currentRow[i] = storage.StaticNullUnknown
				}
			case *parser.InfixExpression:
				// For comparison expressions, evaluate them
				result, err := p.evaluator.Evaluate(inner)
				if err != nil {
					p.currentRow[i] = storage.StaticNullUnknown
				} else {
					p.currentRow[i] = result
				}
			default:
				// For other expressions, use the evaluator
				result, err := p.evaluator.Evaluate(inner)
				if err != nil {
					p.currentRow[i] = storage.StaticNullUnknown
				} else {
					p.currentRow[i] = result
				}
			}

		case *parser.InfixExpression:
			// For comparison and other infix expressions, evaluate them
			result, err := p.evaluator.Evaluate(e)
			if err != nil {
				p.currentRow[i] = storage.StaticNullUnknown
			} else {
				p.currentRow[i] = result
			}

		default:
			// For other expressions, use the evaluator
			result, err := p.evaluator.Evaluate(expr)
			if err != nil {
				p.currentRow[i] = storage.StaticNullUnknown
			} else {
				p.currentRow[i] = result
			}
		}
	}

	return true
}

// Scan copies values from the current row
func (p *ProjectedResult) Scan(dest ...interface{}) error {
	if p.currentRow == nil {
		return fmt.Errorf("no row available")
	}

	if len(dest) != len(p.columns) {
		return fmt.Errorf("expected %d destination arguments in Scan, got %d", len(p.columns), len(dest))
	}

	for i, val := range p.currentRow {
		if err := storage.ScanColumnValueToDestination(val, dest[i]); err != nil {
			return fmt.Errorf("column %s: %w", p.columns[i], err)
		}
	}

	return nil
}

// Row returns the current row
func (p *ProjectedResult) Row() storage.Row {
	return p.currentRow
}

// Close closes the result
func (p *ProjectedResult) Close() error {
	return p.baseResult.Close()
}

// RowsAffected returns the number of rows affected
func (p *ProjectedResult) RowsAffected() int64 {
	return p.baseResult.RowsAffected()
}

// LastInsertID returns the last insert ID
func (p *ProjectedResult) LastInsertID() int64 {
	return p.baseResult.LastInsertID()
}

// Context returns the context
func (p *ProjectedResult) Context() context.Context {
	return p.baseResult.Context()
}

// WithAliases applies aliases to the result
func (p *ProjectedResult) WithAliases(aliases map[string]string) storage.Result {
	// Apply aliases to the base result if it supports them
	if aliasable, ok := p.baseResult.(interface {
		WithAliases(map[string]string) storage.Result
	}); ok {
		baseResult := aliasable.WithAliases(aliases)
		return &ProjectedResult{
			baseResult:  baseResult,
			columns:     p.columns,
			expressions: p.expressions,
			registry:    p.registry,
			baseColumns: baseResult.Columns(),
			evaluator:   p.evaluator,
		}
	}

	// Otherwise, wrap with AliasedResult
	return NewAliasedResult(p, aliases)
}
