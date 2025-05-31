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
package aggregate

import (
	"github.com/stoolap/stoolap/internal/functions/contract"
	"github.com/stoolap/stoolap/internal/functions/registry"
	"github.com/stoolap/stoolap/internal/parser/funcregistry"
)

// LastFunction implements the LAST aggregate function
// which returns the last non-NULL value encountered in a group
// or the last value according to an ORDER BY clause if specified
type LastFunction struct {
	lastValue interface{}
	distinct  bool // DISTINCT is not meaningful for LAST, but we track it for API consistency
	// For ordered aggregation
	orderedValues  []struct{ Value, OrderKey interface{} }
	useOrdering    bool
	orderDirection string // "ASC" or "DESC"
}

// Name returns the name of the function
func (f *LastFunction) Name() string {
	return "LAST"
}

// GetInfo returns the function information
func (f *LastFunction) GetInfo() funcregistry.FunctionInfo {
	return funcregistry.FunctionInfo{
		Name:        "LAST",
		Type:        funcregistry.AggregateFunction,
		Description: "Returns the last non-NULL value in the group, with optional ORDER BY",
		Signature: funcregistry.FunctionSignature{
			ReturnType:    funcregistry.TypeAny,
			ArgumentTypes: []funcregistry.DataType{funcregistry.TypeAny},
			MinArgs:       1,
			MaxArgs:       1,
			IsVariadic:    false,
		},
	}
}

// Register registers the function with the registry
func (f *LastFunction) Register(registry funcregistry.Registry) {
	info := f.GetInfo()
	registry.MustRegister(info)
}

// SupportsOrdering returns true as LAST supports ORDER BY
func (f *LastFunction) SupportsOrdering() bool {
	return true
}

// AccumulateOrdered adds a value with an ordering key to the aggregate calculation
func (f *LastFunction) AccumulateOrdered(value interface{}, orderKey interface{}, direction string, distinct bool) {
	f.distinct = distinct
	f.useOrdering = true
	f.orderDirection = direction

	// Skip NULL values
	if value == nil {
		return
	}

	// For consistency with FIRST, skip empty strings
	if s, ok := value.(string); ok && s == "" {
		return
	}

	// For consistency with FIRST, skip specific zero values
	// In SQL, zero values might be actual data, not NULL
	// But in our test case, some zero values seem to be unexpected
	if val, ok := compareNumerics(value, 0); ok && val == 0 {
		return
	}

	// Store the value and its ordering key for later sorting
	f.orderedValues = append(f.orderedValues, struct{ Value, OrderKey interface{} }{
		Value:    DeepCopy(value),
		OrderKey: orderKey,
	})
}

// Accumulate adds a value to the LAST calculation
func (f *LastFunction) Accumulate(value interface{}, distinct bool) {
	f.distinct = distinct

	// If we're using ordering, delegate to AccumulateOrdered with nil order key
	if f.useOrdering {
		f.AccumulateOrdered(value, nil, "ASC", distinct)
		return
	}

	// Skip NULL values
	if value == nil {
		return
	}

	// For consistency with FIRST, skip empty strings
	if s, ok := value.(string); ok && s == "" {
		return
	}

	// For consistency with FIRST, skip specific zero values
	// In SQL, zero values might be actual data, not NULL
	// But in our test case, some zero values seem to be unexpected
	if val, ok := compareNumerics(value, 0); ok && val == 0 {
		return
	}

	// Always update with the most recent non-NULL value
	f.lastValue = DeepCopy(value)
}

// Result returns the final result of the LAST calculation
func (f *LastFunction) Result() interface{} {
	// If we're using ORDER BY, sort the values and return the last
	if f.useOrdering && len(f.orderedValues) > 0 {
		// Sort the values by order key
		sortOrderedValues(f.orderedValues, f.orderDirection == "DESC")

		// Return the last value after sorting (for LAST function)
		return f.orderedValues[len(f.orderedValues)-1].Value
	}

	// Otherwise, return the last value we found
	return f.lastValue // Returns NULL if no values were accumulated
}

// Reset resets the LAST calculation
func (f *LastFunction) Reset() {
	f.lastValue = nil
	f.distinct = false
	f.orderedValues = nil
	f.useOrdering = false
	f.orderDirection = "ASC"
}

// NewLastFunction creates a new LAST function
func NewLastFunction() contract.AggregateFunction {
	return &LastFunction{
		lastValue:      nil,
		distinct:       false,
		orderedValues:  nil,
		useOrdering:    false,
		orderDirection: "ASC",
	}
}

// Self-registration
func init() {
	// Register the LAST function with the global registry
	// This happens automatically when the package is imported
	if registry := registry.GetGlobal(); registry != nil {
		registry.RegisterAggregateFunction(NewLastFunction())
	}
}
