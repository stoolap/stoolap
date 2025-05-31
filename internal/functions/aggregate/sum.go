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

// SumFunction implements the SUM aggregate function
type SumFunction struct {
	sum         any
	distinct    bool
	values      map[distinctValuePair]struct{} // used for DISTINCT
	initialized bool                           // track if we've seen any values
}

// Name returns the name of the function
func (f *SumFunction) Name() string {
	return "SUM"
}

// GetInfo returns the function information
func (f *SumFunction) GetInfo() funcregistry.FunctionInfo {
	return funcregistry.FunctionInfo{
		Name:        "SUM",
		Type:        funcregistry.AggregateFunction,
		Description: "Returns the sum of all non-NULL values in the specified column. Returns int64 for integer inputs, float64 for floating-point inputs.",
		Signature: funcregistry.FunctionSignature{
			ReturnType:    funcregistry.TypeAny,                          // can return either int64 or float64 based on inputs
			ArgumentTypes: []funcregistry.DataType{funcregistry.TypeAny}, // accepts numeric types
			MinArgs:       1,
			MaxArgs:       1,
			IsVariadic:    false,
		},
	}
}

// Register registers the SUM function with the registry
func (f *SumFunction) Register(registry funcregistry.Registry) {
	info := f.GetInfo()
	registry.MustRegister(info)
}

// Accumulate adds a value to the SUM calculation
func (f *SumFunction) Accumulate(value any, distinct bool) {
	f.distinct = distinct

	// Handle NULL values (SUM ignores NULLs)
	if value == nil {
		return
	}

	var ok bool
	switch v := value.(type) {
	case int, int8, int16, int32, int64, Int64Convertible:
		value, ok = toInt64(v)
	case uint, uint8, uint16, uint32, uint64:
		value, ok = toUint64(v)
	case float32, float64, Float64Convertible:
		value, ok = toFloat64(v)
	default:
		return
	}

	if !ok {
		// If we can't convert to a numeric type, we ignore this value
		return
	}

	// Initialize tracking for first value
	if !f.initialized {
		f.initialized = true
	}

	// Handle DISTINCT case
	if distinct {
		if isDistinct := f.isDistinct(value); !isDistinct {
			// If the value is not distinct, we skip accumulating it
			return
		}
	}

	switch sum := f.sum.(type) {
	case int64:
		switch val := value.(type) {
		case int64:
			f.sum = sum + val
		case uint64:
			f.sum = sum + int64(val)
		case float64:
			f.sum = float64(sum) + val
		}
	case float64:
		switch val := value.(type) {
		case int64:
			f.sum = sum + float64(val)
		case uint64:
			f.sum = sum + float64(val)
		case float64:
			f.sum = sum + val
		}
	default:
		switch val := value.(type) {
		case int64:
			f.sum = val
		case uint64:
			f.sum = int64(val)
		case float64:
			f.sum = val
		}
	}
}

// valuePair is a struct used to store distinct values for the SUM function.
// It holds both int64 and float64 representations to handle distinct checks
// for both integer and floating-point versions of inputs. In SQL standard SUM
// function treats 2.0 and 2 as the same value, so we need to ensure that
// we can compare both integer and floating-point values correctly for avoiding
// duplicate entries.
type distinctValuePair struct {
	intEq   int64
	floatEq float64
}

func (f *SumFunction) isDistinct(value any) bool {
	if f.values == nil {
		f.values = make(map[distinctValuePair]struct{})
	}

	var pair distinctValuePair
	switch v := value.(type) {
	case int64:
		pair.intEq = v
		pair.floatEq = float64(v) // Store as float for comparison
	case uint64:
		pair.intEq = int64(v)     // Convert to int64 for comparison
		pair.floatEq = float64(v) // Store as float for comparison
	case float64:
		pair.intEq = int64(v) // Convert to int64 for comparison
		pair.floatEq = v      // Store as float for comparison
	default:
		return false
	}
	// Check if the value is already in the map
	if _, ok := f.values[pair]; !ok {
		// If not, add it to the map and accumulate the sum
		f.values[pair] = struct{}{}

		return true
	}
	return false
}

// Result returns the final result of the SUM calculation
func (f *SumFunction) Result() any {
	// If no values were accumulated, return NULL
	if !f.initialized {
		return nil
	}

	return f.sum
}

// Reset resets the SUM calculation
func (f *SumFunction) Reset() {
	f.sum = 0
	f.values = nil
	f.distinct = false
	f.initialized = false
}

// NewSumFunction creates a new SUM function
func NewSumFunction() contract.AggregateFunction {
	return &SumFunction{
		sum:         int64(0),
		values:      make(map[distinctValuePair]struct{}),
		distinct:    false,
		initialized: false,
	}
}

// Self-registration
func init() {
	// Register the SUM function with the global registry
	// This happens automatically when the package is imported
	if registry := registry.GetGlobal(); registry != nil {
		registry.RegisterAggregateFunction(NewSumFunction())
	}
}
