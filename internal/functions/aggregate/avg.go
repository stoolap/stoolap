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

// AvgFunction implements the AVG aggregate function
type AvgFunction struct {
	sum      float64
	count    int64
	distinct bool
	values   map[float64]struct{} // used for DISTINCT
}

// Name returns the name of the function
func (f *AvgFunction) Name() string {
	return "AVG"
}

// GetInfo returns the function information
func (f *AvgFunction) GetInfo() funcregistry.FunctionInfo {
	return funcregistry.FunctionInfo{
		Name:        "AVG",
		Type:        funcregistry.AggregateFunction,
		Description: "Returns the average of all non-NULL values in the specified column",
		Signature: funcregistry.FunctionSignature{
			ReturnType:    funcregistry.TypeFloat,
			ArgumentTypes: []funcregistry.DataType{funcregistry.TypeAny}, // accepts numeric types
			MinArgs:       1,
			MaxArgs:       1,
			IsVariadic:    false,
		},
	}
}

// Register registers the AVG function with the registry
func (f *AvgFunction) Register(registry funcregistry.Registry) {
	info := f.GetInfo()
	registry.MustRegister(info)
}

// Accumulate adds a value to the AVG calculation
func (f *AvgFunction) Accumulate(value any, distinct bool) {
	f.distinct = distinct

	// Handle NULL values (AVG ignores NULLs)
	if value == nil {
		return
	}

	// First try direct type conversion for performance
	var numericValue float64

	// Handle direct numeric types
	switch v := value.(type) {
	case float32, float64, Float64Convertible:
		var ok bool
		if numericValue, ok = toFloat64(v); !ok {
			return
		}
	case uint, uint8, uint16, uint32, uint64:
		if v, ok := toUint64(v); ok {
			numericValue = float64(v)
		} else {
			return
		}
	case int, int8, int16, int32, int64, Int64Convertible:
		if v, ok := toInt64(v); ok {
			numericValue = float64(v)
		} else {
			return
		}
	}

	// Handle DISTINCT case
	if distinct {
		if f.values == nil {
			f.values = make(map[float64]struct{})
		}
		if _, exists := f.values[numericValue]; exists {
			return // Already seen, skip
		}
		f.values[numericValue] = struct{}{}
	}

	f.sum += numericValue
	f.count++
}

// Result returns the final result of the AVG calculation
func (f *AvgFunction) Result() any {
	if f.count == 0 {
		return nil // Return NULL for empty sets
	}
	return f.sum / float64(f.count)
}

// Reset resets the AVG calculation
func (f *AvgFunction) Reset() {
	f.sum = 0
	f.count = 0
	f.values = nil
	f.distinct = false
}

// NewAvgFunction creates a new AVG function
func NewAvgFunction() contract.AggregateFunction {
	return &AvgFunction{
		sum:      0,
		count:    0,
		values:   make(map[float64]struct{}),
		distinct: false,
	}
}

// Self-registration
func init() {
	// Register the AVG function with the global registry
	// This happens automatically when the package is imported
	if registry := registry.GetGlobal(); registry != nil {
		registry.RegisterAggregateFunction(NewAvgFunction())
	}
}
