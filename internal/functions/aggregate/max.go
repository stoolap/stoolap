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
	"time"

	"github.com/stoolap/stoolap/internal/functions/contract"
	"github.com/stoolap/stoolap/internal/functions/registry"
	"github.com/stoolap/stoolap/internal/parser/funcregistry"
)

// MaxFunction implements the MAX aggregate function
type MaxFunction struct {
	maxValue any
	dataType funcregistry.DataType
	distinct bool // DISTINCT doesn't change MAX behavior, but we track it for consistency
}

// Name returns the name of the function
func (f *MaxFunction) Name() string {
	return "MAX"
}

// GetInfo returns the function information
func (f *MaxFunction) GetInfo() funcregistry.FunctionInfo {
	return funcregistry.FunctionInfo{
		Name:        "MAX",
		Type:        funcregistry.AggregateFunction,
		Description: "Returns the maximum value of all non-NULL values in the specified column",
		Signature: funcregistry.FunctionSignature{
			ReturnType:    funcregistry.TypeAny,                          // MAX returns the same type as the input
			ArgumentTypes: []funcregistry.DataType{funcregistry.TypeAny}, // accepts any comparable type
			MinArgs:       1,
			MaxArgs:       1,
			IsVariadic:    false,
		},
	}
}

// Register registers the MAX function with the registry
func (f *MaxFunction) Register(registry funcregistry.Registry) {
	info := f.GetInfo()
	registry.MustRegister(info)
}

// Accumulate adds a value to the MAX calculation
func (f *MaxFunction) Accumulate(value any, distinct bool) {
	f.distinct = distinct

	// Handle NULL values (MAX ignores NULLs)
	if value == nil {
		return
	}

	// If this is the first non-NULL value, set it as the maximum
	if f.maxValue == nil {
		f.maxValue = value
		f.dataType = getDataType(value)
		return
	}

	// Compare the current value with the maximum (use isGreaterThan, which is !isLessThan && not equal)
	if !isLessThan(value, f.maxValue) && !isEqual(value, f.maxValue) {
		f.maxValue = value
	}
}

// Result returns the final result of the MAX calculation
func (f *MaxFunction) Result() any {
	return f.maxValue // Returns NULL if no values were accumulated
}

// Reset resets the MAX calculation
func (f *MaxFunction) Reset() {
	f.maxValue = nil
	f.dataType = funcregistry.TypeUnknown
	f.distinct = false
}

// NewMaxFunction creates a new MAX function
func NewMaxFunction() contract.AggregateFunction {
	return &MaxFunction{
		maxValue: nil,
		dataType: funcregistry.TypeUnknown,
		distinct: false,
	}
}

// Self-registration
func init() {
	// Register the MAX function with the global registry
	// This happens automatically when the package is imported
	if registry := registry.GetGlobal(); registry != nil {
		registry.RegisterAggregateFunction(NewMaxFunction())
	}
}

// isEqual compares two values for equality
func isEqual(a, b any) bool {
	// Handle nil cases
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}

	// Handle time.Time specifically (common case)
	if aTime, aOk := a.(time.Time); aOk {
		if bTime, bOk := b.(time.Time); bOk {
			return aTime.Equal(bTime)
		}
		return false // time.Time != other types
	} else if _, bOk := b.(time.Time); bOk {
		return false // other types != time.Time
	}

	// Use type assertions for specific comparisons
	switch a.(type) {
	case bool:
		bBool, ok := b.(bool)
		return ok && a.(bool) == bBool
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float32, float64, Int64Convertible, Float64Convertible:
		switch b.(type) {
		case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float32, float64, Int64Convertible, Float64Convertible:
			cmpResult, ok := compareNumerics(a, b)
			return ok && cmpResult == 0
		}
	case string:
		bStr, ok := b.(string)
		return ok && a.(string) == bStr
	}

	// Different types or non-primitive types that are not directly comparable
	// are not equal
	return false
}
