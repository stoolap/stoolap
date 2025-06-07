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
	"reflect"
	"strings"
	"time"

	"github.com/stoolap/stoolap/internal/functions/contract"
	"github.com/stoolap/stoolap/internal/functions/registry"
	"github.com/stoolap/stoolap/internal/parser/funcregistry"
)

// MinFunction implements the MIN aggregate function
type MinFunction struct {
	minValue interface{}
	dataType funcregistry.DataType
	distinct bool // DISTINCT doesn't change MIN behavior, but we track it for consistency
}

// Name returns the name of the function
func (f *MinFunction) Name() string {
	return "MIN"
}

// GetInfo returns the function information
func (f *MinFunction) GetInfo() funcregistry.FunctionInfo {
	return funcregistry.FunctionInfo{
		Name:        "MIN",
		Type:        funcregistry.AggregateFunction,
		Description: "Returns the minimum value of all non-NULL values in the specified column",
		Signature: funcregistry.FunctionSignature{
			ReturnType:    funcregistry.TypeAny,                          // MIN returns the same type as the input
			ArgumentTypes: []funcregistry.DataType{funcregistry.TypeAny}, // accepts any comparable type
			MinArgs:       1,
			MaxArgs:       1,
			IsVariadic:    false,
		},
	}
}

// Register registers the MIN function with the registry
func (f *MinFunction) Register(registry funcregistry.Registry) {
	info := f.GetInfo()
	registry.MustRegister(info)
}

// Accumulate adds a value to the MIN calculation
func (f *MinFunction) Accumulate(value interface{}, distinct bool) {
	f.distinct = distinct

	// Handle NULL values (MIN ignores NULLs)
	if value == nil {
		return
	}

	// If this is the first non-NULL value, set it as the minimum
	if f.minValue == nil {
		f.minValue = value
		f.dataType = getDataType(value)
		return
	}

	// Compare the current value with the minimum
	if isLessThan(value, f.minValue) {
		f.minValue = value
	}
}

// Result returns the final result of the MIN calculation
func (f *MinFunction) Result() interface{} {
	return f.minValue // Returns NULL if no values were accumulated
}

// Reset resets the MIN calculation
func (f *MinFunction) Reset() {
	f.minValue = nil
	f.dataType = funcregistry.TypeUnknown
	f.distinct = false
}

// NewMinFunction creates a new MIN function
func NewMinFunction() contract.AggregateFunction {
	return &MinFunction{
		minValue: nil,
		dataType: funcregistry.TypeUnknown,
		distinct: false,
	}
}

// Self-registration
func init() {
	// Register the MIN function with the global registry
	// This happens automatically when the package is imported
	if registry := registry.GetGlobal(); registry != nil {
		registry.RegisterAggregateFunction(NewMinFunction())
	}
}

// isLessThan compares two values and returns true if a < b
func isLessThan(a, b any) bool {
	// Handle nil cases
	if a == nil || b == nil {
		return a == nil && b != nil
	}

	// Handle time.Time specifically
	if aTime, aOk := a.(time.Time); aOk {
		if bTime, bOk := b.(time.Time); bOk {
			return aTime.Before(bTime)
		}
		return true // time.Time < other types
	}
	if _, bOk := b.(time.Time); bOk {
		return false // other types > time.Time
	}

	// Convert to comparable numeric values with precision safety
	if result, ok := compareNumerics(a, b); ok {
		if result < 0 {
			return true // a < b
		} else if result > 0 {
			return false // a > b
		}
	}

	// Handle non-numeric types
	switch a := a.(type) {
	case bool:
		if b, ok := b.(bool); ok {
			return !a && b // false < true
		}
		return true // bool < other types
	case string:
		if b, ok := b.(string); ok {
			return strings.Compare(a, b) < 0
		}
		// Check if b is a basic type (bool, numeric). string > basic types
		return !isBasicType(b)
	}

	// For different types, compare type names for stable ordering
	return typeNameOf(a) < typeNameOf(b)
}

// compareNumerics compares two numeric values with proper precision handling
func compareNumerics(a, b any) (int, bool) {
	switch a.(type) {
	case int, int8, int16, int32, int64, Int64Convertible:
		switch b.(type) {
		case int, int8, int16, int32, int64, Int64Convertible:
			if a, ok1 := toInt64(a); ok1 {
				if b, ok2 := toInt64(b); ok2 {
					return compareSigned(a, b), true
				}
			}
		case uint, uint8, uint16, uint32, uint64:
			if a, ok1 := toInt64(a); ok1 {
				if b, ok2 := toUint64(b); ok2 {
					return compareSignedUnsigned(a, b), true
				}
			}
		case float32, float64, Float64Convertible:
			if a, ok1 := toInt64(a); ok1 {
				if b, ok2 := toFloat64(b); ok2 {
					return compareFloat(float64(a), b), true
				}
			}
		}
	case uint, uint8, uint16, uint32, uint64:
		switch b.(type) {
		case int, int8, int16, int32, int64, Int64Convertible:
			if a, ok1 := toUint64(a); ok1 {
				if b, ok2 := toInt64(b); ok2 {
					switch compareSignedUnsigned(b, a) {
					case -1:
						return 1, true // a > b
					case 0:
						return 0, true // a == b
					case 1:
						return -1, true // a < b
					}
				}
			}
		case uint, uint8, uint16, uint32, uint64:
			if a, ok1 := toUint64(a); ok1 {
				if b, ok2 := toUint64(b); ok2 {
					return compareUnsigned(a, b), true
				}
			}
		case float32, float64, Float64Convertible:
			if a, ok1 := toUint64(a); ok1 {
				if b, ok2 := toFloat64(b); ok2 {
					return compareFloat(float64(a), b), true
				}
			}
		}
	case float32, float64, Float64Convertible:
		switch b.(type) {
		case int, int8, int16, int32, int64, Int64Convertible:
			if a, ok1 := toFloat64(a); ok1 {
				if b, ok2 := toInt64(b); ok2 {
					return compareFloat(a, float64(b)), true
				}
			}
		case uint, uint8, uint16, uint32, uint64:
			if a, ok1 := toFloat64(a); ok1 {
				if b, ok2 := toUint64(b); ok2 {
					return compareFloat(a, float64(b)), true
				}
			}
		case float32, float64, Float64Convertible:
			if a, ok1 := toFloat64(a); ok1 {
				if b, ok2 := toFloat64(b); ok2 {
					return compareFloat(a, b), true
				}
			}
		}
	}

	return 0, false
}

func toInt64(v any) (int64, bool) {
	switch x := v.(type) {
	case int:
		return int64(x), true
	case int8:
		return int64(x), true
	case int16:
		return int64(x), true
	case int32:
		return int64(x), true
	case int64:
		return x, true
	case Int64Convertible:
		if i64, ok := x.AsInt64(); ok {
			return i64, true
		}
	}
	return 0, false
}

func toUint64(v any) (uint64, bool) {
	switch x := v.(type) {
	case uint:
		return uint64(x), true
	case uint8:
		return uint64(x), true
	case uint16:
		return uint64(x), true
	case uint32:
		return uint64(x), true
	case uint64:
		return x, true
	}
	return 0, false
}

// toFloat64 converts a value to float64
func toFloat64(value any) (float64, bool) {
	// Handle common types directly without reflection first
	switch v := value.(type) {
	case float32:
		return float64(v), true
	case float64:
		return v, true
	case Float64Convertible:
		if f64, ok := v.AsFloat64(); ok {
			return f64, true
		}
	}

	return 0, false
}

func compareSigned(a, b int64) int {
	if a < b {
		return -1
	} else if a > b {
		return 1
	}
	return 0
}

func compareUnsigned(a, b uint64) int {
	if a < b {
		return -1
	} else if a > b {
		return 1
	}
	return 0
}

func compareSignedUnsigned(a int64, b uint64) int {
	if a < 0 {
		return -1
	}
	ua := uint64(a)
	if ua < b {
		return -1
	} else if ua > b {
		return 1
	}
	return 0
}

func compareFloat(a, b float64) int {
	if a < b {
		return -1
	} else if a > b {
		return 1
	}
	return 0
}

func isBasicType(v interface{}) bool {
	switch v.(type) {
	case bool, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float32, float64:
		return true
	}
	return false
}

// typeNameOf returns the type name as a string without using fmt.Sprintf
func typeNameOf(v interface{}) string {
	if v == nil {
		return "nil"
	}

	switch v.(type) {
	case bool:
		return "bool"
	case int:
		return "int"
	case int8:
		return "int8"
	case int16:
		return "int16"
	case int32:
		return "int32"
	case int64:
		return "int64"
	case uint:
		return "uint"
	case uint8:
		return "uint8"
	case uint16:
		return "uint16"
	case uint32:
		return "uint32"
	case uint64:
		return "uint64"
	case float32:
		return "float32"
	case float64:
		return "float64"
	case string:
		return "string"
	case time.Time:
		return "time.Time"
	}

	// For custom types, use reflection as a last resort
	// This is only used for the fallback case
	return reflect.TypeOf(v).String()
}

// getDataType determines the data type of a value
func getDataType(value any) funcregistry.DataType {
	v := reflect.ValueOf(value)

	switch v.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return funcregistry.TypeInteger
	case reflect.Float32, reflect.Float64:
		return funcregistry.TypeFloat
	case reflect.String:
		return funcregistry.TypeString
	case reflect.Bool:
		return funcregistry.TypeBoolean
	default:
		return funcregistry.TypeAny
	}
}
