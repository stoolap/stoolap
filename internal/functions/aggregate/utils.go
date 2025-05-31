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
	"sort"
	"time"
)

type Int64Convertible interface {
	AsInt64() (int64, bool)
}

type Float64Convertible interface {
	AsFloat64() (float64, bool)
}

type BooleanConvertible interface {
	AsBoolean() (bool, bool)
}

type StringConvertible interface {
	AsString() (string, bool)
}

type TimestampConvertible interface {
	AsTimestamp() (time.Time, bool)
}

// DeepCopy creates a copy of simple values to avoid reference sharing
func DeepCopy(val interface{}) interface{} {
	if val == nil {
		return nil
	}

	switch v := val.(type) {
	case int:
		return v
	case int64:
		return v
	case float64:
		return v
	case string:
		return v
	case bool:
		return v
	default:
		// For complex types, we'd need more sophisticated copying
		// but for simple database types this should be sufficient
		return v
	}
}

// sortOrderedValues sorts a slice of ordered values
func sortOrderedValues(values []struct{ Value, OrderKey interface{} }, descending bool) {
	// Sort the values by their order keys
	sort.SliceStable(values, func(i, j int) bool {
		a, b := values[i].OrderKey, values[j].OrderKey

		// Handle nil values
		if a == nil && b == nil {
			return false // Equal, preserve original order
		}
		if a == nil {
			return !descending // nil is less than non-nil by default
		}
		if b == nil {
			return descending // non-nil is greater than nil by default
		}

		// Compare values based on type
		switch av := a.(type) {
		case int64:
			if bv, ok := b.(int64); ok {
				if descending {
					return av > bv
				}
				return av < bv
			}
		case float64:
			if bv, ok := b.(float64); ok {
				if descending {
					return av > bv
				}
				return av < bv
			}
		case string:
			if bv, ok := b.(string); ok {
				if descending {
					return av > bv
				}
				return av < bv
			}
		case bool:
			if bv, ok := b.(bool); ok {
				if descending {
					return av && !bv // true > false when descending
				}
				return !av && bv // false < true when ascending
			}
		}

		// Default for incomparable types - preserve original order
		return false
	})
}
