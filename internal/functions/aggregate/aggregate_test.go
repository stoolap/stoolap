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
package aggregate_test

import (
	"reflect"
	"testing"

	"github.com/stoolap/stoolap/internal/functions/aggregate"
)

// TestCount tests the COUNT aggregate function with various edge cases
func TestCount(t *testing.T) {
	tests := []struct {
		name     string
		values   []any
		distinct bool
		expected int64
	}{
		{
			name:     "COUNT(*) with mixed values",
			values:   []any{"*", "*", "*"},
			distinct: false,
			expected: 3,
		},
		{
			name:     "COUNT with NULL values - should ignore NULLs",
			values:   []any{1, nil, 2, nil, 3},
			distinct: false,
			expected: 3, // NULL values are ignored in COUNT
		},
		{
			name:     "COUNT DISTINCT with duplicates",
			values:   []any{1, 2, 2, 3, 3, 3},
			distinct: true,
			expected: 3, // Only unique values are counted
		},
		{
			name:     "COUNT DISTINCT with NULLs - should ignore NULLs",
			values:   []any{1, nil, 2, nil, 2},
			distinct: true,
			expected: 2, // NULL values are ignored, duplicates removed
		},
		{
			name:     "COUNT on empty set",
			values:   []any{},
			distinct: false,
			expected: 0, // Empty set returns 0
		},
		{
			name:     "COUNT on all NULL values",
			values:   []any{nil, nil, nil},
			distinct: false,
			expected: 0.0, // All NULLs should return 0
		},
		{
			name:     "COUNT(*) special case",
			values:   []any{"*", "*", "*", "*"},
			distinct: false,
			expected: 4, // COUNT(*) counts rows, not values
		},
		{
			name:     "COUNT with string values",
			values:   []any{"apple", "banana", "apple", "cherry"},
			distinct: false,
			expected: 4,
		},
		{
			name:     "COUNT DISTINCT with string values",
			values:   []any{"apple", "banana", "apple", "cherry"},
			distinct: true,
			expected: 3, // Only unique strings counted
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			countFunc := aggregate.NewCountFunction()

			// Accumulate all values
			for _, value := range tt.values {
				countFunc.Accumulate(value, tt.distinct)
			}

			result := countFunc.Result()
			if result != tt.expected {
				t.Errorf("COUNT test %s failed: expected %v, got %v", tt.name, tt.expected, result)
			}
		})
	}
}

// TestSum tests the SUM aggregate function with various data types and edge cases
func TestSum(t *testing.T) {
	tests := []struct {
		name     string
		values   []any
		distinct bool
		expected any
	}{
		{
			name:     "SUM of integers - should return int64",
			values:   []any{int64(1), int64(2), int64(3)},
			distinct: false,
			expected: int64(6), // Integer inputs return int64
		},
		{
			name:     "SUM of floating point numbers",
			values:   []any{float64(1.5), float32(2.5), float64(3.0)},
			distinct: false,
			expected: 7.0, // Float inputs return float64
		},
		{
			name:     "SUM with NULL values - should ignore NULLs",
			values:   []any{uint64(10), nil, int64(20), nil},
			distinct: false,
			expected: int64(30), // NULL values are ignored in SUM
		},
		{
			name:     "SUM DISTINCT with duplicates",
			values:   []any{int8(5), int64(5), uint8(10), uint64(10)},
			distinct: true,
			expected: int64(15), // Only unique values are summed: 5 + 10 = 15
		},
		{
			name:     "SUM on empty set",
			values:   []any{},
			distinct: false,
			expected: nil, // Empty set returns 0 for SUM
		},
		{
			name:     "SUM on all NULL values",
			values:   []any{nil, nil, nil},
			distinct: false,
			expected: nil, // All NULLs should return 0
		},
		{
			name:     "SUM with mixed integer types",
			values:   []any{int(1), int32(2), int64(3)},
			distinct: false,
			expected: int64(6), // All converted to int64
		},
		{
			name:     "SUM with negative numbers",
			values:   []any{int64(10), int64(-5), int64(3)},
			distinct: false,
			expected: int64(8), // Negative numbers handled correctly
		},
		{
			name:     "SUM DISTINCT with mixed float and int",
			values:   []any{1.0, int64(1), 2.0, int64(2)},
			distinct: true,
			expected: 3.0, // SQL considers numeric values same
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sumFunc := aggregate.NewSumFunction()

			// Accumulate all values
			for _, value := range tt.values {
				sumFunc.Accumulate(value, tt.distinct)
			}

			result := sumFunc.Result()
			if result != tt.expected {
				t.Errorf("SUM test %s failed: expected %v, got %v with type of: %v", tt.name, tt.expected, result, reflect.TypeOf(result))
			}
		})
	}
}

// TestAvg tests the AVG aggregate function with standard SQL behavior
func TestAvg(t *testing.T) {
	tests := []struct {
		name     string
		values   []any
		distinct bool
		expected any
	}{
		{
			name:     "AVG of integers",
			values:   []any{int64(2), int64(4), int64(6)},
			distinct: false,
			expected: 4.0, // AVG always returns float64: (2+4+6)/3 = 4.0
		},
		{
			name:     "AVG of floating point numbers",
			values:   []any{1.5, 2.5, 3.0},
			distinct: false,
			expected: 2.3333333333333335, // (1.5+2.5+3.0)/3
		},
		{
			name:     "AVG with NULL values - should ignore NULLs",
			values:   []any{int64(10), nil, int64(20), nil},
			distinct: false,
			expected: 15.0, // (10+20)/2 = 15.0, NULLs ignored
		},
		{
			name:     "AVG DISTINCT with duplicates",
			values:   []any{int64(5), int64(5), int64(15)},
			distinct: true,
			expected: 10.0, // (5+15)/2 = 10.0, duplicate 5 ignored
		},
		{
			name:     "AVG on empty set",
			values:   []any{},
			distinct: false,
			expected: nil, // Empty set returns NULL for AVG
		},
		{
			name:     "AVG on all NULL values",
			values:   []any{nil, nil, nil},
			distinct: false,
			expected: nil, // All NULLs should return NULL
		},
		{
			name:     "AVG with negative numbers",
			values:   []any{int64(-10), int64(10), int64(0)},
			distinct: false,
			expected: 0.0, // (-10+10+0)/3 = 0.0
		},
		{
			name:     "AVG with single value",
			values:   []any{int64(42)},
			distinct: false,
			expected: 42.0, // Single value returns that value as float
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			avgFunc := aggregate.NewAvgFunction()

			// Accumulate all values
			for _, value := range tt.values {
				avgFunc.Accumulate(value, tt.distinct)
			}

			result := avgFunc.Result()
			if result != tt.expected {
				t.Errorf("AVG test %s failed: expected %v, got %v", tt.name, tt.expected, result)
			}
		})
	}
}

// TestMin tests the MIN aggregate function with various data types
func TestMin(t *testing.T) {
	tests := []struct {
		name     string
		values   []any
		expected any
	}{
		{
			name:     "MIN of integers",
			values:   []any{int64(5), int64(2), int64(8), int64(1)},
			expected: int64(1), // Minimum integer value
		},
		{
			name:     "MIN of floating point numbers",
			values:   []any{5.5, 2.2, 8.8, 1.1},
			expected: 1.1, // Minimum float value
		},
		{
			name:     "MIN with NULL values - should ignore NULLs",
			values:   []any{int64(10), nil, int64(5), nil, int64(15)},
			expected: int64(5), // NULL values are ignored in MIN
		},
		{
			name:     "MIN of strings - lexicographic ordering",
			values:   []any{"zebra", "apple", "banana"},
			expected: "apple", // Lexicographically smallest string
		},
		{
			name:     "MIN on empty set",
			values:   []any{},
			expected: nil, // Empty set returns NULL for MIN
		},
		{
			name:     "MIN on all NULL values",
			values:   []any{nil, nil, nil},
			expected: nil, // All NULLs should return NULL
		},
		{
			name:     "MIN with negative numbers",
			values:   []any{int64(5), int64(-10), int64(0), int64(3)},
			expected: int64(-10), // Most negative number
		},
		{
			name:     "MIN with boolean values",
			values:   []any{true, false, true},
			expected: false, // false < true in boolean comparison
		},
		{
			name:     "MIN with single value",
			values:   []any{int64(42)},
			expected: int64(42), // Single value returns that value
		},
		{
			name:     "MIN with duplicate values",
			values:   []any{int64(5), int64(5), int64(5)},
			expected: int64(5), // All same values return that value
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			minFunc := aggregate.NewMinFunction()

			// Accumulate all values
			for _, value := range tt.values {
				minFunc.Accumulate(value, false) // DISTINCT doesn't affect MIN behavior
			}

			result := minFunc.Result()
			if result != tt.expected {
				t.Errorf("MIN test %s failed: expected %v, got %v", tt.name, tt.expected, result)
			}
		})
	}
}

// TestMax tests the MAX aggregate function with various data types
func TestMax(t *testing.T) {
	tests := []struct {
		name     string
		values   []any
		expected any
	}{
		{
			name:     "MAX of integers",
			values:   []any{int64(5), int64(2), int64(8), int64(1)},
			expected: int64(8), // Maximum integer value
		},
		{
			name:     "MAX of floating point numbers",
			values:   []any{5.5, 2.2, 8.8, 1.1},
			expected: 8.8, // Maximum float value
		},
		{
			name:     "MAX with NULL values - should ignore NULLs",
			values:   []any{int64(10), nil, int64(15), nil, int64(5)},
			expected: int64(15), // NULL values are ignored in MAX
		},
		{
			name:     "MAX of strings - lexicographic ordering",
			values:   []any{"apple", "zebra", "banana"},
			expected: "zebra", // Lexicographically largest string
		},
		{
			name:     "MAX on empty set",
			values:   []any{},
			expected: nil, // Empty set returns NULL for MAX
		},
		{
			name:     "MAX on all NULL values",
			values:   []any{nil, nil, nil},
			expected: nil, // All NULLs should return NULL
		},
		{
			name:     "MAX with negative numbers",
			values:   []any{int64(-5), int64(-10), int64(-1), int64(-3)},
			expected: int64(-1), // Least negative number (closest to 0)
		},
		{
			name:     "MAX with boolean values",
			values:   []any{false, true, false},
			expected: true, // true > false in boolean comparison
		},
		{
			name:     "MAX with single value",
			values:   []any{int64(42)},
			expected: int64(42), // Single value returns that value
		},
		{
			name:     "MAX with duplicate values",
			values:   []any{int64(5), int64(5), int64(5)},
			expected: int64(5), // All same values return that value
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			maxFunc := aggregate.NewMaxFunction()

			// Accumulate all values
			for _, value := range tt.values {
				maxFunc.Accumulate(value, false) // DISTINCT doesn't affect MAX behavior
			}

			result := maxFunc.Result()
			if result != tt.expected {
				t.Errorf("MAX test %s failed: expected %v, got %v", tt.name, tt.expected, result)
			}
		})
	}
}

// TestFirst tests the FIRST aggregate function with SQL standard behavior
func TestFirst(t *testing.T) {
	tests := []struct {
		name     string
		values   []any
		expected any
	}{
		{
			name:     "FIRST with multiple values",
			values:   []any{int8(10), int32(20), int64(30)},
			expected: int8(10), // First non-NULL value encountered
		},
		{
			name:     "FIRST with NULL values at start - should skip NULLs",
			values:   []any{nil, nil, uint64(15), uint32(25)},
			expected: uint64(15), // First non-NULL value
		},
		{
			name:     "FIRST with string values",
			values:   []any{"first", "second", "third"},
			expected: "first", // First string value
		},
		{
			name:     "FIRST on empty set",
			values:   []any{},
			expected: nil, // Empty set returns NULL
		},
		{
			name:     "FIRST on all NULL values",
			values:   []any{nil, nil, nil},
			expected: nil, // All NULLs should return NULL
		},
		{
			name:     "FIRST with empty strings - should skip empty strings",
			values:   []any{"", "", "actual_value"},
			expected: "actual_value", // Skip empty strings like NULLs
		},
		{
			name:     "FIRST with zero values - should skip zeros",
			values:   []any{int64(0), uint32(0), int64(42)},
			expected: int64(42), // Skip zero values based on implementation
		},
		{
			name:     "FIRST with single value",
			values:   []any{int64(42)},
			expected: int64(42), // Single value returns that value
		},
		{
			name:     "FIRST with boolean values",
			values:   []any{false, true, false},
			expected: false, // First boolean value
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			firstFunc := aggregate.NewFirstFunction()

			// Accumulate all values
			for _, value := range tt.values {
				firstFunc.Accumulate(value, false) // DISTINCT doesn't affect FIRST behavior
			}

			result := firstFunc.Result()
			if result != tt.expected {
				t.Errorf("FIRST test %s failed: expected %v, got %v with type of: %v", tt.name, tt.expected, result, reflect.TypeOf(result))
			}
		})
	}
}

// TestLast tests the LAST aggregate function with SQL standard behavior
func TestLast(t *testing.T) {
	tests := []struct {
		name     string
		values   []any
		expected any
	}{
		{
			name:     "LAST with multiple values",
			values:   []any{int64(10), int64(20), int64(30)},
			expected: int64(30), // Last non-NULL value encountered
		},
		{
			name:     "LAST with NULL values at end - should skip NULLs",
			values:   []any{int64(15), int64(25), nil, nil},
			expected: int64(25), // Last non-NULL value
		},
		{
			name:     "LAST with string values",
			values:   []any{"first", "second", "third"},
			expected: "third", // Last string value
		},
		{
			name:     "LAST on empty set",
			values:   []any{},
			expected: nil, // Empty set returns NULL
		},
		{
			name:     "LAST on all NULL values",
			values:   []any{nil, nil, nil},
			expected: nil, // All NULLs should return NULL
		},
		{
			name:     "LAST with empty strings - should skip empty strings",
			values:   []any{"actual_value", "", ""},
			expected: "actual_value", // Skip empty strings like NULLs
		},
		{
			name:     "LAST with zero values - should skip zeros",
			values:   []any{int64(42), int64(0), int64(0)},
			expected: int64(42), // Skip zero values based on implementation
		},
		{
			name:     "LAST with single value",
			values:   []any{int64(42)},
			expected: int64(42), // Single value returns that value
		},
		{
			name:     "LAST with boolean values",
			values:   []any{false, true, false},
			expected: false, // Last boolean value
		},
		{
			name:     "LAST with interleaved NULLs",
			values:   []any{int64(10), nil, int64(20), nil, int64(30)},
			expected: int64(30), // Last non-NULL value in sequence
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lastFunc := aggregate.NewLastFunction()

			// Accumulate all values
			for _, value := range tt.values {
				lastFunc.Accumulate(value, false) // DISTINCT doesn't affect LAST behavior
			}

			result := lastFunc.Result()
			if result != tt.expected {
				t.Errorf("LAST test %s failed: expected %v, got %v", tt.name, tt.expected, result)
			}
		})
	}
}
