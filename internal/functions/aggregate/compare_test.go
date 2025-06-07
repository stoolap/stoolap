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
	"testing"
	"time"
)

type testInt int64
type testFloat float64

func (m testInt) AsInt64() (int64, bool) {
	return int64(m), true
}

func (m testFloat) AsFloat64() (float64, bool) {
	return float64(m), true
}

func TestIsLessThan(t *testing.T) {
	now := time.Now()
	later := now.Add(time.Hour)

	tests := []struct {
		name string
		a, b any
		want bool
	}{
		// Nil cases
		{"nil < int", nil, 1, true},
		{"int < nil", 1, nil, false},
		{"nil < nil", nil, nil, false},

		// Time comparisons
		{"time < time", now, later, true},
		{"time > time", later, now, false},
		{"time < string", now, "abc", true},
		{"string > time", "abc", now, false},

		// Same type comparisons
		{"int < int", 1, 2, true},
		{"int > int", 5, 1, false},
		{"uint < uint", uint(1), uint(2), true},
		{"uint > uint", uint(3), uint(1), false},
		{"float32 < float32", float32(1.1), float32(1.2), true},
		{"float64 < float64", float64(1.1), float64(2.2), true},
		{"bool < bool", false, true, true},
		{"string < string", "abc", "xyz", true},

		// Cross-type numeric comparisons
		{"int32 < uint8", int32(10), uint8(20), true},
		{"uint8 < int32", uint8(5), int32(10), true},
		{"int16 < float64", int16(10), float64(10.5), true},
		{"float32 < int64", float32(1.1), int64(2), true},
		{"int < float32", int(3), float32(4.5), true},
		{"uint < float64", uint(3), float64(3.1), true},
		{"float64 < uint", float64(5.5), uint(6), true},
		{"int64 < uint64", int64(9), uint64(10), true},
		{"int64 < uint64 (negative int)", int64(-5), uint64(1), true},
		{"uint64 < int64", uint64(1), int64(10), true},
		{"uint64 > int64 (negative int)", uint64(10), int64(-1), false},

		// Equal numeric types (should return false)
		{"int == float64", int(3), float64(3.0), false},
		{"uint == int64", uint(100), int64(100), false},

		// Mixed type comparisons with custom interfaces
		{"Int64Convertible < int", testInt(5), 10, true},
		{"Float64Convertible < float64", testFloat(1.5), 2.0, true},
		{"Int64Convertible == int64", testInt(10), int64(10), true},
		{"Float64Convertible == float32", testFloat(3.3), float32(3.3), false},

		// Fallback type name comparison
		{"struct vs int", struct{}{}, 1, typeNameOf(struct{}{}) < typeNameOf(1)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isLessThan(tt.a, tt.b)
			if got != tt.want {
				t.Errorf("isLessThan(%#v, %#v) = %v, want %v", tt.a, tt.b, got, tt.want)
			}
		})
	}
}
