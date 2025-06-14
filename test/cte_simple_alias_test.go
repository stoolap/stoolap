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
package test

import (
	"context"
	"testing"

	"github.com/stoolap/stoolap"
)

func TestCTESimpleAlias(t *testing.T) {
	db, err := stoolap.Open("memory://")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	ctx := context.Background()

	// Create test table
	_, err = db.Exec(ctx, `
		CREATE TABLE test (
			a INTEGER,
			b INTEGER
		)
	`)
	if err != nil {
		t.Fatal(err)
	}

	// Insert test data
	_, err = db.Exec(ctx, `
		INSERT INTO test (a, b) VALUES
		(10, 20),
		(30, 40)
	`)
	if err != nil {
		t.Fatal(err)
	}

	// Test 1: Simple alias without WHERE
	t.Run("Simple alias", func(t *testing.T) {
		query := `
			WITH renamed (x, y) AS (
				SELECT a, b FROM test
			)
			SELECT x + y as sum FROM renamed
		`

		rows, err := db.Query(ctx, query)
		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}
		defer rows.Close()

		var count int
		for rows.Next() {
			var sum int
			err := rows.Scan(&sum)
			if err != nil {
				t.Fatalf("Scan failed: %v", err)
			}
			t.Logf("sum=%d", sum)
			count++
		}
		if count != 2 {
			t.Errorf("Expected 2 rows, got %d", count)
		}
	})

	// Test 2: With WHERE on aliased column
	t.Run("With WHERE", func(t *testing.T) {
		query := `
			WITH renamed (x, y) AS (
				SELECT a, b FROM test
			)
			SELECT x, y FROM renamed WHERE x > 20
		`

		rows, err := db.Query(ctx, query)
		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}
		defer rows.Close()

		var count int
		for rows.Next() {
			var x, y int
			err := rows.Scan(&x, &y)
			if err != nil {
				t.Fatalf("Scan failed: %v", err)
			}
			t.Logf("x=%d, y=%d", x, y)
			count++
		}
		if count != 1 {
			t.Errorf("Expected 1 row, got %d", count)
		}
	})
}
