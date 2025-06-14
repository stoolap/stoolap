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

func TestCTESubqueryAccess(t *testing.T) {
	db, err := stoolap.Open("memory://")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	ctx := context.Background()

	// Create a simple test table
	_, err = db.Exec(ctx, `
		CREATE TABLE test (
			id INTEGER,
			value INTEGER
		)
	`)
	if err != nil {
		t.Fatal(err)
	}

	// Insert test data
	_, err = db.Exec(ctx, `
		INSERT INTO test (id, value) VALUES
		(1, 10),
		(2, 20),
		(3, 30)
	`)
	if err != nil {
		t.Fatal(err)
	}

	// Test 1: Simple CTE access
	t.Run("Simple CTE", func(t *testing.T) {
		query := `
			WITH cte AS (
				SELECT * FROM test WHERE value > 15
			)
			SELECT COUNT(*) FROM cte
		`

		var count int
		row := db.QueryRow(ctx, query)
		err := row.Scan(&count)
		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}
		t.Logf("Simple CTE count: %d", count)
		if count != 2 {
			t.Errorf("Expected 2, got %d", count)
		}
	})

	// Test 2: CTE in scalar subquery
	t.Run("CTE in scalar subquery", func(t *testing.T) {
		query := `
			WITH cte AS (
				SELECT * FROM test WHERE value > 15
			)
			SELECT id, value > (SELECT AVG(value) FROM cte) as above_avg
			FROM test
		`

		rows, err := db.Query(ctx, query)
		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}
		defer rows.Close()

		t.Log("CTE scalar subquery results:")
		for rows.Next() {
			var id int
			var aboveAvg bool
			err := rows.Scan(&id, &aboveAvg)
			if err != nil {
				t.Fatalf("Scan failed: %v", err)
			}
			t.Logf("  id=%d, above_avg=%v", id, aboveAvg)
		}
	})

	// Test 3: Direct subquery (no CTE) for comparison
	t.Run("Direct subquery", func(t *testing.T) {
		query := `
			SELECT id, value > (SELECT AVG(value) FROM test WHERE value > 15) as above_avg
			FROM test
		`

		rows, err := db.Query(ctx, query)
		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}
		defer rows.Close()

		t.Log("Direct subquery results:")
		for rows.Next() {
			var id int
			var aboveAvg bool
			err := rows.Scan(&id, &aboveAvg)
			if err != nil {
				t.Fatalf("Scan failed: %v", err)
			}
			t.Logf("  id=%d, above_avg=%v", id, aboveAvg)
		}
	})

	// Test 4: Debug - what does the CTE subquery return?
	t.Run("Debug CTE subquery", func(t *testing.T) {
		// First, check what the CTE contains
		query1 := `
			WITH cte AS (
				SELECT * FROM test WHERE value > 15
			)
			SELECT AVG(value) FROM cte
		`

		var avg float64
		row := db.QueryRow(ctx, query1)
		err := row.Scan(&avg)
		if err != nil {
			t.Fatalf("CTE AVG query failed: %v", err)
		}
		t.Logf("CTE AVG(value) = %f (should be 25.0)", avg)

		// Now try in a WHERE clause
		query2 := `
			WITH cte AS (
				SELECT * FROM test WHERE value > 15
			)
			SELECT id FROM test
			WHERE value > (SELECT AVG(value) FROM cte)
		`

		rows, err := db.Query(ctx, query2)
		if err != nil {
			t.Fatalf("WHERE subquery failed: %v", err)
		}
		defer rows.Close()

		t.Log("WHERE clause with CTE subquery:")
		count := 0
		for rows.Next() {
			var id int
			err := rows.Scan(&id)
			if err != nil {
				t.Fatalf("Scan failed: %v", err)
			}
			t.Logf("  id=%d", id)
			count++
		}
		t.Logf("Total rows: %d (expecting 1, since only 30 > 25)", count)
	})
}
