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

func TestCTERegistry(t *testing.T) {
	db, err := stoolap.Open("memory://")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	ctx := context.Background()

	// Create test table
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
		(1, 100),
		(2, 200),
		(3, 300)
	`)
	if err != nil {
		t.Fatal(err)
	}

	// Test: Try to understand what's happening with the CTE registry
	// Let's create a simple case that should work
	query := `
		WITH cte AS (
			SELECT * FROM test WHERE value >= 200
		)
		SELECT id, value, 
		       (SELECT COUNT(*) FROM cte) as cte_count,
		       (SELECT AVG(value) FROM cte) as cte_avg
		FROM cte
	`

	rows, err := db.Query(ctx, query)
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
	defer rows.Close()

	t.Log("CTE with subqueries in SELECT:")
	for rows.Next() {
		var id, value, cteCount int
		var cteAvg float64
		err := rows.Scan(&id, &value, &cteCount, &cteAvg)
		if err != nil {
			t.Fatalf("Scan failed: %v", err)
		}
		t.Logf("  id=%d, value=%d, cte_count=%d, cte_avg=%f", id, value, cteCount, cteAvg)
	}

	// Now test the problematic case
	query2 := `
		WITH cte AS (
			SELECT * FROM test WHERE value >= 200
		)
		SELECT id, value
		FROM cte
		WHERE value > (SELECT AVG(value) FROM cte)
	`

	rows2, err := db.Query(ctx, query2)
	if err != nil {
		t.Fatalf("Query 2 failed: %v", err)
	}
	defer rows2.Close()

	t.Log("\nCTE with subquery in WHERE:")
	count := 0
	for rows2.Next() {
		var id, value int
		err := rows2.Scan(&id, &value)
		if err != nil {
			t.Fatalf("Scan failed: %v", err)
		}
		t.Logf("  id=%d, value=%d", id, value)
		count++
	}
	t.Logf("Total rows: %d (expecting 1, since only 300 > 250)", count)
}
