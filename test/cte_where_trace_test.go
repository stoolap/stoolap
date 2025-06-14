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

func TestCTEWhereTrace(t *testing.T) {
	db, err := stoolap.Open("memory://")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	ctx := context.Background()

	// Create test table
	_, err = db.Exec(ctx, `
		CREATE TABLE sales (
			id INTEGER PRIMARY KEY,
			product TEXT,
			region TEXT,
			amount FLOAT,
			year INTEGER
		)
	`)
	if err != nil {
		t.Fatal(err)
	}

	// Insert minimal test data
	_, err = db.Exec(ctx, `
		INSERT INTO sales (id, product, region, amount, year) VALUES
		(5, 'Widget', 'North', 1200, 2024),
		(7, 'Gadget', 'North', 2200, 2024)
	`)
	if err != nil {
		t.Fatal(err)
	}

	// Test 1: Simple WHERE with literal
	t.Run("WHERE with literal", func(t *testing.T) {
		query := `
			WITH recent_sales AS (
				SELECT * FROM sales WHERE year = 2024
			)
			SELECT product, amount FROM recent_sales WHERE amount > 1700
		`

		rows, err := db.Query(ctx, query)
		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}
		defer rows.Close()

		t.Log("WHERE amount > 1700:")
		for rows.Next() {
			var product string
			var amount float64
			err := rows.Scan(&product, &amount)
			if err != nil {
				t.Fatalf("Scan failed: %v", err)
			}
			t.Logf("  product=%s, amount=%f", product, amount)
		}
	})

	// Test 2: Subquery value check
	t.Run("Subquery value", func(t *testing.T) {
		query := `
			WITH recent_sales AS (
				SELECT * FROM sales WHERE year = 2024
			)
			SELECT 
				(SELECT AVG(amount) FROM recent_sales) as avg_from_cte,
				(SELECT COUNT(*) FROM recent_sales) as count_from_cte
		`

		var avg float64
		var count int
		row := db.QueryRow(ctx, query)
		err := row.Scan(&avg, &count)
		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}
		t.Logf("AVG from CTE: %f, COUNT from CTE: %d", avg, count)
	})

	// Test 3: Debug WHERE processing
	t.Run("WHERE processing debug", func(t *testing.T) {
		// Let's see what happens when we process the WHERE clause
		query := `
			WITH recent_sales AS (
				SELECT * FROM sales WHERE year = 2024
			)
			SELECT product, amount,
			       amount as amount_value,
			       1700.0 as literal_value,
			       amount > 1700.0 as gt_literal
			FROM recent_sales
		`

		rows, err := db.Query(ctx, query)
		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}
		defer rows.Close()

		t.Log("Debug WHERE comparisons:")
		for rows.Next() {
			var product string
			var amount, amountValue, literalValue float64
			var gtLiteral bool
			err := rows.Scan(&product, &amount, &amountValue, &literalValue, &gtLiteral)
			if err != nil {
				t.Fatalf("Scan failed: %v", err)
			}
			t.Logf("  product=%s, amount=%f, amount_value=%f, literal=%f, >literal=%v",
				product, amount, amountValue, literalValue, gtLiteral)
		}
	})

	// Test 4: The problematic query
	t.Run("WHERE with subquery", func(t *testing.T) {
		query := `
			WITH recent_sales AS (
				SELECT * FROM sales WHERE year = 2024
			)
			SELECT product, amount
			FROM recent_sales
			WHERE amount > (SELECT AVG(amount) FROM recent_sales)
		`

		rows, err := db.Query(ctx, query)
		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}
		defer rows.Close()

		t.Log("WHERE amount > (SELECT AVG(amount) FROM recent_sales):")
		count := 0
		for rows.Next() {
			var product string
			var amount float64
			err := rows.Scan(&product, &amount)
			if err != nil {
				t.Fatalf("Scan failed: %v", err)
			}
			t.Logf("  product=%s, amount=%f", product, amount)
			count++
		}
		t.Logf("Total rows: %d (expecting 1, since 2200 > 1700)", count)
	})
}
