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

func TestCTETotals(t *testing.T) {
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

	// Insert test data
	_, err = db.Exec(ctx, `
		INSERT INTO sales (id, product, region, amount, year) VALUES
		(1, 'Widget', 'North', 1000, 2023),
		(2, 'Widget', 'South', 1500, 2023),
		(3, 'Gadget', 'North', 2000, 2023),
		(4, 'Gadget', 'South', 2500, 2023),
		(5, 'Widget', 'North', 1200, 2024),
		(6, 'Widget', 'South', 1800, 2024),
		(7, 'Gadget', 'North', 2200, 2024),
		(8, 'Gadget', 'South', 2800, 2024)
	`)
	if err != nil {
		t.Fatal(err)
	}

	// Check totals per product
	t.Run("Product totals", func(t *testing.T) {
		query := `
			SELECT product, SUM(amount) as total
			FROM sales
			GROUP BY product
		`

		rows, err := db.Query(ctx, query)
		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}
		defer rows.Close()

		t.Log("Product totals:")
		for rows.Next() {
			var product string
			var total float64
			err := rows.Scan(&product, &total)
			if err != nil {
				t.Fatalf("Scan failed: %v", err)
			}
			t.Logf("  product=%s, total=%f", product, total)
		}
	})
}
