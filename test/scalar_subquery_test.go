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

	"github.com/stoolap/stoolap/internal/sql/executor"
	"github.com/stoolap/stoolap/internal/storage"
	"github.com/stoolap/stoolap/internal/storage/mvcc"
)

// TestScalarSubqueries tests scalar subqueries in WHERE clauses
func TestScalarSubqueries(t *testing.T) {
	// Create test engine
	cfg := &storage.Config{
		Path: "",
		Persistence: storage.PersistenceConfig{
			Enabled: false,
		},
	}
	eng := mvcc.NewMVCCEngine(cfg)
	if err := eng.Open(); err != nil {
		t.Fatalf("Failed to open MVCC engine: %v", err)
	}
	defer eng.Close()

	// Create executor
	exec := executor.NewExecutor(eng)
	ctx := context.Background()

	// Helper function to execute SQL
	executeSQL := func(query string) error {
		tx, err := eng.BeginTransaction()
		if err != nil {
			return err
		}
		defer tx.Rollback()

		// Execute the query
		_, err = exec.Execute(ctx, tx, query)
		if err != nil {
			return err
		}

		return tx.Commit()
	}

	// Helper function to query and verify
	queryAndVerify := func(query string) ([][]interface{}, error) {
		tx, err := eng.BeginTransaction()
		if err != nil {
			return nil, err
		}
		defer tx.Rollback()

		// Execute the query
		result, err := exec.Execute(ctx, tx, query)
		if err != nil {
			return nil, err
		}
		defer result.Close()

		// Collect results
		var rows [][]interface{}
		for result.Next() {
			row := result.Row()
			values := make([]interface{}, len(row))
			for i, col := range row {
				values[i] = col.AsInterface()
			}
			rows = append(rows, values)
		}

		return rows, nil
	}

	// Create products table
	err := executeSQL(`
		CREATE TABLE products (
			id INTEGER PRIMARY KEY,
			name TEXT,
			price FLOAT,
			category TEXT
		)
	`)
	if err != nil {
		t.Fatalf("Failed to create products table: %v", err)
	}

	// Insert test data
	err = executeSQL(`
		INSERT INTO products (id, name, price, category) VALUES 
		(1, 'Laptop', 1000.0, 'Electronics'),
		(2, 'Mouse', 25.0, 'Electronics'),
		(3, 'Book', 15.0, 'Books'),
		(4, 'Phone', 800.0, 'Electronics'),
		(5, 'Pen', 2.0, 'Stationery')
	`)
	if err != nil {
		t.Fatalf("Failed to insert products: %v", err)
	}

	// Test 1: WHERE column > (SELECT AVG(...))
	rows, err := queryAndVerify(`
		SELECT id, name, price 
		FROM products 
		WHERE price > (SELECT AVG(price) FROM products)
		ORDER BY id
	`)
	if err != nil {
		t.Fatalf("Failed to execute query with > scalar subquery: %v", err)
	}

	// Average price is (1000 + 25 + 15 + 800 + 2) / 5 = 368.4
	// So we expect products with price > 368.4: Laptop and Phone
	expectedProducts := []struct {
		id    int64
		name  string
		price float64
	}{
		{1, "Laptop", 1000.0},
		{4, "Phone", 800.0},
	}

	if len(rows) != len(expectedProducts) {
		t.Errorf("Expected %d products, got %d", len(expectedProducts), len(rows))
	}
	for i, row := range rows {
		if i >= len(expectedProducts) {
			break
		}
		id := row[0].(int64)
		name := row[1].(string)
		price := row[2].(float64)
		if id != expectedProducts[i].id || name != expectedProducts[i].name || price != expectedProducts[i].price {
			t.Errorf("Expected product %d:%s:%f, got %d:%s:%f",
				expectedProducts[i].id, expectedProducts[i].name, expectedProducts[i].price,
				id, name, price)
		}
	}

	// Test 2: WHERE column = (SELECT MAX(...))
	rows, err = queryAndVerify(`
		SELECT id, name, price 
		FROM products 
		WHERE price = (SELECT MAX(price) FROM products)
	`)
	if err != nil {
		t.Fatalf("Failed to execute query with = scalar subquery: %v", err)
	}

	// Should return only Laptop (highest price)
	if len(rows) != 1 {
		t.Errorf("Expected 1 product with max price, got %d", len(rows))
	} else {
		id := rows[0][0].(int64)
		name := rows[0][1].(string)
		if id != 1 || name != "Laptop" {
			t.Errorf("Expected Laptop with max price, got %s", name)
		}
	}

	// Test 3: WHERE column < (SELECT MIN(...))
	rows, err = queryAndVerify(`
		SELECT id, name, price 
		FROM products 
		WHERE price < (SELECT MIN(price) FROM products WHERE category = 'Electronics')
	`)
	if err != nil {
		t.Fatalf("Failed to execute query with < scalar subquery: %v", err)
	}

	// Min price in Electronics is 25.0 (Mouse)
	// So we expect products with price < 25.0: Book and Pen
	expectedCheapProducts := []struct {
		id    int64
		name  string
		price float64
	}{
		{3, "Book", 15.0},
		{5, "Pen", 2.0},
	}

	if len(rows) != len(expectedCheapProducts) {
		t.Errorf("Expected %d products, got %d", len(expectedCheapProducts), len(rows))
	}

	// Test 4: Subquery returning no rows (should be NULL)
	rows, err = queryAndVerify(`
		SELECT id, name 
		FROM products 
		WHERE price = (SELECT MAX(price) FROM products WHERE category = 'NonExistent')
	`)
	if err != nil {
		t.Fatalf("Failed to execute query with NULL scalar subquery: %v", err)
	}

	// Should return no rows because NULL = anything is always false
	if len(rows) != 0 {
		t.Errorf("Expected 0 rows for NULL comparison, got %d", len(rows))
	}

	// Test 5: Error - Subquery returning multiple rows
	err = executeSQL(`
		DELETE FROM products 
		WHERE price = (SELECT price FROM products WHERE category = 'Electronics')
	`)
	if err == nil {
		t.Error("Expected error for scalar subquery returning multiple rows, but got none")
	} else if !contains(err.Error(), "more than one row") {
		t.Errorf("Expected 'more than one row' error, got: %v", err)
	}

	// Test 6: Complex scalar subquery with aggregation
	rows, err = queryAndVerify(`
		SELECT name, price 
		FROM products 
		WHERE price >= (
			SELECT AVG(price) 
			FROM products 
			WHERE category = 'Electronics'
		)
		ORDER BY price DESC
	`)
	if err != nil {
		t.Fatalf("Failed to execute complex scalar subquery: %v", err)
	}

	// Average price for Electronics: (1000 + 25 + 800) / 3 = 608.33
	// Products with price >= 608.33: Laptop and Phone
	if len(rows) != 2 {
		t.Errorf("Expected 2 products above Electronics average, got %d", len(rows))
	}
}

// TestScalarSubqueriesInDML tests scalar subqueries in DELETE and UPDATE statements
func TestScalarSubqueriesInDML(t *testing.T) {
	// Create test engine
	cfg := &storage.Config{
		Path: "",
		Persistence: storage.PersistenceConfig{
			Enabled: false,
		},
	}
	eng := mvcc.NewMVCCEngine(cfg)
	if err := eng.Open(); err != nil {
		t.Fatalf("Failed to open MVCC engine: %v", err)
	}
	defer eng.Close()

	// Create executor
	exec := executor.NewExecutor(eng)
	ctx := context.Background()

	// Helper function to execute SQL
	executeSQL := func(query string) error {
		tx, err := eng.BeginTransaction()
		if err != nil {
			return err
		}
		defer tx.Rollback()

		// Execute the query
		_, err = exec.Execute(ctx, tx, query)
		if err != nil {
			return err
		}

		return tx.Commit()
	}

	// Helper function to count rows
	countRows := func(table string) (int, error) {
		tx, err := eng.BeginTransaction()
		if err != nil {
			return 0, err
		}
		defer tx.Rollback()

		result, err := exec.Execute(ctx, tx, "SELECT COUNT(*) FROM "+table)
		if err != nil {
			return 0, err
		}
		defer result.Close()

		if !result.Next() {
			return 0, nil
		}
		row := result.Row()
		return int(row[0].AsInterface().(int64)), nil
	}

	// Create orders table
	err := executeSQL(`
		CREATE TABLE orders (
			id INTEGER PRIMARY KEY,
			amount FLOAT,
			status TEXT
		)
	`)
	if err != nil {
		t.Fatalf("Failed to create orders table: %v", err)
	}

	// Insert test data
	err = executeSQL(`
		INSERT INTO orders (id, amount, status) VALUES 
		(1, 100.0, 'pending'),
		(2, 500.0, 'pending'),
		(3, 200.0, 'completed'),
		(4, 1000.0, 'pending'),
		(5, 50.0, 'completed')
	`)
	if err != nil {
		t.Fatalf("Failed to insert orders: %v", err)
	}

	// Test 1: DELETE with scalar subquery
	err = executeSQL(`
		DELETE FROM orders 
		WHERE amount < (SELECT AVG(amount) FROM orders WHERE status = 'completed')
	`)
	if err != nil {
		t.Fatalf("Failed to execute DELETE with scalar subquery: %v", err)
	}

	// Average of completed orders: (200 + 50) / 2 = 125
	// Should delete orders with amount < 125: orders 1 and 5
	count, err := countRows("orders")
	if err != nil {
		t.Fatalf("Failed to count orders: %v", err)
	}
	if count != 3 {
		t.Errorf("Expected 3 orders after DELETE, got %d", count)
	}

	// Test 2: UPDATE with scalar subquery
	err = executeSQL(`
		UPDATE orders 
		SET status = 'high_value' 
		WHERE amount > (SELECT AVG(amount) FROM orders)
	`)
	if err != nil {
		t.Fatalf("Failed to execute UPDATE with scalar subquery: %v", err)
	}

	// Verify high_value orders
	tx, _ := eng.BeginTransaction()
	result, _ := exec.Execute(ctx, tx, "SELECT COUNT(*) FROM orders WHERE status = 'high_value'")
	result.Next()
	highValueCount := int(result.Row()[0].AsInterface().(int64))
	result.Close()
	tx.Rollback()

	// After DELETE, we have orders 2(500), 3(200), 4(1000)
	// Average = (500+200+1000)/3 = 566.67
	// Only order 4 (1000) is > 566.67
	if highValueCount != 1 {
		t.Errorf("Expected 1 high_value order, got %d", highValueCount)
	}
}

// Helper function to check if string contains substring
func contains(s, substr string) bool {
	return len(s) >= len(substr) && s[:len(substr)] == substr || len(s) > len(substr) && contains(s[1:], substr)
}
