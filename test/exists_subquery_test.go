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

// TestExistsSubqueries tests EXISTS/NOT EXISTS subqueries
func TestExistsSubqueries(t *testing.T) {
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

	// Create tables
	err := executeSQL(`
		CREATE TABLE customers (
			id INTEGER PRIMARY KEY,
			name TEXT,
			country TEXT
		)
	`)
	if err != nil {
		t.Fatalf("Failed to create customers table: %v", err)
	}

	err = executeSQL(`
		CREATE TABLE orders (
			id INTEGER PRIMARY KEY,
			customer_id INTEGER,
			amount FLOAT
		)
	`)
	if err != nil {
		t.Fatalf("Failed to create orders table: %v", err)
	}

	// Insert test data
	err = executeSQL(`
		INSERT INTO customers (id, name, country) VALUES 
		(1, 'Alice', 'USA'),
		(2, 'Bob', 'UK'),
		(3, 'Charlie', 'USA'),
		(4, 'David', 'Canada')
	`)
	if err != nil {
		t.Fatalf("Failed to insert customers: %v", err)
	}

	err = executeSQL(`
		INSERT INTO orders (id, customer_id, amount) VALUES 
		(1, 1, 100.0),
		(2, 1, 200.0),
		(3, 3, 150.0),
		(4, 4, 300.0)
	`)
	if err != nil {
		t.Fatalf("Failed to insert orders: %v", err)
	}

	// Test 1: EXISTS - Simple non-correlated test
	// For now, we'll test if any orders exist at all
	rows, err := queryAndVerify(`
		SELECT id, name 
		FROM customers
		WHERE EXISTS (
			SELECT * FROM orders
		)
		ORDER BY id
	`)
	if err != nil {
		t.Fatalf("Failed to execute EXISTS query: %v", err)
	}

	// Since orders exist, all customers should be returned
	if len(rows) != 4 {
		t.Errorf("Expected 4 customers (since orders exist), got %d", len(rows))
	}

	// Test 2: EXISTS with no results - Test when subquery returns no rows
	// First, let's delete all orders
	err = executeSQL(`DELETE FROM orders`)
	if err != nil {
		t.Fatalf("Failed to delete orders: %v", err)
	}

	rows, err = queryAndVerify(`
		SELECT id, name 
		FROM customers
		WHERE EXISTS (
			SELECT 1 FROM orders
		)
		ORDER BY id
	`)
	if err != nil {
		t.Fatalf("Failed to execute EXISTS query with empty subquery: %v", err)
	}

	// Since no orders exist, no customers should be returned
	if len(rows) != 0 {
		t.Errorf("Expected 0 customers (since no orders exist), got %d", len(rows))
	}

	// Test 3: NOT EXISTS - Test when subquery returns no rows
	rows, err = queryAndVerify(`
		SELECT id, name 
		FROM customers
		WHERE NOT EXISTS (
			SELECT * FROM orders
		)
		ORDER BY id
	`)
	if err != nil {
		t.Fatalf("Failed to execute NOT EXISTS query: %v", err)
	}

	// Since no orders exist, all customers should be returned
	if len(rows) != 4 {
		t.Errorf("Expected 4 customers (since no orders exist), got %d", len(rows))
	}

	// Test 4: EXISTS with conditions in subquery
	// Re-insert some orders
	err = executeSQL(`
		INSERT INTO orders (id, customer_id, amount) VALUES 
		(1, 1, 100.0),
		(2, 1, 200.0)
	`)
	if err != nil {
		t.Fatalf("Failed to re-insert orders: %v", err)
	}

	rows, err = queryAndVerify(`
		SELECT id, name 
		FROM customers
		WHERE EXISTS (
			SELECT * FROM orders WHERE amount > 150
		)
		ORDER BY id
	`)
	if err != nil {
		t.Fatalf("Failed to execute EXISTS with condition query: %v", err)
	}

	// Since there is at least one order > 150, all customers should be returned
	if len(rows) != 4 {
		t.Errorf("Expected 4 customers (order > 150 exists), got %d", len(rows))
	}
}

// TestExistsSubqueriesInDML tests EXISTS/NOT EXISTS in DELETE and UPDATE statements
func TestExistsSubqueriesInDML(t *testing.T) {
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

	// Create tables
	err := executeSQL(`
		CREATE TABLE products (
			id INTEGER PRIMARY KEY,
			name TEXT,
			category TEXT,
			in_stock BOOLEAN
		)
	`)
	if err != nil {
		t.Fatalf("Failed to create products table: %v", err)
	}

	err = executeSQL(`
		CREATE TABLE inventory (
			id INTEGER PRIMARY KEY,
			product_id INTEGER,
			quantity INTEGER
		)
	`)
	if err != nil {
		t.Fatalf("Failed to create inventory table: %v", err)
	}

	// Insert test data
	err = executeSQL(`
		INSERT INTO products (id, name, category, in_stock) VALUES 
		(1, 'Laptop', 'Electronics', true),
		(2, 'Mouse', 'Electronics', true),
		(3, 'Book', 'Books', true),
		(4, 'Phone', 'Electronics', true)
	`)
	if err != nil {
		t.Fatalf("Failed to insert products: %v", err)
	}

	err = executeSQL(`
		INSERT INTO inventory (id, product_id, quantity) VALUES 
		(1, 1, 10),
		(2, 3, 5),
		(3, 4, 0)
	`)
	if err != nil {
		t.Fatalf("Failed to insert inventory: %v", err)
	}

	// Test 1: DELETE with EXISTS - Delete products when orders table is empty
	// First ensure orders table is empty
	err = executeSQL(`DELETE FROM inventory`)
	if err != nil {
		t.Fatalf("Failed to delete inventory: %v", err)
	}

	err = executeSQL(`
		DELETE FROM products 
		WHERE EXISTS (
			SELECT * FROM inventory
		)
	`)
	if err != nil {
		t.Fatalf("Failed to execute DELETE with EXISTS: %v", err)
	}

	// Since inventory is empty, EXISTS returns false, so no products should be deleted
	count, err := countRows("products")
	if err != nil {
		t.Fatalf("Failed to count products: %v", err)
	}
	if count != 4 {
		t.Errorf("Expected 4 products after DELETE (no deletions), got %d", count)
	}

	// Test 2: UPDATE with EXISTS - Re-add inventory and test UPDATE
	err = executeSQL(`
		INSERT INTO inventory (id, product_id, quantity) VALUES 
		(1, 1, 10)
	`)
	if err != nil {
		t.Fatalf("Failed to insert inventory: %v", err)
	}

	err = executeSQL(`
		UPDATE products 
		SET in_stock = false 
		WHERE EXISTS (
			SELECT 1 FROM inventory
		)
	`)
	if err != nil {
		t.Fatalf("Failed to execute UPDATE with EXISTS: %v", err)
	}

	// Verify that all products are marked as out of stock
	tx, _ := eng.BeginTransaction()
	result, _ := exec.Execute(ctx, tx, "SELECT COUNT(*) FROM products WHERE in_stock = false")
	result.Next()
	row := result.Row()
	outOfStockCount := row[0].AsInterface().(int64)
	result.Close()
	tx.Rollback()

	if outOfStockCount != 4 {
		t.Errorf("Expected all 4 products to be marked as out of stock, got %d", outOfStockCount)
	}
}
