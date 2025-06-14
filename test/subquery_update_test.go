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

// TestUpdateWithINSubquery tests UPDATE statements with IN subqueries
func TestUpdateWithINSubquery(t *testing.T) {
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
			category TEXT,
			price FLOAT,
			discount FLOAT
		)
	`)
	if err != nil {
		t.Fatalf("Failed to create products table: %v", err)
	}

	// Create categories table
	err = executeSQL(`
		CREATE TABLE categories (
			id INTEGER PRIMARY KEY,
			name TEXT,
			is_premium BOOLEAN
		)
	`)
	if err != nil {
		t.Fatalf("Failed to create categories table: %v", err)
	}

	// Insert test data - categories
	err = executeSQL(`
		INSERT INTO categories (id, name, is_premium) VALUES 
		(1, 'Electronics', true),
		(2, 'Books', false),
		(3, 'Clothing', true),
		(4, 'Food', false)
	`)
	if err != nil {
		t.Fatalf("Failed to insert categories: %v", err)
	}

	// Insert test data - products
	err = executeSQL(`
		INSERT INTO products (id, name, category, price, discount) VALUES 
		(1, 'Laptop', 'Electronics', 1000.0, 0.0),
		(2, 'Novel', 'Books', 20.0, 0.0),
		(3, 'Shirt', 'Clothing', 50.0, 0.0),
		(4, 'Phone', 'Electronics', 800.0, 0.0),
		(5, 'Bread', 'Food', 5.0, 0.0)
	`)
	if err != nil {
		t.Fatalf("Failed to insert products: %v", err)
	}

	// Test 1: Update discount for products in premium categories
	err = executeSQL(`
		UPDATE products 
		SET discount = 0.15 
		WHERE category IN (
			SELECT name FROM categories WHERE is_premium = true
		)
	`)
	if err != nil {
		t.Fatalf("Failed to execute UPDATE with IN subquery: %v", err)
	}

	// Verify discounts were updated correctly
	rows, err := queryAndVerify("SELECT id, name, discount FROM products ORDER BY id")
	if err != nil {
		t.Fatalf("Failed to query products after UPDATE: %v", err)
	}

	expectedDiscounts := []struct {
		id       int64
		name     string
		discount float64
	}{
		{1, "Laptop", 0.15}, // Electronics - premium
		{2, "Novel", 0.0},   // Books - not premium
		{3, "Shirt", 0.15},  // Clothing - premium
		{4, "Phone", 0.15},  // Electronics - premium
		{5, "Bread", 0.0},   // Food - not premium
	}

	if len(rows) != len(expectedDiscounts) {
		t.Errorf("Expected %d products, got %d", len(expectedDiscounts), len(rows))
	}
	for i, row := range rows {
		if i >= len(expectedDiscounts) {
			break
		}
		id := row[0].(int64)
		name := row[1].(string)
		discount := row[2].(float64)
		if id != expectedDiscounts[i].id || name != expectedDiscounts[i].name || discount != expectedDiscounts[i].discount {
			t.Errorf("Expected product %d:%s:%f, got %d:%s:%f",
				expectedDiscounts[i].id, expectedDiscounts[i].name, expectedDiscounts[i].discount,
				id, name, discount)
		}
	}

	// Test 2: Update price for products NOT in premium categories
	err = executeSQL(`
		UPDATE products 
		SET price = price * 0.9 
		WHERE category NOT IN (
			SELECT name FROM categories WHERE is_premium = true
		)
	`)
	if err != nil {
		t.Fatalf("Failed to execute UPDATE with NOT IN subquery: %v", err)
	}

	// Verify prices were updated correctly
	rows, err = queryAndVerify("SELECT id, name, price FROM products ORDER BY id")
	if err != nil {
		t.Fatalf("Failed to query products after price UPDATE: %v", err)
	}

	expectedPrices := []struct {
		id    int64
		name  string
		price float64
	}{
		{1, "Laptop", 1000.0}, // Electronics - premium (unchanged)
		{2, "Novel", 18.0},    // Books - not premium (20 * 0.9)
		{3, "Shirt", 50.0},    // Clothing - premium (unchanged)
		{4, "Phone", 800.0},   // Electronics - premium (unchanged)
		{5, "Bread", 4.5},     // Food - not premium (5 * 0.9)
	}

	if len(rows) != len(expectedPrices) {
		t.Errorf("Expected %d products, got %d", len(expectedPrices), len(rows))
	}
	for i, row := range rows {
		if i >= len(expectedPrices) {
			break
		}
		id := row[0].(int64)
		name := row[1].(string)
		price := row[2].(float64)
		if id != expectedPrices[i].id || name != expectedPrices[i].name || price != expectedPrices[i].price {
			t.Errorf("Expected product %d:%s:%f, got %d:%s:%f",
				expectedPrices[i].id, expectedPrices[i].name, expectedPrices[i].price,
				id, name, price)
		}
	}
}
