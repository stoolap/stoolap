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
	"fmt"
	"testing"

	"github.com/stoolap/stoolap"
)

func TestSimpleDistinct(t *testing.T) {
	db, err := stoolap.Open("memory://")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	ctx := context.Background()

	// Create test table
	_, err = db.Exec(ctx, `
		CREATE TABLE test (
			id INTEGER PRIMARY KEY,
			value TEXT
		)
	`)
	if err != nil {
		t.Fatal(err)
	}

	// Insert duplicate values
	_, err = db.Exec(ctx, `
		INSERT INTO test (id, value) VALUES
		(1, 'A'),
		(2, 'B'),
		(3, 'A'),
		(4, 'B'),
		(5, 'A'),
		(6, 'C')
	`)
	if err != nil {
		t.Fatal(err)
	}

	// Test DISTINCT
	query := `SELECT DISTINCT value FROM test ORDER BY value`
	rows, err := db.Query(ctx, query)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()

	var values []string
	for rows.Next() {
		var val string
		rows.Scan(&val)
		values = append(values, val)
	}

	fmt.Printf("DISTINCT values: %v\n", values)
	if len(values) != 3 {
		t.Fatalf("Expected 3 distinct values, got %d: %v", len(values), values)
	}
}
