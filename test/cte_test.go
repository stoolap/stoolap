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

func TestBasicCTE(t *testing.T) {
	// Create in-memory database
	db, err := stoolap.Open("memory://")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	ctx := context.Background()

	// Create test table
	_, err = db.Exec(ctx, `
		CREATE TABLE employees (
			id INTEGER PRIMARY KEY,
			name TEXT,
			department TEXT,
			salary FLOAT
		)
	`)
	if err != nil {
		t.Fatal(err)
	}

	// Insert test data
	_, err = db.Exec(ctx, `
		INSERT INTO employees (id, name, department, salary) VALUES
		(1, 'Alice', 'Engineering', 100000),
		(2, 'Bob', 'Engineering', 90000),
		(3, 'Charlie', 'Sales', 80000),
		(4, 'David', 'Sales', 75000),
		(5, 'Eve', 'HR', 70000)
	`)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("Simple CTE", func(t *testing.T) {
		query := `
			WITH high_earners AS (
				SELECT * FROM employees WHERE salary > 85000
			)
			SELECT * FROM high_earners
		`

		rows, err := db.Query(ctx, query)
		if err != nil {
			t.Fatal(err)
		}
		defer rows.Close()

		var count int
		for rows.Next() {
			var id int
			var name, department string
			var salary float64
			err := rows.Scan(&id, &name, &department, &salary)
			if err != nil {
				t.Fatal(err)
			}
			if salary <= 85000 {
				t.Errorf("Expected salary > 85000, got %f", salary)
			}
			count++
		}
		if count != 2 {
			t.Errorf("Expected 2 high earners, got %d", count)
		}
	})

	t.Run("CTE with WHERE clause", func(t *testing.T) {
		query := `
			WITH engineering_team AS (
				SELECT * FROM employees WHERE department = 'Engineering'
			)
			SELECT name, salary FROM engineering_team WHERE salary > 95000
		`

		rows, err := db.Query(ctx, query)
		if err != nil {
			t.Fatal(err)
		}
		defer rows.Close()

		var count int
		for rows.Next() {
			var name string
			var salary float64
			err := rows.Scan(&name, &salary)
			if err != nil {
				t.Fatal(err)
			}
			if name != "Alice" {
				t.Errorf("Expected name to be Alice, got %s", name)
			}
			if salary != 100000 {
				t.Errorf("Expected salary to be 100000, got %f", salary)
			}
			count++
		}
		if count != 1 {
			t.Errorf("Expected 1 result, got %d", count)
		}
	})

	t.Run("CTE with column selection", func(t *testing.T) {
		query := `
			WITH dept_summary AS (
				SELECT department, COUNT(*) as emp_count, AVG(salary) as avg_salary
				FROM employees
				GROUP BY department
			)
			SELECT * FROM dept_summary WHERE emp_count > 1
		`

		rows, err := db.Query(ctx, query)
		if err != nil {
			t.Fatal(err)
		}
		defer rows.Close()

		results := make(map[string]struct {
			count     int64
			avgSalary float64
		})

		for rows.Next() {
			var department string
			var count int64
			var avgSalary float64
			err := rows.Scan(&department, &count, &avgSalary)
			if err != nil {
				t.Fatal(err)
			}
			results[department] = struct {
				count     int64
				avgSalary float64
			}{count, avgSalary}
		}

		// Verify results
		if len(results) != 2 {
			t.Errorf("Expected 2 departments with more than 1 employee, got %d", len(results))
		}

		// Check Engineering department
		eng, ok := results["Engineering"]
		if !ok {
			t.Error("Expected Engineering department")
		} else {
			if eng.count != 2 {
				t.Errorf("Expected Engineering count to be 2, got %d", eng.count)
			}
			if eng.avgSalary != 95000 {
				t.Errorf("Expected Engineering avg salary to be 95000, got %f", eng.avgSalary)
			}
		}

		// Check Sales department
		sales, ok := results["Sales"]
		if !ok {
			t.Error("Expected Sales department")
		} else {
			if sales.count != 2 {
				t.Errorf("Expected Sales count to be 2, got %d", sales.count)
			}
			if sales.avgSalary != 77500 {
				t.Errorf("Expected Sales avg salary to be 77500, got %f", sales.avgSalary)
			}
		}
	})

	t.Run("CTE with column aliases", func(t *testing.T) {
		query := `
			WITH team_info (dept, total_employees, total_salary) AS (
				SELECT department, COUNT(*), SUM(salary)
				FROM employees
				GROUP BY department
			)
			SELECT dept, total_salary / total_employees as avg_salary
			FROM team_info
			WHERE total_employees > 1
		`

		rows, err := db.Query(ctx, query)
		if err != nil {
			t.Fatal(err)
		}
		defer rows.Close()

		var count int
		for rows.Next() {
			var dept string
			var avgSalary float64
			err := rows.Scan(&dept, &avgSalary)
			if err != nil {
				t.Fatal(err)
			}
			if dept != "Engineering" && dept != "Sales" {
				t.Errorf("Expected dept to be Engineering or Sales, got %s", dept)
			}
			count++
		}
		if count != 2 {
			t.Errorf("Expected 2 departments, got %d", count)
		}
	})
}

func TestMultipleCTEs(t *testing.T) {
	// Create in-memory database
	db, err := stoolap.Open("memory://")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	ctx := context.Background()

	// Create test tables
	_, err = db.Exec(ctx, `
		CREATE TABLE employees (
			id INTEGER PRIMARY KEY,
			name TEXT,
			department_id INTEGER,
			salary FLOAT
		)
	`)
	if err != nil {
		t.Fatal(err)
	}

	_, err = db.Exec(ctx, `
		CREATE TABLE departments (
			id INTEGER PRIMARY KEY,
			name TEXT,
			budget FLOAT
		)
	`)
	if err != nil {
		t.Fatal(err)
	}

	// Insert test data
	_, err = db.Exec(ctx, `
		INSERT INTO departments (id, name, budget) VALUES
		(1, 'Engineering', 500000),
		(2, 'Sales', 300000),
		(3, 'HR', 200000)
	`)
	if err != nil {
		t.Fatal(err)
	}

	_, err = db.Exec(ctx, `
		INSERT INTO employees (id, name, department_id, salary) VALUES
		(1, 'Alice', 1, 100000),
		(2, 'Bob', 1, 90000),
		(3, 'Charlie', 2, 80000),
		(4, 'David', 2, 75000),
		(5, 'Eve', 3, 70000)
	`)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("Multiple CTEs", func(t *testing.T) {
		query := `
			WITH high_earners AS (
				SELECT * FROM employees WHERE salary > 80000
			),
			big_departments AS (
				SELECT * FROM departments WHERE budget > 250000
			)
			SELECT e.name, e.salary, d.name as department
			FROM high_earners e
			JOIN big_departments d ON e.department_id = d.id
		`

		rows, err := db.Query(ctx, query)
		if err != nil {
			t.Fatal(err)
		}
		defer rows.Close()

		var count int
		for rows.Next() {
			var name, department string
			var salary float64
			err := rows.Scan(&name, &salary, &department)
			if err != nil {
				t.Fatal(err)
			}
			if salary <= 80000 {
				t.Errorf("Expected salary > 80000, got %f", salary)
			}
			if department != "Engineering" && department != "Sales" {
				t.Errorf("Expected department to be Engineering or Sales, got %s", department)
			}
			count++
		}
		if count != 2 {
			t.Errorf("Expected 2 results, got %d", count)
		}
	})

	t.Run("CTE referencing another CTE", func(t *testing.T) {
		query := `
			WITH dept_stats AS (
				SELECT department_id, COUNT(*) as emp_count, AVG(salary) as avg_salary
				FROM employees
				GROUP BY department_id
			),
			expensive_depts AS (
				SELECT * FROM dept_stats WHERE avg_salary > 75000
			)
			SELECT d.name, e.emp_count, e.avg_salary
			FROM expensive_depts e
			JOIN departments d ON e.department_id = d.id
		`

		rows, err := db.Query(ctx, query)
		if err != nil {
			t.Fatal(err)
		}
		defer rows.Close()

		results := make(map[string]struct {
			count     int64
			avgSalary float64
		})

		for rows.Next() {
			var name string
			var count int64
			var avgSalary float64
			err := rows.Scan(&name, &count, &avgSalary)
			if err != nil {
				t.Fatal(err)
			}
			results[name] = struct {
				count     int64
				avgSalary float64
			}{count, avgSalary}
		}

		// Verify results
		if len(results) != 2 {
			t.Errorf("Expected 2 expensive departments, got %d", len(results))
		}

		// Check Engineering department
		eng, ok := results["Engineering"]
		if !ok {
			t.Error("Expected Engineering department")
		} else {
			if eng.count != 2 {
				t.Errorf("Expected Engineering count to be 2, got %d", eng.count)
			}
			if eng.avgSalary != 95000 {
				t.Errorf("Expected Engineering avg salary to be 95000, got %f", eng.avgSalary)
			}
		}

		// Check Sales department
		sales, ok := results["Sales"]
		if !ok {
			t.Error("Expected Sales department")
		} else {
			if sales.count != 2 {
				t.Errorf("Expected Sales count to be 2, got %d", sales.count)
			}
			if sales.avgSalary != 77500 {
				t.Errorf("Expected Sales avg salary to be 77500, got %f", sales.avgSalary)
			}
		}
	})
}

func TestCTEWithSubqueries(t *testing.T) {
	// Create in-memory database
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

	t.Run("CTE with subquery in WHERE", func(t *testing.T) {
		query := `
			WITH recent_sales AS (
				SELECT * FROM sales WHERE year = 2024
			)
			SELECT product, SUM(amount) as total
			FROM recent_sales
			WHERE amount > (SELECT AVG(amount) FROM recent_sales)
			GROUP BY product
		`

		rows, err := db.Query(ctx, query)
		if err != nil {
			t.Fatal(err)
		}
		defer rows.Close()

		results := make(map[string]float64)
		for rows.Next() {
			var product string
			var total float64
			err := rows.Scan(&product, &total)
			if err != nil {
				t.Fatal(err)
			}
			results[product] = total
		}

		// Average of 2024 sales is (1200+1800+2200+2800)/4 = 2000
		// So we should only get Gadget sales (2200+2800 = 5000)
		if len(results) != 1 {
			t.Errorf("Expected 1 result, got %d", len(results))
		}
		if total, ok := results["Gadget"]; !ok {
			t.Error("Expected Gadget in results")
		} else if total != 5000 {
			t.Errorf("Expected Gadget total to be 5000, got %f", total)
		}
	})

	t.Run("CTE with EXISTS", func(t *testing.T) {
		query := `
			WITH high_value_products AS (
				SELECT product, SUM(amount) as total
				FROM sales
				GROUP BY product
				HAVING SUM(amount) > 5000
			)
			SELECT DISTINCT region
			FROM sales s
			WHERE EXISTS (
				SELECT 1 FROM high_value_products hvp
				WHERE hvp.product = 'Gadget'
			)
		`

		rows, err := db.Query(ctx, query)
		if err != nil {
			t.Fatal(err)
		}
		defer rows.Close()

		var regions []string
		for rows.Next() {
			var region string
			err := rows.Scan(&region)
			if err != nil {
				t.Fatal(err)
			}
			regions = append(regions, region)
		}

		// Since Gadget exists in high_value_products, we should get all regions
		if len(regions) != 2 {
			t.Errorf("Expected 2 regions, got %d", len(regions))
		}
		foundNorth := false
		foundSouth := false
		for _, r := range regions {
			if r == "North" {
				foundNorth = true
			}
			if r == "South" {
				foundSouth = true
			}
		}
		if !foundNorth {
			t.Error("Expected North region in results")
		}
		if !foundSouth {
			t.Error("Expected South region in results")
		}
	})
}
