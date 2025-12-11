---
layout: doc
title: Common Table Expressions (CTEs)
category: SQL Features
order: 9
---

# Common Table Expressions (CTEs)

Common Table Expressions (CTEs) provide a way to define temporary named result sets that can be used within a SELECT, INSERT, UPDATE, or DELETE statement. CTEs make complex queries more readable and maintainable by breaking them into simpler, reusable parts.

## Basic Syntax

```sql
WITH cte_name [(column1, column2, ...)] AS (
    -- CTE query definition
    SELECT ...
)
-- Main query that uses the CTE
SELECT * FROM cte_name;
```

## Simple CTE Examples

### Basic CTE

```sql
WITH high_value_orders AS (
    SELECT * FROM orders WHERE total_amount > 1000
)
SELECT * FROM high_value_orders;
```

### CTE with Column Aliases

You can specify custom column names for the CTE:

```sql
WITH dept_summary (dept_name, employee_count, avg_salary) AS (
    SELECT department, COUNT(*), AVG(salary)
    FROM employees
    GROUP BY department
)
SELECT * FROM dept_summary WHERE avg_salary > 50000;
```

### CTE with WHERE Clause

```sql
WITH engineering_employees AS (
    SELECT id, name, salary 
    FROM employees 
    WHERE department = 'Engineering'
)
SELECT * FROM engineering_employees WHERE salary > 80000;
```

## Multiple CTEs

You can define multiple CTEs in a single query by separating them with commas:

```sql
WITH 
high_salary AS (
    SELECT * FROM employees WHERE salary > 100000
),
low_salary AS (
    SELECT * FROM employees WHERE salary < 50000
)
SELECT 'High' as category, COUNT(*) as count FROM high_salary
UNION ALL
SELECT 'Low' as category, COUNT(*) as count FROM low_salary;
```

## Nested CTEs

CTEs can reference other CTEs defined earlier in the same WITH clause:

```sql
WITH 
dept_totals AS (
    SELECT department, SUM(salary) as total_salary
    FROM employees
    GROUP BY department
),
above_average_depts AS (
    SELECT * FROM dept_totals
    WHERE total_salary > (SELECT AVG(total_salary) FROM dept_totals)
)
SELECT * FROM above_average_depts;
```

## CTEs with Joins

CTEs work seamlessly with JOIN operations:

```sql
WITH 
customer_orders AS (
    SELECT customer_id, COUNT(*) as order_count, SUM(total) as total_spent
    FROM orders
    GROUP BY customer_id
)
SELECT c.name, co.order_count, co.total_spent
FROM customers c
JOIN customer_orders co ON c.id = co.customer_id
WHERE co.total_spent > 10000;
```

## CTEs with Subqueries

CTEs can contain subqueries in their definition:

```sql
WITH top_customers AS (
    SELECT * FROM customers
    WHERE id IN (
        SELECT customer_id
        FROM orders
        GROUP BY customer_id
        HAVING SUM(total) > 5000
    )
)
SELECT * FROM top_customers;
```

## Recursive CTEs

Stoolap supports recursive CTEs using the `WITH RECURSIVE` syntax. Recursive CTEs are useful for hierarchical data, generating sequences, and graph traversal.

### Basic Syntax

```sql
WITH RECURSIVE cte_name AS (
    -- Base case (anchor member)
    SELECT ...
    UNION ALL
    -- Recursive case (references cte_name)
    SELECT ... FROM cte_name WHERE ...
)
SELECT * FROM cte_name;
```

### Generating Number Sequences

```sql
-- Generate numbers 1 to 10
WITH RECURSIVE numbers AS (
    SELECT 1 as n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < 10
)
SELECT * FROM numbers;
```

### Hierarchical Data (Organization Chart)

```sql
-- Find all employees under a manager
WITH RECURSIVE org_chart AS (
    -- Base case: start with the CEO
    SELECT id, name, manager_id, 1 as level
    FROM employees
    WHERE manager_id IS NULL
    UNION ALL
    -- Recursive case: find direct reports
    SELECT e.id, e.name, e.manager_id, oc.level + 1
    FROM employees e
    JOIN org_chart oc ON e.manager_id = oc.id
)
SELECT * FROM org_chart ORDER BY level, name;
```

### Category Hierarchy

```sql
-- Find all subcategories under a parent category
WITH RECURSIVE category_tree AS (
    SELECT id, name, parent_id, name as path
    FROM categories
    WHERE parent_id IS NULL
    UNION ALL
    SELECT c.id, c.name, c.parent_id, ct.path || ' > ' || c.name
    FROM categories c
    JOIN category_tree ct ON c.parent_id = ct.id
)
SELECT * FROM category_tree;
```

### Fibonacci Sequence

```sql
-- Generate Fibonacci numbers
WITH RECURSIVE fib AS (
    SELECT 1 as n, 1 as fib_n, 1 as fib_n1
    UNION ALL
    SELECT n + 1, fib_n1, fib_n + fib_n1
    FROM fib
    WHERE n < 10
)
SELECT n, fib_n as fibonacci FROM fib;
```

### Recursive CTE Guidelines

1. **Termination Condition**: Always include a WHERE clause in the recursive part to prevent infinite loops
2. **UNION ALL**: Required between the anchor and recursive members
3. **Single Reference**: The recursive member can only reference the CTE once
4. **No Aggregates**: Aggregate functions are not allowed in the recursive member

## Performance Considerations

1. **Materialization**: In Stoolap, CTEs are evaluated once and their results are stored in memory for the duration of the query
2. **No Indexes**: CTE results don't have indexes, so filtering should be done in the CTE definition when possible
3. **Memory Usage**: Large CTEs consume memory, so be mindful of the result set size

## Use Cases

### Data Aggregation

```sql
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', order_date) as month,
        SUM(total) as revenue,
        COUNT(*) as order_count
    FROM orders
    GROUP BY DATE_TRUNC('month', order_date)
)
SELECT 
    month,
    revenue,
    order_count,
    revenue / order_count as avg_order_value
FROM monthly_sales
ORDER BY month;
```

### Hierarchical Data

```sql
WITH RECURSIVE org_chart AS (
    -- Anchor: CEO (no manager)
    SELECT id, name, manager_id, 0 as level
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive: Employees with managers
    SELECT e.id, e.name, e.manager_id, oc.level + 1
    FROM employees e
    JOIN org_chart oc ON e.manager_id = oc.id
)
SELECT * FROM org_chart ORDER BY level, name;
```

### Complex Calculations

```sql
WITH 
order_stats AS (
    SELECT 
        customer_id,
        COUNT(*) as order_count,
        AVG(total) as avg_order,
        MAX(total) as max_order
    FROM orders
    GROUP BY customer_id
),
customer_categories AS (
    SELECT 
        customer_id,
        CASE 
            WHEN order_count > 10 AND avg_order > 100 THEN 'VIP'
            WHEN order_count > 5 THEN 'Regular'
            ELSE 'Occasional'
        END as category
    FROM order_stats
)
SELECT c.name, cc.category, os.order_count, os.avg_order
FROM customers c
JOIN customer_categories cc ON c.id = cc.customer_id
JOIN order_stats os ON c.id = os.customer_id
ORDER BY os.order_count DESC;
```

## Full CTE Support

Stoolap provides comprehensive support for Common Table Expressions (CTEs) with the following features:

### Supported Features
- Single and multiple CTEs in SELECT statements
- CTE column aliases with custom names
- WHERE clauses within CTEs
- CTEs in subqueries (IN, NOT IN, EXISTS, NOT EXISTS, scalar subqueries)
- CTEs as table sources in JOINs
- Aggregate functions on CTEs (MIN, MAX, COUNT, SUM, AVG)
- HAVING clauses with aggregate functions
- CTEs referencing other CTEs (nested CTEs)
- Complex expressions including comparisons and calculations
- CTEs with scalar subqueries in SELECT expressions

### Current Limitations
1. **DML Operations**: CTEs in UPDATE, DELETE, or INSERT statements are not yet supported
2. **Performance**: CTE results are materialized in memory, which may impact performance for very large datasets

### Performance Considerations
- CTEs are evaluated once and their results are stored efficiently in memory
- Aggregate operations (COUNT, SUM, AVG, MIN, MAX) are optimized for performance
- For large datasets, CTEs provide better performance than repeated subqueries due to single materialization
- The query optimizer automatically applies optimizations when possible

## Best Practices

1. **Use Descriptive Names**: Give CTEs meaningful names that describe their purpose
2. **Keep It Simple**: Each CTE should have a single, clear purpose
3. **Order Matters**: Define CTEs in logical order, with dependencies appearing first
4. **Performance**: For large datasets, consider whether a temporary table might be more appropriate
5. **Column Aliases**: Use column aliases in CTEs to make the results clearer

## See Also

- [Subqueries]({% link _docs/sql-features/subqueries.md %})
- [SELECT Statement]({% link _docs/sql-commands/sql-commands.md %}#select)
- [Query Optimization]({% link _docs/performance/optimization.md %})