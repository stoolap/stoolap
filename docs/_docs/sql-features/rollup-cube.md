---
title: ROLLUP and CUBE
category: SQL Features
order: 12
---

# ROLLUP and CUBE

ROLLUP and CUBE are extensions to GROUP BY that generate multiple levels of aggregation in a single query. They're useful for generating reports with subtotals and grand totals.

## ROLLUP

ROLLUP creates a hierarchical set of subtotals, rolling up from the most detailed level to a grand total.

### Syntax

```sql
SELECT columns, aggregate_functions
FROM table
GROUP BY ROLLUP(column1, column2, ...);
```

### Example

```sql
CREATE TABLE sales (
    id INTEGER PRIMARY KEY,
    region TEXT,
    category TEXT,
    amount FLOAT
);

INSERT INTO sales VALUES
(1, 'East', 'Electronics', 100),
(2, 'East', 'Electronics', 150),
(3, 'East', 'Clothing', 50),
(4, 'West', 'Electronics', 200),
(5, 'West', 'Clothing', 75),
(6, 'West', 'Clothing', 60);

SELECT region, category, SUM(amount) as total
FROM sales
GROUP BY ROLLUP(region, category);
```

Result:
```
region | category    | total
-------+-------------+-------
East   | Clothing    | 50.0
East   | Electronics | 250.0
West   | Electronics | 200.0
West   | Clothing    | 135.0
East   | NULL        | 300.0    -- Subtotal for East
West   | NULL        | 335.0    -- Subtotal for West
NULL   | NULL        | 635.0    -- Grand total
```

### How ROLLUP Works

For `ROLLUP(region, category)`, it produces:
1. Detail rows: `(region, category)` - each combination
2. Region subtotals: `(region, NULL)` - totals per region
3. Grand total: `(NULL, NULL)` - overall total

The NULL values indicate the aggregation level.

### Column Order Matters

ROLLUP aggregates from right to left:

```sql
-- ROLLUP(a, b, c) produces:
-- (a, b, c) - detail
-- (a, b, NULL) - subtotal by a, b
-- (a, NULL, NULL) - subtotal by a
-- (NULL, NULL, NULL) - grand total
```

## CUBE

CUBE generates all possible combinations of grouping columns, providing a complete multi-dimensional analysis.

### Syntax

```sql
SELECT columns, aggregate_functions
FROM table
GROUP BY CUBE(column1, column2, ...);
```

### Example

```sql
SELECT region, category, SUM(amount) as total
FROM sales
GROUP BY CUBE(region, category);
```

Result:
```
region | category    | total
-------+-------------+-------
East   | Clothing    | 50.0
East   | Electronics | 250.0
West   | Electronics | 200.0
West   | Clothing    | 135.0
East   | NULL        | 300.0    -- Subtotal for East
West   | NULL        | 335.0    -- Subtotal for West
NULL   | Clothing    | 185.0    -- Subtotal for Clothing
NULL   | Electronics | 450.0    -- Subtotal for Electronics
NULL   | NULL        | 635.0    -- Grand total
```

### How CUBE Works

For `CUBE(region, category)`, it produces all 2^n combinations:
1. `(region, category)` - detail rows
2. `(region, NULL)` - totals by region
3. `(NULL, category)` - totals by category
4. `(NULL, NULL)` - grand total

## ROLLUP vs CUBE

| Feature | ROLLUP | CUBE |
|---------|--------|------|
| Subtotals | Hierarchical only | All combinations |
| Groupings | n + 1 | 2^n |
| Use case | Hierarchical reports | Cross-tabulation |

### Grouping Count Example

For 3 columns (a, b, c):

**ROLLUP(a, b, c)** produces 4 groupings:
- (a, b, c)
- (a, b, NULL)
- (a, NULL, NULL)
- (NULL, NULL, NULL)

**CUBE(a, b, c)** produces 8 groupings:
- (a, b, c)
- (a, b, NULL)
- (a, NULL, c)
- (NULL, b, c)
- (a, NULL, NULL)
- (NULL, b, NULL)
- (NULL, NULL, c)
- (NULL, NULL, NULL)

## Use Cases

### ROLLUP for Hierarchical Reports

Time-based hierarchies (year > quarter > month):

```sql
SELECT
    EXTRACT(YEAR FROM order_date) as year,
    EXTRACT(QUARTER FROM order_date) as quarter,
    SUM(amount) as total
FROM orders
GROUP BY ROLLUP(
    EXTRACT(YEAR FROM order_date),
    EXTRACT(QUARTER FROM order_date)
);
```

Geographic hierarchies (country > region > city):

```sql
SELECT country, region, city, SUM(sales) as total
FROM stores
GROUP BY ROLLUP(country, region, city);
```

### CUBE for Cross-Tabulation

Analyze sales by multiple dimensions:

```sql
SELECT product_type, customer_segment, SUM(revenue) as total
FROM sales
GROUP BY CUBE(product_type, customer_segment);
```

This gives totals for:
- Each product_type + customer_segment combination
- Each product_type (all segments)
- Each customer_segment (all products)
- Grand total

## Working with NULL Markers

NULL in the result indicates an aggregation level. To distinguish from actual NULL data:

```sql
SELECT
    COALESCE(region, '(All Regions)') as region,
    COALESCE(category, '(All Categories)') as category,
    SUM(amount) as total
FROM sales
GROUP BY ROLLUP(region, category);
```

Result:
```
region         | category          | total
---------------+-------------------+-------
East           | Clothing          | 50.0
East           | Electronics       | 250.0
West           | Electronics       | 200.0
West           | Clothing          | 135.0
East           | (All Categories)  | 300.0
West           | (All Categories)  | 335.0
(All Regions)  | (All Categories)  | 635.0
```

## Performance Considerations

- **CUBE** generates 2^n groupings, which can be expensive for many columns
- **ROLLUP** is more efficient for hierarchical data (n+1 groupings)
- Consider adding a WHERE clause to reduce input data
- Indexes don't help with ROLLUP/CUBE aggregations

### Recommended Limits

| Columns | ROLLUP Groupings | CUBE Groupings |
|---------|------------------|----------------|
| 2 | 3 | 4 |
| 3 | 4 | 8 |
| 4 | 5 | 16 |
| 5 | 6 | 32 |
| 6 | 7 | 64 |

For more than 4-5 columns, CUBE may produce too many rows.

## Complete Example

```sql
-- Sales analysis with ROLLUP
CREATE TABLE quarterly_sales (
    id INTEGER PRIMARY KEY,
    year INTEGER,
    quarter INTEGER,
    product TEXT,
    revenue FLOAT
);

INSERT INTO quarterly_sales VALUES
(1, 2024, 1, 'Widget', 10000),
(2, 2024, 1, 'Gadget', 15000),
(3, 2024, 2, 'Widget', 12000),
(4, 2024, 2, 'Gadget', 18000),
(5, 2024, 3, 'Widget', 11000),
(6, 2024, 3, 'Gadget', 16000);

-- Hierarchical report: Year > Quarter > Product
SELECT
    year,
    quarter,
    product,
    SUM(revenue) as total_revenue,
    COUNT(*) as transactions
FROM quarterly_sales
GROUP BY ROLLUP(year, quarter, product)
ORDER BY year, quarter, product;
```

This produces a report with:
- Detail rows per product per quarter
- Quarterly totals (all products)
- Yearly totals (all quarters, all products)
- Grand total
