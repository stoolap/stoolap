---
layout: doc
title: RETURNING Clause
category: SQL Features
order: 19
---

# RETURNING Clause

The RETURNING clause returns the affected rows from INSERT, UPDATE, and DELETE operations without requiring a separate SELECT query.

## INSERT RETURNING

Return the inserted rows:

```sql
INSERT INTO users VALUES (1, 'Alice', 100)
RETURNING id, name;
-- Returns: id=1, name='Alice'
```

Works with multi-row inserts:

```sql
INSERT INTO items VALUES (1, 100), (2, 200), (3, 300)
RETURNING id, value;
-- Returns all 3 inserted rows
```

Return a subset of columns:

```sql
INSERT INTO records VALUES (1, 'a', 'b', 'c')
RETURNING id;
-- Returns only the id column
```

## UPDATE RETURNING

Return the updated rows with their new values:

```sql
UPDATE counters SET count = count + 5 WHERE id = 1
RETURNING id, count;
-- Returns: id=1, count=<new value>
```

Returns all rows that matched the WHERE clause:

```sql
UPDATE scores SET score = score * 2 WHERE score >= 200
RETURNING id, score;
-- Returns multiple updated rows
```

If no rows match, returns an empty result set:

```sql
UPDATE data SET value = 999 WHERE id = 999
RETURNING id, value;
-- Returns empty result set
```

## DELETE RETURNING

Return the deleted rows (with their values before deletion):

```sql
DELETE FROM items WHERE id = 1
RETURNING id, name;
-- Returns the deleted row's data
```

Return multiple deleted rows:

```sql
DELETE FROM products WHERE category = 'discontinued'
RETURNING id, name, price;
-- Returns all deleted rows
```

## Subqueries in RETURNING

A `RETURNING` value may hold a subquery. It sees the tables as the statement found them, and a column of the target in it names the row returned:

```sql
UPDATE products SET price = price * 1.1 WHERE category_id = 3
RETURNING id, price, (SELECT name FROM categories WHERE categories.id = products.category_id);
```

See [Subqueries in RETURNING]({% link _docs/sql-features/subqueries.md %}#subqueries-in-returning) for the rules.

## Use Cases

- **Get auto-generated IDs** after INSERT without a separate SELECT
- **Verify updates** by seeing the new values in a single round-trip
- **Audit deletes** by capturing deleted data for logging
- **Build pipelines** by chaining DML results into application logic
