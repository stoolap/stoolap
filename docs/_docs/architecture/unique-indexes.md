---
layout: doc
title: Unique Indexes
category: Architecture
order: 8
---

# Unique Indexes in Stoolap

This document explains Stoolap's unique index implementation, how uniqueness constraints are enforced, and best practices for using unique indexes.

## Overview of Unique Indexes

Unique indexes ensure that no duplicate values exist in the indexed columns. In Stoolap, they serve two primary purposes:

1. **Data Integrity** - Enforce uniqueness constraints on data
2. **Performance** - Provide fast access paths for lookups

Unique indexes can be created on a single column or across multiple columns (composite unique indexes).

## Creating Unique Indexes

### Basic Syntax

```sql
-- Create a unique index on a single column
CREATE UNIQUE INDEX idx_users_email ON users (email);

-- Create a unique index on multiple columns
CREATE UNIQUE INDEX idx_order_items ON order_items (order_id, product_id);
```

### Primary Keys

Primary keys automatically create an underlying unique index:

```sql
-- Creating a table with a primary key
CREATE TABLE products (
    product_id INT PRIMARY KEY,
    name VARCHAR(255),
    price DECIMAL(10,2)
);

-- This is equivalent to:
CREATE TABLE products (
    product_id INT,
    name VARCHAR(255),
    price DECIMAL(10,2),
    CONSTRAINT pk_products PRIMARY KEY (product_id)
);
```

### Unique Constraints

Unique constraints can also be defined when creating or altering tables:

```sql
-- Adding a unique constraint when creating a table
CREATE TABLE users (
    id INT PRIMARY KEY,
    email VARCHAR(255) UNIQUE,
    username VARCHAR(50)
);

-- Adding a unique constraint to an existing table
ALTER TABLE users ADD CONSTRAINT unique_username UNIQUE (username);
```

### Composite Unique Indexes

Multiple columns can be combined in a single unique index:

```sql
-- Ensure each user has a unique combination of first and last name
CREATE UNIQUE INDEX idx_users_name ON users (first_name, last_name);

-- Ensure each subscription has at most one active plan per user
CREATE UNIQUE INDEX idx_active_subscription ON subscriptions (user_id, plan_id) 
WHERE status = 'active';
```

## How Unique Indexes Work in Stoolap

Stoolap's unique indexes are implemented using specialized versions of the standard indexing mechanisms:

### Implementation Details

Unique indexes in Stoolap use a combination of:

1. **MVCC-Aware Uniqueness Check** - Performs visibility-aware uniqueness validation
2. **Lock-Free Concurrent Operations** - Uses optimistic concurrency for high throughput
3. **Transaction Isolation Compliance** - Respects transaction boundaries for uniqueness
4. **Efficient Duplicate Detection** - Quickly detects and reports duplicates

### Uniqueness Enforcement Process

When inserting or updating a row that affects a unique index:

1. Check if the new values already exist in the index
2. Apply MVCC visibility rules to only consider visible rows
3. If no visible duplicate exists, proceed with the operation
4. If a duplicate is found, abort the operation with a uniqueness violation error
5. For committed transactions, the uniqueness check is final
6. For active transactions, uniqueness is enforced optimistically

### NULL Handling

Stoolap follows the SQL standard for handling NULL values in unique indexes:

- Multiple NULL values are allowed in a unique index (they are not considered equal)
- In a composite unique index, rows are considered duplicate only if all non-NULL values match

## Performance Characteristics

### Benefits

- **Fast Lookups** - Unique indexes provide O(log n) lookup performance
- **Join Optimization** - Joins on unique columns can use optimized paths
- **Data Integrity** - Prevents invalid duplicate data

### Considerations

- **Write Overhead** - Each unique index adds validation overhead to writes
- **Memory Usage** - Unique indexes require additional memory
- **Concurrency Impact** - Uniqueness checks can affect concurrent write performance

## Unique Indexes and MVCC

Stoolap's unique indexes are fully integrated with the MVCC system:

### Visibility Rules

- Uniqueness is enforced based on the transaction's view of the database
- A row deleted in one transaction but not yet committed won't affect uniqueness checks in another transaction
- Different isolation levels may see different versions of rows, but uniqueness is always enforced correctly

### Conflict Resolution

When two transactions attempt to insert the same unique value:

1. The first transaction to commit succeeds
2. The second transaction will fail with a uniqueness violation
3. This implements "first-committer-wins" semantics consistent with MVCC

## Using Unique Indexes Effectively

### Best Practices

1. **Choose appropriate columns** - Use unique indexes for natural keys and business constraints
2. **Consider composite indexes** - Often business uniqueness involves multiple columns
3. **Watch for overhead** - Be aware of the performance impact on write operations
4. **Handle violations gracefully** - Implement proper error handling for uniqueness violations
5. **Monitor index size** - Large unique indexes can impact performance

### Common Use Cases

- **Primary Keys** - Ensure row uniqueness with primary key indexes
- **Natural Keys** - Enforce uniqueness on business identifiers (email, product code)
- **Composite Business Rules** - Enforce complex uniqueness rules (one active subscription per user)
- **Preventing Duplicates** - Ensure data quality by preventing duplicate entries

## Error Handling

When a uniqueness violation occurs, Stoolap returns an error:

```
ERROR: Duplicate value in unique index 'idx_users_email' for value 'user@example.com'
```

Applications should handle these errors gracefully:

```rust
// Example Rust code for handling uniqueness violations
match db.execute("INSERT INTO users (email) VALUES ($1)", (email,)) {
    Ok(_) => Ok(()),
    Err(e) => {
        if e.to_string().contains("Duplicate value in unique index") {
            // Handle uniqueness violation
            Err(format!("email address already exists: {}", e))
        } else {
            // Handle other errors
            Err(e.to_string())
        }
    }
}
```

## Implementation Details

Stoolap's unique index implementation is found in these key components:

- **src/storage/mvcc/btree_index.rs** - B-tree index implementation
- **src/storage/mvcc/hash_index.rs** - Hash index implementation
- **src/storage/mvcc/multi_column_index.rs** - Multi-column index implementation
- **src/storage/mvcc/table.rs** - Table-level uniqueness enforcement
- **src/executor/ddl.rs** - SQL-level uniqueness handling

The uniqueness check algorithm:

```rust
// Simplified pseudocode for uniqueness checking
fn check_uniqueness(
    txn: &Transaction,
    table: &Table,
    index: &UniqueIndex,
    values: &[Value],
) -> Result<(), StoolapError> {
    // Get all potentially matching rows
    let matches = index.find_matches(values)?;

    // Filter for visible rows based on MVCC rules
    let visible_matches = filter_visible(txn, matches);

    // If any visible matches exist (excluding the current row in updates),
    // then we have a uniqueness violation
    if !visible_matches.is_empty() {
        return Err(StoolapError::UniqueConstraintViolation(
            index.name.clone(),
            format!("{:?}", values),
        ));
    }

    Ok(())
}
```

## Limitations

Current limitations of Stoolap's unique index implementation:

1. **No Partial Unique Indexes** - Cannot create unique indexes that apply only to a subset of rows
2. **No Deferrable Constraints** - Uniqueness is always enforced immediately, not at transaction end
3. **Performance with Many Indexes** - Multiple unique indexes can impact write performance