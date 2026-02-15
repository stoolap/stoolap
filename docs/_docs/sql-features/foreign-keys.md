---
layout: doc
title: Foreign Keys
category: SQL Features
order: 14
---

# Foreign Keys

This document explains foreign key constraints in Stoolap, including syntax, referential actions, cascading behavior, and transaction semantics.

## Overview

Stoolap supports foreign key (FK) constraints that enforce referential integrity between tables. A foreign key ensures that values in a child table column always reference an existing row in a parent table. This prevents orphaned records and maintains data consistency.

| Feature | Description |
|---------|-------------|
| **Zero cost for non-FK tables** | All checks short-circuit when no FK constraints exist |
| **Index-based lookups** | Parent existence checks use PK or secondary indexes (O(1) or O(log N)) |
| **Transaction-aware** | FK checks see uncommitted rows within the same transaction |
| **Automatic FK indexes** | Indexes are auto-created on FK columns for efficient cascade operations |
| **WAL and snapshot durable** | FK metadata survives restart and WAL truncation |

## Syntax

### Column-Level REFERENCES

```sql
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    product_id INTEGER REFERENCES products(id) ON DELETE CASCADE,
    amount FLOAT
);
```

When the referenced column is omitted, it defaults to the parent table's primary key:

```sql
-- References customers(id) — the primary key
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers
);
```

### Table-Level FOREIGN KEY

```sql
CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    FOREIGN KEY(order_id) REFERENCES orders(id) ON DELETE CASCADE,
    FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE RESTRICT
);
```

### Multiple FK Columns

A table can have multiple foreign key columns referencing different parent tables:

```sql
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    dept_id INTEGER REFERENCES departments(id),
    mgr_id INTEGER REFERENCES managers(id),
    name TEXT
);
```

## Referential Actions

Referential actions define what happens to child rows when the referenced parent row is deleted or updated.

### ON DELETE Actions

| Action | Behavior |
|--------|----------|
| `RESTRICT` | Block the delete if child rows exist (default) |
| `NO ACTION` | Same as RESTRICT for immediate constraint checking |
| `CASCADE` | Automatically delete all child rows referencing the parent |
| `SET NULL` | Set the FK column to NULL in all child rows referencing the parent |

### ON UPDATE Actions

| Action | Behavior |
|--------|----------|
| `RESTRICT` | Block the update if child rows reference the old PK value (default) |
| `NO ACTION` | Same as RESTRICT for immediate constraint checking |
| `CASCADE` | Update the FK column in all child rows to the new PK value |
| `SET NULL` | Set the FK column to NULL in all child rows referencing the old PK value |

### Examples

```sql
-- RESTRICT (default): block parent delete if children exist
CREATE TABLE children (
    id INTEGER PRIMARY KEY,
    parent_id INTEGER REFERENCES parents(id)
);

-- CASCADE: delete children when parent is deleted
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id) ON DELETE CASCADE
);

-- SET NULL: nullify FK when parent is deleted
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY,
    assignee_id INTEGER REFERENCES users(id) ON DELETE SET NULL
);

-- Combined actions
CREATE TABLE line_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id) ON DELETE CASCADE ON UPDATE CASCADE
);
```

## NULL FK Values

A NULL value in a FK column means "no reference" and is always allowed, regardless of the referential action. This follows the SQL standard:

```sql
-- NULL FK is valid — the row has no parent reference
INSERT INTO children (id, parent_id, name) VALUES (1, NULL, 'No Parent');
```

## Multi-Level Cascading

CASCADE operations recurse through the FK hierarchy. If a grandparent is deleted, the cascade propagates through parent tables to child tables:

```sql
CREATE TABLE grandparents (id INTEGER PRIMARY KEY, name TEXT);

CREATE TABLE parents (
    id INTEGER PRIMARY KEY,
    gp_id INTEGER REFERENCES grandparents(id) ON DELETE CASCADE
);

CREATE TABLE children (
    id INTEGER PRIMARY KEY,
    parent_id INTEGER REFERENCES parents(id) ON DELETE CASCADE
);

-- Deleting a grandparent cascades through parents to children
DELETE FROM grandparents WHERE id = 1;
-- Deletes matching parents AND their children
```

A maximum recursion depth of 16 levels is enforced to prevent infinite loops from circular references.

### Mixed Actions in Multi-Level Hierarchies

RESTRICT at any level in the cascade chain blocks the entire operation:

```sql
CREATE TABLE grandparents (id INTEGER PRIMARY KEY, name TEXT);

CREATE TABLE parents (
    id INTEGER PRIMARY KEY,
    gp_id INTEGER REFERENCES grandparents(id) ON DELETE CASCADE
);

-- RESTRICT here blocks the cascade from grandparents
CREATE TABLE children (
    id INTEGER PRIMARY KEY,
    parent_id INTEGER REFERENCES parents(id) ON DELETE RESTRICT
);

-- This fails: CASCADE would delete parents, but RESTRICT blocks because children exist
DELETE FROM grandparents WHERE id = 1;
-- Error: cannot cascade-delete row — still referenced by table 'children'
```

## Transaction Semantics

### FK Checks See Uncommitted Rows

Within a single transaction, FK checks see uncommitted inserts and deletes:

```sql
BEGIN;
-- Insert a new parent (not yet committed)
INSERT INTO parents VALUES (100, 'New Parent');
-- This succeeds: the FK check sees the uncommitted parent row
INSERT INTO children VALUES (1, 100, 'Child');
COMMIT;
```

```sql
BEGIN;
-- Delete a parent (not yet committed)
DELETE FROM parents WHERE id = 1;
-- This fails: the FK check sees the uncommitted delete
INSERT INTO children VALUES (2, 1, 'Child');
-- Error: referenced row does not exist
ROLLBACK;
```

### CASCADE Atomicity

CASCADE effects participate in the caller's transaction. A ROLLBACK undoes both the parent operation and all cascaded child changes:

```sql
BEGIN;
-- CASCADE deletes matching children
DELETE FROM parents WHERE id = 1;
-- Both the parent delete AND child cascade are undone
ROLLBACK;
-- All rows are back to their original state
```

## DDL Interactions

### DROP TABLE

DROP TABLE is blocked if any child table has rows with non-NULL FK values referencing the table, regardless of the FK action (RESTRICT, CASCADE, or SET NULL). DDL operations do not cascade to child rows:

```sql
-- Blocked: child rows reference this table
DROP TABLE parents;
-- Error: cannot drop/truncate table 'parents' — rows in 'children' still reference it

-- Solution: delete child rows first, then drop
DELETE FROM children WHERE parent_id IS NOT NULL;
DROP TABLE parents;
```

When a parent table is dropped (after child references are cleared), FK constraints referencing it are automatically stripped from child table schemas:

```sql
DROP TABLE parents;
-- children.parent_id is no longer a FK — inserts with any value succeed
INSERT INTO children VALUES (1, 999, 'No FK');
```

### TRUNCATE TABLE

TRUNCATE is blocked when child tables have referencing rows, same as DROP TABLE:

```sql
-- Blocked if children reference this table
TRUNCATE TABLE parents;

-- Child table truncation is always allowed
TRUNCATE TABLE children;
```

## CREATE TABLE Validation

FK constraints are validated at table creation time:

| Check | Error |
|-------|-------|
| Parent table must exist | `references non-existent table` |
| Referenced column must exist in parent | `references non-existent column` |
| Referenced column must be PRIMARY KEY or UNIQUE | `neither PRIMARY KEY nor UNIQUE` |
| SET NULL requires nullable FK column | `SET NULL but is NOT NULL` |

```sql
-- Rejected: SET NULL on NOT NULL column
CREATE TABLE bad (
    id INTEGER PRIMARY KEY,
    parent_id INTEGER NOT NULL REFERENCES parents(id) ON DELETE SET NULL
);
-- Error: foreign key column 'parent_id' has ON DELETE SET NULL but is NOT NULL
```

## Automatic FK Indexes

When a FK constraint is defined, Stoolap automatically creates an index on the FK column (unless the column already has a PRIMARY KEY or UNIQUE index). This ensures efficient:

- CASCADE DELETE / SET NULL operations (find child rows by FK value)
- RESTRICT checks (quickly determine if child rows exist)

```sql
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    dept_id INTEGER REFERENCES departments(id)
);

-- An index named 'fk_employees_dept_id' is automatically created
SHOW INDEXES FROM employees;
```

## Performance

| Operation | Complexity | Notes |
|-----------|------------|-------|
| INSERT FK check | O(1) or O(log N) | PK fast path or index lookup on parent |
| DELETE RESTRICT check | O(log N) | Index lookup on child FK column |
| DELETE CASCADE | O(K log N) | K = number of child rows to delete |
| DELETE SET NULL | O(K log N) | K = number of child rows to update |
| Non-FK table operations | Zero cost | All checks short-circuit immediately |

## Persistence

FK constraints are persisted through both WAL (Write-Ahead Log) and snapshots. After a crash and recovery, or after WAL truncation:

- FK constraints are fully restored from the latest snapshot
- All referential actions (CASCADE, SET NULL, RESTRICT) work as before
- Auto-created FK indexes are also persisted and restored

## Limitations

- FK constraints reference single columns (composite FK not yet supported)
- ON UPDATE CASCADE does not recursively cascade to grandchild tables when the child's FK column is also its PK
- Self-referencing FK (a table referencing itself) follows the same rules but requires careful ordering of inserts
