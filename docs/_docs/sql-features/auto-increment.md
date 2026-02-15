---
layout: doc
title: Auto Increment
category: SQL Features
order: 20
---

# Auto Increment

Auto increment automatically generates sequential integer values for primary key columns when no explicit value is provided.

## Syntax

```sql
CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    name TEXT NOT NULL,
    price FLOAT
);
```

Both `AUTO_INCREMENT` and `AUTOINCREMENT` are accepted.

## Usage

Omit the auto-increment column to generate the next value:

```sql
INSERT INTO products (name, price) VALUES ('Widget', 9.99);
-- id = 1 (auto-generated)

INSERT INTO products (name, price) VALUES ('Gadget', 19.99);
-- id = 2 (auto-generated)
```

## Explicit Values

You can still provide explicit values for auto-increment columns:

```sql
INSERT INTO products (id, name, price) VALUES (100, 'Special', 49.99);
-- id = 100 (explicitly set)

INSERT INTO products (name, price) VALUES ('Next', 5.99);
-- id = 101 (counter updated to max + 1)
```

## Counter Behavior

The auto-increment counter always tracks the maximum ID seen:

```sql
INSERT INTO products (id, name, price) VALUES (1, 'A', 10);    -- id = 1
INSERT INTO products (id, name, price) VALUES (100, 'B', 20);  -- id = 100
INSERT INTO products (name, price) VALUES ('C', 30);            -- id = 101
INSERT INTO products (id, name, price) VALUES (50, 'D', 40);   -- id = 50 (explicit)
INSERT INTO products (name, price) VALUES ('E', 50);            -- id = 102 (not 51)
```

The counter never regresses. It always generates `max(all_existing_ids) + 1`.

## Constraints

- Auto increment is only supported on `INTEGER PRIMARY KEY` columns
- The counter is maintained per table
- Gaps in the sequence are allowed (e.g., after deletions or explicit inserts)
- The counter is not reset by DELETE (only by DROP TABLE + recreate)
