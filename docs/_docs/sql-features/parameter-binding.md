---
layout: doc
title: Parameter Binding
category: SQL Features
order: 10
---

# Parameter Binding

Stoolap supports parameterized queries using positional parameters (`$1`, `$2`, etc.) to prevent SQL injection and improve performance through query plan reuse.

## Syntax

### Positional Parameters ($N)

Parameters use the `$N` syntax where N is the 1-based position:

```sql
-- Single parameter
SELECT * FROM users WHERE id = $1;

-- Multiple parameters
SELECT * FROM users WHERE age > $1 AND department = $2;

-- Parameters in INSERT
INSERT INTO users (id, name, age) VALUES ($1, $2, $3);

-- Parameters in UPDATE
UPDATE products SET price = $1 WHERE id = $2;
```

### Question Mark Parameters (?)

The `?` placeholder style is also supported. Parameters are bound in order of appearance:

```sql
-- Question mark placeholders
SELECT * FROM users WHERE id = ?;
INSERT INTO users (id, name) VALUES (?, ?);
UPDATE products SET price = ? WHERE id = ?;
```

Both styles work identically. Use whichever is more natural for your application.

## Using Parameters in Rust

### Basic Usage

```rust
use stoolap::Database;

let db = Database::open("memory://")?;

// Create a table
db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)", ())?;

// Insert with tuple parameters
db.execute("INSERT INTO users VALUES ($1, $2, $3)", (1, "Alice", 30))?;

// Query with parameters
for row in db.query("SELECT * FROM users WHERE age > $1", (25,))? {
    let row = row?;
    let name: String = row.get("name")?;
    println!("Name: {}", name);
}
```

### Using the params! Macro

For more complex cases, use the `params!` macro:

```rust
use stoolap::{Database, params};

let db = Database::open("memory://")?;

// Using params! macro
db.execute("INSERT INTO users VALUES ($1, $2, $3)", params![2, "Bob", 25])?;

// With variables
let name = "Charlie";
let age = 35;
db.execute("INSERT INTO users VALUES ($1, $2, $3)", params![3, name, age])?;
```

### Named Parameters

Stoolap also supports named parameters with the `:name` syntax:

```rust
use stoolap::{Database, named_params};

let db = Database::open("memory://")?;

// Using named_params! macro
db.execute_named(
    "INSERT INTO users VALUES (:id, :name, :age)",
    named_params!{ id: 1, name: "Alice", age: 30 }
)?;

// Query with named parameters
for row in db.query_named(
    "SELECT * FROM users WHERE age > :min_age",
    named_params!{ min_age: 25 }
)? {
    // Process rows...
}
```

## Supported Data Types

| Rust Type | SQL Type |
|-----------|----------|
| `i64`, `i32`, `i16`, `i8` | INTEGER |
| `f64`, `f32` | FLOAT |
| `String`, `&str` | TEXT |
| `bool` | BOOLEAN |
| `Option<T>` | NULL or T |

## With Transactions

```rust
use stoolap::Database;

let db = Database::open("memory://")?;

db.execute("BEGIN", ())?;

db.execute("INSERT INTO accounts VALUES ($1, $2)", (1, 1000))?;
db.execute("UPDATE accounts SET balance = balance - $1 WHERE id = $2", (100, 1))?;

db.execute("COMMIT", ())?;
```

## Query Methods

### query() - Multiple Rows

```rust
for row in db.query("SELECT * FROM users WHERE age > $1", (18,))? {
    let row = row?;
    let id: i64 = row.get("id")?;
    let name: String = row.get("name")?;
}
```

### query_one() - Single Value

```rust
let count: i64 = db.query_one("SELECT COUNT(*) FROM users WHERE active = $1", (true,))?;
let name: String = db.query_one("SELECT name FROM users WHERE id = $1", (1,))?;
```

### query_opt() - Optional Value

```rust
let name: Option<String> = db.query_opt("SELECT name FROM users WHERE id = $1", (999,))?;

match name {
    Some(n) => println!("Found: {}", n),
    None => println!("Not found"),
}
```

## Prepared Statements

For repeated queries, use prepared statements:

```rust
let stmt = db.prepare("SELECT * FROM users WHERE id = $1")?;

for id in 1..=100 {
    for row in stmt.query((id,))? {
        let row = row?;
        // Process row...
    }
}
```

## Benefits of Parameters

1. **SQL Injection Prevention**: Parameters are never interpolated into SQL strings
2. **Query Plan Caching**: Same query structure allows plan reuse
3. **Type Safety**: Rust type system ensures correct parameter types
4. **Performance**: Reduced parsing overhead for repeated queries

## Best Practices

1. **Always use parameters for user input** - Never concatenate user data into SQL
2. **Match parameter count** - Number of `$N` placeholders must match parameters provided
3. **Use correct types** - Match Rust types to expected SQL column types
4. **Prefer prepared statements** - For queries executed multiple times

## Common Patterns

### Bulk Insert

```rust
db.execute("BEGIN", ())?;
for (id, name, age) in data {
    db.execute("INSERT INTO users VALUES ($1, $2, $3)", (id, name, age))?;
}
db.execute("COMMIT", ())?;
```

### Dynamic Queries

```rust
// Build query based on optional filters
let mut conditions = vec!["1=1".to_string()];
let mut params: Vec<Value> = vec![];

if let Some(min_age) = filter.min_age {
    conditions.push(format!("age > ${}", params.len() + 1));
    params.push(Value::Integer(min_age));
}

let query = format!("SELECT * FROM users WHERE {}", conditions.join(" AND "));
// Execute with collected parameters
```
