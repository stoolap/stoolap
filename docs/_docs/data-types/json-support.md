---
layout: doc
title: JSON Support
category: Data Types
order: 3
---

# JSON Support

This document details Stoolap's current JSON data type support, capabilities, and best practices for working with JSON data based on the implemented test cases.

## Introduction to JSON in Stoolap

Stoolap provides native support for JSON (JavaScript Object Notation) data, allowing you to store structured data alongside your conventional relational data. The current implementation includes:

- JSON storage and validation
- JSON extraction and path queries
- JSON construction functions (JSON_OBJECT, JSON_ARRAY)
- JSON type inspection (JSON_TYPE, JSON_TYPEOF)
- Equality comparison for JSON values

## JSON Data Type

Stoolap implements a dedicated JSON data type:

```sql
CREATE TABLE products (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  attributes JSON
);
```

The JSON data type in Stoolap supports:

- **Objects** - Collection of key-value pairs: `{"name": "value", "name2": "value2"}`
- **Arrays** - Ordered collection of values: `[1, 2, 3, "text", true]`
- **Nested structures** - Complex combinations of objects and arrays
- **Primitive values** - Numbers, strings, booleans, and null
- **NULL constraints** - `NOT NULL` constraints can be applied to JSON columns

## Basic JSON Operations

### Inserting JSON Data

```sql
-- Insert as a JSON string
INSERT INTO products (id, name, attributes)
VALUES (1, 'Smartphone', '{"brand": "Example", "color": "black", "specs": {"ram": 8, "storage": 128}}');

-- Insert null into nullable JSON column
INSERT INTO products (id, name, attributes)
VALUES (2, 'Headphones', NULL);

-- Using parameter binding with JSON
INSERT INTO products (id, name, attributes) VALUES ($1, $2, $3);
-- With values: 3, 'Tablet', '{"brand":"Example","model":"T500"}'
```

### Retrieving JSON Data

```sql
-- Fetch entire JSON values
SELECT id, name, attributes FROM products;

-- Filter by non-JSON columns
SELECT id, attributes FROM products WHERE name = 'Smartphone';
```

### Updating JSON Data

```sql
-- Update entire JSON value
UPDATE products 
SET attributes = '{"brand": "Example", "color": "red", "specs": {"ram": 16, "storage": 256}}'
WHERE id = 1;
```

## JSON Validation

As shown in the test files, Stoolap validates JSON syntax during insertion:

```sql
-- Valid JSON will be accepted
INSERT INTO products (id, name, attributes) VALUES (4, 'Valid', '{"brand":"Example"}');

-- Invalid JSON will be rejected
INSERT INTO products (id, name, attributes) VALUES (5, 'Invalid', '{brand:"Example"}');
-- Error: Invalid JSON format
```

Stoolap validates these examples of properly formatted JSON:

```
{"name":"John","age":30}
[1,2,3,4]
{"user":{"name":"John","age":30}}
[{"name":"John"},{"name":"Jane"}]
[]
{}
{"":""}
```

And these examples of invalid JSON:

```
{name:"John"}        -- Missing quotes around property name
{"name":"John"       -- Missing closing brace
{"name":"John",}     -- Trailing comma
{"name":John}        -- Missing quotes around string value
{name}               -- Invalid format
[1,2,3,}             -- Mismatched brackets
```

## JSON Operators

Stoolap supports PostgreSQL-style JSON access operators:

| Operator | Returns | Description |
|----------|---------|-------------|
| `->` | JSON | Extracts a JSON value by key or index |
| `->>` | TEXT | Extracts a value as text |

```sql
-- Extract as JSON (preserves type)
SELECT attributes -> 'specs' FROM products;

-- Extract as text
SELECT attributes ->> 'brand' FROM products;

-- Nested access
SELECT attributes -> 'specs' ->> 'ram' FROM products;

-- Filter by extracted value
SELECT * FROM products WHERE attributes ->> 'brand' = 'Example';
```

These operators are shorthand for `JSON_EXTRACT`. Use `->` when you need to chain further JSON access; use `->>` when you need the final text value.

## JSON Functions

Stoolap provides several functions for working with JSON data:

### JSON_EXTRACT

Extracts a value from JSON using a path:

```sql
SELECT JSON_EXTRACT('{"name": "John", "age": 30}', '$.name');
-- Returns: "John"

SELECT JSON_EXTRACT('{"user": {"name": "John"}}', '$.user.name');
-- Returns: "John"

SELECT JSON_EXTRACT('[1, 2, 3]', '$[0]');
-- Returns: 1
```

### JSON_TYPE / JSON_TYPEOF

Returns the type of a JSON value. Supports both single-argument and two-argument forms:

```sql
-- Single argument: type of the root value
SELECT JSON_TYPE('{"name": "John"}');  -- Returns: object
SELECT JSON_TYPE('[1, 2, 3]');         -- Returns: array
SELECT JSON_TYPE('"hello"');           -- Returns: string
SELECT JSON_TYPE('123');               -- Returns: number
SELECT JSON_TYPE('true');              -- Returns: boolean
SELECT JSON_TYPE('null');              -- Returns: null

-- Two arguments: type of value at path
SELECT JSON_TYPE('{"name": "John", "age": 30}', '$.name');  -- Returns: string
SELECT JSON_TYPE('{"name": "John", "age": 30}', '$.age');   -- Returns: number
SELECT JSON_TYPE('{"user": {"active": true}}', '$.user');   -- Returns: object
SELECT JSON_TYPE('{"tags": ["a", "b"]}', '$.tags');         -- Returns: array
```

Using with table columns:

```sql
SELECT name, JSON_TYPE(metadata, '$.price') AS price_type
FROM products;
```

### JSON_VALID

Checks if a string is valid JSON:

```sql
SELECT JSON_VALID('{"name": "John"}');  -- Returns: true
SELECT JSON_VALID('{invalid}');         -- Returns: false
```

### JSON_KEYS

Returns the keys of a JSON object as an array:

```sql
SELECT JSON_KEYS('{"a": 1, "b": 2, "c": 3}');
-- Returns: ["a", "b", "c"]
```

### JSON_OBJECT

Creates a JSON object from key-value pairs:

```sql
SELECT JSON_OBJECT('name', 'Alice', 'age', 30);
-- Returns: {"name":"Alice","age":30}

SELECT JSON_OBJECT('id', 1, 'active', true, 'score', 95.5);
-- Returns: {"id":1,"active":true,"score":95.5}

-- Empty object
SELECT JSON_OBJECT();
-- Returns: {}
```

### JSON_ARRAY

Creates a JSON array from the provided values:

```sql
SELECT JSON_ARRAY(1, 2, 3);
-- Returns: [1,2,3]

SELECT JSON_ARRAY('a', 'b', 'c');
-- Returns: ["a","b","c"]

SELECT JSON_ARRAY(1, 'mixed', true, null);
-- Returns: [1,"mixed",true,null]

-- Empty array
SELECT JSON_ARRAY();
-- Returns: []
```

### JSON_ARRAY_LENGTH

Returns the length of a JSON array. Supports both single-argument and two-argument forms:

```sql
-- Single argument: length of root array
SELECT JSON_ARRAY_LENGTH('[1, 2, 3, 4, 5]');
-- Returns: 5

SELECT JSON_ARRAY_LENGTH('[]');
-- Returns: 0

-- Two arguments: length of array at path
SELECT JSON_ARRAY_LENGTH('{"tags": ["a", "b", "c"]}', '$.tags');
-- Returns: 3

SELECT JSON_ARRAY_LENGTH('{"users": [{"name": "John"}, {"name": "Jane"}]}', '$.users');
-- Returns: 2
```

## Application Integration

When using Rust with Stoolap, you can work with JSON data using `serde_json`:

```rust
use stoolap::Database;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Product {
    brand: String,
    color: String,
}

let db = Database::open("memory://")?;

// Insert JSON from a Rust struct
let product = Product {
    brand: "Example".to_string(),
    color: "blue".to_string(),
};
let product_json = serde_json::to_string(&product)?;

db.execute(
    "INSERT INTO products (id, name, attributes) VALUES ($1, $2, $3)",
    (6, "Widget", &product_json)
)?;

// Query and parse JSON data
for row in db.query("SELECT attributes FROM products WHERE id = $1", (6,))? {
    let row = row?;
    let attributes: String = row.get_by_name("attributes")?;
    let parsed_product: Product = serde_json::from_str(&attributes)?;
}
```

## Current Limitations

The current JSON implementation has some limitations:

- No JSON modification functions (JSON_SET, JSON_INSERT, JSON_REPLACE, JSON_REMOVE)
- No JSON path query functions (JSON_CONTAINS, JSON_CONTAINS_PATH)
- No indexing of JSON properties

## Example

Here's an example showcasing JSON functionality:

```sql
-- Create a table with JSON column
CREATE TABLE json_extended (
    id INTEGER NOT NULL,
    data JSON
);

-- Insert test data
INSERT INTO json_extended (id, data) VALUES 
(1, '{"name":"John","age":30,"address":{"city":"New York","zip":"10001"},"tags":["developer","manager"]}'),
(2, '{"name":"Alice","age":25,"address":{"city":"Boston","zip":"02108"},"tags":["designer","artist"]}'),
(3, '{"name":"Bob","age":null,"address":null,"tags":[]}'),
(4, '[1,2,3,4,5]'),
(5, '{"numbers":[1,2,3,4,5],"nested":{"a":1,"b":2}}');

-- Simple equality comparison (supported)
SELECT id FROM json_extended WHERE data = '{"name":"John","age":30,"address":{"city":"New York","zip":"10001"},"tags":["developer","manager"]}';
```

## Best Practices

Based on the current implementation:

### Schema Design

- **Hybrid approach**: Store frequently queried fields in regular columns, use JSON for flexible/nested data
- **Don't overuse**: Don't use JSON to avoid proper data modeling
- **Use extraction**: Use JSON_EXTRACT to query nested values efficiently

### Implementation Tips

- **Validate JSON**: Always validate JSON in your application before insertion
- **Size management**: Keep JSON documents reasonably sized
- **Type safety**: Use proper typing when working with JSON in your application code

## Future JSON Features

The following features may be implemented in future releases:

- JSON modification functions (JSON_SET, JSON_INSERT, JSON_REPLACE, JSON_REMOVE)
- JSON path query functions (JSON_CONTAINS, JSON_CONTAINS_PATH)
- JSON aggregation functions (JSON_ARRAYAGG, JSON_OBJECTAGG)
- Indexing on JSON paths