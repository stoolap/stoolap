---
layout: doc
title: JSON Support
category: Data Types
order: 3
---

# JSON Support

This document details Stoolap's current JSON data type support, capabilities, and best practices for working with JSON data based on the implemented test cases.

## Introduction to JSON in Stoolap

Stoolap provides native support for JSON (JavaScript Object Notation) data, allowing you to store structured data alongside your conventional relational data. The current implementation focuses on:

- Basic JSON storage and validation
- Support for JSON data types in tables
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

Returns the type of a JSON value:

```sql
SELECT JSON_TYPE('{"name": "John"}');  -- Returns: object
SELECT JSON_TYPE('[1, 2, 3]');         -- Returns: array
SELECT JSON_TYPE('"hello"');           -- Returns: string
SELECT JSON_TYPE('123');               -- Returns: number
SELECT JSON_TYPE('true');              -- Returns: boolean
SELECT JSON_TYPE('null');              -- Returns: null
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
    let attributes: String = row.get("attributes")?;
    let parsed_product: Product = serde_json::from_str(&attributes)?;
}
```

## Current Limitations

The current JSON implementation has some limitations:

- No JSON modification functions (JSON_SET, JSON_INSERT, etc.)
- No JSON construction functions (JSON_OBJECT, JSON_ARRAY, etc.)
- No indexing of JSON properties

These features may be implemented in future releases.

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
- **Keep it simple**: Since advanced JSON operations aren't yet supported, use simple JSON structures

### Implementation Tips

- **Validate JSON**: Always validate JSON in your application before insertion
- **Size management**: Keep JSON documents reasonably sized
- **Type safety**: Use proper typing when working with JSON in your application code

## Future JSON Features

The following features may be implemented in future releases:

- JSON modification functions (JSON_SET, JSON_INSERT, etc.)
- JSON construction functions (JSON_OBJECT, JSON_ARRAY, etc.)
- JSON comparison functions (JSON_CONTAINS, etc.)
- Indexing on JSON paths