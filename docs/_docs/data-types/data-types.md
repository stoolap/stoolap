---
layout: doc
title: Data Types in Stoolap
category: Data Types
order: 1
---

# Data Types in Stoolap

This document describes the data types supported in Stoolap and how they behave in practice.

## Supported Data Types

Stoolap supports the following data types:

### INTEGER

64-bit signed integer values:

```sql
-- Column definition
CREATE TABLE example (
    id INTEGER PRIMARY KEY,
    count INTEGER,
    large_number INTEGER
);

-- Example values
INSERT INTO example VALUES (1, 42, 9223372036854775807);  -- Max int64
INSERT INTO example VALUES (2, -100, -9223372036854775808);  -- Min int64
```

Features:
- Full range of 64-bit integer values
- Support for PRIMARY KEY constraint
- Auto-increment support

### FLOAT

64-bit floating-point numbers:

```sql
-- Column definition
CREATE TABLE example (
    id INTEGER PRIMARY KEY,
    price FLOAT,
    temperature FLOAT
);

-- Example values
INSERT INTO example VALUES (1, 99.99, -273.15);
INSERT INTO example VALUES (2, 3.14159265359, 1.7976931348623157e+308);  -- Max float64
```

Features:
- Full range of 64-bit floating-point values
- Support for scientific notation

### TEXT

UTF-8 encoded string values:

```sql
-- Column definition
CREATE TABLE example (
    id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT
);

-- Example values
INSERT INTO example VALUES (1, 'Simple text', 'This is a longer description');
INSERT INTO example VALUES (2, 'Unicode: こんにちは', 'Special chars: !@#$%^&*()');
```

Features:
- UTF-8 encoding
- No practical length limit (constrained by available memory)
- Support for quotes and special characters

### BOOLEAN

Boolean true/false values:

```sql
-- Column definition
CREATE TABLE example (
    id INTEGER PRIMARY KEY,
    is_active BOOLEAN,
    is_deleted BOOLEAN
);

-- Example values
INSERT INTO example VALUES (1, true, false);
INSERT INTO example VALUES (2, FALSE, TRUE);  -- Case-insensitive
```

Features:
- Case-insensitive `TRUE`/`FALSE` literals
- Conversion to/from integers (1 = true, 0 = false)

### TIMESTAMP

Date and time values:

```sql
-- Column definition
CREATE TABLE example (
    id INTEGER PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Example values
INSERT INTO example VALUES (1, '2023-01-01 12:00:00', '2023-01-02T15:30:45');
INSERT INTO example VALUES (2, CURRENT_TIMESTAMP, NULL);
```

Features:
- ISO 8601 compatible format
- Support for date and time components
- `NOW()` and `CURRENT_TIMESTAMP` functions for current time
- Date and time functions (`DATE_TRUNC()`, `TIME_TRUNC()`) as shown in tests

### JSON

JSON-formatted data:

```sql
-- Column definition
CREATE TABLE example (
    id INTEGER PRIMARY KEY,
    data JSON
);

-- Example values
INSERT INTO example VALUES (1, '{"name": "John", "age": 30}');
INSERT INTO example VALUES (2, '[1, 2, 3, 4, 5]');
INSERT INTO example VALUES (3, '{"nested": {"a": 1, "b": 2}, "array": [1, 2, 3]}');
```

Features:
- Support for JSON objects and arrays
- Nested structures
- Validation of JSON syntax on insert
- Basic equality comparison
- More details in the dedicated [JSON Support](json-support) documentation

### VECTOR

Fixed-dimension floating-point vectors for similarity search:

```sql
-- Column definition with dimension count
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    content TEXT,
    embedding VECTOR(384)
);

-- Example values (bracket-delimited float arrays)
INSERT INTO embeddings VALUES (1, 'Hello', '[0.1, 0.2, 0.3, ...]');
INSERT INTO embeddings VALUES (2, 'World', '[0.4, 0.5, 0.6, ...]');
```

Features:
- Fixed dimensions specified at table creation (`VECTOR(N)`)
- Stored as packed little-endian f32 arrays for compact storage
- Dimension validation on insert
- Distance functions: `VEC_DISTANCE_L2`, `VEC_DISTANCE_COSINE`, `VEC_DISTANCE_IP`
- HNSW index support for O(log N) approximate nearest neighbor search
- More details in the dedicated [Vector Search](vector-search) documentation

## Column Constraints

Stoolap supports several column constraints:

### PRIMARY KEY

Uniquely identifies each row in a table:

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT
);

-- With AUTO_INCREMENT
CREATE TABLE orders (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    product TEXT
);
```

### NOT NULL

Ensures a column cannot contain NULL values:

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL
);
```

### UNIQUE

Ensures all values in a column are distinct:

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email TEXT UNIQUE,
    username TEXT UNIQUE
);

-- Duplicate values will be rejected
INSERT INTO users VALUES (1, 'alice@test.com', 'alice');
INSERT INTO users VALUES (2, 'alice@test.com', 'bob');  -- Error: unique constraint failed
```

### DEFAULT

Specifies a default value when none is provided:

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT DEFAULT 'Unknown',
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert without specifying defaulted columns
INSERT INTO users (id) VALUES (1);
-- Result: id=1, name='Unknown', active=true, created_at=<current time>
```

Supported default values:
- Literal values: `'text'`, `123`, `3.14`, `true`, `false`
- `NULL`
- `CURRENT_TIMESTAMP` or `NOW()` for timestamps

### CHECK

Validates that values satisfy a condition (column-level constraint):

```sql
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    age INTEGER CHECK(age >= 18 AND age <= 120),
    salary FLOAT CHECK(salary > 0),
    status TEXT CHECK(status IN ('active', 'inactive', 'pending'))
);

-- Valid insert
INSERT INTO employees VALUES (1, 25, 50000, 'active');

-- Invalid insert - fails CHECK constraint
INSERT INTO employees VALUES (2, -5, 50000, 'active');
-- Error: CHECK constraint failed for column age: (age >= 18 AND age <= 120)
```

Note: CHECK must be specified as a column constraint (inline with column definition), not as a table-level constraint.

## NULL Values

Stoolap fully supports NULL values:

```sql
-- Column definition with nullable columns
CREATE TABLE example (
    id INTEGER PRIMARY KEY,
    name TEXT,       -- Implicitly nullable
    value INTEGER,   -- Implicitly nullable
    required TEXT NOT NULL  -- Explicitly non-nullable
);

-- Example values with NULL
INSERT INTO example (id, name, value, required) VALUES (1, NULL, NULL, 'Required');
```

Features:
- Any column can be NULL unless specifically marked as NOT NULL
- NULL handling in indexes
- IS NULL and IS NOT NULL operators
- NULL propagation in expressions
- NULL is distinct from any value, including another NULL
- More details in the dedicated [NULL Handling](../sql-features/null-handling) documentation

## Type Conversions

Stoolap supports type casting between compatible types:

```sql
-- Explicit CAST
SELECT CAST(42 AS TEXT);
SELECT CAST('42' AS INTEGER);
SELECT CAST('2023-01-01' AS TIMESTAMP);

-- Implicit conversion
SELECT '42' + 1;  -- Converts '42' to INTEGER
```

More details on type conversions can be found in the dedicated [CAST Operations](../sql-features/cast-operations) documentation.

## Examples

### Basic Types

```sql
-- Create table with all basic types
CREATE TABLE data_types_test (
    id INTEGER PRIMARY KEY,
    int_val INTEGER,
    float_val FLOAT,
    text_val TEXT,
    bool_val BOOLEAN,
    timestamp_val TIMESTAMP
);

-- Insert test values
INSERT INTO data_types_test VALUES (
    1,                    -- INTEGER
    42,                   -- INTEGER
    3.14,                 -- FLOAT
    'Hello, world!',      -- TEXT
    TRUE,                 -- BOOLEAN
    '2023-01-01 12:00:00' -- TIMESTAMP
);
```

### Timestamp Operations

```sql
-- Create table for timestamp testing
CREATE TABLE timestamp_test (
    id INTEGER PRIMARY KEY,
    event_time TIMESTAMP
);

-- Insert timestamps in different formats
INSERT INTO timestamp_test VALUES (1, '2023-05-15 14:30:45');
INSERT INTO timestamp_test VALUES (2, '2023-05-15T14:30:45');
INSERT INTO timestamp_test VALUES (3, '2023-05-15');

-- Query with time functions
SELECT id, DATE_TRUNC('day', event_time) FROM timestamp_test;
```

### JSON Data

```sql
-- Create table with JSON column
CREATE TABLE json_test (
    id INTEGER PRIMARY KEY,
    data JSON
);

-- Insert different JSON structures
INSERT INTO json_test VALUES (1, '{"name":"John","age":30}');
INSERT INTO json_test VALUES (2, '[1,2,3,4]');
INSERT INTO json_test VALUES (3, '{"user":{"name":"John","age":30}}');
```

## Data Type Storage and Performance

Based on implementation details in the code:

- INTEGER and BOOLEAN types are stored efficiently with native Rust types
- TEXT strings use UTF-8 encoding for maximum compatibility
- TIMESTAMP values are stored as Unix time with nanosecond precision
- JSON values are validated on insert but stored as string representation
- VECTOR values are stored as packed little-endian f32 bytes for zero-copy distance computation
- Row data is compressed using LZ4 in snapshots for compact storage

## Best Practices

- Use the most appropriate data type for your data
- Use INTEGER for IDs and counters
- Use BOOLEAN for true/false flags rather than INTEGER
- Use JSON only for genuinely structured/schemaless data
- Use VECTOR for embedding and similarity search workloads
- Consider type-specific optimizations in WHERE clauses
- Use TIMESTAMP for date and time values rather than storing as TEXT
