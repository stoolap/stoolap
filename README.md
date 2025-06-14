<div align="center">
  <img src="logo.svg" alt="Stoolap Logo" width="360">
</div>

<p align="center">
  <a href="https://goreportcard.com/report/github.com/stoolap/stoolap"><img src="https://goreportcard.com/badge/github.com/stoolap/stoolap?style=flat-square" alt="Go Report Card"></a>
  <a href="https://pkg.go.dev/github.com/stoolap/stoolap"><img src="https://img.shields.io/badge/go.dev-reference-007d9c?style=flat-square" alt="go.dev reference"></a>
  <a href="https://github.com/stoolap/stoolap/releases"><img src="https://img.shields.io/github/v/release/stoolap/stoolap?style=flat-square" alt="GitHub release"></a>
  <a href="https://github.com/stoolap/stoolap/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="Apache License 2.0"></a>
</p>

# Stoolap

**NewSQL That Starts Simple!** 
Stoolap is a high-performance, columnar SQL database written in pure Go with zero dependencies. It combines OLTP (transaction) and OLAP (analytical) capabilities in a single engine, making it suitable for hybrid transactional/analytical processing (HTAP) workloads with MVCC support.

## Key Features

- **Pure Go Implementation**: Zero external dependencies for maximum portability
- **ACID Transactions**: Full transaction support with true MVCC (Multi-Version Concurrency Control)
- **Lock-Free Design**: No table-level locks, even for SNAPSHOT isolation
- **Fast Analytical Processing**: Columnar indexes optimized for analytical queries
- **HTAP Capabilities**: Hybrid transactional/analytical processing in one engine
- **Memory-First Design**: Optimized for in-memory performance with optional persistence
- **Vectorized Execution**: SIMD-accelerated operations for high throughput
- **SQL Support**: Rich SQL functionality including JOINs, aggregations, and more
- **JSON Support**: Native JSON data type with optimized storage
- **Go SQL Driver**: Standard database/sql compatible driver
- **PostgreSQL Compatibility**: Wire protocol server for PostgreSQL client compatibility

## Installation

```bash
go get github.com/stoolap/stoolap
```

Or clone the repository and build from source:

```bash
git clone https://github.com/stoolap/stoolap.git
cd stoolap
go build -o stoolap ./cmd/stoolap
go build -o stoolap-pgserver ./cmd/stoolap-pgserver
```

## Usage

### Command Line Interface

Stoolap comes with a built-in CLI for interactive SQL queries:

```bash
# Start CLI with in-memory database
./stoolap -db memory://

# Start CLI with persistent storage
./stoolap -db file:///path/to/data
```

The CLI provides a familiar SQL interface with command history, tab completion, and formatted output.

### Go Application Integration

Stoolap can be used in Go applications using the standard database/sql interface:

```go
package main

import (
	"database/sql"
	"fmt"
	"log"

	_ "github.com/stoolap/stoolap/pkg/driver"
)

func main() {
	// Connect to an in-memory database
	db, err := sql.Open("stoolap", "memory://")
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	// Create a table
	_, err = db.Exec(`
		CREATE TABLE users (
			id INTEGER PRIMARY KEY,
			name TEXT NOT NULL,
			email TEXT,
			created_at TIMESTAMP
		)
	`)
	if err != nil {
		log.Fatal(err)
	}

	// Insert data
	_, err = db.Exec(`
		INSERT INTO users (id, name, email, created_at)
		VALUES (1, 'John Doe', 'john@example.com', NOW())
	`)
	if err != nil {
		log.Fatal(err)
	}

	// Query data
	rows, err := db.Query("SELECT id, name, email FROM users")
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()

	// Process results
	for rows.Next() {
		var id int
		var name, email string
		if err := rows.Scan(&id, &name, &email); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("User: %d, %s, %s\n", id, name, email)
	}
}
```

### PostgreSQL Wire Protocol Server

Stoolap includes a PostgreSQL wire protocol server that allows any PostgreSQL client or driver to connect:

```bash
# Start server with in-memory database (default port 5432)
./stoolap-pgserver

# Start server with persistent database
./stoolap-pgserver -d file:///path/to/database.db

# Start server on custom port
./stoolap-pgserver -b :5433
```

Connect using any PostgreSQL client:

```bash
# Using psql
psql -h localhost -p 5432 -d stoolap

# Using Python (psycopg2)
import psycopg2
conn = psycopg2.connect(host="localhost", port=5432, database="stoolap")

# Using Node.js (pg)
const { Client } = require('pg')
const client = new Client({ host: 'localhost', port: 5432, database: 'stoolap' })
```

The PostgreSQL server supports:
- Full transaction support (BEGIN/COMMIT/ROLLBACK)
- READ COMMITTED and SNAPSHOT isolation levels
- Standard PostgreSQL clients and ORMs
- Prepared statements (basic support)

### Connection Strings

Stoolap supports two storage modes:

- **In-Memory**: `memory://` - Fast, non-persistent storage for maximum performance
- **File-Based**: `file:///path/to/data` - Durable storage with persistence

## Supported SQL Features

### Data Types

- `INTEGER`: 64-bit signed integers
- `FLOAT`: 64-bit floating point numbers
- `TEXT`: UTF-8 encoded strings
- `BOOLEAN`: TRUE/FALSE values
- `TIMESTAMP`: Date and time values
- `JSON`: JSON-formatted data

### SQL Commands

- **DDL**: CREATE/ALTER/DROP TABLE, CREATE/DROP INDEX, CREATE/DROP VIEW
- **DML**: SELECT, INSERT, UPDATE, DELETE, MERGE
- **Queries**: JOINs, GROUP BY, ORDER BY, LIMIT, OFFSET, DISTINCT
- **Subqueries**: Scalar subqueries, IN/NOT IN, EXISTS/NOT EXISTS, correlated subqueries
- **Common Table Expressions (CTEs)**: WITH clause for recursive and non-recursive CTEs
  - Single and multiple CTEs in one query
  - CTEs with column aliases
  - CTEs in JOINs and subqueries
  - CTEs with aggregations and HAVING clauses
  - Nested CTEs (CTEs referencing other CTEs)
- **Temporal Queries**: AS OF TRANSACTION/TIMESTAMP for time travel queries
- **Transactions**: BEGIN/COMMIT/ROLLBACK with isolation levels
- **Indexing**: CREATE INDEX, unique constraints, primary keys, multi-column indexes
- **Functions**: Aggregate (SUM, AVG, COUNT, MIN, MAX) and scalar functions

### Transaction Isolation

Stoolap supports two isolation levels:

- **READ COMMITTED** (default): High concurrency, sees committed changes immediately
- **SNAPSHOT**: Consistent view from transaction start, with write-write conflict detection

```sql
-- Start a transaction with SNAPSHOT isolation
BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT;
SELECT * FROM accounts WHERE id = 1;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
COMMIT;
```

### Temporal Queries (AS OF)

Stoolap supports SQL:2011 temporal queries for accessing historical data:

```sql
-- Query data as of a specific transaction
SELECT * FROM users AS OF TRANSACTION 42;

-- Query data as of a specific timestamp
SELECT * FROM orders AS OF TIMESTAMP '2025-06-10 14:30:00';

-- Compare current and historical data
SELECT 
    current.balance as current_balance,
    historical.balance as yesterday_balance
FROM accounts current
CROSS JOIN (
    SELECT balance FROM accounts 
    AS OF TIMESTAMP '2025-06-09 23:59:59'
    WHERE id = 1
) historical
WHERE current.id = 1;
```

## Performance Optimizations

Stoolap includes numerous performance optimizations:

- **Lock-Free MVCC**: Concurrent writers without serialization bottlenecks
- **Columnar Indexes**: Optimized for analytical queries and compression
- **SIMD Operations**: Vectorized execution for arithmetic, comparisons, and functions
- **Specialized Data Structures**: Custom concurrent maps and B-trees for high throughput
- **Expression Pushdown**: Filters pushed to storage layer for faster execution
- **Type-Specific Optimization**: Optimized operations for different data types
- **Version Chain Management**: Automatic garbage collection of old versions

## Development Status

Stoolap is under active development. While it provides ACID compliance and a rich feature set, it should be considered experimental for production use.

## Documentation

Comprehensive documentation is available at:

- [Official Documentation](https://stoolap.io): Complete documentation with architecture, features, and usage examples

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.

```
Copyright 2025 Stoolap Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
