---
layout: doc
title: PHP Driver
category: Drivers
order: 3
icon: php
---

# PHP Driver

High-performance PHP driver for Stoolap. Built as a native PHP extension (C) for minimal overhead.

## Installation

### Prebuilt Binaries

Download a package matching your PHP version and platform from [GitHub Releases](https://github.com/stoolap/stoolap-php/releases). Each archive contains both the PHP extension and the stoolap library.

```bash
tar xzf stoolap-v0.4.0-php8.4-linux-x86_64.tar.gz
cd stoolap-v0.4.0-php8.4-linux-x86_64

# Install the shared library
sudo cp libstoolap.so /usr/local/lib/
sudo ldconfig  # Linux only

# Install the PHP extension
sudo cp stoolap.so $(php-config --extension-dir)/
```

### Using PIE

Requires the stoolap shared library (`libstoolap.so` / `libstoolap.dylib`) installed on your system.

```bash
pie install stoolap/stoolap-php
```

### From Source

Requires a C compiler and the stoolap shared library.

```bash
cd ext
phpize
./configure --with-stoolap=/path/to/libstoolap
make
sudo make install
```

### Enable the Extension

```ini
; php.ini or conf.d/stoolap.ini
extension=stoolap
```

Or load it per-invocation:

```bash
php -d extension=stoolap.so your_script.php
```

## Quick Start

```php
use Stoolap\Database;

$db = Database::open(':memory:');

$db->exec('
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT
    )
');

// Insert with positional parameters ($1, $2, ...)
$db->execute(
    'INSERT INTO users (id, name, email) VALUES ($1, $2, $3)',
    [1, 'Alice', 'alice@example.com']
);

// Insert with named parameters (:key)
$db->execute(
    'INSERT INTO users (id, name, email) VALUES (:id, :name, :email)',
    ['id' => 2, 'name' => 'Bob', 'email' => 'bob@example.com']
);

// Query rows as associative arrays
$users = $db->query('SELECT * FROM users ORDER BY id');
// [['id' => 1, 'name' => 'Alice', 'email' => 'alice@example.com'], ...]

// Query single row
$user = $db->queryOne('SELECT * FROM users WHERE id = $1', [1]);
// ['id' => 1, 'name' => 'Alice', 'email' => 'alice@example.com']

// Query in raw columnar format (faster, no per-row key creation)
$raw = $db->queryRaw('SELECT id, name FROM users ORDER BY id');
// ['columns' => ['id', 'name'], 'rows' => [[1, 'Alice'], [2, 'Bob']]]

$db->close();
```

## Opening a Database

```php
// In-memory
$db = Database::open(':memory:');
$db = Database::open('');
$db = Database::openInMemory();

// File-based (data persists across restarts)
$db = Database::open('./mydata');
$db = Database::open('file:///absolute/path/to/db');
```

## Database Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `Database::open($dsn)` | `Database` | Open a database |
| `Database::openInMemory()` | `Database` | Open an in-memory database |
| `exec($sql)` | `int` | Execute DDL/DML, returns affected rows |
| `execute($sql, $params)` | `int` | Execute parameterized DML, returns affected rows |
| `executeBatch($sql, $paramsArray)` | `int` | Execute with multiple param sets in a transaction |
| `query($sql, $params?)` | `array` | Query rows as associative arrays |
| `queryOne($sql, $params?)` | `?array` | Query first row or null |
| `queryRaw($sql, $params?)` | `array` | Query in columnar format |
| `prepare($sql)` | `Statement` | Create a prepared statement |
| `begin()` | `Transaction` | Begin a read-write transaction |
| `beginSnapshot()` | `Transaction` | Begin a snapshot (read) transaction |
| `clone()` | `Database` | Clone the database handle |
| `close()` | `void` | Close the database |
| `version()` | `string` | Get stoolap engine version |

`queryOne()` automatically appends `LIMIT 1` when the SQL has no LIMIT clause. For prepared statements, add `LIMIT 1` yourself since the SQL is fixed at prepare time.

## Persistence

File-based databases persist data to disk using WAL (Write-Ahead Logging) and immutable cold volumes. A background checkpoint cycle seals hot rows into columnar volume files, compacts them, and truncates the WAL. Data survives process restarts.

```php
$db = Database::open('./mydata');

$db->exec('CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT)');
$db->execute('INSERT INTO kv VALUES ($1, $2)', ['hello', 'world']);
$db->close();

// Reopen: data is still there
$db2 = Database::open('./mydata');
$row = $db2->queryOne('SELECT * FROM kv WHERE key = $1', ['hello']);
// ['key' => 'hello', 'value' => 'world']
$db2->close();
```

## Raw Query Format

`queryRaw()` returns `['columns' => [...], 'rows' => [[...], ...]]` instead of an array of associative arrays. Faster when you don't need named keys.

```php
$raw = $db->queryRaw('SELECT id, name, email FROM users ORDER BY id');
// $raw['columns'] => ['id', 'name', 'email']
// $raw['rows']    => [[1, 'Alice', 'alice@example.com'], [2, 'Bob', 'bob@example.com']]
```

## Batch Execution

Execute the same SQL with multiple parameter sets in a single atomic transaction. SQL is parsed once and reused for every row.

```php
$db->executeBatch(
    'INSERT INTO users (id, name, email) VALUES ($1, $2, $3)',
    [
        [1, 'Alice', 'alice@example.com'],
        [2, 'Bob', 'bob@example.com'],
        [3, 'Charlie', 'charlie@example.com'],
    ]
);
// Returns total affected rows (3)
```

On error, all changes are rolled back (atomic). Also available on transactions via `$tx->executeBatch()`.

## Prepared Statements

Prepared statements parse SQL once and reuse the execution plan on every call. No parsing overhead per execution.

```php
$insert = $db->prepare('INSERT INTO users VALUES ($1, $2, $3)');
$insert->execute([1, 'Alice', 'alice@example.com']);
$insert->execute([2, 'Bob', 'bob@example.com']);

$lookup = $db->prepare('SELECT * FROM users WHERE id = $1');
$user = $lookup->queryOne([1]);
// ['id' => 1, 'name' => 'Alice', 'email' => 'alice@example.com']
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `execute($params?)` | `int` | Execute DML, returns affected rows |
| `query($params?)` | `array` | Query rows as associative arrays |
| `queryOne($params?)` | `?array` | Query single row or null |
| `queryRaw($params?)` | `array` | Query in columnar format |
| `sql()` | `string` | Get the SQL text |
| `finalize()` | `void` | Release the prepared statement |

Statements are automatically finalized on garbage collection.

## Transactions

```php
$tx = $db->begin();
try {
    $tx->execute(
        'INSERT INTO users VALUES ($1, $2, $3)',
        [1, 'Alice', 'alice@example.com']
    );
    $tx->execute(
        'INSERT INTO users VALUES ($1, $2, $3)',
        [2, 'Bob', 'bob@example.com']
    );

    // Read within the transaction (sees uncommitted changes)
    $rows = $tx->query('SELECT * FROM users');
    $one = $tx->queryOne('SELECT * FROM users WHERE id = $1', [1]);
    $raw = $tx->queryRaw('SELECT id, name FROM users');

    $tx->commit();
} catch (\Exception $e) {
    $tx->rollback();
    throw $e;
}
```

Transactions auto-rollback on garbage collection if not committed.

### Batch in Transaction

```php
$tx = $db->begin();
$tx->executeBatch(
    'INSERT INTO users VALUES ($1, $2, $3)',
    [
        [1, 'Alice', 'alice@example.com'],
        [2, 'Bob', 'bob@example.com'],
    ]
);
$tx->commit(); // or $tx->rollback() to undo
```

### Transaction Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `exec($sql)` | `int` | Execute DDL/DML without params |
| `execute($sql, $params)` | `int` | Execute parameterized DML |
| `executeBatch($sql, $paramsArray)` | `int` | Execute with multiple param sets |
| `query($sql, $params?)` | `array` | Query rows as associative arrays |
| `queryOne($sql, $params?)` | `?array` | Query single row or null |
| `queryRaw($sql, $params?)` | `array` | Query in columnar format |
| `commit()` | `void` | Commit the transaction |
| `rollback()` | `void` | Rollback the transaction |

## Parameters

Both positional and named parameters are supported across all methods:

```php
// Positional ($1, $2, ...)
$db->query('SELECT * FROM users WHERE id = $1 AND name = $2', [1, 'Alice']);

// Named (:key)
$db->query(
    'SELECT * FROM users WHERE id = :id AND name = :name',
    ['id' => 1, 'name' => 'Alice']
);
```

## Error Handling

All methods throw `Stoolap\StoolapException` (extends `\RuntimeException`) on errors:

```php
use Stoolap\StoolapException;

try {
    $db->execute('INSERT INTO users VALUES ($1, $2)', [1, null]); // NOT NULL violation
} catch (StoolapException $e) {
    echo $e->getMessage();
}
```

## Type Mapping

| PHP | Stoolap | Notes |
|-----|---------|-------|
| `int` | `INTEGER` | |
| `float` | `FLOAT` | |
| `string` | `TEXT` | |
| `bool` | `BOOLEAN` | |
| `null` | `NULL` | |
| `array` / `object` | `JSON` | Auto-encoded/decoded |
| `DateTimeInterface` | `TIMESTAMP` | Converted to nanoseconds |

## PHP-FPM / Web Server Support

Stoolap uses exclusive file locking, so only one OS process can open a database at a time. The PHP extension solves this transparently for multi-process environments (php-fpm, Apache mod_php) with a built-in daemon proxy.

When loaded under php-fpm, CGI, or Apache, the extension automatically forks a background daemon process. PHP workers communicate with the daemon via shared memory + kernel wait primitives (futex on Linux, __ulock on macOS) for near-zero IPC overhead (~0.5μs per call). No configuration required.

### Architecture

```
php-fpm master
  ├── worker 1 ──┐
  ├── worker 2 ──┼── shared memory + futex/ulock ──► stoolap daemon
  ├── worker 3 ──┤                                    ├── DB: file:///data/app
  └── worker N ──┘                                    └── DB: memory://cache
```

- One daemon process serves all workers and all databases
- Each worker gets a `clone()`'d database handle via the daemon
- Daemon auto-starts with php-fpm and auto-exits when php-fpm stops
- All Stoolap features work identically in daemon mode (transactions, prepared statements, batch operations)

### IPC Overhead

| Operation | Direct (CLI) | Daemon (FPM) | Overhead |
|---|---|---|---|
| SELECT by PK | 0.5 μs | 1.0 μs | ~0.5 μs |
| UPDATE by PK | 0.8 μs | 1.2 μs | ~0.5 μs |
| Prepared execute | 0.6 μs | 1.1 μs | ~0.5 μs |
| Prepared queryOne | 1.1 μs | 1.7 μs | ~0.6 μs |

Analytical queries (GROUP BY, JOIN, window functions) are unaffected since they're dominated by engine execution time, not IPC.

### Environment Variables

| Variable | Values | Description |
|---|---|---|
| `STOOLAP_DAEMON` | `1` / `on` | Force daemon mode (useful for CLI testing) |
| `STOOLAP_DAEMON` | `0` / `off` | Force direct mode (disable daemon) |
| `STOOLAP_DAEMON_DEBUG` | `1` | Enable daemon stderr logging |

When not set, daemon mode is auto-detected from the SAPI name (fpm, cgi, apache).

## Building from Source

Requires:
- [PHP](https://www.php.net) >= 8.1 with development headers (`php-dev` / `php-devel`)
- C compiler (gcc, clang, or MSVC)
- The stoolap shared library (`libstoolap.dylib` / `libstoolap.so` / `stoolap.dll`), either from [Stoolap releases](https://github.com/stoolap/stoolap/releases) or built from source

```bash
git clone https://github.com/stoolap/stoolap-php.git
cd stoolap-php/ext
phpize
./configure --with-stoolap=/path/to/libstoolap
make
```
