---
layout: doc
title: Date and Time Handling
category: Data Types
order: 2
---

# Date and Time Handling

This document explains how to work with dates and times in Stoolap based on the implementation and test files.

## TIMESTAMP Data Type

Stoolap uses the `TIMESTAMP` data type for storing date and time values. When defining a table schema, you can use this type for any columns that need to store temporal data:

```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    event_name TEXT,
    event_date TIMESTAMP,
    created_at TIMESTAMP
);
```

A TIMESTAMP is computed with between `0000-01-01 00:00:00` and `9999-12-31 23:59:59.999999999`: a comparison, a `CAST` or a literal such as `'9999-12-31'` reads it in full. Moving a timestamp by an INTERVAL or a number of days (`+`, `-`, `DATE_ADD`, `DATE_SUB`) to a point outside that range is an error.

A table stores a TIMESTAMP as nanoseconds since 1970-01-01, which holds `1677-09-21 00:12:43.145224192` to `2262-04-11 23:47:16.854775807` UTC. Writing one outside that range with `INSERT`, `UPDATE` or an upsert, or giving one to the existing rows with `ALTER TABLE ... ADD COLUMN ... DEFAULT` or `MODIFY COLUMN`, is an error and changes nothing, as is reading a table `AS OF TIMESTAMP` a time outside it, `TIME_TRUNC` of one, and handing one to a driver through the C API. For "no end", store NULL rather than a far date such as `9999-12-31`.

Before version 0.5.0 such a value was accepted and, once its table was written to disk, read back as another time; `'9999-12-31 23:59:59'`, for example, came back as `1816-03-30 05:56:07.066277376`. A value written that way is still in its table. A row of a table that is not yet on disk and holds one makes the table refuse to be written to disk, with the row named in the error, until that row is updated.

## Supported Date and Time Formats

Based on the implementation, Stoolap recognizes various date and time formats:

### ISO 8601 Format
```
2023-05-15T14:30:45Z          -- ISO format with UTC timezone (Z)
2023-05-15T14:30:45+00:00     -- ISO format with explicit timezone
2023-05-15T14:30:45           -- ISO format without timezone
```

### SQL-Style Format
```
2023-05-15 14:30:45.123       -- SQL format with milliseconds
2023-05-15 14:30:45           -- SQL format without fractional seconds
2023-05-15 14:30:45+02:00     -- SQL format with a time zone
```

Either form may end with a time zone after the time: `Z`, `+HH`, `+HHMM` or `+HH:MM` (or with `-`), optionally after a space. The value is that instant in UTC, so `'2023-05-15 16:30:45+02:00'`, `'2023-05-15T09:30:45-05'` and `'2023-05-15 14:30:45Z'` are all `2023-05-15 14:30:45` UTC. A zone outside `-23:59` to `+23:59`, or one written as a name such as `UTC`, does not read: the text is no timestamp.

A typed literal, `TIMESTAMP '...'` or `DATE '...'`, whose text does not read as a timestamp is an error, as writing that text into a TIMESTAMP column is.

### Date Only
```
2023-05-15                    -- Date only (time defaults to 00:00:00)
```

### Alternative Formats
```
2023/05/15 14:30:45           -- Alternative format with slashes
2023/05/15                    -- Alternative date only
05/15/2023                    -- US format (month/day/year)
15/05/2023                    -- European format (day/month/year)
```

A date in either of the last two forms that both can read, such as `03/04/2024`, is read as the US form, month before day: March 4. A year is always written with four digits (`0024-03-04` for the year 24), so `24-03-04` or `3/4/24` reads as no TIMESTAMP and `CAST` gives NULL.

### Time Only
```
14:30:45.123                  -- Time with milliseconds
14:30:45                      -- Standard time
14:30                         -- Hours and minutes only
```

A time of day alone, read as a TIMESTAMP, is that time on 1970-01-01 (UTC), the same whenever and wherever it is read: `'12:00:00'` is `1970-01-01 12:00:00`. A `TIME` column stores a TIMESTAMP, so its values compare as times of day: `tm < '13:00'` keeps every row before one o'clock, whichever day it was written. For today at a time of day, write it from the clock: `CURRENT_DATE + INTERVAL '12 hours'`.

Before version 0.5.0 a time of day alone took the date it was read on. A `TIME` column written by an earlier version holds those dates; this moves its values onto 1970-01-01, keeping whole seconds:

```sql
UPDATE events SET start_time = CAST(TO_CHAR(start_time, 'HH24:MI:SS') AS TIMESTAMP);
```

## Date and Time Functions

### NOW() / CURRENT_TIMESTAMP / CURRENT_DATE

```sql
SELECT NOW();                -- Current timestamp
SELECT CURRENT_TIMESTAMP;    -- Same as NOW()
SELECT CURRENT_DATE;         -- Current date (at midnight UTC)
```

### EXTRACT(field FROM timestamp)

Extracts a date/time component using SQL standard syntax:

```sql
SELECT EXTRACT(YEAR FROM '2024-03-15');              -- 2024
SELECT EXTRACT(MONTH FROM '2024-03-15');             -- 3
SELECT EXTRACT(DAY FROM '2024-03-15');               -- 15
SELECT EXTRACT(HOUR FROM '2024-03-15 14:30:00');     -- 14
SELECT EXTRACT(MINUTE FROM '2024-03-15 14:30:00');   -- 30
SELECT EXTRACT(SECOND FROM '2024-03-15 14:30:45');   -- 45
SELECT EXTRACT(MILLISECOND FROM '2024-03-15 14:30:45.123'); -- 123
SELECT EXTRACT(MICROSECOND FROM '2024-03-15 14:30:45.123456'); -- 123456
SELECT EXTRACT(DOW FROM '2024-03-15');               -- 5 (day of week, 0=Sunday)
SELECT EXTRACT(ISODOW FROM '2024-03-15');            -- 5 (ISO day of week, 1=Monday)
SELECT EXTRACT(DOY FROM '2024-03-15');               -- 75 (day of year)
SELECT EXTRACT(WEEK FROM '2024-03-15');              -- 11 (ISO week)
SELECT EXTRACT(QUARTER FROM '2024-05-15');           -- 2
SELECT EXTRACT(EPOCH FROM '2024-03-15 00:00:00');    -- Unix timestamp
```

### YEAR / MONTH / DAY / HOUR / MINUTE / SECOND

Shorthand functions for extracting date/time parts:

```sql
SELECT YEAR('2024-03-15');                 -- 2024
SELECT MONTH('2024-03-15');                -- 3
SELECT DAY('2024-03-15');                  -- 15
SELECT HOUR('2024-03-15 14:30:45');        -- 14
SELECT MINUTE('2024-03-15 14:30:45');      -- 30
SELECT SECOND('2024-03-15 14:30:45');      -- 45
```

### DATE_TRUNC(unit, timestamp)

Truncates a timestamp to the specified precision:

```sql
SELECT DATE_TRUNC('day', event_date) FROM events;
SELECT DATE_TRUNC('month', event_date) FROM events;
```

Supported units: `year`, `quarter`, `month`, `week`, `day`, `hour`, `minute`, `second`

### TIME_TRUNC(interval, timestamp)

Truncates a timestamp to a specific time interval, useful for time series data:

```sql
SELECT TIME_TRUNC('15m', event_time) FROM events;
SELECT TIME_TRUNC('1h', event_time) FROM events;
```

The interval parameter accepts duration strings like `15m`, `30m`, `1h`, `4h`, `1d`.

### DATE_ADD / DATE_SUB

Add or subtract intervals from a timestamp:

```sql
SELECT DATE_ADD('2024-03-15', 10);                -- Add 10 days (default)
SELECT DATE_ADD('2024-03-15', 2, 'month');        -- Add 2 months
SELECT DATE_SUB('2024-03-15', 1, 'month');        -- Subtract 1 month
```

Supported units: every INTERVAL unit below. A month or year moves the calendar and keeps the time of day, fractions of a second included.

### DATEDIFF / DATE_DIFF

Returns the difference between two dates in days:

```sql
SELECT DATEDIFF('2024-03-15', '2024-03-01');      -- 14
SELECT DATE_DIFF('2024-03-15', '2024-03-01');     -- 14 (alias)
```

### TO_CHAR(timestamp, format)

Formats a timestamp as a string:

```sql
SELECT TO_CHAR('2024-03-15 14:30:45', 'YYYY-MM-DD');     -- '2024-03-15'
SELECT TO_CHAR('2024-03-15 14:30:45', 'DD MON YYYY');    -- '15 MAR 2024'
SELECT TO_CHAR('2024-03-15 14:30:45', 'HH24:MI:SS');     -- '14:30:45'
```

Format patterns: `YYYY`, `YY`, `MM`, `MON`, `MONTH`, `DD`, `DY`, `DAY`, `HH24`, `HH`/`HH12`, `MI`, `SS`

## Date and Time Arithmetic with INTERVAL

Stoolap supports PostgreSQL-style INTERVAL literals for date and time arithmetic:

### INTERVAL Syntax

```sql
-- Subtract 24 hours from current time
SELECT NOW() - INTERVAL '24 hours';

-- Add 7 days to a timestamp
SELECT event_date + INTERVAL '7 days' FROM events;

-- Subtract 30 minutes
SELECT NOW() - INTERVAL '30 minutes';
```

### Supported INTERVAL Units

- `microsecond`, `microseconds` or `us`
- `millisecond`, `milliseconds` or `ms`
- `second`, `seconds`, `sec` or `secs` - For second-based intervals
- `minute`, `minutes`, `min` or `mins` - For minute-based intervals
- `hour` or `hours` - For hour-based intervals
- `day` or `days` - For day-based intervals
- `week` or `weeks` - For week-based intervals (7 days)
- `month` or `months` - Calendar months; the day is clamped to the last day of the target month
- `year` or `years` - Calendar years (12 months)

Units are read in any case, in an INTERVAL literal, in interval text added to a timestamp, in `DATE_ADD` and in `generate_series`. An INTERVAL literal with any other unit is an error. Interval text from a column or a parameter that is not an interval reads as NULL; its number outside the range is an error.

### INTERVAL Examples

```sql
-- Find events from the last 24 hours
SELECT * FROM events 
WHERE event_date > NOW() - INTERVAL '24 hours';

-- Find events from the last week
SELECT * FROM events 
WHERE event_date > NOW() - INTERVAL '7 days';

-- Calculate event duration
SELECT event_name,
       end_time - start_time AS duration
FROM events;

-- Add specific intervals to timestamps
SELECT event_date + INTERVAL '90 seconds' FROM events;
SELECT event_date + INTERVAL '2 weeks' FROM events;
```

### Multiplying and Negating an INTERVAL

An INTERVAL literal can be multiplied by a number, a column or a parameter, and negated:

```sql
-- One reading per minute, starting at a fixed time
SELECT TIMESTAMP '2025-01-01 00:00:00' + value * INTERVAL '1 minute'
FROM generate_series(0, 59);

-- Shift by a per-row amount
SELECT start_time + retries * INTERVAL '30 seconds' FROM jobs;

-- Fractional multiples of fixed units are rounded to the microsecond
SELECT TIMESTAMP '2025-01-01 00:00:00' + 1.5 * INTERVAL '1 hour';  -- 01:30:00

-- Calendar units take whole multiples
SELECT TIMESTAMP '2025-01-31 00:00:00' + 3 * INTERVAL '1 month';   -- 2025-04-30

-- Negation
SELECT NOW() + -INTERVAL '1 hour';

-- On its own, a product is interval text
SELECT 3 * INTERVAL '1 second';    -- '3 second'
SELECT 1.5 * INTERVAL '1 hour';    -- '5400 second'
```

A NULL multiplier gives NULL. These are errors:

- dividing an INTERVAL or taking its remainder (`INTERVAL '1 hour' / 2`, `% 2`);
- adding, subtracting or multiplying two INTERVALs, or subtracting a value from an INTERVAL;
- a fractional multiple of `month` or `year` (`1.5 * INTERVAL '1 month'`);
- a multiplier that is not a number, and a result out of range.

### Timestamp Literals

You can also use typed timestamp literals:

```sql
-- TIMESTAMP literal
SELECT TIMESTAMP '2025-01-01 12:00:00';

-- Add interval to a specific timestamp
SELECT TIMESTAMP '2025-01-01 12:00:00' + INTERVAL '25 hours';

-- DATE and TIME literals are also supported
SELECT DATE '2025-01-01';
SELECT TIME '12:00:00';
```

## Examples

### Basic Timestamp Operations

```sql
-- Create a table with timestamp columns
CREATE TABLE timestamp_test (
    id INTEGER PRIMARY KEY,
    event_time TIMESTAMP
);

-- Insert timestamps in different formats
INSERT INTO timestamp_test VALUES (1, '2023-05-15 14:30:45');
INSERT INTO timestamp_test VALUES (2, '2023-05-15T14:30:45');
INSERT INTO timestamp_test VALUES (3, '2023-05-15');

-- Query with date truncation
SELECT id, DATE_TRUNC('day', event_time) FROM timestamp_test;
```

### Time Series Aggregation

```sql
-- Aggregating time series data into 15-minute intervals
SELECT 
    TIME_TRUNC('15m', timestamp) AS time_bucket,
    FIRST(open) AS open_price,
    MAX(high) AS high_price,
    MIN(low) AS low_price,
    LAST(close) AS close_price,
    SUM(volume) AS total_volume
FROM candle_data
GROUP BY TIME_TRUNC('15m', timestamp)
ORDER BY time_bucket;
```

## Time Zone Handling

Stoolap normalizes timestamps to UTC internally for consistent storage and comparison. When you provide a timestamp without a timezone specification, it is interpreted as being in UTC.

```sql
-- These are equivalent
INSERT INTO events VALUES (1, '2023-05-15T14:30:45Z');
INSERT INTO events VALUES (2, '2023-05-15T14:30:45+00:00');
INSERT INTO events VALUES (3, '2023-05-15 14:30:45'); -- Assumed UTC
INSERT INTO events VALUES (4, '2023-05-15 16:30:45+02:00');
```

## Date and Time Comparisons

You can use the standard comparison operators with timestamp values:

```sql
-- Equality comparison
SELECT * FROM events WHERE event_date = '2023-05-15';

-- Range queries
SELECT * FROM events 
WHERE event_date >= '2023-05-01' AND event_date < '2023-06-01';

-- Less than
SELECT * FROM events WHERE event_date < NOW();
```

## Indexing Timestamps

Timestamp columns can be indexed for efficient filtering:

```sql
-- Create an index on a timestamp column
CREATE INDEX idx_event_date ON events(event_date);

-- This query can now use the index
SELECT * FROM events WHERE event_date >= '2023-05-01';
```

## Limitations

1. **Limited Time Zone Support**: Timestamps are normalized to UTC internally. There are no explicit functions for time zone conversion.

## Best Practices

1. **Consistent Format Usage**: Use ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ) for consistency.

2. **Index Timestamp Columns**: If you filter frequently on timestamp columns, create indexes on them.

3. **Use DATE_TRUNC for Date-Only Comparisons**: When you need to compare only the date part, use DATE_TRUNC to remove the time component.

4. **Use TIME_TRUNC for Time Series**: For time series data, use TIME_TRUNC to bucket data into regular intervals for analysis.

5. **Store in UTC**: Always store timestamps in UTC for consistency, especially when working with data across different time zones.