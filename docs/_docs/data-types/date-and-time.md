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
```

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

### Time Only
```
14:30:45.123                  -- Time with milliseconds
14:30:45                      -- Standard time
14:30                         -- Hours and minutes only
2:30:45 PM                    -- 12-hour format
```

## Date and Time Functions

### NOW()

Returns the current date and time:

```sql
-- Insert the current timestamp
INSERT INTO events (id, event_name, created_at)
VALUES (1, 'Meeting', NOW());

-- Filter for records created today
SELECT * FROM events WHERE created_at >= NOW();
```

### DATE_TRUNC(unit, timestamp)

Truncates a timestamp to the specified precision:

```sql
-- Truncate to day (removes time component)
SELECT DATE_TRUNC('day', event_date) FROM events;

-- Truncate to month (sets day to 1 and time to 00:00:00)
SELECT DATE_TRUNC('month', event_date) FROM events;
```

Supported units:
- `year` - Truncates to the beginning of the year
- `month` - Truncates to the beginning of the month
- `day` - Truncates to the beginning of the day
- `hour` - Truncates to the beginning of the hour
- `minute` - Truncates to the beginning of the minute
- `second` - Truncates to the beginning of the second

### TIME_TRUNC(interval, timestamp)

Truncates a timestamp to a specific time interval, useful for time series data:

```sql
-- Truncate to 15-minute intervals
SELECT TIME_TRUNC('15m', event_time) FROM events;

-- Truncate to 1-hour intervals
SELECT TIME_TRUNC('1h', event_time) FROM events;
```

The interval parameter accepts duration strings like:
- `15m` - 15 minutes
- `30m` - 30 minutes
- `1h` - 1 hour
- `4h` - 4 hours
- `1d` - 1 day

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

- `second` or `seconds` - For second-based intervals
- `minute` or `minutes` - For minute-based intervals  
- `hour` or `hours` - For hour-based intervals
- `day` or `days` - For day-based intervals
- `week` or `weeks` - For week-based intervals (7 days)
- `month` or `months` - For month-based intervals (approximated as 30 days)
- `year` or `years` - For year-based intervals (approximated as 365 days)

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

Based on the implementation, Stoolap has the following limitations for date and time handling:

1. **Limited Time Zone Support**: While timestamps are normalized to UTC internally, there are no explicit functions for time zone conversion.

2. **No Date/Time Extraction**: Functions to extract parts from dates (like EXTRACT(YEAR FROM date)) are not implemented.

3. **No Custom Formatting**: No functions to format timestamps in custom output formats.

4. **Approximate Month and Year Intervals**: INTERVAL calculations for months and years use approximations (30 days for months, 365 days for years) rather than calendar-aware calculations.

## Best Practices

1. **Consistent Format Usage**: Use ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ) for consistency.

2. **Index Timestamp Columns**: If you filter frequently on timestamp columns, create indexes on them.

3. **Use DATE_TRUNC for Date-Only Comparisons**: When you need to compare only the date part, use DATE_TRUNC to remove the time component.

4. **Use TIME_TRUNC for Time Series**: For time series data, use TIME_TRUNC to bucket data into regular intervals for analysis.

5. **Store in UTC**: Always store timestamps in UTC for consistency, especially when working with data across different time zones.