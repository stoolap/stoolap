---
layout: post
title: "Migrating a Single-Node Trading App from PostgreSQL to Stoolap"
author: Stefan Keller
date: 2026-03-13
category: engineering
---

We recently migrated one of our Node.js trading apps from PostgreSQL (with TimescaleDB) to Stoolap. It runs in production on a single machine, ingests market data, stores candlestick OHLCV bars, computes technical indicators, and serves a web dashboard with live WebSocket updates. Before the migration, it had been on PostgreSQL for over a year and was processing thousands of writes per minute.

This post is a narrow case study, not a general argument against PostgreSQL. PostgreSQL and TimescaleDB handled the workload well; the reason we changed was that this app runs on a single machine, and we wanted to see how much operational complexity we could remove by embedding the database directly in the Node.js process.

## Why Move Away from PostgreSQL?

PostgreSQL with TimescaleDB worked well for the data side, but for this particular deployment the operational overhead kept feeling heavier than the workload needed. We were still running a PostgreSQL service, handling connection pooling, managing credentials, and keeping the TimescaleDB extension updated. For a single-node trading app on one machine, that was more moving parts than we wanted.

We wanted the database to live inside the Node.js process instead of beside it. In practice, that meant removing the separate database service and opening a local database file.

## SQL Compatibility

Going in, I expected to spend most of the migration rewriting queries. PostgreSQL has a rich SQL dialect, and I assumed we would run into more incompatibilities than we did.

That assumption was mostly wrong. Here's what we used in the original codebase that worked in Stoolap without query rewrites:

- `ON CONFLICT ... DO UPDATE SET` with `EXCLUDED` pseudo-table
- `RETURNING` on INSERT, UPDATE, and DELETE
- `ILIKE` and `NOT ILIKE` for case-insensitive search
- `DISTINCT ON (columns)` for per-group deduplication
- Window functions like `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)`
- Aggregate functions including `AVG`, `STDDEV`, `SUM`, `MIN`, `MAX`
- `TIME_TRUNC` for timestamp bucketing
- `NOW() - INTERVAL '30 days'` for relative time filters
- ISO timestamps with fractional seconds (`2024-01-15T14:30:00.000Z`)
- Subqueries, CTEs with `WITH`, `HAVING`, `GROUP BY`

In this codebase, the only SQL-level change we had to make was renaming `DATE_TRUNC` to `TIME_TRUNC`. For the queries we were using, it was a one-name-change find-and-replace.

We also had a bunch of PostgreSQL `::` cast expressions (`value::numeric`) that we replaced with standard `CAST(value AS type)`, though many of those casts turned out to be unnecessary in this codebase. We ended up deleting most of those lines entirely.

## Rebuilding Candlestick Aggregation

This was the real work. Not because Stoolap forced it, but because removing TimescaleDB meant we lost continuous aggregates, the feature that automatically rolled up 1-minute candles into 5-minute, 15-minute, 1-hour, and higher timeframes.

We replaced it with a period chain:

```js
const PERIOD_CHAIN = [
  { period: '3m',  parent: '1m',  minutes: 3 },
  { period: '5m',  parent: '1m',  minutes: 5 },
  { period: '15m', parent: '5m',  minutes: 15 },
  { period: '1h',  parent: '15m', minutes: 60 },
  { period: '4h',  parent: '1h',  minutes: 240 },
  { period: '1d',  parent: '4h',  minutes: 1440 },
];
```

Each timeframe has its own table (`candlesticks_t1m`, `candlesticks_t3m`, etc.) and aggregates from its parent. A 2-second timer fires one SQL query per timeframe:

```sql
INSERT INTO candlesticks_t5m (time, exchange, symbol, open, high, low, close, volume)
SELECT
  TIME_TRUNC('5m', time) AS time, exchange, symbol,
  FIRST(open ORDER BY time) AS open,
  MAX(high) AS high,
  MIN(low) AS low,
  LAST(close ORDER BY time) AS close,
  SUM(volume) AS volume
FROM candlesticks_t1m
WHERE time >= $1
GROUP BY TIME_TRUNC('5m', time), exchange, symbol
ON CONFLICT (exchange, symbol, time) DO UPDATE SET
  open = EXCLUDED.open, high = EXCLUDED.high,
  low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume
```

This query is doing a lot: `FIRST` and `LAST` ordered aggregates, `TIME_TRUNC` bucketing, `GROUP BY` on a function expression, and `INSERT ... SELECT ... ON CONFLICT DO UPDATE`. It runs every 2 seconds across 6 timeframes for all trading pairs. I honestly expected at least one of those features to need a workaround, but in this workload they all worked as written.

We added a `dirty` flag to skip the sync when no new 1-minute candles have arrived, and a `syncing` mutex to prevent the periodic sync from colliding with full backfill operations.

## Numbers

As a rough measure of migration size, the final diff across 25 files was 391 lines added and 650 lines removed. The codebase ended up 259 lines smaller overall, and that's including the new candlestick aggregation system that did not exist before.

Most of the deleted lines were PostgreSQL-specific: `::` cast expressions, comment blocks explaining PostgreSQL behavior, compatibility aliases, and workaround methods that no longer applied.

## Takeaways

I expected this migration to involve simpler queries, fewer SQL features, and more application-side workarounds. Instead, a lot of the work was deleting PostgreSQL-specific code paths that the new setup no longer needed.

The SQL coverage surprised me. In this migration, we did not have to drop `ON CONFLICT` upserts, `RETURNING`, window functions, ordered aggregates, `DISTINCT ON`, `ILIKE`, `INTERVAL`, CTEs, or subqueries. That was more SQL coverage than I expected from an embedded database.

Performance has been fine so far for this workload. The app processes live market data from multiple exchanges with sub-second latency, and the 2-second candlestick sync has not shown up as a meaningful CPU cost in our monitoring so far. We've been running this setup in production for a few weeks without issues, but that's still early and I would not generalize from it yet.

This is not a blanket recommendation. If you need a separate database server, mature operational tooling, or TimescaleDB features like continuous aggregates, PostgreSQL is still the more obvious fit.

If you have a similar single-node deployment where PostgreSQL is mostly acting as a local service, it may be worth evaluating whether an embedded database would simplify the stack.
