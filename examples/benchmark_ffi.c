/*
 * Copyright 2025 Stoolap Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * C FFI benchmark matching examples/benchmark.rs exactly.
 *
 * Build & run:
 *   cargo build --profile release-ffi --features ffi
 *   cc -O2 -o benchmark_ffi examples/benchmark_ffi.c \
 *      -I include -L target/release-ffi -lstoolap
 *   DYLD_LIBRARY_PATH=target/release-ffi ./benchmark_ffi      # macOS
 *   LD_LIBRARY_PATH=target/release-ffi ./benchmark_ffi         # Linux
 */

#include "stoolap.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ROW_COUNT       10000
#define ITERATIONS      500
#define ITERATIONS_MEDIUM 250
#define ITERATIONS_HEAVY  50
#define WARMUP          10

/* =========================================================================
 * Timing helpers
 * ========================================================================= */

static uint64_t now_nanos(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void print_result(const char* name, uint64_t total_nanos, int iterations) {
    double avg_us = (double)total_nanos / 1000.0 / (double)iterations;
    double ops = 1000000.0 / avg_us;
    printf("%-25s | %15.3f | %12.0f\n", name, avg_us, ops);
}

/* =========================================================================
 * Simple xorshift64 PRNG (deterministic, no dependency)
 * ========================================================================= */

static uint64_t rng_state = 0x123456789ABCDEF0ULL;

static uint64_t xorshift64(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

static int64_t rng_range(int64_t lo, int64_t hi) {
    return lo + (int64_t)(xorshift64() % (uint64_t)(hi - lo));
}

static double rng_double(double lo, double hi) {
    return lo + (double)(xorshift64() % 1000000ULL) / 1000000.0 * (hi - lo);
}

static int rng_bool(double prob) {
    return (double)(xorshift64() % 10000ULL) / 10000.0 < prob;
}

/* =========================================================================
 * Helpers
 * ========================================================================= */

#define CHECK(rc, ctx) do { \
    if ((rc) != STOOLAP_OK) { \
        fprintf(stderr, "FATAL: %s failed: %s\n", (ctx), stoolap_errmsg(db)); \
        exit(1); \
    } \
} while(0)

#define CHECK_STMT(rc, stmt_ptr, ctx) do { \
    if ((rc) != STOOLAP_OK) { \
        fprintf(stderr, "FATAL: %s failed: %s\n", (ctx), stoolap_stmt_errmsg(stmt_ptr)); \
        exit(1); \
    } \
} while(0)

#define CHECK_TX(rc, tx_ptr, ctx) do { \
    if ((rc) != STOOLAP_OK) { \
        fprintf(stderr, "FATAL: %s failed: %s\n", (ctx), stoolap_tx_errmsg(tx_ptr)); \
        exit(1); \
    } \
} while(0)

/* Drain all rows from a result set. */
static void drain_rows(StoolapRows* rows) {
    while (stoolap_rows_next(rows) == STOOLAP_ROW) {}
    stoolap_rows_close(rows);
}

/* Query + drain (no params). Returns immediately on error. */
static void query_drain(StoolapDB* db, StoolapStmt* stmt) {
    StoolapRows* rows = NULL;
    stoolap_stmt_query(stmt, NULL, 0, &rows);
    if (rows) drain_rows(rows);
}

/* =========================================================================
 * Main benchmark
 * ========================================================================= */

int main(void) {
    StoolapDB* db = NULL;
    StoolapStmt* stmt = NULL;
    StoolapRows* rows = NULL;
    int32_t rc;
    uint64_t t0, elapsed;
    int i;

    printf("Starting Stoolap-C (FFI) benchmark...\n");
    printf("Configuration: %d rows, %d iterations per test\n\n", ROW_COUNT, ITERATIONS);

    rc = stoolap_open("memory://", &db);
    CHECK(rc, "open");

    /* ── Schema ── */
    rc = stoolap_exec(db,
        "CREATE TABLE users ("
        "  id INTEGER PRIMARY KEY,"
        "  name TEXT NOT NULL,"
        "  email TEXT NOT NULL,"
        "  age INTEGER NOT NULL,"
        "  balance REAL NOT NULL,"
        "  active BOOLEAN NOT NULL,"
        "  created_at TEXT NOT NULL"
        ")", NULL);
    CHECK(rc, "create users");

    rc = stoolap_exec(db, "CREATE INDEX idx_users_age ON users(age)", NULL);
    CHECK(rc, "create idx_age");
    rc = stoolap_exec(db, "CREATE INDEX idx_users_active ON users(active)", NULL);
    CHECK(rc, "create idx_active");

    /* ── Populate ── */
    rc = stoolap_prepare(db,
        "INSERT INTO users (id, name, email, age, balance, active, created_at) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7)", &stmt);
    CHECK(rc, "prepare insert");

    for (i = 1; i <= ROW_COUNT; i++) {
        int64_t age = rng_range(18, 80);
        double balance = rng_double(0.0, 100000.0);
        int32_t active = rng_bool(0.7) ? 1 : 0;
        char name[64], email[64];
        snprintf(name, sizeof(name), "User_%d", i);
        snprintf(email, sizeof(email), "user%d@example.com", i);

        StoolapValue params[7] = {
            { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = (int64_t)i } },
            { .value_type = STOOLAP_TYPE_TEXT, 0, { .text = { name, (int64_t)strlen(name) } } },
            { .value_type = STOOLAP_TYPE_TEXT, 0, { .text = { email, (int64_t)strlen(email) } } },
            { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = age } },
            { .value_type = STOOLAP_TYPE_FLOAT, 0, { .float64 = balance } },
            { .value_type = STOOLAP_TYPE_BOOLEAN, 0, { .boolean = active } },
            { .value_type = STOOLAP_TYPE_TEXT, 0, { .text = { "2024-01-01 00:00:00", 19 } } },
        };
        rc = stoolap_stmt_exec(stmt, params, 7, NULL);
        CHECK_STMT(rc, stmt, "insert user");
    }
    stoolap_stmt_finalize(stmt);

    printf("Benchmarking Stoolap-C (FFI)...\n\n");
    printf("============================================================\n");
    printf("STOOLAP-C FFI BENCHMARK (%d rows, %d iterations, in-memory)\n", ROW_COUNT, ITERATIONS);
    printf("============================================================\n\n");
    printf("%-25s | %15s | %12s\n", "Operation", "Avg (us)", "ops/sec");
    printf("---------------------------------------------------------------\n");

    /* ── SELECT by ID (prepared) ── */
    rc = stoolap_prepare(db, "SELECT * FROM users WHERE id = $1", &stmt);
    CHECK(rc, "prepare select_by_id");
    /* warmup */
    for (i = 0; i < WARMUP; i++) {
        StoolapValue p = { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = (int64_t)(i % ROW_COUNT) + 1 } };
        stoolap_stmt_query(stmt, &p, 1, &rows);
        if (rows) { stoolap_rows_next(rows); stoolap_rows_close(rows); rows = NULL; }
    }
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) {
        StoolapValue p = { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = (int64_t)(i % ROW_COUNT) + 1 } };
        stoolap_stmt_query(stmt, &p, 1, &rows);
        if (rows) { stoolap_rows_next(rows); stoolap_rows_close(rows); rows = NULL; }
    }
    elapsed = now_nanos() - t0;
    print_result("SELECT by ID", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── SELECT by index (exact) ── */
    rc = stoolap_prepare(db, "SELECT * FROM users WHERE age = $1", &stmt);
    CHECK(rc, "prepare select_by_index");
    for (i = 0; i < WARMUP; i++) {
        StoolapValue p = { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = (int64_t)(i % 62) + 18 } };
        stoolap_stmt_query(stmt, &p, 1, &rows);
        if (rows) { drain_rows(rows); rows = NULL; }
    }
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) {
        StoolapValue p = { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = (int64_t)(i % 62) + 18 } };
        stoolap_stmt_query(stmt, &p, 1, &rows);
        if (rows) { drain_rows(rows); rows = NULL; }
    }
    elapsed = now_nanos() - t0;
    print_result("SELECT by index (exact)", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── SELECT by index (range) ── */
    rc = stoolap_prepare(db, "SELECT * FROM users WHERE age >= $1 AND age <= $2", &stmt);
    CHECK(rc, "prepare select_by_index_range");
    for (i = 0; i < WARMUP; i++) {
        StoolapValue p[2] = {
            { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 30 } },
            { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 40 } },
        };
        stoolap_stmt_query(stmt, p, 2, &rows);
        if (rows) { drain_rows(rows); rows = NULL; }
    }
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) {
        StoolapValue p[2] = {
            { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 30 } },
            { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 40 } },
        };
        stoolap_stmt_query(stmt, p, 2, &rows);
        if (rows) { drain_rows(rows); rows = NULL; }
    }
    elapsed = now_nanos() - t0;
    print_result("SELECT by index (range)", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── SELECT complex ── */
    rc = stoolap_prepare(db,
        "SELECT id, name, balance FROM users WHERE age >= 25 AND age <= 45 "
        "AND active = true ORDER BY balance DESC LIMIT 100", &stmt);
    CHECK(rc, "prepare select_complex");
    for (i = 0; i < WARMUP; i++) query_drain(db, stmt);
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("SELECT complex", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── SELECT * (full scan) ── */
    rc = stoolap_prepare(db, "SELECT * FROM users", &stmt);
    CHECK(rc, "prepare select_all");
    for (i = 0; i < 10; i++) query_drain(db, stmt);
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS_HEAVY; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("SELECT * (full scan)", elapsed, ITERATIONS_HEAVY);
    stoolap_stmt_finalize(stmt);

    /* ── UPDATE by ID ── */
    rc = stoolap_prepare(db, "UPDATE users SET balance = $1 WHERE id = $2", &stmt);
    CHECK(rc, "prepare update_by_id");
    for (i = 0; i < WARMUP; i++) {
        StoolapValue p[2] = {
            { .value_type = STOOLAP_TYPE_FLOAT, 0, { .float64 = rng_double(0.0, 100000.0) } },
            { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = (int64_t)(i % ROW_COUNT) + 1 } },
        };
        stoolap_stmt_exec(stmt, p, 2, NULL);
    }
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) {
        StoolapValue p[2] = {
            { .value_type = STOOLAP_TYPE_FLOAT, 0, { .float64 = rng_double(0.0, 100000.0) } },
            { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = (int64_t)(i % ROW_COUNT) + 1 } },
        };
        stoolap_stmt_exec(stmt, p, 2, NULL);
    }
    elapsed = now_nanos() - t0;
    print_result("UPDATE by ID", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── UPDATE complex ── */
    rc = stoolap_prepare(db,
        "UPDATE users SET balance = $1 WHERE age >= $2 AND age <= $3 AND active = true", &stmt);
    CHECK(rc, "prepare update_complex");
    for (i = 0; i < WARMUP; i++) {
        StoolapValue p[3] = {
            { .value_type = STOOLAP_TYPE_FLOAT, 0, { .float64 = rng_double(0.0, 100000.0) } },
            { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 27 } },
            { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 28 } },
        };
        stoolap_stmt_exec(stmt, p, 3, NULL);
    }
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) {
        StoolapValue p[3] = {
            { .value_type = STOOLAP_TYPE_FLOAT, 0, { .float64 = rng_double(0.0, 100000.0) } },
            { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 27 } },
            { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 28 } },
        };
        stoolap_stmt_exec(stmt, p, 3, NULL);
    }
    elapsed = now_nanos() - t0;
    print_result("UPDATE complex", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── INSERT single ── */
    rc = stoolap_prepare(db,
        "INSERT INTO users (id, name, email, age, balance, active, created_at) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7)", &stmt);
    CHECK(rc, "prepare insert_single");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) {
        int64_t id = (int64_t)(ROW_COUNT + 1000 + i);
        char name[64], email[64];
        snprintf(name, sizeof(name), "New_%lld", (long long)id);
        snprintf(email, sizeof(email), "new%lld@example.com", (long long)id);
        int64_t age = rng_range(18, 80);

        StoolapValue p[7] = {
            { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = id } },
            { .value_type = STOOLAP_TYPE_TEXT, 0, { .text = { name, (int64_t)strlen(name) } } },
            { .value_type = STOOLAP_TYPE_TEXT, 0, { .text = { email, (int64_t)strlen(email) } } },
            { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = age } },
            { .value_type = STOOLAP_TYPE_FLOAT, 0, { .float64 = 100.0 } },
            { .value_type = STOOLAP_TYPE_BOOLEAN, 0, { .boolean = 1 } },
            { .value_type = STOOLAP_TYPE_TEXT, 0, { .text = { "2024-01-01 00:00:00", 19 } } },
        };
        stoolap_stmt_exec(stmt, p, 7, NULL);
    }
    elapsed = now_nanos() - t0;
    print_result("INSERT single", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── DELETE by ID ── */
    rc = stoolap_prepare(db, "DELETE FROM users WHERE id = $1", &stmt);
    CHECK(rc, "prepare delete_by_id");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) {
        StoolapValue p = { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = (int64_t)(ROW_COUNT + 1000 + i) } };
        stoolap_stmt_exec(stmt, &p, 1, NULL);
    }
    elapsed = now_nanos() - t0;
    print_result("DELETE by ID", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── DELETE complex ── */
    rc = stoolap_prepare(db,
        "DELETE FROM users WHERE age >= $1 AND age <= $2 AND active = true", &stmt);
    CHECK(rc, "prepare delete_complex");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) {
        StoolapValue p[2] = {
            { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 25 } },
            { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 26 } },
        };
        stoolap_stmt_exec(stmt, p, 2, NULL);
    }
    elapsed = now_nanos() - t0;
    print_result("DELETE complex", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── Aggregation (GROUP BY) ── */
    rc = stoolap_prepare(db,
        "SELECT age, COUNT(*), AVG(balance) FROM users GROUP BY age", &stmt);
    CHECK(rc, "prepare agg");
    for (i = 0; i < 10; i++) query_drain(db, stmt);
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS_MEDIUM; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("Aggregation (GROUP BY)", elapsed, ITERATIONS_MEDIUM);
    stoolap_stmt_finalize(stmt);

    printf("============================================================\n\n");
    printf("%-25s | %15s | %12s\n", "Advanced Operations", "Avg (us)", "ops/sec");
    printf("---------------------------------------------------------------\n");

    /* ── Create orders table ── */
    rc = stoolap_exec(db,
        "CREATE TABLE orders ("
        "  id INTEGER PRIMARY KEY,"
        "  user_id INTEGER NOT NULL,"
        "  amount REAL NOT NULL,"
        "  status TEXT NOT NULL,"
        "  order_date TEXT NOT NULL"
        ")", NULL);
    CHECK(rc, "create orders");
    rc = stoolap_exec(db, "CREATE INDEX idx_orders_user_id ON orders(user_id)", NULL);
    CHECK(rc, "create idx_orders_user_id");
    rc = stoolap_exec(db, "CREATE INDEX idx_orders_status ON orders(status)", NULL);
    CHECK(rc, "create idx_orders_status");

    /* Populate orders */
    {
        StoolapStmt* insert_order = NULL;
        rc = stoolap_prepare(db,
            "INSERT INTO orders (id, user_id, amount, status, order_date) "
            "VALUES ($1, $2, $3, $4, $5)", &insert_order);
        CHECK(rc, "prepare insert_order");

        const char* statuses[] = { "pending", "completed", "shipped", "cancelled" };
        for (i = 1; i <= ROW_COUNT * 3; i++) {
            int64_t user_id = rng_range(1, ROW_COUNT + 1);
            double amount = rng_double(10.0, 1000.0);
            const char* status = statuses[xorshift64() % 4];
            StoolapValue p[5] = {
                { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = (int64_t)i } },
                { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = user_id } },
                { .value_type = STOOLAP_TYPE_FLOAT, 0, { .float64 = amount } },
                { .value_type = STOOLAP_TYPE_TEXT, 0, { .text = { status, (int64_t)strlen(status) } } },
                { .value_type = STOOLAP_TYPE_TEXT, 0, { .text = { "2024-01-15", 10 } } },
            };
            stoolap_stmt_exec(insert_order, p, 5, NULL);
        }
        stoolap_stmt_finalize(insert_order);
    }

    /* ── INNER JOIN ── */
    rc = stoolap_prepare(db,
        "SELECT u.name, o.amount FROM users u INNER JOIN orders o ON u.id = o.user_id "
        "WHERE o.status = 'completed' LIMIT 100", &stmt);
    CHECK(rc, "prepare inner_join");
    for (i = 0; i < WARMUP; i++) query_drain(db, stmt);
    t0 = now_nanos();
    for (i = 0; i < 100; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("INNER JOIN", elapsed, 100);
    stoolap_stmt_finalize(stmt);

    /* ── LEFT JOIN + GROUP BY ── */
    rc = stoolap_prepare(db,
        "SELECT u.name, COUNT(o.id) as order_count, SUM(o.amount) as total "
        "FROM users u LEFT JOIN orders o ON u.id = o.user_id "
        "GROUP BY u.id, u.name LIMIT 100", &stmt);
    CHECK(rc, "prepare left_join");
    for (i = 0; i < WARMUP; i++) query_drain(db, stmt);
    t0 = now_nanos();
    for (i = 0; i < 100; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("LEFT JOIN + GROUP BY", elapsed, 100);
    stoolap_stmt_finalize(stmt);

    /* ── Scalar subquery ── */
    rc = stoolap_prepare(db,
        "SELECT name, balance, (SELECT AVG(balance) FROM users) as avg_balance "
        "FROM users WHERE balance > (SELECT AVG(balance) FROM users) LIMIT 100", &stmt);
    CHECK(rc, "prepare scalar_sub");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("Scalar subquery", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── IN subquery ── */
    rc = stoolap_prepare(db,
        "SELECT * FROM users WHERE id IN "
        "(SELECT user_id FROM orders WHERE status = 'completed') LIMIT 100", &stmt);
    CHECK(rc, "prepare in_sub");
    t0 = now_nanos();
    for (i = 0; i < 10; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("IN subquery", elapsed, 10);
    stoolap_stmt_finalize(stmt);

    /* ── EXISTS subquery ── */
    rc = stoolap_prepare(db,
        "SELECT * FROM users u WHERE EXISTS "
        "(SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.amount > 500) LIMIT 100", &stmt);
    CHECK(rc, "prepare exists_sub");
    for (i = 0; i < WARMUP; i++) query_drain(db, stmt);
    t0 = now_nanos();
    for (i = 0; i < 100; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("EXISTS subquery", elapsed, 100);
    stoolap_stmt_finalize(stmt);

    /* ── CTE + JOIN ── */
    rc = stoolap_prepare(db,
        "WITH high_value AS (SELECT user_id, SUM(amount) as total FROM orders "
        "GROUP BY user_id HAVING SUM(amount) > 1000) "
        "SELECT u.name, h.total FROM users u INNER JOIN high_value h ON u.id = h.user_id LIMIT 100", &stmt);
    CHECK(rc, "prepare cte");
    t0 = now_nanos();
    for (i = 0; i < 20; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("CTE + JOIN", elapsed, 20);
    stoolap_stmt_finalize(stmt);

    /* ── Window ROW_NUMBER ── */
    rc = stoolap_prepare(db,
        "SELECT name, balance, ROW_NUMBER() OVER (ORDER BY balance DESC) as rank "
        "FROM users LIMIT 100", &stmt);
    CHECK(rc, "prepare window_rn");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("Window ROW_NUMBER", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── Window ROW_NUMBER (PK) ── */
    rc = stoolap_prepare(db,
        "SELECT name, ROW_NUMBER() OVER (ORDER BY id) as rank FROM users LIMIT 100", &stmt);
    CHECK(rc, "prepare window_rn_pk");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("Window ROW_NUMBER (PK)", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── Window PARTITION BY ── */
    rc = stoolap_prepare(db,
        "SELECT name, age, balance, RANK() OVER (PARTITION BY age ORDER BY balance DESC) as age_rank "
        "FROM users LIMIT 100", &stmt);
    CHECK(rc, "prepare window_part");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("Window PARTITION BY", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── UNION ALL ── */
    rc = stoolap_prepare(db,
        "SELECT name, 'high' as category FROM users WHERE balance > 50000 "
        "UNION ALL SELECT name, 'low' as category FROM users WHERE balance <= 50000 LIMIT 100", &stmt);
    CHECK(rc, "prepare union");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("UNION ALL", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── CASE expression ── */
    rc = stoolap_prepare(db,
        "SELECT name, CASE WHEN balance > 75000 THEN 'platinum' "
        "WHEN balance > 50000 THEN 'gold' WHEN balance > 25000 THEN 'silver' "
        "ELSE 'bronze' END as tier FROM users LIMIT 100", &stmt);
    CHECK(rc, "prepare case");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("CASE expression", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── Complex JOIN+GROUP+HAVING ── */
    rc = stoolap_prepare(db,
        "SELECT u.name, COUNT(DISTINCT o.id) as orders, SUM(o.amount) as total "
        "FROM users u INNER JOIN orders o ON u.id = o.user_id "
        "WHERE u.active = true AND o.status IN ('completed', 'shipped') "
        "GROUP BY u.id, u.name HAVING COUNT(o.id) > 1 LIMIT 50", &stmt);
    CHECK(rc, "prepare complex_join");
    t0 = now_nanos();
    for (i = 0; i < 20; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("Complex JOIN+GROUP+HAVING", elapsed, 20);
    stoolap_stmt_finalize(stmt);

    /* ── Batch INSERT in transaction (prepared stmt + SQL BEGIN/COMMIT, matches Rust) ── */
    {
        StoolapStmt* batch_stmt = NULL;
        rc = stoolap_prepare(db,
            "INSERT INTO orders (id, user_id, amount, status, order_date) "
            "VALUES ($1, $2, $3, $4, $5)", &batch_stmt);
        CHECK(rc, "prepare batch_insert");

        t0 = now_nanos();
        for (i = 0; i < ITERATIONS; i++) {
            int64_t base_id = (int64_t)(ROW_COUNT * 10 + i * 100);
            stoolap_exec(db, "BEGIN", NULL);
            int j;
            for (j = 0; j < 100; j++) {
                StoolapValue p[5] = {
                    { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = base_id + j } },
                    { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 1 } },
                    { .value_type = STOOLAP_TYPE_FLOAT, 0, { .float64 = 100.0 } },
                    { .value_type = STOOLAP_TYPE_TEXT, 0, { .text = { "pending", 7 } } },
                    { .value_type = STOOLAP_TYPE_TEXT, 0, { .text = { "2024-02-01", 10 } } },
                };
                stoolap_stmt_exec(batch_stmt, p, 5, NULL);
            }
            stoolap_exec(db, "COMMIT", NULL);
        }
        elapsed = now_nanos() - t0;
        print_result("Batch INSERT (100 rows)", elapsed, ITERATIONS);
        stoolap_stmt_finalize(batch_stmt);
    }

    printf("============================================================\n\n");
    printf("%-25s | %15s | %12s\n", "Bottleneck Hunters", "Avg (us)", "ops/sec");
    printf("---------------------------------------------------------------\n");

    /* ── DISTINCT (no ORDER) ── */
    rc = stoolap_prepare(db, "SELECT DISTINCT age FROM users", &stmt);
    CHECK(rc, "prepare distinct");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("DISTINCT (no ORDER)", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── DISTINCT + ORDER BY ── */
    rc = stoolap_prepare(db, "SELECT DISTINCT age FROM users ORDER BY age", &stmt);
    CHECK(rc, "prepare distinct_order");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("DISTINCT + ORDER BY", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── COUNT DISTINCT ── */
    rc = stoolap_prepare(db, "SELECT COUNT(DISTINCT age) FROM users", &stmt);
    CHECK(rc, "prepare count_distinct");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("COUNT DISTINCT", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── LIKE prefix ── */
    rc = stoolap_prepare(db, "SELECT * FROM users WHERE name LIKE 'User_1%' LIMIT 100", &stmt);
    CHECK(rc, "prepare like_prefix");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("LIKE prefix (User_1%)", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── LIKE contains ── */
    rc = stoolap_prepare(db, "SELECT * FROM users WHERE email LIKE '%50%' LIMIT 100", &stmt);
    CHECK(rc, "prepare like_contains");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("LIKE contains (%50%)", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── OR conditions ── */
    rc = stoolap_prepare(db,
        "SELECT * FROM users WHERE age = 25 OR age = 50 OR age = 75 LIMIT 100", &stmt);
    CHECK(rc, "prepare or_cond");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("OR conditions (3 vals)", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── IN list ── */
    rc = stoolap_prepare(db,
        "SELECT * FROM users WHERE age IN (20, 25, 30, 35, 40, 45, 50) LIMIT 100", &stmt);
    CHECK(rc, "prepare in_list");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("IN list (7 values)", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── NOT IN subquery ── */
    rc = stoolap_prepare(db,
        "SELECT * FROM users WHERE id NOT IN "
        "(SELECT user_id FROM orders WHERE status = 'cancelled') LIMIT 100", &stmt);
    CHECK(rc, "prepare not_in");
    t0 = now_nanos();
    for (i = 0; i < 10; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("NOT IN subquery", elapsed, 10);
    stoolap_stmt_finalize(stmt);

    /* ── NOT EXISTS ── */
    rc = stoolap_prepare(db,
        "SELECT * FROM users u WHERE NOT EXISTS "
        "(SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.status = 'cancelled') LIMIT 100", &stmt);
    CHECK(rc, "prepare not_exists");
    for (i = 0; i < WARMUP; i++) query_drain(db, stmt);
    t0 = now_nanos();
    for (i = 0; i < 100; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("NOT EXISTS subquery", elapsed, 100);
    stoolap_stmt_finalize(stmt);

    /* ── OFFSET pagination ── */
    rc = stoolap_prepare(db,
        "SELECT * FROM users ORDER BY id LIMIT 100 OFFSET 5000", &stmt);
    CHECK(rc, "prepare offset");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("OFFSET pagination (5000)", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── Multi-column ORDER BY ── */
    rc = stoolap_prepare(db,
        "SELECT * FROM users ORDER BY age DESC, balance ASC, name LIMIT 100", &stmt);
    CHECK(rc, "prepare multi_order");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("Multi-col ORDER BY (3)", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── Self JOIN ── */
    rc = stoolap_prepare(db,
        "SELECT u1.name, u2.name, u1.age FROM users u1 "
        "INNER JOIN users u2 ON u1.age = u2.age AND u1.id < u2.id LIMIT 100", &stmt);
    CHECK(rc, "prepare self_join");
    for (i = 0; i < WARMUP; i++) query_drain(db, stmt);
    t0 = now_nanos();
    for (i = 0; i < 100; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("Self JOIN (same age)", elapsed, 100);
    stoolap_stmt_finalize(stmt);

    /* ── Multiple window functions ── */
    rc = stoolap_prepare(db,
        "SELECT name, balance, ROW_NUMBER() OVER (ORDER BY balance DESC) as rn, "
        "RANK() OVER (ORDER BY balance DESC) as rnk, "
        "LAG(balance) OVER (ORDER BY balance DESC) as prev_bal FROM users LIMIT 100", &stmt);
    CHECK(rc, "prepare multi_window");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("Multi window funcs (3)", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── Nested subquery (3 levels) ── */
    rc = stoolap_prepare(db,
        "SELECT * FROM users WHERE id IN "
        "(SELECT user_id FROM orders WHERE amount > "
        "(SELECT AVG(amount) FROM orders)) LIMIT 100", &stmt);
    CHECK(rc, "prepare nested_sub");
    t0 = now_nanos();
    for (i = 0; i < 20; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("Nested subquery (3 lvl)", elapsed, 20);
    stoolap_stmt_finalize(stmt);

    /* ── Multiple aggregates ── */
    rc = stoolap_prepare(db,
        "SELECT COUNT(*), SUM(balance), AVG(balance), MIN(balance), MAX(balance), "
        "COUNT(DISTINCT age) FROM users", &stmt);
    CHECK(rc, "prepare multi_agg");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("Multi aggregates (6)", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── COALESCE + IS NOT NULL ── */
    rc = stoolap_prepare(db,
        "SELECT name, COALESCE(balance, 0) as bal FROM users "
        "WHERE balance IS NOT NULL LIMIT 100", &stmt);
    CHECK(rc, "prepare coalesce");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("COALESCE + IS NOT NULL", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── Expression in WHERE (funcs) ── */
    rc = stoolap_prepare(db,
        "SELECT * FROM users WHERE LENGTH(name) > 7 AND UPPER(name) LIKE 'USER_%' LIMIT 100", &stmt);
    CHECK(rc, "prepare expr_where");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("Expr in WHERE (funcs)", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── Math expressions ── */
    rc = stoolap_prepare(db,
        "SELECT name, balance * 1.1 as new_bal, ROUND(balance / 1000, 2) as k_bal, "
        "ABS(balance - 50000) as diff FROM users LIMIT 100", &stmt);
    CHECK(rc, "prepare math");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("Math expressions", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── String concat ── */
    rc = stoolap_prepare(db,
        "SELECT name || ' (' || email || ')' as full_info FROM users LIMIT 100", &stmt);
    CHECK(rc, "prepare concat");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("String concat (||)", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── Large result (no LIMIT) ── */
    rc = stoolap_prepare(db,
        "SELECT id, name, balance FROM users WHERE active = true", &stmt);
    CHECK(rc, "prepare large_result");
    t0 = now_nanos();
    for (i = 0; i < 20; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("Large result (no LIMIT)", elapsed, 20);
    stoolap_stmt_finalize(stmt);

    /* ── Multiple CTEs ── */
    rc = stoolap_prepare(db,
        "WITH young AS (SELECT * FROM users WHERE age < 30), "
        "rich AS (SELECT * FROM users WHERE balance > 70000) "
        "SELECT y.name, r.name FROM young y INNER JOIN rich r ON y.id = r.id LIMIT 50", &stmt);
    CHECK(rc, "prepare multi_cte");
    for (i = 0; i < WARMUP; i++) query_drain(db, stmt);
    t0 = now_nanos();
    for (i = 0; i < 100; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("Multiple CTEs (2)", elapsed, 100);
    stoolap_stmt_finalize(stmt);

    /* ── Correlated subquery in SELECT ── */
    rc = stoolap_prepare(db,
        "SELECT u.name, (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) as order_count "
        "FROM users u LIMIT 100", &stmt);
    CHECK(rc, "prepare corr_select");
    for (i = 0; i < 5; i++) query_drain(db, stmt);
    t0 = now_nanos();
    for (i = 0; i < 100; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("Correlated in SELECT", elapsed, 100);
    stoolap_stmt_finalize(stmt);

    /* ── BETWEEN (non-indexed) ── */
    rc = stoolap_prepare(db,
        "SELECT * FROM users WHERE balance BETWEEN 25000 AND 75000 LIMIT 100", &stmt);
    CHECK(rc, "prepare between");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("BETWEEN (non-indexed)", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── GROUP BY (2 columns) ── */
    rc = stoolap_prepare(db,
        "SELECT age, active, COUNT(*), AVG(balance) FROM users GROUP BY age, active", &stmt);
    CHECK(rc, "prepare multi_group");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("GROUP BY (2 columns)", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── CROSS JOIN (limited) ── */
    rc = stoolap_prepare(db,
        "SELECT u.name, o.status FROM users u "
        "CROSS JOIN (SELECT DISTINCT status FROM orders) o LIMIT 100", &stmt);
    CHECK(rc, "prepare cross_join");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("CROSS JOIN (limited)", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── Derived table ── */
    rc = stoolap_prepare(db,
        "SELECT t.age_group, COUNT(*) FROM "
        "(SELECT CASE WHEN age < 30 THEN 'young' WHEN age < 50 THEN 'middle' "
        "ELSE 'senior' END as age_group FROM users) t GROUP BY t.age_group", &stmt);
    CHECK(rc, "prepare derived");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("Derived table (FROM sub)", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── Window ROWS frame ── */
    rc = stoolap_prepare(db,
        "SELECT name, balance, SUM(balance) OVER "
        "(ORDER BY balance ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING) as rolling_sum "
        "FROM users LIMIT 100", &stmt);
    CHECK(rc, "prepare window_frame");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("Window ROWS frame", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── HAVING complex ── */
    rc = stoolap_prepare(db,
        "SELECT age FROM users GROUP BY age HAVING COUNT(*) > 100 AND AVG(balance) > 40000", &stmt);
    CHECK(rc, "prepare having");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("HAVING complex", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    /* ── Compare with subquery ── */
    rc = stoolap_prepare(db,
        "SELECT * FROM users WHERE balance > (SELECT AVG(amount) * 100 FROM orders) LIMIT 100", &stmt);
    CHECK(rc, "prepare compare_sub");
    t0 = now_nanos();
    for (i = 0; i < ITERATIONS; i++) query_drain(db, stmt);
    elapsed = now_nanos() - t0;
    print_result("Compare with subquery", elapsed, ITERATIONS);
    stoolap_stmt_finalize(stmt);

    printf("============================================================\n");

    stoolap_close(db);
    return 0;
}
