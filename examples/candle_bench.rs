// Copyright 2025 Stoolap Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Time-series workload bench: many (key, key, time) series in one table,
//! bulk loaded, sealed, reopened, and queried the way a market data
//! application does. Reports wall latency (median, p99, max), process CPU
//! time and resident memory per phase, so a storage or index change can be
//! judged on all three at once.
//!
//! Run: `cargo run --release --example candle_bench`
//! Knobs (env): PAIRS (174), MINUTES (12500), THREADS (4), CKPT (30 s).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use stoolap::Database;

fn env_or(name: &str, default: i64) -> i64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[cfg(unix)]
fn cpu_secs() -> f64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    let t = |tv: libc::timeval| tv.tv_sec as f64 + tv.tv_usec as f64 / 1e6;
    t(ru.ru_utime) + t(ru.ru_stime)
}

/// Process CPU time is reported as zero where getrusage is not available.
#[cfg(not(unix))]
fn cpu_secs() -> f64 {
    0.0
}

fn rss_mb() -> f64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok();
    out.and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

fn fmt_ts(secs: i64) -> String {
    chrono::DateTime::from_timestamp(secs, 0)
        .map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_default()
}

struct Lat(Vec<f64>);

impl Lat {
    fn from_ms(mut v: Vec<f64>) -> Self {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Lat(v)
    }
    fn pct(&self, p: f64) -> f64 {
        if self.0.is_empty() {
            return 0.0;
        }
        let i = ((self.0.len() as f64 - 1.0) * p).round() as usize;
        self.0[i]
    }
    fn max(&self) -> f64 {
        self.0.last().copied().unwrap_or(0.0)
    }
}

fn row(phase: &str, metric: &str, value: String) {
    println!("{phase:<10} {metric:<44} {value}");
}

/// Run `sql` `n` times, return latencies (ms) and rows of the last run.
fn timed(db: &Database, sql: &str, n: usize) -> (Lat, usize, f64) {
    let mut ms = Vec::with_capacity(n);
    let mut rows = 0;
    let cpu0 = cpu_secs();
    for _ in 0..n {
        let t = Instant::now();
        rows = db.query(sql, ()).unwrap().count();
        ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let cpu_ms = (cpu_secs() - cpu0) * 1000.0 / n as f64;
    (Lat::from_ms(ms), rows, cpu_ms)
}

struct Shapes {
    pair_day: String,
    pair_latest: String,
    pair_since: String,
    all_recent: String,
    rollup: String,
    day_summary: String,
    distinct: String,
}

fn shapes(last: i64) -> Shapes {
    let day_start = fmt_ts(last - 86400);
    let day_end = fmt_ts(last);
    let ten_min = fmt_ts(last - 600);
    let cols = "time, open, high, low, close, volume";
    let pair = "exchange = 'ex1' AND symbol = 'S7USDT'";
    Shapes {
        pair_day: format!(
            "SELECT {cols} FROM c WHERE {pair} AND time >= '{day_start}' AND time <= '{day_end}' ORDER BY time DESC LIMIT 500"
        ),
        pair_latest: format!("SELECT {cols} FROM c WHERE {pair} ORDER BY time DESC LIMIT 200"),
        pair_since: format!(
            "SELECT {cols} FROM c WHERE {pair} AND time >= '{day_start}' ORDER BY time DESC LIMIT 200"
        ),
        all_recent: format!(
            "SELECT exchange, symbol, {cols} FROM c WHERE time >= '{ten_min}' ORDER BY time DESC LIMIT 100"
        ),
        rollup: format!(
            "SELECT TIME_TRUNC('5m', time), exchange, symbol, FIRST(open ORDER BY time), MAX(high), MIN(low), \
             LAST(close ORDER BY time), ROUND(SUM(volume), 8) FROM c WHERE time >= '{ten_min}' \
             GROUP BY TIME_TRUNC('5m', time), exchange, symbol"
        ),
        day_summary: format!(
            "SELECT exchange, symbol, FIRST(open ORDER BY time), LAST(close ORDER BY time), MAX(high), MIN(low) \
             FROM c WHERE time >= '{day_start}' GROUP BY exchange, symbol"
        ),
        distinct: "SELECT DISTINCT exchange, symbol FROM c ORDER BY exchange, symbol".to_string(),
    }
}

fn query_set(phase: &str, db: &Database, s: &Shapes) {
    for (name, sql, n) in [
        ("pair + day range DESC LIMIT 500", &s.pair_day, 7),
        ("pair latest DESC LIMIT 200", &s.pair_latest, 7),
        ("pair + time >= day DESC LIMIT 200", &s.pair_since, 7),
        ("all pairs last 10 min DESC LIMIT 100", &s.all_recent, 7),
        ("5m rollup last 10 min", &s.rollup, 5),
        ("day summary per pair", &s.day_summary, 5),
        ("DISTINCT pairs", &s.distinct, 3),
    ] {
        let (lat, rows, cpu) = timed(db, sql, n);
        row(
            phase,
            name,
            format!(
                "p50 {:>9.2} ms  max {:>9.2} ms  cpu {:>8.2} ms  rows={rows}",
                lat.pct(0.5),
                lat.max(),
                cpu
            ),
        );
    }
}

/// Fire the per-pair "latest" query for every pair across `threads` workers.
fn fan_out(phase: &str, db: &Database, pairs: i64, threads: usize) {
    let sqls: Vec<String> = (0..pairs)
        .map(|p| {
            format!(
                "SELECT time, open, high, low, close, volume FROM c WHERE exchange = 'ex{}' AND symbol = 'S{}USDT' \
                 ORDER BY time DESC LIMIT 200",
                p % 3,
                p / 3
            )
        })
        .collect();
    let sqls = Arc::new(sqls);
    let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let cpu0 = cpu_secs();
    let wall = Instant::now();
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let db = db.clone();
            let sqls = Arc::clone(&sqls);
            let next = Arc::clone(&next);
            std::thread::spawn(move || {
                let mut ms = Vec::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= sqls.len() {
                        break;
                    }
                    let t = Instant::now();
                    db.query(&sqls[i], ()).unwrap().count();
                    ms.push(t.elapsed().as_secs_f64() * 1000.0);
                }
                ms
            })
        })
        .collect();
    let mut all = Vec::new();
    for h in handles {
        all.extend(h.join().unwrap());
    }
    let wall_ms = wall.elapsed().as_secs_f64() * 1000.0;
    let cpu_ms = (cpu_secs() - cpu0) * 1000.0;
    let lat = Lat::from_ms(all);
    row(
        phase,
        &format!("fan-out {pairs} pair queries on {threads} threads"),
        format!(
            "wall {:>8.0} ms  cpu {:>8.0} ms  p50 {:>8.2} ms  p99 {:>8.2} ms  max {:>8.2} ms",
            wall_ms,
            cpu_ms,
            lat.pct(0.5),
            lat.pct(0.99),
            lat.max()
        ),
    );
}

fn main() {
    let pairs = env_or("PAIRS", 174);
    let minutes = env_or("MINUTES", 12_500);
    let threads = env_or("THREADS", 4) as usize;
    let ckpt = env_or("CKPT", 30);
    let dir = std::env::temp_dir().join(format!("candle_bench_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let dsn = format!("file://{}?checkpoint_interval={ckpt}", dir.display());
    println!(
        "rows={} pairs={pairs} minutes={minutes} threads={threads} checkpoint_interval={ckpt}s",
        pairs * minutes
    );
    row("start", "rss", format!("{:.0} MB", rss_mb()));

    let mut db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY AUTO_INCREMENT, time TIMESTAMP NOT NULL, \
         exchange TEXT NOT NULL, symbol TEXT NOT NULL, open FLOAT, high FLOAT, low FLOAT, close FLOAT, \
         volume FLOAT, UNIQUE(exchange, symbol, time))",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_c_time ON c(time) USING BTREE", ())
        .unwrap();

    // Phase: bulk load in time order, ten minutes of every pair per statement
    let base = 1_704_067_200i64;
    let t0 = Instant::now();
    let cpu0 = cpu_secs();
    let mut stmt = String::new();
    for m0 in (0..minutes).step_by(10) {
        stmt.clear();
        stmt.push_str(
            "INSERT INTO c (time, exchange, symbol, open, high, low, close, volume) VALUES ",
        );
        let mut first = true;
        for m in m0..(m0 + 10).min(minutes) {
            let t = fmt_ts(base + m * 60);
            for p in 0..pairs {
                if !first {
                    stmt.push(',');
                }
                first = false;
                let px = 100.0 + p as f64 + (m % 97) as f64 * 0.01;
                stmt.push_str(&format!(
                    "('{t}', 'ex{}', 'S{}USDT', {px}, {}, {}, {}, {})",
                    p % 3,
                    p / 3,
                    px + 0.5,
                    px - 0.5,
                    px + 0.1,
                    1000 + m % 50
                ));
            }
        }
        db.execute(&stmt, ()).unwrap();
    }
    row(
        "load",
        "bulk insert",
        format!(
            "wall {:>8.0} ms  cpu {:>8.0} ms  rss {:.0} MB",
            t0.elapsed().as_secs_f64() * 1000.0,
            (cpu_secs() - cpu0) * 1000.0,
            rss_mb()
        ),
    );
    let last = base + (minutes - 1) * 60;
    let s = shapes(last);

    // Phase: first query after load pays any lazy index build
    let (lat, _, cpu) = timed(&db, &s.pair_day, 1);
    row(
        "hot",
        "first pair query after load",
        format!("{:>9.2} ms  cpu {:>8.2} ms", lat.max(), cpu),
    );
    query_set("hot", &db, &s);
    fan_out("hot", &db, pairs, threads);
    row("hot", "rss", format!("{:.0} MB", rss_mb()));

    // Phase: seal while a reader keeps querying; report the stall it sees
    let stop = Arc::new(AtomicBool::new(false));
    let reader = {
        let db = db.clone();
        let stop = Arc::clone(&stop);
        let sql = s.pair_since.clone();
        std::thread::spawn(move || {
            let mut ms = Vec::new();
            while !stop.load(Ordering::Relaxed) {
                let t = Instant::now();
                db.query(&sql, ()).unwrap().count();
                ms.push(t.elapsed().as_secs_f64() * 1000.0);
                std::thread::sleep(Duration::from_millis(5));
            }
            ms
        })
    };
    std::thread::sleep(Duration::from_millis(200));
    let t = Instant::now();
    let cpu0 = cpu_secs();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let seal_ms = t.elapsed().as_secs_f64() * 1000.0;
    let seal_cpu = (cpu_secs() - cpu0) * 1000.0;
    std::thread::sleep(Duration::from_millis(200));
    stop.store(true, Ordering::Relaxed);
    let reader_lat = Lat::from_ms(reader.join().unwrap());
    row(
        "seal",
        "PRAGMA CHECKPOINT of the loaded rows",
        format!(
            "wall {seal_ms:>8.0} ms  cpu {seal_cpu:>8.0} ms  rss {:.0} MB",
            rss_mb()
        ),
    );
    row(
        "seal",
        "reader latency during seal",
        format!(
            "p50 {:>8.2} ms  p99 {:>8.2} ms  max {:>8.0} ms  ({} queries)",
            reader_lat.pct(0.5),
            reader_lat.pct(0.99),
            reader_lat.max(),
            reader_lat.0.len()
        ),
    );

    // Phase: cold rows, eager columns (as sealed in this process)
    query_set("cold", &db, &s);
    fan_out("cold", &db, pairs, threads);
    row("cold", "rss", format!("{:.0} MB", rss_mb()));

    // Phase: reopen, columns deferred (as after a restart or a cold reload)
    drop(db);
    let t = Instant::now();
    db = Database::open(&dsn).unwrap();
    row(
        "reopen",
        "Database::open",
        format!(
            "{:>9.0} ms  rss {:.0} MB",
            t.elapsed().as_secs_f64() * 1000.0,
            rss_mb()
        ),
    );
    query_set("reopen", &db, &s);
    fan_out("reopen", &db, pairs, threads);
    row("reopen", "rss after queries", format!("{:.0} MB", rss_mb()));

    // Phase: retention delete of the oldest ten minutes
    let t = Instant::now();
    let deleted = db
        .execute(
            "DELETE FROM c WHERE time < $1",
            (fmt_ts(base + 600).as_str(),),
        )
        .unwrap();
    row(
        "delete",
        "DELETE time < first 10 min",
        format!(
            "{:>9.0} ms  rows={deleted}",
            t.elapsed().as_secs_f64() * 1000.0
        ),
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}
