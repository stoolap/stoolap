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

//! Product Recommendation Engine — Real-World Vector Search Example
//!
//! Simulates an e-commerce product catalog with semantic embeddings.
//! Demonstrates:
//!   - Schema with VECTOR columns alongside regular columns
//!   - HNSW index for fast approximate nearest-neighbor search
//!   - Hybrid search: vector similarity + SQL filters (category, price range)
//!   - `<=>` operator shorthand for L2 distance
//!   - Transactions for bulk loading
//!   - Window functions for ranking
//!
//! Run:  cargo run --example product_search --release

use std::collections::BTreeSet;
use std::time::Instant;
use stoolap::api::Database;

// ─────────────────────────────────────────────────────────────
// Product catalog data
// ─────────────────────────────────────────────────────────────

struct Product {
    name: &'static str,
    category: &'static str,
    price: f64,
    description: &'static str,
}

/// 50 realistic products across 5 categories.
/// Each product gets a deterministic 32-dim embedding derived from its attributes.
const PRODUCTS: &[Product] = &[
    // ── Electronics ──
    Product {
        name: "Wireless Noise-Cancelling Headphones",
        category: "Electronics",
        price: 299.99,
        description: "over-ear bluetooth ANC headphones",
    },
    Product {
        name: "True Wireless Earbuds",
        category: "Electronics",
        price: 149.99,
        description: "in-ear bluetooth earbuds with ANC",
    },
    Product {
        name: "Studio Monitor Headphones",
        category: "Electronics",
        price: 199.99,
        description: "wired open-back studio headphones",
    },
    Product {
        name: "Portable Bluetooth Speaker",
        category: "Electronics",
        price: 79.99,
        description: "waterproof portable speaker",
    },
    Product {
        name: "Soundbar with Subwoofer",
        category: "Electronics",
        price: 349.99,
        description: "home theater soundbar system",
    },
    Product {
        name: "USB-C Hub Adapter",
        category: "Electronics",
        price: 49.99,
        description: "multiport USB hub HDMI ethernet",
    },
    Product {
        name: "Mechanical Gaming Keyboard",
        category: "Electronics",
        price: 129.99,
        description: "RGB mechanical keyboard cherry switches",
    },
    Product {
        name: "Ergonomic Wireless Mouse",
        category: "Electronics",
        price: 69.99,
        description: "vertical ergonomic bluetooth mouse",
    },
    Product {
        name: "4K Webcam",
        category: "Electronics",
        price: 119.99,
        description: "ultra HD webcam with microphone",
    },
    Product {
        name: "Portable SSD 1TB",
        category: "Electronics",
        price: 89.99,
        description: "external solid state drive USB-C",
    },
    // ── Clothing ──
    Product {
        name: "Merino Wool Sweater",
        category: "Clothing",
        price: 89.99,
        description: "warm knit wool pullover winter",
    },
    Product {
        name: "Down Puffer Jacket",
        category: "Clothing",
        price: 199.99,
        description: "insulated winter jacket waterproof",
    },
    Product {
        name: "Fleece Zip-Up Hoodie",
        category: "Clothing",
        price: 59.99,
        description: "soft fleece hooded sweatshirt",
    },
    Product {
        name: "Slim Fit Chinos",
        category: "Clothing",
        price: 49.99,
        description: "stretch cotton casual pants",
    },
    Product {
        name: "Denim Jacket",
        category: "Clothing",
        price: 79.99,
        description: "classic blue denim trucker jacket",
    },
    Product {
        name: "Linen Summer Shirt",
        category: "Clothing",
        price: 44.99,
        description: "lightweight breathable linen shirt",
    },
    Product {
        name: "Running Shorts",
        category: "Clothing",
        price: 34.99,
        description: "moisture-wicking athletic shorts",
    },
    Product {
        name: "Waterproof Rain Coat",
        category: "Clothing",
        price: 129.99,
        description: "lightweight packable rain jacket",
    },
    Product {
        name: "Cashmere Scarf",
        category: "Clothing",
        price: 69.99,
        description: "soft luxury cashmere winter scarf",
    },
    Product {
        name: "Thermal Base Layer",
        category: "Clothing",
        price: 39.99,
        description: "warm compression thermal underwear",
    },
    // ── Books ──
    Product {
        name: "Database Internals",
        category: "Books",
        price: 49.99,
        description: "distributed systems storage engines algorithms",
    },
    Product {
        name: "Designing Data-Intensive Applications",
        category: "Books",
        price: 44.99,
        description: "distributed systems data architecture scalability",
    },
    Product {
        name: "The Rust Programming Language",
        category: "Books",
        price: 39.99,
        description: "rust programming systems memory safety",
    },
    Product {
        name: "Clean Architecture",
        category: "Books",
        price: 34.99,
        description: "software design patterns architecture",
    },
    Product {
        name: "Hands-On Machine Learning",
        category: "Books",
        price: 59.99,
        description: "ML deep learning neural networks python",
    },
    Product {
        name: "Introduction to Algorithms",
        category: "Books",
        price: 79.99,
        description: "algorithms data structures computer science",
    },
    Product {
        name: "Site Reliability Engineering",
        category: "Books",
        price: 44.99,
        description: "SRE operations monitoring infrastructure",
    },
    Product {
        name: "The Art of PostgreSQL",
        category: "Books",
        price: 49.99,
        description: "SQL database queries optimization postgres",
    },
    Product {
        name: "Programming Rust",
        category: "Books",
        price: 54.99,
        description: "rust systems programming performance concurrency",
    },
    Product {
        name: "Deep Learning with Python",
        category: "Books",
        price: 49.99,
        description: "neural networks keras tensorflow ML",
    },
    // ── Sports ──
    Product {
        name: "Carbon Fiber Road Bike",
        category: "Sports",
        price: 2499.99,
        description: "lightweight racing bicycle road cycling",
    },
    Product {
        name: "Trail Running Shoes",
        category: "Sports",
        price: 129.99,
        description: "waterproof trail running footwear grip",
    },
    Product {
        name: "Yoga Mat Premium",
        category: "Sports",
        price: 49.99,
        description: "non-slip exercise yoga mat thick",
    },
    Product {
        name: "Adjustable Dumbbells Set",
        category: "Sports",
        price: 299.99,
        description: "weight training home gym dumbbells",
    },
    Product {
        name: "Hiking Backpack 40L",
        category: "Sports",
        price: 119.99,
        description: "waterproof hiking daypack camping",
    },
    Product {
        name: "Tennis Racket Pro",
        category: "Sports",
        price: 189.99,
        description: "carbon fiber tennis racquet control",
    },
    Product {
        name: "Swimming Goggles",
        category: "Sports",
        price: 24.99,
        description: "anti-fog swim goggles UV protection",
    },
    Product {
        name: "Foam Roller Recovery",
        category: "Sports",
        price: 29.99,
        description: "muscle recovery massage foam roller",
    },
    Product {
        name: "Resistance Bands Set",
        category: "Sports",
        price: 19.99,
        description: "exercise bands strength training portable",
    },
    Product {
        name: "Climbing Harness",
        category: "Sports",
        price: 79.99,
        description: "rock climbing safety harness lightweight",
    },
    // ── Home & Kitchen ──
    Product {
        name: "Espresso Machine",
        category: "Home",
        price: 599.99,
        description: "automatic espresso coffee maker barista",
    },
    Product {
        name: "Cast Iron Dutch Oven",
        category: "Home",
        price: 79.99,
        description: "enameled cast iron pot cooking",
    },
    Product {
        name: "Air Purifier HEPA",
        category: "Home",
        price: 199.99,
        description: "HEPA air purifier large room allergen",
    },
    Product {
        name: "Robot Vacuum Cleaner",
        category: "Home",
        price: 349.99,
        description: "smart robot vacuum lidar mapping",
    },
    Product {
        name: "Stand Mixer",
        category: "Home",
        price: 279.99,
        description: "kitchen stand mixer baking dough",
    },
    Product {
        name: "French Press Coffee",
        category: "Home",
        price: 29.99,
        description: "glass french press coffee brewer",
    },
    Product {
        name: "Memory Foam Pillow",
        category: "Home",
        price: 49.99,
        description: "ergonomic sleeping pillow neck support",
    },
    Product {
        name: "Stainless Steel Water Bottle",
        category: "Home",
        price: 24.99,
        description: "insulated vacuum water bottle travel",
    },
    Product {
        name: "Smart LED Light Bulbs 4-Pack",
        category: "Home",
        price: 39.99,
        description: "wifi smart bulbs color RGB dimming",
    },
    Product {
        name: "Bamboo Cutting Board Set",
        category: "Home",
        price: 34.99,
        description: "natural bamboo kitchen cutting boards",
    },
];

// ─────────────────────────────────────────────────────────────
// Embedding generation — deterministic, based on product text
// ─────────────────────────────────────────────────────────────

const DIMS: usize = 32;

/// Simple hash-based embedding: deterministic 32-dim vector from text.
/// Products with similar words get similar embeddings.
fn text_to_embedding(texts: &[&str]) -> Vec<f32> {
    let mut vec = vec![0.0f32; DIMS];

    // Bag-of-characters + word-level features
    for text in texts {
        for (i, word) in text.split_whitespace().enumerate() {
            let word_lower = word.to_lowercase();
            // Hash each word into a few dimensions
            let h = simple_hash(word_lower.as_bytes());
            let dim1 = (h as usize) % DIMS;
            let dim2 = ((h >> 8) as usize) % DIMS;
            let dim3 = ((h >> 16) as usize) % DIMS;

            let weight = 1.0 / (1.0 + i as f32 * 0.1); // earlier words matter more
            vec[dim1] += weight;
            vec[dim2] += weight * 0.7;
            vec[dim3] -= weight * 0.3;

            // Character n-gram features for subword similarity
            let bytes = word_lower.as_bytes();
            for chunk in bytes.windows(3) {
                let ch = simple_hash(chunk);
                let d = (ch as usize) % DIMS;
                vec[d] += 0.3 * weight;
            }
        }
    }

    // L2-normalize
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut vec {
            *v /= norm;
        }
    }
    vec
}

fn simple_hash(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

fn vec_to_sql(v: &[f32]) -> String {
    let inner: Vec<String> = v.iter().map(|x| format!("{:.6}", x)).collect();
    format!("[{}]", inner.join(","))
}

fn category_count() -> usize {
    PRODUCTS
        .iter()
        .map(|p| p.category)
        .collect::<BTreeSet<_>>()
        .len()
}

fn avg_query_time_ms(db: &Database, sql: &str, runs: usize) -> f64 {
    let runs = runs.max(1);
    let mut total_ms = 0.0;
    for _ in 0..runs {
        let start = Instant::now();
        for row in db.query(sql, ()).unwrap() {
            row.unwrap();
        }
        total_ms += start.elapsed().as_secs_f64() * 1000.0;
    }
    total_ms / runs as f64
}

// ─────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────

fn main() {
    println!();
    println!("============================================================================");
    println!("  PRODUCT RECOMMENDATION ENGINE — Stoolap Vector Search Demo");
    println!("============================================================================");
    println!();
    println!(
        "  {} products across {} categories, {}-dimensional embeddings",
        PRODUCTS.len(),
        category_count(),
        DIMS
    );
    println!("  Embeddings are deterministic and generated locally for demo repeatability.");
    println!("  Replace text_to_embedding() with a real model/API in production.");
    println!();

    // ── 1. Create database and schema ──
    let db = Database::open_in_memory().expect("failed to open database");

    db.execute(
        &format!(
            "CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price FLOAT NOT NULL,
                description TEXT,
                embedding VECTOR({DIMS})
            )"
        ),
        (),
    )
    .unwrap();

    // ── 2. Insert products with embeddings ──
    println!("  Loading product catalog...");
    let start = Instant::now();

    db.execute("BEGIN", ()).unwrap();
    for (i, p) in PRODUCTS.iter().enumerate() {
        let emb = text_to_embedding(&[p.name, p.description, p.category]);
        let emb_sql = vec_to_sql(&emb);
        db.execute(
            &format!(
                "INSERT INTO products (id, name, category, price, description, embedding) \
                 VALUES ({}, '{}', '{}', {}, '{}', '{}')",
                i + 1,
                p.name.replace('\'', "''"),
                p.category,
                p.price,
                p.description.replace('\'', "''"),
                emb_sql
            ),
            (),
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();
    println!(
        "  Inserted {} products in {:.0}ms",
        PRODUCTS.len(),
        start.elapsed().as_secs_f64() * 1000.0
    );

    // ── 3. Benchmark query latency before building the index ──
    let benchmark_emb = text_to_embedding(&["wireless bluetooth audio headphones"]);
    let benchmark_sql_vec = vec_to_sql(&benchmark_emb);
    let benchmark_sql = format!(
        "SELECT id, VEC_DISTANCE_COSINE(embedding, '{benchmark_sql_vec}') AS dist \
         FROM products \
         ORDER BY dist \
         LIMIT 5"
    );
    let benchmark_runs = 50usize;
    let baseline_ms = avg_query_time_ms(&db, &benchmark_sql, benchmark_runs);
    println!(
        "  Baseline top-k query (no index): {:.4}ms avg over {} runs",
        baseline_ms, benchmark_runs
    );

    // ── 4. Create HNSW index ──
    println!("  Building HNSW index (cosine metric)...");
    let start = Instant::now();
    db.execute(
        "CREATE INDEX idx_product_embedding ON products(embedding) \
         USING HNSW WITH (m = 16, ef_construction = 200, ef_search = 64, metric = 'cosine')",
        (),
    )
    .unwrap();
    println!(
        "  Index built in {:.0}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    let indexed_ms = avg_query_time_ms(&db, &benchmark_sql, benchmark_runs);
    println!(
        "  Same top-k query with HNSW: {:.4}ms avg over {} runs",
        indexed_ms, benchmark_runs
    );
    if baseline_ms > 0.0 && indexed_ms > 0.0 {
        let ratio = baseline_ms / indexed_ms;
        if ratio >= 1.0 {
            println!("  Observed speedup: {:.2}x", ratio);
        } else {
            println!(
                "  Observed slowdown: {:.2}x (tiny dataset effect)",
                1.0 / ratio
            );
        }
    } else {
        println!("  Timings are below clock precision in this run.");
    }
    println!(
        "  Note: {} rows is intentionally small; larger datasets show clearer HNSW gains.",
        PRODUCTS.len()
    );
    println!();

    // ── 5. Scenario: "I liked these headphones, show me similar products" ──
    println!("  ────────────────────────────────────────────────────────────");
    println!("  Scenario 1: \"I liked Wireless Noise-Cancelling Headphones\"");
    println!("               Find 5 most similar products");
    println!("  ────────────────────────────────────────────────────────────");

    let query_emb = text_to_embedding(&[
        "Wireless Noise-Cancelling Headphones",
        "over-ear bluetooth ANC headphones",
        "Electronics",
    ]);
    let query_sql = vec_to_sql(&query_emb);

    let sql = format!(
        "SELECT name, category, price, VEC_DISTANCE_COSINE(embedding, '{query_sql}') AS similarity \
         FROM products \
         WHERE name != 'Wireless Noise-Cancelling Headphones' \
         ORDER BY similarity \
         LIMIT 5"
    );

    let start = Instant::now();
    println!();
    println!(
        "  {:<45} {:>10} {:>10} {:>10}",
        "Product", "Category", "Price", "Distance"
    );
    println!("  {}", "-".repeat(78));
    for row in db.query(&sql, ()).unwrap() {
        let row = row.unwrap();
        let name = row.get::<String>(0).unwrap();
        let cat = row.get::<String>(1).unwrap();
        let price = row.get::<f64>(2).unwrap();
        let dist = row.get::<f64>(3).unwrap();
        println!("  {:<45} {:>10} {:>9.2} {:>10.4}", name, cat, price, dist);
    }
    println!();
    println!(
        "  Query time: {:.2}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );

    // ── 6. Scenario: "Find books similar to Database Internals under $50" ──
    println!();
    println!("  ────────────────────────────────────────────────────────────");
    println!("  Scenario 2: \"Books like Database Internals, under $50\"");
    println!("               Hybrid search: vector similarity + price filter");
    println!("  ────────────────────────────────────────────────────────────");

    let query_emb = text_to_embedding(&[
        "Database Internals",
        "distributed systems storage engines algorithms",
        "Books",
    ]);
    let query_sql = vec_to_sql(&query_emb);

    let sql = format!(
        "SELECT name, category, price, VEC_DISTANCE_COSINE(embedding, '{query_sql}') AS similarity \
         FROM products \
         WHERE category = 'Books' AND price < 50.00 \
           AND name != 'Database Internals' \
         ORDER BY similarity \
         LIMIT 5"
    );

    let start = Instant::now();
    println!();
    println!(
        "  {:<45} {:>10} {:>10} {:>10}",
        "Product", "Category", "Price", "Distance"
    );
    println!("  {}", "-".repeat(78));
    for row in db.query(&sql, ()).unwrap() {
        let row = row.unwrap();
        let name = row.get::<String>(0).unwrap();
        let cat = row.get::<String>(1).unwrap();
        let price = row.get::<f64>(2).unwrap();
        let dist = row.get::<f64>(3).unwrap();
        println!("  {:<45} {:>10} {:>9.2} {:>10.4}", name, cat, price, dist);
    }
    println!();
    println!(
        "  Query time: {:.2}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );

    // ── 7. Scenario: Natural language query — "warm winter clothing" ──
    println!();
    println!("  ────────────────────────────────────────────────────────────");
    println!("  Scenario 3: Search \"warm winter clothing\"");
    println!("               Natural language query across all products");
    println!("  ────────────────────────────────────────────────────────────");

    let query_emb = text_to_embedding(&["warm winter clothing jacket insulated"]);
    let query_sql = vec_to_sql(&query_emb);

    let sql = format!(
        "SELECT name, category, price, VEC_DISTANCE_COSINE(embedding, '{query_sql}') AS similarity \
         FROM products \
         ORDER BY similarity \
         LIMIT 5"
    );

    let start = Instant::now();
    println!();
    println!(
        "  {:<45} {:>10} {:>10} {:>10}",
        "Product", "Category", "Price", "Distance"
    );
    println!("  {}", "-".repeat(78));
    for row in db.query(&sql, ()).unwrap() {
        let row = row.unwrap();
        let name = row.get::<String>(0).unwrap();
        let cat = row.get::<String>(1).unwrap();
        let price = row.get::<f64>(2).unwrap();
        let dist = row.get::<f64>(3).unwrap();
        println!("  {:<45} {:>10} {:>9.2} {:>10.4}", name, cat, price, dist);
    }
    println!();
    println!(
        "  Query time: {:.2}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );

    // ── 8. Scenario: "Budget fitness gear under $50" ──
    println!();
    println!("  ────────────────────────────────────────────────────────────");
    println!("  Scenario 4: Search \"fitness exercise gear\" under $50");
    println!("               Hybrid: semantic + price range filter");
    println!("  ────────────────────────────────────────────────────────────");

    let query_emb = text_to_embedding(&["fitness exercise workout training gym"]);
    let query_sql = vec_to_sql(&query_emb);

    let sql = format!(
        "SELECT name, category, price, VEC_DISTANCE_COSINE(embedding, '{query_sql}') AS similarity \
         FROM products \
         WHERE price <= 50.00 \
         ORDER BY similarity \
         LIMIT 5"
    );

    let start = Instant::now();
    println!();
    println!(
        "  {:<45} {:>10} {:>10} {:>10}",
        "Product", "Category", "Price", "Distance"
    );
    println!("  {}", "-".repeat(78));
    for row in db.query(&sql, ()).unwrap() {
        let row = row.unwrap();
        let name = row.get::<String>(0).unwrap();
        let cat = row.get::<String>(1).unwrap();
        let price = row.get::<f64>(2).unwrap();
        let dist = row.get::<f64>(3).unwrap();
        println!("  {:<45} {:>10} {:>9.2} {:>10.4}", name, cat, price, dist);
    }
    println!();
    println!(
        "  Query time: {:.2}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );

    // ── 9. Scenario: Analytics — top product per category by price rank ──
    println!();
    println!("  ────────────────────────────────────────────────────────────");
    println!("  Scenario 5: Analytics — most similar to \"coffee\" per category");
    println!("               Uses window function RANK() OVER (PARTITION BY)");
    println!("  ────────────────────────────────────────────────────────────");

    let query_emb = text_to_embedding(&["coffee espresso brewing maker"]);
    let query_sql = vec_to_sql(&query_emb);

    let sql = format!(
        "SELECT name, category, price, similarity, rnk FROM ( \
            SELECT name, category, price, \
                   VEC_DISTANCE_COSINE(embedding, '{query_sql}') AS similarity, \
                   RANK() OVER (PARTITION BY category ORDER BY VEC_DISTANCE_COSINE(embedding, '{query_sql}')) AS rnk \
            FROM products \
         ) sub \
         WHERE rnk = 1 \
         ORDER BY similarity"
    );

    let start = Instant::now();
    println!();
    println!(
        "  {:<45} {:>10} {:>10} {:>10}",
        "Best Match", "Category", "Price", "Distance"
    );
    println!("  {}", "-".repeat(78));
    for row in db.query(&sql, ()).unwrap() {
        let row = row.unwrap();
        let name = row.get::<String>(0).unwrap();
        let cat = row.get::<String>(1).unwrap();
        let price = row.get::<f64>(2).unwrap();
        let dist = row.get::<f64>(3).unwrap();
        println!("  {:<45} {:>10} {:>9.2} {:>10.4}", name, cat, price, dist);
    }
    println!();
    println!(
        "  Query time: {:.2}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );

    // ── 10. Scenario: `<=>` shorthand operator (L2 distance) ──
    println!();
    println!("  ────────────────────────────────────────────────────────────");
    println!("  Scenario 6: L2 similarity with `<=>` operator");
    println!("               `<=>` is shorthand for VEC_DISTANCE_L2");
    println!("  ────────────────────────────────────────────────────────────");
    println!();

    let query_emb = text_to_embedding(&["portable bluetooth speaker travel audio"]);
    let query_sql = vec_to_sql(&query_emb);

    let sql = format!(
        "SELECT name, category, price, embedding <=> '{query_sql}' AS l2_distance \
         FROM products \
         ORDER BY l2_distance \
         LIMIT 5"
    );

    let start = Instant::now();
    println!(
        "  {:<45} {:>10} {:>10} {:>10}",
        "Product", "Category", "Price", "L2 Dist"
    );
    println!("  {}", "-".repeat(78));
    for row in db.query(&sql, ()).unwrap() {
        let row = row.unwrap();
        let name = row.get::<String>(0).unwrap();
        let cat = row.get::<String>(1).unwrap();
        let price = row.get::<f64>(2).unwrap();
        let dist = row.get::<f64>(3).unwrap();
        println!("  {:<45} {:>10} {:>9.2} {:>10.4}", name, cat, price, dist);
    }
    println!();
    println!(
        "  Query time: {:.2}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );

    // ── 11. Scenario: Category summary with aggregation ──
    println!();
    println!("  ────────────────────────────────────────────────────────────");
    println!("  Scenario 7: Catalog summary — aggregation query");
    println!("  ────────────────────────────────────────────────────────────");
    println!();

    let sql = "SELECT category, COUNT(*) AS count, \
               ROUND(AVG(price), 2) AS avg_price, \
               ROUND(MIN(price), 2) AS min_price, \
               ROUND(MAX(price), 2) AS max_price \
               FROM products \
               GROUP BY category \
               ORDER BY count DESC";

    println!(
        "  {:<12} {:>6} {:>12} {:>12} {:>12}",
        "Category", "Count", "Avg Price", "Min Price", "Max Price"
    );
    println!("  {}", "-".repeat(58));
    for row in db.query(sql, ()).unwrap() {
        let row = row.unwrap();
        let cat = row.get::<String>(0).unwrap();
        let count = row.get::<i64>(1).unwrap();
        let avg = row.get::<f64>(2).unwrap();
        let min = row.get::<f64>(3).unwrap();
        let max = row.get::<f64>(4).unwrap();
        println!(
            "  {:<12} {:>6} {:>12.2} {:>12.2} {:>12.2}",
            cat, count, avg, min, max
        );
    }

    println!();
    println!("============================================================================");
    println!("  All scenarios completed successfully.");
    println!("============================================================================");
    println!();
}
