-- Comprehensive v0.3.7 test database for migration testing
-- Exercises all data types, index types, and features

-- ============================================================
-- Table 1: users - basic types with multiple index types
-- ============================================================
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    age INTEGER,
    balance FLOAT,
    active BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL,
    metadata JSON
);

-- BTree index (auto for INTEGER)
CREATE INDEX idx_users_age ON users (age);
-- Hash index (auto for TEXT)
CREATE INDEX idx_users_email ON users (email);
-- Bitmap index (auto for BOOLEAN)
CREATE INDEX idx_users_active ON users (active);
-- BTree index on TIMESTAMP
CREATE INDEX idx_users_created ON users (created_at);

BEGIN;
INSERT INTO users VALUES (1, 'Alice', 'alice@example.com', 30, 1000.50, true, '2024-01-15T10:30:00Z', '{"role": "admin", "tags": ["vip", "early"]}');
INSERT INTO users VALUES (2, 'Bob', 'bob@example.com', 25, 2500.75, true, '2024-02-20T14:00:00Z', '{"role": "user", "tags": ["new"]}');
INSERT INTO users VALUES (3, 'Charlie', 'charlie@example.com', 35, 500.00, false, '2024-03-10T09:15:00Z', '{"role": "user", "tags": []}');
INSERT INTO users VALUES (4, 'Diana', 'diana@example.com', 28, 3200.25, true, '2024-04-05T16:45:00Z', '{"role": "moderator", "tags": ["trusted"]}');
INSERT INTO users VALUES (5, 'Eve', 'eve@example.com', 40, 150.00, false, '2024-05-01T08:30:00Z', '{"role": "user", "tags": ["flagged"]}');
INSERT INTO users VALUES (6, 'Frank', 'frank@example.com', 22, 4100.00, true, '2024-06-12T13:20:00Z', '{"role": "user", "tags": []}');
INSERT INTO users VALUES (7, 'Grace', 'grace@example.com', 33, 780.50, true, '2024-07-18T17:00:00Z', '{"role": "admin", "tags": ["vip"]}');
INSERT INTO users VALUES (8, 'Hank', 'hank@example.com', 45, 6000.00, false, '2024-08-25T10:00:00Z', '{"role": "user", "tags": ["premium"]}');
INSERT INTO users VALUES (9, 'Ivy', 'ivy@example.com', 27, 920.30, true, '2024-09-30T15:30:00Z', '{"role": "user", "tags": []}');
INSERT INTO users VALUES (10, 'Jack', 'jack@example.com', 31, 1500.00, true, '2024-10-08T09:00:00Z', '{"role": "moderator", "tags": ["trusted", "senior"]}');
INSERT INTO users VALUES (11, 'Karen', 'karen@example.com', 29, 2200.00, true, '2024-11-15T11:30:00Z', '{"role": "user", "tags": ["new"]}');
INSERT INTO users VALUES (12, 'Leo', 'leo@example.com', 38, 3800.00, true, '2024-12-01T14:15:00Z', '{"role": "admin", "tags": ["vip", "founder"]}');
INSERT INTO users VALUES (13, 'Mia', 'mia@example.com', 24, 450.00, false, '2025-01-10T08:00:00Z', '{"role": "user", "tags": []}');
INSERT INTO users VALUES (14, 'Noah', 'noah@example.com', 42, 5500.00, true, '2025-01-20T16:00:00Z', '{"role": "user", "tags": ["premium"]}');
INSERT INTO users VALUES (15, 'Olivia', 'olivia@example.com', 26, 1800.00, true, '2025-02-05T12:30:00Z', '{"role": "moderator", "tags": ["trusted"]}');
COMMIT;

-- Updates and deletes to test WAL state
UPDATE users SET balance = 1100.00, metadata = '{"role": "admin", "tags": ["vip", "early", "updated"]}' WHERE id = 1;
UPDATE users SET active = false WHERE id = 6;
DELETE FROM users WHERE id = 5;
DELETE FROM users WHERE id = 13;

-- ============================================================
-- Table 2: orders - timestamps, floats, joins
-- ============================================================
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    product TEXT NOT NULL,
    amount FLOAT NOT NULL,
    quantity INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    notes JSON
);

CREATE INDEX idx_orders_user ON orders (user_id);
CREATE INDEX idx_orders_status ON orders (status);
CREATE INDEX idx_orders_created ON orders (created_at);
CREATE INDEX idx_orders_amount ON orders (amount);

BEGIN;
INSERT INTO orders VALUES (1, 1, 'Widget', 29.99, 2, 'completed', '2025-01-15T10:30:00Z', '{"priority": "normal"}');
INSERT INTO orders VALUES (2, 1, 'Gadget', 49.99, 1, 'completed', '2025-01-16T14:00:00Z', '{"priority": "high"}');
INSERT INTO orders VALUES (3, 2, 'Widget', 29.99, 5, 'shipped', '2025-01-17T09:15:00Z', '{"tracking": "TR001"}');
INSERT INTO orders VALUES (4, 3, 'Doohickey', 99.99, 1, 'pending', '2025-01-18T16:45:00Z', null);
INSERT INTO orders VALUES (5, 4, 'Widget', 29.99, 3, 'completed', '2025-01-19T11:00:00Z', '{"priority": "low"}');
INSERT INTO orders VALUES (6, 2, 'Gadget', 49.99, 2, 'completed', '2025-01-20T08:30:00Z', null);
INSERT INTO orders VALUES (7, 2, 'Thingamajig', 15.50, 10, 'shipped', '2025-01-21T13:20:00Z', '{"tracking": "TR002", "fragile": true}');
INSERT INTO orders VALUES (8, 6, 'Doohickey', 99.99, 1, 'cancelled', '2025-01-22T17:00:00Z', '{"reason": "out_of_stock"}');
INSERT INTO orders VALUES (9, 7, 'Widget', 29.99, 4, 'completed', '2025-01-23T10:00:00Z', null);
INSERT INTO orders VALUES (10, 8, 'Gadget', 49.99, 1, 'pending', '2025-01-24T15:30:00Z', null);
INSERT INTO orders VALUES (11, 1, 'Thingamajig', 15.50, 3, 'completed', '2025-02-01T09:00:00Z', null);
INSERT INTO orders VALUES (12, 4, 'Doohickey', 99.99, 2, 'shipped', '2025-02-02T14:30:00Z', '{"tracking": "TR003"}');
INSERT INTO orders VALUES (13, 9, 'Widget', 29.99, 1, 'completed', '2025-02-03T11:15:00Z', null);
INSERT INTO orders VALUES (14, 10, 'Gadget', 49.99, 3, 'completed', '2025-02-04T16:00:00Z', '{"priority": "normal"}');
INSERT INTO orders VALUES (15, 3, 'Widget', 29.99, 2, 'pending', '2025-02-05T08:45:00Z', null);
INSERT INTO orders VALUES (16, 11, 'Gadget', 49.99, 1, 'completed', '2025-02-06T10:00:00Z', null);
INSERT INTO orders VALUES (17, 12, 'Doohickey', 99.99, 3, 'shipped', '2025-02-07T13:00:00Z', '{"tracking": "TR004"}');
INSERT INTO orders VALUES (18, 14, 'Widget', 29.99, 6, 'completed', '2025-02-08T09:30:00Z', null);
INSERT INTO orders VALUES (19, 15, 'Thingamajig', 15.50, 2, 'completed', '2025-02-09T11:00:00Z', null);
INSERT INTO orders VALUES (20, 7, 'Doohickey', 99.99, 1, 'pending', '2025-02-10T14:00:00Z', null);
COMMIT;

-- ============================================================
-- Table 3: products - nullable columns, unique constraints
-- ============================================================
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    price FLOAT NOT NULL,
    weight FLOAT,
    in_stock BOOLEAN NOT NULL,
    category TEXT NOT NULL,
    tags JSON,
    listed_at TIMESTAMP
);

CREATE UNIQUE INDEX idx_products_sku ON products (sku);
CREATE INDEX idx_products_category ON products (category);
CREATE INDEX idx_products_price ON products (price);

BEGIN;
INSERT INTO products VALUES (1, 'WDG-001', 'Widget', 'A standard widget for everyday use', 29.99, 0.5, true, 'hardware', '["popular", "essential"]', '2024-06-01T00:00:00Z');
INSERT INTO products VALUES (2, 'GDG-001', 'Gadget', 'Premium electronic gadget', 49.99, 0.3, true, 'electronics', '["premium", "new"]', '2024-07-15T00:00:00Z');
INSERT INTO products VALUES (3, 'DHK-001', 'Doohickey', 'Industrial-grade doohickey', 99.99, 2.1, true, 'industrial', '["heavy-duty"]', '2024-08-20T00:00:00Z');
INSERT INTO products VALUES (4, 'THG-001', 'Thingamajig', 'Versatile thingamajig', 15.50, 0.1, true, 'accessories', '["budget", "popular"]', '2024-09-01T00:00:00Z');
INSERT INTO products VALUES (5, 'WDG-002', 'Widget Pro', 'Enhanced widget with extra features', 59.99, 0.7, false, 'hardware', '["premium"]', '2024-10-10T00:00:00Z');
INSERT INTO products VALUES (6, 'GDG-002', 'Gadget Mini', null, 24.99, 0.15, true, 'electronics', null, null);
INSERT INTO products VALUES (7, 'SPR-001', 'Sprocket', 'Precision-engineered sprocket', 12.75, 0.8, true, 'hardware', '["industrial"]', '2024-11-05T00:00:00Z');
INSERT INTO products VALUES (8, 'BLT-001', 'Bolt Pack', 'Pack of 100 standard bolts', 8.99, 1.5, true, 'hardware', '["bulk", "essential"]', '2024-12-01T00:00:00Z');
INSERT INTO products VALUES (9, 'CBL-001', 'Cable Set', 'Assorted cable set', 34.99, 0.6, true, 'electronics', '["essential"]', '2025-01-01T00:00:00Z');
INSERT INTO products VALUES (10, 'DHK-002', 'Doohickey Lite', 'Lightweight doohickey for home use', 49.99, 0.9, true, 'industrial', '["new", "popular"]', '2025-01-15T00:00:00Z');
COMMIT;

-- ============================================================
-- Table 4: events - large-ish dataset with timestamps
-- ============================================================
CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL,
    source TEXT NOT NULL,
    severity INTEGER NOT NULL,
    message TEXT NOT NULL,
    payload JSON,
    occurred_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_events_type ON events (event_type);
CREATE INDEX idx_events_severity ON events (severity);
CREATE INDEX idx_events_occurred ON events (occurred_at);

-- Insert 200 events in a transaction
BEGIN;
INSERT INTO events VALUES (1, 'login', 'web', 1, 'User login successful', '{"user_id": 1, "ip": "192.168.1.1"}', '2025-01-01T00:00:01Z');
INSERT INTO events VALUES (2, 'login', 'mobile', 1, 'User login successful', '{"user_id": 2, "ip": "10.0.0.1"}', '2025-01-01T00:01:00Z');
INSERT INTO events VALUES (3, 'error', 'api', 3, 'Internal server error', '{"endpoint": "/api/users", "status": 500}', '2025-01-01T00:02:00Z');
INSERT INTO events VALUES (4, 'purchase', 'web', 1, 'Order placed', '{"order_id": 1, "total": 59.98}', '2025-01-01T00:03:00Z');
INSERT INTO events VALUES (5, 'logout', 'web', 1, 'User logout', '{"user_id": 1}', '2025-01-01T00:04:00Z');
INSERT INTO events VALUES (6, 'error', 'worker', 4, 'Job failed: timeout', '{"job_id": "j-001", "retry": 3}', '2025-01-01T00:05:00Z');
INSERT INTO events VALUES (7, 'login', 'web', 1, 'User login successful', '{"user_id": 3, "ip": "172.16.0.1"}', '2025-01-01T00:06:00Z');
INSERT INTO events VALUES (8, 'config', 'admin', 2, 'Settings updated', '{"setting": "max_connections", "old": 100, "new": 200}', '2025-01-01T00:07:00Z');
INSERT INTO events VALUES (9, 'purchase', 'mobile', 1, 'Order placed', '{"order_id": 2, "total": 49.99}', '2025-01-01T00:08:00Z');
INSERT INTO events VALUES (10, 'error', 'api', 3, 'Rate limit exceeded', '{"ip": "203.0.113.5", "limit": 1000}', '2025-01-01T00:09:00Z');
INSERT INTO events VALUES (11, 'login', 'api', 1, 'API key auth', '{"key_id": "k-001"}', '2025-01-01T00:10:00Z');
INSERT INTO events VALUES (12, 'deploy', 'ci', 2, 'Deployment started', '{"version": "1.2.3", "env": "staging"}', '2025-01-01T00:11:00Z');
INSERT INTO events VALUES (13, 'deploy', 'ci', 2, 'Deployment completed', '{"version": "1.2.3", "env": "staging", "duration_s": 45}', '2025-01-01T00:12:00Z');
INSERT INTO events VALUES (14, 'error', 'web', 3, 'Page not found', '{"path": "/old-page", "status": 404}', '2025-01-01T00:13:00Z');
INSERT INTO events VALUES (15, 'purchase', 'web', 1, 'Order placed', '{"order_id": 3, "total": 149.95}', '2025-01-01T00:14:00Z');
INSERT INTO events VALUES (16, 'login', 'web', 1, 'User login', '{"user_id": 4}', '2025-01-01T01:00:00Z');
INSERT INTO events VALUES (17, 'error', 'api', 5, 'Database connection lost', '{"db": "primary", "retry_in": 5}', '2025-01-01T01:01:00Z');
INSERT INTO events VALUES (18, 'login', 'mobile', 1, 'User login', '{"user_id": 5}', '2025-01-01T01:02:00Z');
INSERT INTO events VALUES (19, 'config', 'admin', 2, 'Feature flag toggled', '{"flag": "dark_mode", "enabled": true}', '2025-01-01T01:03:00Z');
INSERT INTO events VALUES (20, 'purchase', 'api', 1, 'Order placed', '{"order_id": 4, "total": 99.99}', '2025-01-01T01:04:00Z');
INSERT INTO events VALUES (21, 'logout', 'web', 1, 'User logout', '{"user_id": 4}', '2025-01-01T01:05:00Z');
INSERT INTO events VALUES (22, 'error', 'worker', 4, 'Memory limit exceeded', '{"worker_id": "w-003", "mem_mb": 512}', '2025-01-01T01:06:00Z');
INSERT INTO events VALUES (23, 'login', 'web', 1, 'User login', '{"user_id": 6}', '2025-01-01T01:07:00Z');
INSERT INTO events VALUES (24, 'deploy', 'ci', 2, 'Deployment started', '{"version": "1.2.4", "env": "production"}', '2025-01-01T01:08:00Z');
INSERT INTO events VALUES (25, 'deploy', 'ci', 2, 'Deployment completed', '{"version": "1.2.4", "env": "production", "duration_s": 120}', '2025-01-01T01:09:00Z');
INSERT INTO events VALUES (26, 'purchase', 'web', 1, 'Order placed', '{"order_id": 5, "total": 89.97}', '2025-01-01T02:00:00Z');
INSERT INTO events VALUES (27, 'error', 'api', 3, 'Authentication failed', '{"user_id": 99, "reason": "invalid_token"}', '2025-01-01T02:01:00Z');
INSERT INTO events VALUES (28, 'login', 'web', 1, 'User login', '{"user_id": 7}', '2025-01-01T02:02:00Z');
INSERT INTO events VALUES (29, 'config', 'admin', 2, 'Cache cleared', '{"cache": "query_cache", "entries": 1500}', '2025-01-01T02:03:00Z');
INSERT INTO events VALUES (30, 'error', 'web', 3, 'Request timeout', '{"path": "/api/search", "timeout_ms": 30000}', '2025-01-01T02:04:00Z');
INSERT INTO events VALUES (31, 'purchase', 'mobile', 1, 'Order placed', '{"order_id": 6, "total": 99.98}', '2025-01-01T02:05:00Z');
INSERT INTO events VALUES (32, 'login', 'api', 1, 'Service auth', '{"service": "payment-gateway"}', '2025-01-01T02:06:00Z');
INSERT INTO events VALUES (33, 'error', 'worker', 3, 'Job retrying', '{"job_id": "j-005", "attempt": 2}', '2025-01-01T02:07:00Z');
INSERT INTO events VALUES (34, 'logout', 'mobile', 1, 'User logout', '{"user_id": 5}', '2025-01-01T02:08:00Z');
INSERT INTO events VALUES (35, 'deploy', 'ci', 2, 'Rollback initiated', '{"version": "1.2.4", "env": "production", "reason": "error_spike"}', '2025-01-01T02:09:00Z');
INSERT INTO events VALUES (36, 'login', 'web', 1, 'User login', '{"user_id": 8}', '2025-01-01T03:00:00Z');
INSERT INTO events VALUES (37, 'purchase', 'web', 1, 'Order placed', '{"order_id": 7, "total": 155.00}', '2025-01-01T03:01:00Z');
INSERT INTO events VALUES (38, 'error', 'api', 4, 'Disk space low', '{"mount": "/data", "free_gb": 2}', '2025-01-01T03:02:00Z');
INSERT INTO events VALUES (39, 'config', 'admin', 2, 'Backup scheduled', '{"time": "03:00", "type": "full"}', '2025-01-01T03:03:00Z');
INSERT INTO events VALUES (40, 'login', 'mobile', 1, 'User login', '{"user_id": 9}', '2025-01-01T03:04:00Z');
INSERT INTO events VALUES (41, 'purchase', 'api', 1, 'Bulk order', '{"order_id": 8, "items": 50, "total": 750.00}', '2025-01-01T03:05:00Z');
INSERT INTO events VALUES (42, 'error', 'web', 3, 'SSL certificate expiring', '{"domain": "api.example.com", "days_left": 7}', '2025-01-01T03:06:00Z');
INSERT INTO events VALUES (43, 'logout', 'web', 1, 'Session expired', '{"user_id": 3}', '2025-01-01T03:07:00Z');
INSERT INTO events VALUES (44, 'login', 'web', 1, 'User login', '{"user_id": 10}', '2025-01-01T03:08:00Z');
INSERT INTO events VALUES (45, 'deploy', 'ci', 2, 'Hotfix deployed', '{"version": "1.2.4.1", "env": "production"}', '2025-01-01T03:09:00Z');
INSERT INTO events VALUES (46, 'error', 'api', 3, 'Invalid request body', '{"endpoint": "/api/orders", "error": "missing field: quantity"}', '2025-01-01T04:00:00Z');
INSERT INTO events VALUES (47, 'purchase', 'web', 1, 'Order placed', '{"order_id": 9, "total": 29.99}', '2025-01-01T04:01:00Z');
INSERT INTO events VALUES (48, 'login', 'web', 1, 'User login', '{"user_id": 11}', '2025-01-01T04:02:00Z');
INSERT INTO events VALUES (49, 'config', 'admin', 2, 'Rate limits updated', '{"endpoint": "/api/*", "rpm": 2000}', '2025-01-01T04:03:00Z');
INSERT INTO events VALUES (50, 'error', 'worker', 5, 'Deadlock detected', '{"table": "orders", "txn_ids": [101, 102]}', '2025-01-01T04:04:00Z');
COMMIT;

-- ============================================================
-- Table 5: documents - TEXT heavy with JSON, nullable fields
-- ============================================================
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    author TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    version INTEGER NOT NULL,
    published BOOLEAN NOT NULL,
    metadata JSON,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP
);

CREATE INDEX idx_docs_author ON documents (author);
CREATE INDEX idx_docs_type ON documents (doc_type);
CREATE INDEX idx_docs_published ON documents (published);

BEGIN;
INSERT INTO documents VALUES (1, 'Getting Started Guide', 'Welcome to the platform. This guide covers the basics of setting up your account and making your first API call.', 'Alice', 'guide', 3, true, '{"category": "onboarding", "reading_time_min": 5}', '2024-06-01T10:00:00Z', '2025-01-10T15:00:00Z');
INSERT INTO documents VALUES (2, 'API Reference v2', 'Complete API reference documentation including all endpoints, request/response schemas, and authentication methods.', 'Bob', 'reference', 2, true, '{"category": "api", "reading_time_min": 30}', '2024-07-15T10:00:00Z', '2025-01-05T12:00:00Z');
INSERT INTO documents VALUES (3, 'Architecture Overview', 'High-level architecture diagram and explanation of the system components, data flow, and integration points.', 'Charlie', 'design', 1, true, '{"category": "architecture", "reading_time_min": 15}', '2024-08-20T10:00:00Z', null);
INSERT INTO documents VALUES (4, 'Migration Guide v3', 'Step-by-step migration guide from version 2.x to 3.x including breaking changes and workarounds.', 'Diana', 'guide', 1, false, '{"category": "migration", "reading_time_min": 20}', '2025-01-20T10:00:00Z', null);
INSERT INTO documents VALUES (5, 'Performance Tuning', 'Best practices for optimizing query performance, indexing strategies, and connection pooling configuration.', 'Alice', 'guide', 5, true, '{"category": "performance", "reading_time_min": 25}', '2024-03-01T10:00:00Z', '2025-02-01T09:00:00Z');
INSERT INTO documents VALUES (6, 'Security Whitepaper', 'Detailed security analysis covering encryption at rest, TLS configuration, RBAC policies, and audit logging.', 'Eve', 'whitepaper', 2, true, '{"category": "security", "reading_time_min": 45}', '2024-09-15T10:00:00Z', '2025-01-25T14:00:00Z');
INSERT INTO documents VALUES (7, 'Release Notes 1.2', 'New features: vector search, JSON path queries, window functions. Bug fixes: 42 issues resolved.', 'Frank', 'release', 1, true, '{"category": "releases", "version": "1.2.0"}', '2025-01-30T10:00:00Z', null);
INSERT INTO documents VALUES (8, 'Troubleshooting FAQ', 'Common issues and their solutions including connection errors, slow queries, and data recovery procedures.', 'Grace', 'guide', 4, true, '{"category": "support", "reading_time_min": 10}', '2024-04-10T10:00:00Z', '2025-02-03T11:00:00Z');
COMMIT;

-- ============================================================
-- Table 6: vectors - vector embeddings with HNSW index
-- ============================================================
CREATE TABLE vectors (
    id INTEGER PRIMARY KEY,
    label TEXT NOT NULL,
    embedding VECTOR(4) NOT NULL,
    category TEXT NOT NULL,
    score FLOAT
);

CREATE INDEX idx_vectors_hnsw ON vectors(embedding) USING HNSW;
CREATE INDEX idx_vectors_category ON vectors (category);

BEGIN;
INSERT INTO vectors VALUES (1, 'cat', '[0.1, 0.2, 0.3, 0.4]', 'animal', 0.95);
INSERT INTO vectors VALUES (2, 'dog', '[0.15, 0.25, 0.28, 0.42]', 'animal', 0.92);
INSERT INTO vectors VALUES (3, 'car', '[0.8, 0.1, 0.05, 0.05]', 'vehicle', 0.88);
INSERT INTO vectors VALUES (4, 'bike', '[0.75, 0.15, 0.05, 0.08]', 'vehicle', 0.85);
INSERT INTO vectors VALUES (5, 'rose', '[0.3, 0.7, 0.1, 0.2]', 'plant', 0.90);
INSERT INTO vectors VALUES (6, 'oak', '[0.25, 0.65, 0.15, 0.25]', 'plant', 0.87);
INSERT INTO vectors VALUES (7, 'fish', '[0.12, 0.18, 0.35, 0.45]', 'animal', 0.91);
INSERT INTO vectors VALUES (8, 'bird', '[0.14, 0.22, 0.32, 0.38]', 'animal', 0.89);
INSERT INTO vectors VALUES (9, 'truck', '[0.82, 0.08, 0.04, 0.06]', 'vehicle', 0.86);
INSERT INTO vectors VALUES (10, 'tulip', '[0.28, 0.72, 0.08, 0.18]', 'plant', 0.93);
INSERT INTO vectors VALUES (11, 'whale', '[0.11, 0.19, 0.38, 0.50]', 'animal', 0.94);
INSERT INTO vectors VALUES (12, 'bus', '[0.78, 0.12, 0.06, 0.07]', 'vehicle', 0.82);
INSERT INTO vectors VALUES (13, 'fern', '[0.22, 0.68, 0.18, 0.22]', 'plant', 0.84);
INSERT INTO vectors VALUES (14, 'hawk', '[0.16, 0.20, 0.30, 0.36]', 'animal', 0.88);
INSERT INTO vectors VALUES (15, 'train', '[0.85, 0.06, 0.03, 0.04]', 'vehicle', 0.91);
COMMIT;

-- ============================================================
-- Table 7: metrics - numeric heavy, aggregation testing
-- ============================================================
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY,
    host TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value FLOAT NOT NULL,
    tags JSON,
    collected_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_metrics_host ON metrics (host);
CREATE INDEX idx_metrics_name ON metrics (metric_name);
CREATE INDEX idx_metrics_collected ON metrics (collected_at);

BEGIN;
INSERT INTO metrics VALUES (1, 'srv-01', 'cpu_usage', 45.2, '{"core": 0}', '2025-01-01T00:00:00Z');
INSERT INTO metrics VALUES (2, 'srv-01', 'cpu_usage', 52.8, '{"core": 1}', '2025-01-01T00:00:00Z');
INSERT INTO metrics VALUES (3, 'srv-01', 'memory_used', 8192.0, '{"unit": "MB"}', '2025-01-01T00:00:00Z');
INSERT INTO metrics VALUES (4, 'srv-01', 'disk_io', 150.5, '{"device": "sda"}', '2025-01-01T00:00:00Z');
INSERT INTO metrics VALUES (5, 'srv-02', 'cpu_usage', 78.1, '{"core": 0}', '2025-01-01T00:00:00Z');
INSERT INTO metrics VALUES (6, 'srv-02', 'cpu_usage', 82.3, '{"core": 1}', '2025-01-01T00:00:00Z');
INSERT INTO metrics VALUES (7, 'srv-02', 'memory_used', 14336.0, '{"unit": "MB"}', '2025-01-01T00:00:00Z');
INSERT INTO metrics VALUES (8, 'srv-02', 'disk_io', 320.7, '{"device": "sda"}', '2025-01-01T00:00:00Z');
INSERT INTO metrics VALUES (9, 'srv-03', 'cpu_usage', 12.5, '{"core": 0}', '2025-01-01T00:00:00Z');
INSERT INTO metrics VALUES (10, 'srv-03', 'memory_used', 4096.0, '{"unit": "MB"}', '2025-01-01T00:00:00Z');
INSERT INTO metrics VALUES (11, 'srv-01', 'cpu_usage', 48.7, '{"core": 0}', '2025-01-01T00:01:00Z');
INSERT INTO metrics VALUES (12, 'srv-01', 'cpu_usage', 55.1, '{"core": 1}', '2025-01-01T00:01:00Z');
INSERT INTO metrics VALUES (13, 'srv-01', 'memory_used', 8300.0, '{"unit": "MB"}', '2025-01-01T00:01:00Z');
INSERT INTO metrics VALUES (14, 'srv-02', 'cpu_usage', 75.9, '{"core": 0}', '2025-01-01T00:01:00Z');
INSERT INTO metrics VALUES (15, 'srv-02', 'cpu_usage', 80.2, '{"core": 1}', '2025-01-01T00:01:00Z');
INSERT INTO metrics VALUES (16, 'srv-02', 'memory_used', 14400.0, '{"unit": "MB"}', '2025-01-01T00:01:00Z');
INSERT INTO metrics VALUES (17, 'srv-03', 'cpu_usage', 15.3, '{"core": 0}', '2025-01-01T00:01:00Z');
INSERT INTO metrics VALUES (18, 'srv-03', 'memory_used', 4100.0, '{"unit": "MB"}', '2025-01-01T00:01:00Z');
INSERT INTO metrics VALUES (19, 'srv-01', 'network_rx', 1024.5, '{"interface": "eth0", "unit": "Mbps"}', '2025-01-01T00:00:00Z');
INSERT INTO metrics VALUES (20, 'srv-01', 'network_tx', 512.3, '{"interface": "eth0", "unit": "Mbps"}', '2025-01-01T00:00:00Z');
INSERT INTO metrics VALUES (21, 'srv-02', 'network_rx', 2048.1, '{"interface": "eth0", "unit": "Mbps"}', '2025-01-01T00:00:00Z');
INSERT INTO metrics VALUES (22, 'srv-02', 'network_tx', 1536.8, '{"interface": "eth0", "unit": "Mbps"}', '2025-01-01T00:00:00Z');
INSERT INTO metrics VALUES (23, 'srv-03', 'disk_io', 45.2, '{"device": "nvme0"}', '2025-01-01T00:00:00Z');
INSERT INTO metrics VALUES (24, 'srv-03', 'network_rx', 256.0, '{"interface": "eth0", "unit": "Mbps"}', '2025-01-01T00:00:00Z');
INSERT INTO metrics VALUES (25, 'srv-03', 'network_tx', 128.5, '{"interface": "eth0", "unit": "Mbps"}', '2025-01-01T00:00:00Z');
COMMIT;

-- ============================================================
-- Table 8: nullable_types - edge cases for all types with NULLs
-- ============================================================
CREATE TABLE nullable_types (
    id INTEGER PRIMARY KEY,
    int_val INTEGER,
    float_val FLOAT,
    text_val TEXT,
    bool_val BOOLEAN,
    ts_val TIMESTAMP,
    json_val JSON
);

BEGIN;
INSERT INTO nullable_types VALUES (1, 42, 3.14, 'hello', true, '2025-01-01T00:00:00Z', '{"key": "value"}');
INSERT INTO nullable_types VALUES (2, null, null, null, null, null, null);
INSERT INTO nullable_types VALUES (3, 0, 0.0, '', false, '1970-01-01T00:00:00Z', '{}');
INSERT INTO nullable_types VALUES (4, -1, -99.99, 'negative', true, '2000-12-31T23:59:59Z', '[]');
INSERT INTO nullable_types VALUES (5, 2147483647, 1.7976931348623157e+308, 'max values', true, '2099-12-31T23:59:59Z', '{"nested": {"deep": true}}');
INSERT INTO nullable_types VALUES (6, null, 0.0, null, false, null, '{"a": 1}');
INSERT INTO nullable_types VALUES (7, 100, null, 'partial', null, '2025-06-15T12:00:00Z', null);
INSERT INTO nullable_types VALUES (8, null, null, null, null, null, null);
INSERT INTO nullable_types VALUES (9, -2147483648, -1.7976931348623157e+308, 'min values', false, '1900-01-01T00:00:00Z', '{"min": true}');
INSERT INTO nullable_types VALUES (10, 1, 1.0, 'one', true, '2025-03-15T08:30:00Z', '{"count": 1}');
COMMIT;

-- ============================================================
-- Force snapshot to persist current state to disk
-- ============================================================
PRAGMA SNAPSHOT;

-- ============================================================
-- Post-snapshot changes (WAL-only, not in snapshot)
-- These test WAL replay after snapshot loading during migration
-- ============================================================

-- New user added after snapshot
INSERT INTO users VALUES (16, 'Quinn', 'quinn@example.com', 34, 2750.00, true, '2025-02-20T10:00:00Z', '{"role": "user", "tags": ["post-snapshot"]}');

-- Update existing user after snapshot
UPDATE users SET balance = 5000.00 WHERE id = 2;

-- Delete a user after snapshot
DELETE FROM users WHERE id = 3;

-- New orders after snapshot
INSERT INTO orders VALUES (21, 16, 'Widget', 29.99, 1, 'pending', '2025-02-20T10:30:00Z', '{"source": "post-snapshot"}');
INSERT INTO orders VALUES (22, 2, 'Gadget', 49.99, 2, 'completed', '2025-02-21T11:00:00Z', null);

-- Update an order after snapshot
UPDATE orders SET status = 'shipped' WHERE id = 4;

-- New product after snapshot
INSERT INTO products VALUES (11, 'NUT-001', 'Nut Pack', 'Assorted nuts and washers', 6.99, 0.4, true, 'hardware', '["bulk"]', '2025-02-15T00:00:00Z');

-- New vector after snapshot
INSERT INTO vectors VALUES (16, 'snake', '[0.13, 0.21, 0.33, 0.41]', 'animal', 0.86);

-- New events after snapshot
INSERT INTO events VALUES (51, 'login', 'web', 1, 'Post-snapshot login', '{"user_id": 16}', '2025-02-20T10:00:00Z');
INSERT INTO events VALUES (52, 'purchase', 'web', 1, 'Post-snapshot order', '{"order_id": 21}', '2025-02-20T10:30:00Z');
INSERT INTO events VALUES (53, 'error', 'api', 4, 'Post-snapshot error', '{"critical": true}', '2025-02-20T11:00:00Z');

-- New metric after snapshot
INSERT INTO metrics VALUES (26, 'srv-04', 'cpu_usage', 33.3, '{"core": 0}', '2025-02-20T00:00:00Z');

-- New nullable row after snapshot
INSERT INTO nullable_types VALUES (11, null, 99.99, 'post-snapshot', true, '2025-02-20T00:00:00Z', '{"after": "snapshot"}');

-- ============================================================
-- Verification queries (expected results for migration test)
-- These reflect the FINAL state (snapshot + WAL changes)
-- ============================================================
-- users: 15 original - 2 deleted (id=5,13) + 1 new (id=16) - 1 deleted (id=3) = 13
SELECT 'users' AS tbl, COUNT(*) AS cnt FROM users;
-- orders: 20 original + 2 new = 22
SELECT 'orders' AS tbl, COUNT(*) AS cnt FROM orders;
-- products: 10 original + 1 new = 11
SELECT 'products' AS tbl, COUNT(*) AS cnt FROM products;
-- events: 50 original + 3 new = 53
SELECT 'events' AS tbl, COUNT(*) AS cnt FROM events;
SELECT 'documents' AS tbl, COUNT(*) AS cnt FROM documents;
-- vectors: 15 original + 1 new = 16
SELECT 'vectors' AS tbl, COUNT(*) AS cnt FROM vectors;
-- metrics: 25 original + 1 new = 26
SELECT 'metrics' AS tbl, COUNT(*) AS cnt FROM metrics;
-- nullable_types: 10 original + 1 new = 11
SELECT 'nullable_types' AS tbl, COUNT(*) AS cnt FROM nullable_types;

-- Aggregation checks (post-WAL state)
-- Bob's balance updated to 5000, Charlie deleted
SELECT SUM(balance) FROM users WHERE active = true;
SELECT AVG(amount) FROM orders;
SELECT MAX(severity) FROM events;
SELECT COUNT(DISTINCT category) FROM products;

-- Verify specific post-snapshot changes
SELECT name, balance FROM users WHERE id = 2;
SELECT name FROM users WHERE id = 3;
SELECT status FROM orders WHERE id = 4;
SELECT name FROM products WHERE sku = 'NUT-001';
SELECT label FROM vectors WHERE id = 16;
