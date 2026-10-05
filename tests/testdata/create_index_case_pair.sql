-- Two indexes whose names differ only in case, on different columns.
-- Created with stoolap 0.4.2 at e0e2aadc, which accepted the second name.
-- index_case_pair_checkpoint: closed with a checkpoint.
-- index_case_pair_wal: closed with checkpoint_on_close=off and
-- checkpoint_interval=0, so the log holds every statement.
CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER);
INSERT INTO t VALUES (1, 10, 100), (2, 20, 200), (3, 10, 300);
CREATE INDEX IdxK ON t(a);
CREATE INDEX idxk ON t(b);
