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

// Runs SQL through the browser build (wasm-pack --target nodejs) and fails on
// a wasm trap, which is how a Rust panic surfaces there.
// Usage: node smoke.cjs <wasm-pack output dir>

const path = require("path");
const { StoolapDB } = require(path.resolve(process.argv[2], "stoolap.js"));

const db = new StoolapDB();
let failed = 0;

function run(sql, expect) {
  let out;
  try {
    out = JSON.parse(db.execute(sql));
  } catch (e) {
    console.error(`TRAP  ${sql}\n      ${e}`);
    failed++;
    return;
  }
  const ok =
    expect === "any" || (expect === "error" ? out.type === "error" : out.type !== "error");
  console.log(`${ok ? "ok  " : "FAIL"}  ${sql}  ->  ${JSON.stringify(out)}`);
  if (!ok) failed++;
}

run("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, s TEXT)");
run("INSERT INTO t VALUES (1, 10, 'a'), (2, 20, 'b'), (3, 30, 'c')");
run("ALTER TABLE t MODIFY COLUMN v FLOAT");
run("SELECT id, v FROM t ORDER BY id");
run("BEGIN");
run("INSERT INTO t VALUES (4, 40, 'd')");
run("ALTER TABLE t MODIFY COLUMN s TEXT NOT NULL", "any");
run("ROLLBACK");
run("BEGIN;");
run("INSERT INTO t VALUES (5, 50, 'e');");
run("COMMIT;");
run("BEGIN;");
run("INSERT INTO t VALUES (6, 60, 'f')");
run("ROLLBACK;");
run("SELECT NOW(), CURRENT_TIMESTAMP, CURRENT_DATE, CURRENT_TIME, RANDOM()");
run("SELECT SLEEP(1)", "error");
run("UPDATE t SET v = v + 1 WHERE id = 2");
run("DELETE FROM t WHERE id = 3");
run("VACUUM");
run("ANALYZE t");
run("EXPLAIN ANALYZE SELECT * FROM t WHERE v > 0");
run("PRAGMA CHECKPOINT", "any");
const ids = JSON.parse(db.execute("SELECT id FROM t ORDER BY id")).rows.map((r) => r[0]);
if (JSON.stringify(ids) !== "[1,2,5]") {
  console.error(`FAIL  ids ${JSON.stringify(ids)}, expected [1,2,5]`);
  failed++;
}

if (failed) {
  console.error(`${failed} statement(s) failed`);
  process.exit(1);
}
