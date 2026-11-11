---
layout: doc
title: Names and Case
category: SQL Features
order: 23
---

# Names and Case

Names of tables, columns, views, indexes, common table expressions, aliases, windows and savepoints ignore case, quoted or not. Two spellings name the same object when their lowercase forms are equal, with the Unicode lowercase mapping: `"ÄÖ"` names the table created as `"äö"`, and `"ΣΟΦΌΣ"` names `"σοφός"`, its final sigma included.

```sql
CREATE TABLE "äö" (id INTEGER PRIMARY KEY, "Äb" INTEGER);
INSERT INTO "ÄÖ" VALUES (1, 10);
ALTER TABLE "ÄÖ" ADD COLUMN v INTEGER;
SELECT "äB" FROM "äö";          -- 10
DROP TABLE "ÄÖ";
```

A table is stored under its lowercase name, so `SHOW TABLES` lists `äö`. A column keeps the spelling it was created with, which is the name a query result shows for `SELECT *`.

The rule is lowercasing, not a looser match. `İ` lowercases to `i` followed by a combining dot, so `"İX"` names `"i̇x"` and not `"ix"`. `ß` has no uppercase of its own, so `"STRASSE"` and `"straße"` are two different names.
