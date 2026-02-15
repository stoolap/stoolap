---
layout: doc
title: Collation
category: SQL Features
order: 21
---

# Collation

Stoolap provides the `COLLATE()` function for controlling string comparison and sorting behavior. Three collation modes are supported.

## Collation Modes

| Mode | Behavior |
|------|----------|
| `BINARY` | Exact byte-level comparison (case-sensitive) |
| `NOCASE` | Case-insensitive comparison |
| `NOACCENT` | Accent-insensitive comparison (strips diacritical marks) |

## Syntax

```sql
COLLATE(value, 'mode')
```

## BINARY (Case-Sensitive)

Performs exact, case-sensitive comparison:

```sql
SELECT * FROM items WHERE COLLATE(name, 'BINARY') = 'Apple';
-- Matches 'Apple' only, NOT 'apple' or 'APPLE'
```

## NOCASE (Case-Insensitive)

Ignores letter casing during comparison:

```sql
SELECT * FROM items
WHERE COLLATE(name, 'NOCASE') = COLLATE('apple', 'NOCASE');
-- Matches 'Apple', 'apple', 'APPLE'
```

## NOACCENT (Accent-Insensitive)

Strips diacritical marks (accents) before comparison:

```sql
SELECT * FROM items
WHERE COLLATE(LOWER(name), 'NOACCENT') = COLLATE('cafe', 'NOACCENT');
-- Matches 'Cafe', 'cafe', 'CAFE'
```

Works with various accented characters:

```sql
-- Matches 'Nacao', 'nacao', 'NACAO'
SELECT * FROM items
WHERE COLLATE(LOWER(name), 'NOACCENT') = COLLATE('nacao', 'NOACCENT');
```

## Combining Collations

Collation functions can be nested for combined effects:

```sql
-- Case-insensitive AND accent-insensitive
SELECT * FROM items
WHERE COLLATE(COLLATE(name, 'NOCASE'), 'NOACCENT')
    = COLLATE(COLLATE('nacao', 'NOCASE'), 'NOACCENT');
```

## Collation in ORDER BY

Use collation to control sort order:

```sql
-- Case-insensitive sort: 'Apple', 'apple', 'APPLE' sort together
SELECT * FROM items ORDER BY COLLATE(name, 'NOCASE');
```
