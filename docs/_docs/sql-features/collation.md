---
layout: doc
title: Collation
category: SQL Features
order: 21
---

# Collation

Stoolap provides the `COLLATE()` function for controlling string comparison and sorting behavior. Four collation modes are supported.

## Collation Modes

| Mode | Aliases | Behavior |
|------|---------|----------|
| `BINARY` | | Exact byte-level comparison (case-sensitive) |
| `NOCASE` | `CASE_INSENSITIVE` | Case-insensitive comparison |
| `NOACCENT` | `ACCENT_INSENSITIVE` | Accent-insensitive comparison (strips diacritical marks) |
| `NUMERIC` | | Numeric-aware string comparison |

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

## NUMERIC (Numeric-Aware)

Compares strings by extracting and comparing their numeric content. This is useful for sorting version numbers, file names with numbers, or any strings where numeric ordering matters:

```sql
-- Without NUMERIC collation: 'file10' sorts before 'file2' (lexicographic)
-- With NUMERIC collation: 'file2' sorts before 'file10' (numeric-aware)
SELECT * FROM files ORDER BY COLLATE(name, 'NUMERIC');

-- Comparing version strings
SELECT * FROM packages
WHERE COLLATE(version, 'NUMERIC') > COLLATE('1.9', 'NUMERIC');
-- Matches '1.10', '2.0', etc. (would miss '1.10' with lexicographic comparison)
```

## Collation in ORDER BY

Use collation to control sort order:

```sql
-- Case-insensitive sort: 'Apple', 'apple', 'APPLE' sort together
SELECT * FROM items ORDER BY COLLATE(name, 'NOCASE');

-- Numeric-aware sort for natural ordering
SELECT * FROM items ORDER BY COLLATE(name, 'NUMERIC');
```
