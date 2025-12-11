---
layout: doc
title: Savepoints
category: SQL Features
order: 12
---

# Savepoints

Savepoints allow you to create named points within a transaction that you can later roll back to, without rolling back the entire transaction. This provides fine-grained control over transaction rollback.

## Syntax

```sql
-- Create a savepoint
SAVEPOINT savepoint_name;

-- Roll back to a savepoint (undo changes after the savepoint)
ROLLBACK TO SAVEPOINT savepoint_name;

-- Release a savepoint (remove it, keep changes)
RELEASE SAVEPOINT savepoint_name;
```

## Basic Example

```sql
BEGIN TRANSACTION;

INSERT INTO accounts (id, balance) VALUES (1, 1000);
SAVEPOINT after_insert;

UPDATE accounts SET balance = 500 WHERE id = 1;
-- Oops, wrong update!

ROLLBACK TO SAVEPOINT after_insert;
-- Balance is back to 1000

UPDATE accounts SET balance = 900 WHERE id = 1;
-- Correct update

COMMIT;
-- Final balance: 900
```

## Use Cases

### Error Recovery

Roll back partial work when an error occurs:

```sql
BEGIN TRANSACTION;

-- Step 1: Create user
INSERT INTO users (id, name) VALUES (1, 'Alice');
SAVEPOINT user_created;

-- Step 2: Create profile
INSERT INTO profiles (user_id, bio) VALUES (1, 'Hello');
SAVEPOINT profile_created;

-- Step 3: Create settings (might fail)
INSERT INTO settings (user_id, theme) VALUES (1, 'dark');
-- If this fails, we can roll back to profile_created
-- ROLLBACK TO SAVEPOINT profile_created;

COMMIT;
```

### Batch Processing

Process items in batches with recovery points:

```sql
BEGIN TRANSACTION;

-- Process batch 1
INSERT INTO processed VALUES (1), (2), (3);
SAVEPOINT batch_1;

-- Process batch 2
INSERT INTO processed VALUES (4), (5), (6);
SAVEPOINT batch_2;

-- Process batch 3 - if this fails, roll back to batch_2
INSERT INTO processed VALUES (7), (8), (9);

COMMIT;
```

### Try/Retry Logic

Implement retry logic within a single transaction:

```sql
BEGIN TRANSACTION;

SAVEPOINT attempt_start;

-- First attempt
INSERT INTO orders (product_id, quantity) VALUES (100, 5);
-- Check if inventory is sufficient...
-- If not: ROLLBACK TO SAVEPOINT attempt_start;
-- Then try with smaller quantity

COMMIT;
```

## Multiple Savepoints

You can create multiple savepoints in a transaction:

```sql
BEGIN TRANSACTION;

INSERT INTO log VALUES ('Starting process');
SAVEPOINT sp1;

INSERT INTO data VALUES (1, 'first');
SAVEPOINT sp2;

INSERT INTO data VALUES (2, 'second');
SAVEPOINT sp3;

INSERT INTO data VALUES (3, 'third');

-- Roll back only the third insert
ROLLBACK TO SAVEPOINT sp3;
-- Now only first and second are in data

COMMIT;
```

## Nested Savepoints

Savepoints work like a stack - you can create nested recovery points:

```sql
BEGIN TRANSACTION;

SAVEPOINT outer;
INSERT INTO t VALUES (1);

  SAVEPOINT inner;
  INSERT INTO t VALUES (2);

  -- Roll back inner work only
  ROLLBACK TO SAVEPOINT inner;
  -- (1) is still there, (2) is gone

INSERT INTO t VALUES (3);

COMMIT;
-- Final: (1), (3)
```

## RELEASE SAVEPOINT

The RELEASE command removes a savepoint and makes changes permanent within the transaction:

```sql
BEGIN TRANSACTION;

INSERT INTO t VALUES (1);
SAVEPOINT sp1;

INSERT INTO t VALUES (2);
RELEASE SAVEPOINT sp1;
-- sp1 no longer exists, (2) is committed to the transaction

-- This would fail: ROLLBACK TO SAVEPOINT sp1;

COMMIT;
```

Note: RELEASE does not commit to the database - it only removes the savepoint. The transaction must still be committed.

## Rolling Back Past a Released Savepoint

If you release a savepoint and then create another one, you cannot roll back past it:

```sql
BEGIN TRANSACTION;

SAVEPOINT sp1;
INSERT INTO t VALUES (1);

RELEASE SAVEPOINT sp1;

SAVEPOINT sp2;
INSERT INTO t VALUES (2);

-- Can roll back to sp2
ROLLBACK TO SAVEPOINT sp2;
-- But cannot roll back to sp1 (it was released)

COMMIT;
```

## Transaction Rollback

ROLLBACK (without a savepoint) cancels the entire transaction:

```sql
BEGIN TRANSACTION;

INSERT INTO t VALUES (1);
SAVEPOINT sp1;
INSERT INTO t VALUES (2);

ROLLBACK;
-- Both (1) and (2) are gone
```

## Best Practices

### Use Descriptive Names

```sql
SAVEPOINT before_price_update;
SAVEPOINT after_inventory_check;
SAVEPOINT user_validation_complete;
```

### Clean Up Savepoints

Release savepoints when no longer needed to free resources:

```sql
BEGIN TRANSACTION;

SAVEPOINT sp1;
-- Do work...
RELEASE SAVEPOINT sp1;  -- No longer need to roll back here

-- Continue with transaction...
COMMIT;
```

### Don't Overuse

Savepoints have overhead. Use them when:
- You need partial rollback capability
- Error recovery is important
- Batch processing requires recovery points

Don't use them:
- For every single statement
- When full transaction rollback is acceptable

## Limitations

- Savepoint names must be unique within a transaction
- Rolling back to a savepoint also releases all savepoints created after it
- Savepoints are only valid within their transaction
- After COMMIT or ROLLBACK, all savepoints are gone

## Complete Example

```sql
-- Transfer funds with error handling
BEGIN TRANSACTION;

-- Debit source account
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
SAVEPOINT after_debit;

-- Try to credit destination
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

-- Check if destination account exists
SELECT COUNT(*) FROM accounts WHERE id = 2;
-- If count is 0, the destination doesn't exist:
-- ROLLBACK TO SAVEPOINT after_debit;
-- Then handle the error appropriately

-- If successful, commit
COMMIT;
```
