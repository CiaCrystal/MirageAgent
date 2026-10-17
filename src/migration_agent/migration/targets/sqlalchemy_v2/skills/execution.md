---
description: "engine.execute / connectionless execution / autocommit / string SQL -> Connection + text() + explicit commit"
---
# Statement execution

- `engine.execute(stmt)` was removed (2.0: `AttributeError: 'Engine' object has no attribute 'execute'`). Execute on a
  `Connection`: `with engine.connect() as conn: conn.execute(stmt)`.
- Library-level autocommit is gone. DML / DDL on a connection must be committed: `conn.commit()`, or use
  `with engine.begin() as conn:` (commits at the end of the block, rolls back on error).
- Plain strings are not executable (2.0: `ObjectNotExecutableError: Not an executable object: '...'`):
  `conn.execute(text("SELECT ... WHERE id = :id"), {"id": 1})`, or `conn.exec_driver_sql("...")` for driver-level SQL.
- Parameters: one dict, or a list of dicts for executemany — `conn.execute(t.insert(), {"x": 1})`; keyword arguments
  (`conn.execute(t.insert(), x=1)`) raise `TypeError` in 2.0.
- `stmt.execute()` / `t.select().execute()` / `stmt.scalar()` (bound execution) were removed: `conn.execute(stmt)`,
  `conn.scalar(stmt)`.

```python
# 1.4
engine.execute("INSERT INTO foo (id) VALUES (1)")
# 2.0
with engine.begin() as conn:
    conn.execute(text("INSERT INTO foo (id) VALUES (1)"))
```
