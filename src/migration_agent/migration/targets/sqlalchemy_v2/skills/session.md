---
description: "Session(autocommit=True), subtransactions, autobegin and explicit commit"
---
# Session

- `Session(engine, autocommit=True)` / `sessionmaker(autocommit=True)` raise in 2.0
  (`ArgumentError: autocommit=True is no longer supported`). Use the default session (it "autobegins" a transaction)
  and call `session.commit()`; or `with Session(engine) as s, s.begin(): ...` for a block that commits at the end.
- `session.begin(subtransactions=True)` was removed; use `session.begin_nested()` for SAVEPOINTs, or restructure so a
  single outer `session.begin()` owns the transaction.
- `Session.get(Model, pk)` replaces `session.query(Model).get(pk)`.
