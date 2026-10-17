---
description: "select([...]) list form, Query.from_self / select_entity_from, session.query(...).get()"
---
# select() and the ORM Query

- `select([a, b])` -> `select(a, b)` (2.0: `ArgumentError: Column expression, FROM clause, or other columns clause
  element expected`); `select([table])` -> `select(table)`; `whereclause=` / `from_obj=` keywords -> `.where()` /
  `.select_from()`.
- `Query.from_self()` was removed: build a subquery and select from an alias — `subq = q.subquery();
  session.query(aliased(User, subq))` (or `select(aliased(User, subq))`).
- `session.query(User).get(5)` still works in 2.0 but is legacy (`LegacyAPIWarning`); the 2.0 form is
  `session.get(User, 5)`.
- `session.query(...)` itself is legacy but supported in 2.0; `session.execute(select(User)).scalars()` is the new style.
  Do not rewrite working Query code just for style.
