---
description: "MetaData(bind=), metadata.create_all() without engine, autoload=True, engine.table_names() / has_table()"
---
# Bound metadata and reflection

- `MetaData(bind=engine)` was removed (2.0: `TypeError: ... unexpected keyword argument 'bind'`): `MetaData()` and pass
  the engine where it is needed — `metadata.create_all(engine)`, `metadata.drop_all(engine)`, `metadata.reflect(engine)`.
- `Table("t", metadata, autoload=True)` -> `Table("t", metadata, autoload_with=engine)`.
- `engine.table_names()` -> `inspect(engine).get_table_names()`; `engine.has_table("t")` -> `inspect(engine).has_table("t")`
  (both deprecated in 1.4 with `SADeprecationWarning`, removed in 2.0). `from sqlalchemy import inspect`.
