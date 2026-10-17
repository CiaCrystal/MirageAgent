---
description: "field arguments: default/missing -> dump_default/load_default, metadata kwargs, Number/Mapping, Nested('self')"
---
# Field arguments

- `fields.String(default="x")` -> `fields.String(dump_default="x")`; `missing=` -> `load_default=` (4.0:
  `TypeError: Field.__init__() got an unexpected keyword argument 'default'`).
- Extra keyword arguments are no longer stored as metadata: `fields.String(description="d")` ->
  `fields.String(metadata={"description": "d"})`.
- `fields.Number()` / `fields.Mapping()` cannot be used directly: use `fields.Integer`, `fields.Float`, `fields.Decimal`,
  or `fields.Dict`.
- `fields.Nested("self")` -> `fields.Nested(lambda: MySchema(...))` (pass `exclude=` / `only=` to the schema).
