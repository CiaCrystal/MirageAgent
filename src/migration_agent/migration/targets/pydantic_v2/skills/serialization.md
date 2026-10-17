---
description: ".dict / .json / .copy / .parse_obj / .parse_raw / .from_orm / .schema / .construct -> model_* methods"
---
# Model methods

| v1 | v2 |
|---|---|
| `.dict()` | `.model_dump()` |
| `.json()` | `.model_dump_json()` |
| `.copy()` | `.model_copy()` |
| `.parse_obj(d)` | `.model_validate(d)` |
| `.parse_raw(s)` | `.model_validate_json(s)` |
| `.from_orm(o)` | `.model_validate(o)` (needs `from_attributes=True`) |
| `.schema()` / `.schema_json()` | `.model_json_schema()` / `json.dumps(...)` |
| `.construct()` | `.model_construct()` |
| `.__fields__` | `.model_fields` (class attribute; access on instances warns) |
| `.__fields_set__` | `.model_fields_set` |

`pydantic.parse_obj_as(T, x)` -> `TypeAdapter(T).validate_python(x)`.
