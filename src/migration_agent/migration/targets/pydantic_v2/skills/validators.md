---
description: "@validator / @root_validator -> field_validator / model_validator (pre, always, each_item, values)"
---
# Validators

- `@validator('f')` -> `@field_validator('f')` — needs `@classmethod`; `pre=True` -> `mode='before'`; `always` was removed
  (use `validate_default=True` on the field); `each_item=True` was removed (validate the items with an annotated type).
- `@root_validator` -> `@model_validator(mode='before'|'after')` — in `after` mode the first parameter is `self` (the
  instance), not `cls, values`; `skip_on_failure` no longer exists.
- Validator signature: the old `values` dict becomes `info: ValidationInfo` (`info.data` holds already-validated fields).

```python
# v1
@validator("name", pre=True)
def strip(cls, v):
    return v.strip()


# v2
@field_validator("name", mode="before")
@classmethod
def strip(cls, v):
    return v.strip()
```
