---
description: "class Config -> model_config = ConfigDict(...) and renamed config keys (orm_mode, extra, frozen ...)"
---
# Model configuration

- `class Config:` -> `model_config = ConfigDict(...)` (`ConfigDict` is imported from pydantic).
- Renamed keys: `orm_mode` -> `from_attributes`; `allow_population_by_field_name` -> `populate_by_name`;
  `anystr_strip_whitespace` -> `str_strip_whitespace`; `min_anystr_length` / `max_anystr_length` -> `str_min_length` /
  `str_max_length`; `validate_all` -> `validate_default`; `allow_mutation = False` -> `frozen=True`;
  `schema_extra` -> `json_schema_extra`; `Config.fields` -> `Field(...)` on the field.
- `extra = Extra.forbid` -> `extra='forbid'` (the `Extra` enum is deprecated).
