---
description: "decorators: pass_many -> pass_collection, @validates receives data_key, keyword-only decorator args"
---
# Decorators

- `@pre_load(pass_many=True)` (and post_load / pre_dump / post_dump / validates_schema) -> `pass_collection=True`.
- Methods decorated with `@validates("field")` receive `data_key` as a keyword argument: `def check(self, value, **kwargs)`.
- `pre_load`, `post_load` and `validates_schema` methods receive `unknown` as a keyword argument: keep `**kwargs`.
- Decorator arguments are keyword-only.
