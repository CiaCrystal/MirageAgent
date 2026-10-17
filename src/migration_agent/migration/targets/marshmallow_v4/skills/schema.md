---
description: "Schema context, class Meta options (fields/additional/ordered), validators returning False"
---
# Schema options and context

- `Schema(context={...})` and `self.context` were removed: pass state with a `contextvars.ContextVar` (or the
  experimental `marshmallow.experimental.context.Context` manager). `fields.Function` / `fields.Method` callables no
  longer receive a `context` argument.
- `class Meta: fields = (...)` / `additional = (...)` no longer create undeclared fields: declare them explicitly.
- `class Meta: ordered = True` was removed; field order is preserved by default.
- Validators must raise `ValidationError`; a validator that returns `False` no longer fails validation in 4.0.
