---
description: "Field(...) arguments and Optional defaults (const, regex, min_items, allow_mutation, example)"
---
# Fields

- `Optional[X]` without a default is now REQUIRED: write `x: Optional[X] = None` explicitly.
- `Field(..., const=True)` -> a `Literal[...]` type.
- `Field(..., regex=...)` -> `Field(..., pattern=...)`.
- `Field(..., min_items/max_items)` -> `Field(..., min_length/max_length)`.
- `Field(..., allow_mutation=False)` -> `Field(..., frozen=True)`.
- `Field(..., example=...)` -> `Field(..., examples=[...])`.
- `ValidationError.errors()[i]['loc'] / ['msg']`: same structure, but `msg` texts and `type` strings changed — when a test
  asserts the message text, compare with the exact v2 wording.
