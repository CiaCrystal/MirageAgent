---
description: "declarative_base moved to sqlalchemy.orm; DeclarativeBase; mapper() -> registry"
---
# Declarative

- `from sqlalchemy.ext.declarative import declarative_base` -> `from sqlalchemy.orm import declarative_base`
  (the old import still works in 2.0 with `MovedIn20Warning`; migrate it). 2.0 also offers `class Base(DeclarativeBase): pass`.
- `declared_attr` likewise moves to `sqlalchemy.orm`.
- Classical `mapper(Cls, table)` -> `registry().map_imperatively(Cls, table)`.
- Typed annotations on mapped classes without `Mapped[]` raise in 2.0 unless the base sets `__allow_unmapped__ = True`.
