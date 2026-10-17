---
description: "Row behaves like a named tuple: row['col'] -> row._mapping['col'] / row.col; 'x' in row"
---
# Result rows

- `row["col"]` raises in 2.0 (`TypeError: tuple indices must be integers or slices, not str`). Use `row.col`,
  `row._mapping["col"]`, or iterate `result.mappings()` to get dict-like rows.
- `dict(row)` -> `dict(row._mapping)` (or `row._asdict()`).
- `"col" in row` is now False (membership tests values, like a tuple); test `"col" in row._mapping` / `row.keys()`.
- ORM results: `session.execute(select(User)).scalars().all()` returns entities; without `.scalars()` each element is a
  one-element `Row`.
