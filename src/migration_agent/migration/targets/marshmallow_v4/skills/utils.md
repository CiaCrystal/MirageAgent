---
description: "removed marshmallow.utils helpers (from_iso_datetime, isoformat, rfcformat ...) and __version__"
---
# Removed utilities

- `marshmallow.utils.from_iso_datetime / from_iso_date / from_iso_time` -> `datetime.datetime.fromisoformat` /
  `datetime.date.fromisoformat` / `datetime.time.fromisoformat`.
- `isoformat`, `to_iso_time`, `to_iso_datetime`, `from_rfc`, `rfcformat`, `get_fixed_timezone`, `pprint` were removed:
  use the standard library (`.isoformat()`, `email.utils`, `pprint.pprint`).
- `marshmallow.__version__` was removed: `importlib.metadata.version("marshmallow")`.
