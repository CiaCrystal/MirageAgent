---
description: "Flask-SQLAlchemy and other integration packages that must be upgraded together with SQLAlchemy 2.0"
---
# Integration packages

- Flask-SQLAlchemy 2.x predates SQLAlchemy 2.0; upgrade it to 3.x (3.0.2 updated compatibility with SQLAlchemy 2;
  3.1 requires SQLAlchemy >= 2.0.16): `pip install -U flask-sqlalchemy`.
  In 3.0 `db.get_engine()` / `get_tables_for_bind()` / `get_binds()` are deprecated (removed in 3.1): use `db.engine` /
  `db.engines`.
- Other packages pinned to `sqlalchemy<2` must be upgraded, not SQLAlchemy downgraded. Upgrading third-party
  dependencies with pip is allowed; check the failing import / traceback to see which package is too old.
