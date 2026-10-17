import json
from typing import Any, Dict, List

from .models import User
from .orders import Order


class _OrmUser:
    """Stand-in for an ORM row."""

    def __init__(self, id: int, name: str, email: str):
        self.id = id
        self.name = name
        self.email = email
        self.nickname = None
        self.tags = ["VIP "]


def load_user(payload: Dict[str, Any]) -> User:
    return User.parse_obj(payload)


def load_user_json(raw: str) -> User:
    return User.parse_raw(raw)


def user_from_db(row: _OrmUser) -> User:
    return User.from_orm(row)


def rename_user(user: User, new_name: str) -> User:
    return user.copy(update={"name": new_name})


def load_orders(raw: str) -> List[Order]:
    return [Order.parse_obj(o) for o in json.loads(raw)]


def user_schema_fields() -> List[str]:
    return list(User.__fields__.keys())
