import json

from toyshop import service
from toyshop.models import User


def test_load_user_from_dict_and_json():
    payload = {"id": 3, "name": "Linus", "email": "L@K.org", "nickname": "torvalds"}
    assert service.load_user(payload).display_name() == "torvalds"
    assert service.load_user_json(json.dumps(payload)).email == "l@k.org"


def test_user_from_orm_row_and_rename():
    row = service._OrmUser(7, "Guido", "g@python.org")
    u = service.user_from_db(row)
    assert isinstance(u, User)
    assert u.tags == ["vip"]
    renamed = service.rename_user(u, "BDFL")
    assert renamed.name == "BDFL" and renamed.id == 7
    assert u.name == "Guido"


def test_user_schema_fields_order():
    assert service.user_schema_fields() == ["id", "name", "email", "nickname", "tags", "address"]
