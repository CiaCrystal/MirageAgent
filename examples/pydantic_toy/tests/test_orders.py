from decimal import Decimal

import pytest
from pydantic import ValidationError
from toyshop.orders import LineItem, Order

ITEMS = [{"sku": "A", "quantity": 2, "unit_price": "1.50"}, {"sku": "B", "quantity": 1, "unit_price": "2.00"}]


def test_order_total_must_match_items():
    with pytest.raises(ValidationError) as exc:
        Order(order_id="o1", user_id=1, items=ITEMS, total=Decimal("9.99"))
    assert "does not match" in str(exc.value)


def test_order_requires_at_least_one_item():
    with pytest.raises(ValidationError):
        Order(order_id="o1", user_id=1, items=[], total=Decimal("0"))


def test_order_is_immutable_and_serialises():
    o = Order(order_id="o1", user_id=1, items=ITEMS, total=Decimal("5.00"))
    with pytest.raises(Exception):
        o.total = Decimal("1")
    rec = o.to_record()
    assert rec == {"order_id": "o1", "user_id": 1, "total": "5.00", "n_items": 2}
    assert '"order_id":"o1"' in o.to_json().replace(" ", "")
    assert isinstance(o.items[0], LineItem)
