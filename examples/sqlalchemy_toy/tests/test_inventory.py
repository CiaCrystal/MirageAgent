import pytest

from inventory import db, store


@pytest.fixture(autouse=True)
def fresh_db():
    db.metadata.drop_all(db.engine)
    db.Base.metadata.drop_all(db.engine)
    db.init_db()


def test_add_and_list():
    store.add_item("apple", 3)
    store.add_item("pear", 2)
    assert store.item_names() == ["apple", "pear"]


def test_empty_list():
    assert store.item_names() == []


def test_total_qty():
    store.add_item("apple", 3)
    store.add_item("pear", 2)
    assert store.total_qty() == 5


def test_total_qty_empty():
    assert store.total_qty() == 0


def test_tables():
    assert store.tables() == ["items", "suppliers"]


def test_supplier_roundtrip():
    pk = store.add_supplier("acme")
    assert store.supplier_name(pk) == "acme"


def test_two_suppliers():
    a = store.add_supplier("a")
    b = store.add_supplier("b")
    assert a != b and store.supplier_name(b) == "b"


def test_items_and_suppliers_independent():
    store.add_item("x", 1)
    store.add_supplier("s")
    assert store.item_names() == ["x"] and store.tables() == ["items", "suppliers"]


def test_qty_after_many():
    for i in range(5):
        store.add_item(f"i{i}", i)
    assert store.total_qty() == 10


def test_names_order():
    for n in ("c", "a", "b"):
        store.add_item(n, 1)
    assert store.item_names() == ["c", "a", "b"]
