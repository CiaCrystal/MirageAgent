from sqlalchemy import select

from .db import Session, Supplier, engine, items


def add_item(name, qty):
    engine.execute(items.insert(), name=name, qty=qty)


def item_names():
    rows = engine.execute(select([items.c.name]).order_by(items.c.id))
    return [row["name"] for row in rows]


def total_qty():
    with engine.connect() as conn:
        return conn.execute("SELECT COALESCE(SUM(qty), 0) FROM items").scalar()


def tables():
    return sorted(engine.table_names())


def add_supplier(name):
    session = Session()
    session.add(Supplier(name=name))
    session.commit()
    return session.query(Supplier).filter_by(name=name).one().id


def supplier_name(pk):
    session = Session()
    return session.query(Supplier).get(pk).name
