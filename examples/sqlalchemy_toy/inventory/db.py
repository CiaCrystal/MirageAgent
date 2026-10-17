from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine("sqlite://")
metadata = MetaData(bind=engine)

items = Table(
    "items",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(50)),
    Column("qty", Integer),
)

Base = declarative_base()


class Supplier(Base):
    __tablename__ = "suppliers"
    id = Column(Integer, primary_key=True)
    name = Column(String(50))


Session = sessionmaker(bind=engine)


def init_db():
    metadata.create_all()
    Base.metadata.create_all(engine)
