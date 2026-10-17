from decimal import Decimal
from typing import List

from pydantic import BaseModel, Field, root_validator


class LineItem(BaseModel):
    sku: str
    quantity: int = Field(..., gt=0)
    unit_price: Decimal


class Order(BaseModel):
    order_id: str
    user_id: int
    items: List[LineItem] = Field(..., min_items=1)
    total: Decimal

    class Config:
        allow_mutation = False

    @root_validator
    def total_matches_items(cls, values):
        items = values.get("items") or []
        expected = sum((i.unit_price * i.quantity for i in items), Decimal(0))
        if values.get("total") != expected:
            raise ValueError(f"total {values.get('total')} does not match items {expected}")
        return values

    def to_record(self) -> dict:
        """Flat dict suitable for a CSV row."""
        data = self.dict(exclude={"items"})
        data["n_items"] = len(self.items)
        data["total"] = str(data["total"])
        return data

    def to_json(self) -> str:
        return self.json(exclude_none=True)
