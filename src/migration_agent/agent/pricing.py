"""Configured token prices and estimated request costs."""

from pydantic import BaseModel, Field


class Pricing(BaseModel):
    currency: str = "USD"
    input_per_million: float = Field(ge=0)
    cache_read_per_million: float = Field(ge=0)
    cache_write_per_million: float = Field(ge=0)
    output_per_million: float = Field(ge=0)
    source: str
    note: str = "Estimated cost at configured rates, not a provider invoice."

    def cost(self, tokens_in: int, cache_read: int, cache_write: int, tokens_out: int) -> float:
        return (
            tokens_in * self.input_per_million
            + cache_read * self.cache_read_per_million
            + cache_write * self.cache_write_per_million
            + tokens_out * self.output_per_million
        ) / 1_000_000
