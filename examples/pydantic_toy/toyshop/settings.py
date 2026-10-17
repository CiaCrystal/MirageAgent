from pydantic import BaseSettings, Field


class ShopSettings(BaseSettings):
    currency: str = "EUR"
    tax_rate: float = Field(0.2, ge=0, le=1)
    debug: bool = False

    class Config:
        env_prefix = "TOYSHOP_"


def get_settings() -> ShopSettings:
    return ShopSettings()
