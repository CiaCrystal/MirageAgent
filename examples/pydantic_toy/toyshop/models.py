from typing import List, Optional

from pydantic import BaseModel, Field, validator


class Address(BaseModel):
    street: str
    city: str
    postcode: str = Field(..., min_length=3, max_length=10)


class User(BaseModel):
    id: int
    name: str
    email: str
    nickname: Optional[str]
    tags: List[str] = []
    address: Optional[Address] = None

    class Config:
        orm_mode = True
        anystr_strip_whitespace = True

    @validator("email")
    def email_must_contain_at(cls, v):
        if "@" not in v:
            raise ValueError("email must contain @")
        return v.lower()

    @validator("tags", pre=True, each_item=True)
    def normalise_tag(cls, v):
        return str(v).strip().lower()

    def display_name(self) -> str:
        return self.nickname or self.name
