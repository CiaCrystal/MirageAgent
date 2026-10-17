import datetime

import pytest
from marshmallow import ValidationError

from petshop.dates import parse_when
from petshop.schemas import OwnerSchema, PetSchema, greet


def test_load_minimal_pet():
    pet = PetSchema().load({"name": "Tom"})
    assert pet["name"] == "Tom" and pet["species"] == "cat" and pet["tagged"]


def test_dump_default_species():
    assert PetSchema().dump({"name": "Tom"})["species"] == "cat"


def test_negative_age_rejected():
    with pytest.raises(ValidationError) as e:
        PetSchema().load({"name": "Tom", "age": -1})
    assert "age" in e.value.messages


def test_blank_name_rejected():
    with pytest.raises(ValidationError) as e:
        PetSchema().load({"name": "   "})
    assert e.value.messages["name"] == ["Name must not be blank."]


def test_names_are_stripped_for_many():
    pets = PetSchema(many=True).load([{"name": " a "}, {"name": "b "}])
    assert [p["name"] for p in pets] == ["a", "b"]


def test_weight_is_a_number():
    assert PetSchema().load({"name": "Tom", "weight": 4.5})["weight"] == 4.5


def test_owner_defaults_to_none():
    assert PetSchema().load({"name": "Tom"})["owner"] is None


def test_owner_email_optional():
    assert OwnerSchema().load({"name": "Ann"}) == {"name": "Ann", "email": None}


def test_owner_field_order():
    assert list(OwnerSchema().dump({"email": "a@b.c", "name": "Ann"})) == ["name", "email"]


def test_owner_name_description():
    assert OwnerSchema().fields["name"].metadata["description"] == "Owner name"


def test_friends_nested_self():
    pet = PetSchema().load({"name": "Tom", "friends": [{"name": "Jerry"}]})
    assert pet["friends"][0]["name"] == "Jerry"


def test_greet_uses_prefix():
    assert greet({"name": "Tom"}, prefix="Hello") == {"greeting": "Hello Tom"}


def test_parse_when():
    assert parse_when("2024-01-02T03:04:05") == datetime.datetime(2024, 1, 2, 3, 4, 5)
