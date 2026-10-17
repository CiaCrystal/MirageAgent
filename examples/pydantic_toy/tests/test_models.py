import pytest
from pydantic import ValidationError
from toyshop.models import Address, User


def test_user_minimal_payload_is_valid():
    u = User(id=1, name="Ada", email="ADA@example.com")
    assert u.email == "ada@example.com"
    assert u.nickname is None
    assert u.tags == []


def test_email_validator_rejects_bad_value():
    with pytest.raises(ValidationError) as exc:
        User(id=1, name="Ada", email="nope")
    assert "email must contain @" in str(exc.value)


def test_whitespace_is_stripped_and_tags_normalised():
    u = User(id=2, name="  Grace ", email="g@x.io", tags=["  Admin", "VIP "])
    assert u.name == "Grace"
    assert u.tags == ["admin", "vip"]


def test_postcode_length_constraint():
    with pytest.raises(ValidationError):
        Address(street="1 Main St", city="Oslo", postcode="12")
