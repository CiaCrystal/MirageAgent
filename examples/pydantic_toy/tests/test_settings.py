from toyshop.settings import get_settings


def test_settings_defaults():
    s = get_settings()
    assert s.currency == "EUR" and s.tax_rate == 0.2 and s.debug is False


def test_settings_read_env_with_prefix(monkeypatch):
    monkeypatch.setenv("TOYSHOP_CURRENCY", "USD")
    monkeypatch.setenv("TOYSHOP_DEBUG", "true")
    s = get_settings()
    assert s.currency == "USD" and s.debug is True
