from unittest.mock import Mock

import pytest

from hear.models import database


def test_missing_database_url_fails_only_when_engine_requested(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(database.settings, "DATABASE_URL", "")
    monkeypatch.setattr(database.DatabaseRuntime, "_engine", None)
    monkeypatch.setattr(database, "create_engine", factory)
    with pytest.raises(RuntimeError, match="DATABASE_URL is required"):
        database.DatabaseRuntime.get_engine()
    factory.assert_not_called()


def test_database_engine_is_created_once_on_first_use(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(database.settings, "DATABASE_URL", "postgresql://example/db")
    monkeypatch.setattr(database.DatabaseRuntime, "_engine", None)
    monkeypatch.setattr(database, "create_engine", factory)
    assert database.DatabaseRuntime.get_engine() is factory.return_value
    assert database.DatabaseRuntime.get_engine() is factory.return_value
    factory.assert_called_once()
