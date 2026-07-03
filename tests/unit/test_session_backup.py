"""#6: the pre-migration session backup must not fail silently.

_backup_old_format_session copies an old-format session.json to session.json.bak
before a migrating load (which shutdown later overwrites). If the backup WRITE
fails (permission denied, read-only dir, disk full) the original is the user's
only pre-migration copy -- silently skipping the backup then overwriting it on
exit is data loss with no trace. The write failure must at least be logged.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

main = pytest.importorskip("gasgiant.app.main")
StudioApp = main.StudioApp


def _old_format_session(path: Path) -> None:
    path.write_text(json.dumps({"preset_format": 1, "name": "session"}), encoding="utf-8")


def test_backup_write_failure_is_logged_and_migration_proceeds(monkeypatch, tmp_path, caplog):
    session = tmp_path / "session.json"
    _old_format_session(session)
    monkeypatch.setattr(main, "SESSION_PATH", session)

    def boom(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
        raise OSError("read-only filesystem")

    monkeypatch.setattr(Path, "write_bytes", boom)

    app = StudioApp.__new__(StudioApp)
    with caplog.at_level(logging.WARNING):
        app._backup_old_format_session()  # must NOT raise

    assert any(r.levelname == "WARNING" for r in caplog.records), (
        "a failed backup write must be logged, not silently swallowed"
    )


def test_backup_written_for_old_format_session(monkeypatch, tmp_path):
    session = tmp_path / "session.json"
    _old_format_session(session)
    monkeypatch.setattr(main, "SESSION_PATH", session)

    app = StudioApp.__new__(StudioApp)
    app._backup_old_format_session()

    assert (tmp_path / "session.json.bak").is_file(), "old-format session should be backed up"


def test_unreadable_session_returns_quietly(monkeypatch, tmp_path):
    session = tmp_path / "session.json"
    session.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(main, "SESSION_PATH", session)

    app = StudioApp.__new__(StudioApp)
    app._backup_old_format_session()  # must not raise on unreadable/corrupt session

    assert not (tmp_path / "session.json.bak").exists()
