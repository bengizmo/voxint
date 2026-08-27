"""Unit tests for the savepoint_adopt_or_conflict helper."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from voxint.idempotency import savepoint_adopt_or_conflict


class TestSavepointAdoptOrConflict:
    def _mock_session(self) -> MagicMock:
        session = MagicMock()
        session.begin_nested.return_value.__enter__ = MagicMock()
        session.begin_nested.return_value.__exit__ = MagicMock(return_value=False)
        return session

    def test_fresh_insert(self) -> None:
        session = self._mock_session()
        row = object()

        result = savepoint_adopt_or_conflict(
            session,
            lookup=lambda: None,
            adopt_or_conflict=lambda _: pytest.fail("should not be called"),
            persist=lambda: row,
        )

        assert result is row
        session.begin_nested.assert_called_once()

    def test_adopt_existing_on_first_lookup(self) -> None:
        session = self._mock_session()
        existing = object()

        result = savepoint_adopt_or_conflict(
            session,
            lookup=lambda: existing,
            adopt_or_conflict=lambda e: e,
            persist=lambda: pytest.fail("should not be called"),
        )

        assert result is existing
        session.begin_nested.assert_not_called()

    def test_conflict_on_first_lookup(self) -> None:
        session = self._mock_session()

        def _conflict(e: object) -> object:
            raise ValueError("conflict")

        with pytest.raises(ValueError, match="conflict"):
            savepoint_adopt_or_conflict(
                session,
                lookup=lambda: object(),
                adopt_or_conflict=_conflict,
                persist=lambda: pytest.fail("should not be called"),
            )

    def test_adopt_on_integrity_error_race(self) -> None:
        session = self._mock_session()
        winner = object()
        call_count = 0

        def _lookup() -> object | None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None
            return winner

        def _persist() -> object:
            raise IntegrityError("dup", {}, Exception())

        result = savepoint_adopt_or_conflict(
            session,
            lookup=_lookup,
            adopt_or_conflict=lambda e: e,
            persist=_persist,
        )

        assert result is winner
        assert call_count == 2

    def test_reraise_integrity_error_when_no_winner(self) -> None:
        session = self._mock_session()

        def _persist() -> object:
            raise IntegrityError("fk violation", {}, Exception())

        with pytest.raises(IntegrityError):
            savepoint_adopt_or_conflict(
                session,
                lookup=lambda: None,
                adopt_or_conflict=lambda _: pytest.fail("should not be called"),
                persist=_persist,
            )

    def test_conflict_on_integrity_error_race(self) -> None:
        session = self._mock_session()
        winner = object()
        call_count = 0

        def _lookup() -> object | None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None
            return winner

        def _conflict(e: object) -> object:
            raise ValueError("payload mismatch")

        def _persist() -> object:
            raise IntegrityError("dup", {}, Exception())

        with pytest.raises(ValueError, match="payload mismatch"):
            savepoint_adopt_or_conflict(
                session,
                lookup=_lookup,
                adopt_or_conflict=_conflict,
                persist=_persist,
            )
