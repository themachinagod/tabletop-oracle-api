"""Unit tests for the UNSET sentinel."""

from __future__ import annotations

from tabletop_oracle.schemas.sentinel import UNSET, _UnsetType


class TestUnsetSentinel:
    """Tests for the UNSET sentinel value."""

    def test_singleton(self) -> None:
        """UNSET is a singleton — multiple calls return the same instance."""
        a = _UnsetType()
        b = _UnsetType()
        assert a is b
        assert a is UNSET

    def test_repr(self) -> None:
        """UNSET repr is the string 'UNSET'."""
        assert repr(UNSET) == "UNSET"

    def test_falsy(self) -> None:
        """UNSET is falsy."""
        assert not UNSET
        assert bool(UNSET) is False

    def test_identity_comparison(self) -> None:
        """UNSET can be compared with 'is' operator."""
        value = UNSET
        assert value is UNSET

    def test_equality(self) -> None:
        """UNSET equals itself and nothing else."""
        assert UNSET == UNSET
        assert UNSET != None  # noqa: E711
        assert UNSET != ""
        assert UNSET != 0
        assert UNSET != False  # noqa: E712

    def test_hashable(self) -> None:
        """UNSET is hashable and can be used in sets."""
        s = {UNSET}
        assert UNSET in s

    def test_not_none(self) -> None:
        """UNSET is not None — they are distinct concepts."""
        assert UNSET is not None
