"""Unit tests for filtering and sorting utilities."""

import pytest
from sqlalchemy import Boolean, Integer, String, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from tabletop_oracle.api.filtering import (
    FilterOperator,
    FilterSpec,
    SortParam,
    apply_filters,
    apply_sort,
    build_filter_spec,
    parse_filter_value,
    parse_sort,
)

# --- Test model for SQLAlchemy clause generation ---


class _TestBase(DeclarativeBase):
    pass


class _FakeModel(_TestBase):
    """Minimal model for testing filter/sort clause generation."""

    __tablename__ = "fake_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    score: Mapped[int] = mapped_column(Integer)
    active: Mapped[bool] = mapped_column(Boolean)


# --- SortParam parsing ---


class TestParseSort:
    """Tests for sort query parameter parsing."""

    def test_ascending_sort(self) -> None:
        """Field without prefix parses as ascending."""
        result = parse_sort("name", {"name", "score"})
        assert result is not None
        assert result.field_name == "name"
        assert result.descending is False

    def test_descending_sort(self) -> None:
        """Field with - prefix parses as descending."""
        result = parse_sort("-score", {"name", "score"})
        assert result is not None
        assert result.field_name == "score"
        assert result.descending is True

    def test_none_value_returns_none(self) -> None:
        """None sort value returns None."""
        assert parse_sort(None, {"name"}) is None

    def test_empty_string_returns_none(self) -> None:
        """Empty string sort value returns None."""
        assert parse_sort("", {"name"}) is None

    def test_invalid_field_raises_400(self) -> None:
        """Sort field not in allowed set raises HTTP 400."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            parse_sort("invalid_field", {"name", "score"})
        assert exc_info.value.status_code == 400

    def test_invalid_descending_field_raises_400(self) -> None:
        """Descending sort on invalid field raises HTTP 400."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            parse_sort("-invalid", {"name", "score"})
        assert exc_info.value.status_code == 400


class TestApplySort:
    """Tests for applying sort to SQLAlchemy queries."""

    def test_ascending_order_by(self) -> None:
        """Ascending sort adds ORDER BY ASC clause."""
        query = select(_FakeModel)
        sort_param = SortParam(field_name="name", descending=False)
        result = apply_sort(query, sort_param, _FakeModel)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "ORDER BY" in compiled
        assert "fake_items.name ASC" in compiled

    def test_descending_order_by(self) -> None:
        """Descending sort adds ORDER BY DESC clause."""
        query = select(_FakeModel)
        sort_param = SortParam(field_name="score", descending=True)
        result = apply_sort(query, sort_param, _FakeModel)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "fake_items.score DESC" in compiled

    def test_none_sort_param_returns_unchanged(self) -> None:
        """None sort_param returns the query unchanged."""
        query = select(_FakeModel)
        result = apply_sort(query, None, _FakeModel)
        assert str(result.compile()) == str(query.compile())


# --- Filter value parsing ---


class TestParseFilterValue:
    """Tests for raw filter value parsing."""

    def test_bool_true_values(self) -> None:
        """Boolean filter recognises true/1/yes."""
        assert parse_filter_value("true", FilterOperator.BOOL) is True
        assert parse_filter_value("True", FilterOperator.BOOL) is True
        assert parse_filter_value("1", FilterOperator.BOOL) is True
        assert parse_filter_value("yes", FilterOperator.BOOL) is True

    def test_bool_false_values(self) -> None:
        """Boolean filter treats other values as False."""
        assert parse_filter_value("false", FilterOperator.BOOL) is False
        assert parse_filter_value("0", FilterOperator.BOOL) is False
        assert parse_filter_value("no", FilterOperator.BOOL) is False

    def test_in_operator_splits_csv(self) -> None:
        """IN operator splits comma-separated values."""
        result = parse_filter_value("light,medium,heavy", FilterOperator.IN)
        assert result == ["light", "medium", "heavy"]

    def test_in_operator_strips_whitespace(self) -> None:
        """IN operator strips whitespace from values."""
        result = parse_filter_value("a , b , c", FilterOperator.IN)
        assert result == ["a", "b", "c"]

    def test_in_operator_skips_empty(self) -> None:
        """IN operator skips empty segments."""
        result = parse_filter_value("a,,b", FilterOperator.IN)
        assert result == ["a", "b"]

    def test_eq_returns_raw_string(self) -> None:
        """EQ operator returns the raw string unchanged."""
        assert parse_filter_value("catan", FilterOperator.EQ) == "catan"

    def test_search_returns_raw_string(self) -> None:
        """SEARCH operator returns the raw string unchanged."""
        assert parse_filter_value("catan", FilterOperator.SEARCH) == "catan"


class TestBuildFilterSpec:
    """Tests for building FilterSpec from raw input."""

    def test_builds_eq_spec(self) -> None:
        """Builds equality filter spec."""
        spec = build_filter_spec("name", "Catan", FilterOperator.EQ)
        assert spec.field_name == "name"
        assert spec.operator == FilterOperator.EQ
        assert spec.value == "Catan"

    def test_builds_in_spec(self) -> None:
        """Builds multi-value IN filter spec."""
        spec = build_filter_spec("complexity", "light,medium", FilterOperator.IN)
        assert spec.value == ["light", "medium"]

    def test_builds_bool_spec(self) -> None:
        """Builds boolean filter spec."""
        spec = build_filter_spec("active", "true", FilterOperator.BOOL)
        assert spec.value is True


# --- Applying filters to queries ---


class TestApplyFilters:
    """Tests for applying filter specs to SQLAlchemy queries."""

    def test_equality_filter(self) -> None:
        """EQ filter adds WHERE column = value."""
        query = select(_FakeModel)
        filters = [FilterSpec(field_name="name", operator=FilterOperator.EQ, value="Catan")]
        result = apply_filters(query, filters, _FakeModel)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "fake_items.name = 'Catan'" in compiled

    def test_in_filter_multiple_values(self) -> None:
        """IN filter with multiple values adds OR conditions."""
        query = select(_FakeModel)
        filters = [
            FilterSpec(field_name="name", operator=FilterOperator.IN, value=["Catan", "Azul"])
        ]
        result = apply_filters(query, filters, _FakeModel)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "fake_items.name = 'Catan' OR fake_items.name = 'Azul'" in compiled

    def test_in_filter_single_value(self) -> None:
        """IN filter with single value uses equality."""
        query = select(_FakeModel)
        filters = [FilterSpec(field_name="name", operator=FilterOperator.IN, value=["Catan"])]
        result = apply_filters(query, filters, _FakeModel)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "fake_items.name = 'Catan'" in compiled

    def test_gte_filter(self) -> None:
        """GTE filter adds WHERE column >= value."""
        query = select(_FakeModel)
        filters = [FilterSpec(field_name="score", operator=FilterOperator.GTE, value=10)]
        result = apply_filters(query, filters, _FakeModel)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "fake_items.score >= 10" in compiled

    def test_lte_filter(self) -> None:
        """LTE filter adds WHERE column <= value."""
        query = select(_FakeModel)
        filters = [FilterSpec(field_name="score", operator=FilterOperator.LTE, value=100)]
        result = apply_filters(query, filters, _FakeModel)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "fake_items.score <= 100" in compiled

    def test_gt_filter(self) -> None:
        """GT filter adds WHERE column > value."""
        query = select(_FakeModel)
        filters = [FilterSpec(field_name="score", operator=FilterOperator.GT, value=5)]
        result = apply_filters(query, filters, _FakeModel)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "fake_items.score > 5" in compiled

    def test_lt_filter(self) -> None:
        """LT filter adds WHERE column < value."""
        query = select(_FakeModel)
        filters = [FilterSpec(field_name="score", operator=FilterOperator.LT, value=50)]
        result = apply_filters(query, filters, _FakeModel)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "fake_items.score < 50" in compiled

    def test_search_filter(self) -> None:
        """SEARCH filter adds ILIKE clause."""
        query = select(_FakeModel)
        filters = [FilterSpec(field_name="name", operator=FilterOperator.SEARCH, value="cat")]
        result = apply_filters(query, filters, _FakeModel)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "LIKE" in compiled.upper()
        assert "%cat%" in compiled

    def test_bool_filter(self) -> None:
        """BOOL filter adds IS clause."""
        query = select(_FakeModel)
        filters = [FilterSpec(field_name="active", operator=FilterOperator.BOOL, value=True)]
        result = apply_filters(query, filters, _FakeModel)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        compiled_lower = compiled.lower()
        has_bool_clause = (
            "fake_items.active is true" in compiled_lower
            or "fake_items.active is 1" in compiled_lower
        )
        assert has_bool_clause

    def test_multiple_filters_combined(self) -> None:
        """Multiple filters produce multiple WHERE conditions."""
        query = select(_FakeModel)
        filters = [
            FilterSpec(field_name="name", operator=FilterOperator.EQ, value="Catan"),
            FilterSpec(field_name="score", operator=FilterOperator.GTE, value=5),
        ]
        result = apply_filters(query, filters, _FakeModel)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "fake_items.name = 'Catan'" in compiled
        assert "fake_items.score >= 5" in compiled

    def test_empty_filters_returns_unchanged(self) -> None:
        """Empty filter list returns the query unchanged."""
        query = select(_FakeModel)
        result = apply_filters(query, [], _FakeModel)
        assert str(result.compile()) == str(query.compile())
