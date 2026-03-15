"""Unit tests for TokenUsageService and TokenAttribution.

Tests service methods with a mocked AsyncSession: record() creates
log entries, get_query_total() and get_daily_total() return aggregated
token counts, get_session_query_count() counts user_question messages.
Also verifies TokenAttribution immutability and default field values.
"""

from __future__ import annotations

import uuid
from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.models.enums import ModelCapability
from tabletop_oracle.services.model.token_usage import TokenAttribution, TokenUsageService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_attribution(
    *,
    user_id: uuid.UUID | None = None,
    session_id: uuid.UUID | None = None,
    message_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
) -> TokenAttribution:
    """Build a TokenAttribution with sensible defaults."""
    return TokenAttribution(
        user_id=user_id or uuid.uuid4(),
        session_id=session_id,
        message_id=message_id,
        document_id=document_id,
    )


def _mock_session(scalar_result: int | None = None) -> AsyncMock:
    """Build a mock AsyncSession.

    Supports both add/flush (for record) and execute/scalar_one (for queries).
    """
    result_mock = MagicMock()
    result_mock.scalar_one.return_value = scalar_result
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result_mock)
    session.add = MagicMock()
    session.flush = AsyncMock()
    return session


# ---------------------------------------------------------------------------
# TokenAttribution — construction
# ---------------------------------------------------------------------------


class TestTokenAttributionConstruction:
    """Tests TokenAttribution creation and default values."""

    def test_all_fields_set(self) -> None:
        """Attribution with all fields populated."""
        uid = uuid.uuid4()
        sid = uuid.uuid4()
        mid = uuid.uuid4()
        did = uuid.uuid4()

        attr = TokenAttribution(
            user_id=uid,
            session_id=sid,
            message_id=mid,
            document_id=did,
        )

        assert attr.user_id == uid
        assert attr.session_id == sid
        assert attr.message_id == mid
        assert attr.document_id == did

    def test_optional_fields_default_to_none(self) -> None:
        """Session, message, and document IDs default to None."""
        uid = uuid.uuid4()
        attr = TokenAttribution(user_id=uid)

        assert attr.user_id == uid
        assert attr.session_id is None
        assert attr.message_id is None
        assert attr.document_id is None

    def test_query_context_attribution(self) -> None:
        """Attribution for a query context has session and message but no document."""
        uid = uuid.uuid4()
        sid = uuid.uuid4()
        mid = uuid.uuid4()

        attr = TokenAttribution(user_id=uid, session_id=sid, message_id=mid)

        assert attr.session_id == sid
        assert attr.message_id == mid
        assert attr.document_id is None

    def test_ingestion_context_attribution(self) -> None:
        """Attribution for ingestion has document but no session or message."""
        uid = uuid.uuid4()
        did = uuid.uuid4()

        attr = TokenAttribution(user_id=uid, document_id=did)

        assert attr.session_id is None
        assert attr.message_id is None
        assert attr.document_id == did


# ---------------------------------------------------------------------------
# TokenAttribution — immutability
# ---------------------------------------------------------------------------


class TestTokenAttributionImmutability:
    """Tests that TokenAttribution is frozen (immutable)."""

    def test_cannot_modify_user_id(self) -> None:
        """Attempting to modify user_id raises FrozenInstanceError."""
        attr = _make_attribution()

        with pytest.raises(FrozenInstanceError):
            attr.user_id = uuid.uuid4()  # type: ignore[misc]

    def test_cannot_modify_session_id(self) -> None:
        """Attempting to modify session_id raises FrozenInstanceError."""
        attr = _make_attribution(session_id=uuid.uuid4())

        with pytest.raises(FrozenInstanceError):
            attr.session_id = uuid.uuid4()  # type: ignore[misc]

    def test_cannot_modify_message_id(self) -> None:
        """Attempting to modify message_id raises FrozenInstanceError."""
        attr = _make_attribution(message_id=uuid.uuid4())

        with pytest.raises(FrozenInstanceError):
            attr.message_id = uuid.uuid4()  # type: ignore[misc]

    def test_cannot_modify_document_id(self) -> None:
        """Attempting to modify document_id raises FrozenInstanceError."""
        attr = _make_attribution(document_id=uuid.uuid4())

        with pytest.raises(FrozenInstanceError):
            attr.document_id = uuid.uuid4()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# record — happy path
# ---------------------------------------------------------------------------


class TestRecord:
    """Tests for TokenUsageService.record()."""

    @pytest.mark.asyncio
    async def test_record_adds_entry_to_session(self) -> None:
        """Record creates a TokenUsageLog and adds it to the DB session."""
        db = _mock_session()
        service = TokenUsageService(db=db)
        attr = _make_attribution(
            session_id=uuid.uuid4(),
            message_id=uuid.uuid4(),
        )

        await service.record(
            capability=ModelCapability.INTENT_ANALYSIS,
            provider="anthropic",
            model_id="claude-sonnet-4-20250514",
            input_tokens=150,
            output_tokens=50,
            attribution=attr,
        )

        db.add.assert_called_once()
        entry = db.add.call_args[0][0]
        assert entry.capability == ModelCapability.INTENT_ANALYSIS
        assert entry.provider == "anthropic"
        assert entry.model_id == "claude-sonnet-4-20250514"
        assert entry.input_tokens == 150
        assert entry.output_tokens == 50
        assert entry.user_id == attr.user_id
        assert entry.session_id == attr.session_id
        assert entry.message_id == attr.message_id
        assert entry.document_id is None

    @pytest.mark.asyncio
    async def test_record_flushes_session(self) -> None:
        """Record flushes the session to persist the entry."""
        db = _mock_session()
        service = TokenUsageService(db=db)

        await service.record(
            capability=ModelCapability.ANSWER_SYNTHESIS,
            provider="openai",
            model_id="gpt-4o",
            input_tokens=200,
            output_tokens=100,
            attribution=_make_attribution(),
        )

        db.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_record_with_document_attribution(self) -> None:
        """Record with document context (ingestion scenario)."""
        db = _mock_session()
        service = TokenUsageService(db=db)
        did = uuid.uuid4()
        attr = _make_attribution(document_id=did)

        await service.record(
            capability=ModelCapability.CONCEPT_EXTRACTION,
            provider="openai",
            model_id="gpt-4o-mini",
            input_tokens=500,
            output_tokens=200,
            attribution=attr,
        )

        entry = db.add.call_args[0][0]
        assert entry.document_id == did
        assert entry.session_id is None
        assert entry.message_id is None

    @pytest.mark.asyncio
    async def test_record_all_capabilities(self) -> None:
        """Record works for every ModelCapability value."""
        db = _mock_session()
        service = TokenUsageService(db=db)

        for capability in ModelCapability:
            await service.record(
                capability=capability,
                provider="test",
                model_id="test-model",
                input_tokens=10,
                output_tokens=5,
                attribution=_make_attribution(),
            )

        assert db.add.call_count == len(ModelCapability)


# ---------------------------------------------------------------------------
# get_query_total
# ---------------------------------------------------------------------------


class TestGetQueryTotal:
    """Tests for TokenUsageService.get_query_total()."""

    @pytest.mark.asyncio
    async def test_returns_sum_of_tokens(self) -> None:
        """Returns total tokens for a message."""
        db = _mock_session(scalar_result=1500)
        service = TokenUsageService(db=db)
        sid = uuid.uuid4()
        mid = uuid.uuid4()

        total = await service.get_query_total(session_id=sid, message_id=mid)

        assert total == 1500
        db.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_entries(self) -> None:
        """Returns 0 when no token usage entries exist for the message."""
        db = _mock_session(scalar_result=0)
        service = TokenUsageService(db=db)

        total = await service.get_query_total(
            session_id=uuid.uuid4(),
            message_id=uuid.uuid4(),
        )

        assert total == 0

    @pytest.mark.asyncio
    async def test_return_type_is_int(self) -> None:
        """Result is always an integer."""
        db = _mock_session(scalar_result=42)
        service = TokenUsageService(db=db)

        total = await service.get_query_total(
            session_id=uuid.uuid4(),
            message_id=uuid.uuid4(),
        )

        assert isinstance(total, int)


# ---------------------------------------------------------------------------
# get_daily_total
# ---------------------------------------------------------------------------


class TestGetDailyTotal:
    """Tests for TokenUsageService.get_daily_total()."""

    @pytest.mark.asyncio
    async def test_returns_daily_sum(self) -> None:
        """Returns total tokens consumed today."""
        db = _mock_session(scalar_result=50000)
        service = TokenUsageService(db=db)

        total = await service.get_daily_total()

        assert total == 50000
        db.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_usage_today(self) -> None:
        """Returns 0 when no token usage has occurred today."""
        db = _mock_session(scalar_result=0)
        service = TokenUsageService(db=db)

        total = await service.get_daily_total()

        assert total == 0

    @pytest.mark.asyncio
    async def test_return_type_is_int(self) -> None:
        """Result is always an integer."""
        db = _mock_session(scalar_result=999)
        service = TokenUsageService(db=db)

        total = await service.get_daily_total()

        assert isinstance(total, int)


# ---------------------------------------------------------------------------
# get_session_query_count
# ---------------------------------------------------------------------------


class TestGetSessionQueryCount:
    """Tests for TokenUsageService.get_session_query_count()."""

    @pytest.mark.asyncio
    async def test_returns_query_count(self) -> None:
        """Returns number of user_question messages in a session."""
        db = _mock_session(scalar_result=7)
        service = TokenUsageService(db=db)
        sid = uuid.uuid4()

        count = await service.get_session_query_count(session_id=sid)

        assert count == 7
        db.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_session(self) -> None:
        """Returns 0 when no user_question messages exist."""
        db = _mock_session(scalar_result=0)
        service = TokenUsageService(db=db)

        count = await service.get_session_query_count(session_id=uuid.uuid4())

        assert count == 0

    @pytest.mark.asyncio
    async def test_return_type_is_int(self) -> None:
        """Result is always an integer."""
        db = _mock_session(scalar_result=3)
        service = TokenUsageService(db=db)

        count = await service.get_session_query_count(session_id=uuid.uuid4())

        assert isinstance(count, int)


# ---------------------------------------------------------------------------
# Multiple operations — independence
# ---------------------------------------------------------------------------


class TestMultipleOperations:
    """Tests that multiple service calls issue independent queries."""

    @pytest.mark.asyncio
    async def test_successive_records_each_add_and_flush(self) -> None:
        """Each record call adds a new entry and flushes."""
        db = _mock_session()
        service = TokenUsageService(db=db)

        for _ in range(3):
            await service.record(
                capability=ModelCapability.ANSWER_SYNTHESIS,
                provider="openai",
                model_id="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                attribution=_make_attribution(),
            )

        assert db.add.call_count == 3
        assert db.flush.await_count == 3

    @pytest.mark.asyncio
    async def test_successive_queries_each_execute(self) -> None:
        """Each aggregation call issues its own query."""
        db = _mock_session(scalar_result=0)
        service = TokenUsageService(db=db)

        await service.get_query_total(uuid.uuid4(), uuid.uuid4())
        await service.get_daily_total()
        await service.get_session_query_count(uuid.uuid4())

        assert db.execute.await_count == 3
