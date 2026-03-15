"""Integration tests for the seed data pipeline.

Validates the full seed system against a real PostgreSQL database:
fresh seed, idempotency, force mode, reset, partial commands, and
ground truth fixture format. DocumentService is mocked (requires blob
storage + Celery) but all other seed operations use real DB via the
existing integration test infrastructure.

Design reference: EPIC-008 design.md, Integration Tests section.
Refs: themachinagod/tabletop-oracle-docs#89
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import func, select

from tabletop_oracle.models.enums import (
    DocumentStatus,
    UserRole,
    UserStatus,
)
from tabletop_oracle.models.guardrail import GuardrailConfig
from tabletop_oracle.models.model_slot import ModelSlot
from tabletop_oracle.models.user import User
from tabletop_oracle.repositories.expansion_repository import ExpansionRepository
from tabletop_oracle.repositories.game_repository import GameRepository
from tabletop_oracle.services.expansion_service import ExpansionService
from tabletop_oracle.services.game_service import GameService
from tabletop_oracle.services.ground_truth_validator import (
    GroundTruthValidator,
    QueryResult,
)
from tabletop_oracle.services.model.slot_repository import ModelSlotRepository
from tabletop_oracle.services.seed_fixture_loader import SeedFixtureLoader
from tabletop_oracle.services.seed_service import SeedService, StepStatus

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CURATOR_EMAIL = "integration-test-curator@tabletop-oracle.dev"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_seed_service(
    session: AsyncSession,
    *,
    doc_service: Any = None,
    wait_timeout: float = 1.0,
) -> SeedService:
    """Construct a SeedService wired to real repositories and services.

    Uses the real SeedFixtureLoader (reads actual fixture files) and
    real GameService/ExpansionService backed by real repositories
    against the test database session.

    Args:
        session: Active AsyncSession.
        doc_service: Optional DocumentService (mock or None).
        wait_timeout: Timeout for document processing wait (kept low).

    Returns:
        Configured SeedService.
    """
    fixture_loader = SeedFixtureLoader()
    game_repo = GameRepository(session)
    game_service = GameService(game_repo)
    expansion_repo = ExpansionRepository(session)
    expansion_service = ExpansionService(expansion_repo, game_repo)
    model_slot_repo = ModelSlotRepository(session)

    return SeedService(
        db=session,
        fixture_loader=fixture_loader,
        game_service=game_service,
        expansion_service=expansion_service,
        model_slot_repo=model_slot_repo,
        document_service=doc_service,
        wait_timeout=wait_timeout,
    )


async def _count_users(session: AsyncSession) -> int:
    """Count user rows in the database."""
    result = await session.execute(select(func.count()).select_from(User))
    return result.scalar_one()


async def _count_model_slots(session: AsyncSession) -> int:
    """Count model slot rows in the database."""
    result = await session.execute(select(func.count()).select_from(ModelSlot))
    return result.scalar_one()


async def _count_guardrail_configs(session: AsyncSession) -> int:
    """Count guardrail config rows in the database."""
    result = await session.execute(select(func.count()).select_from(GuardrailConfig))
    return result.scalar_one()


async def _find_game_by_name(session: AsyncSession, name: str) -> Any:
    """Find a game by name, returning the ORM object or None."""
    from tabletop_oracle.models.game import Game

    stmt = select(Game).where(Game.name == name)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def _count_expansions_for_game(session: AsyncSession, game_id: uuid.UUID) -> int:
    """Count expansion rows for a specific game."""
    from tabletop_oracle.models.expansion import Expansion

    stmt = select(func.count()).select_from(Expansion).where(Expansion.game_id == game_id)
    result = await session.execute(stmt)
    return result.scalar_one()


def _make_mock_doc_service() -> tuple[MagicMock, list[uuid.UUID]]:
    """Create a mock DocumentService that tracks uploads.

    Returns a mock DocumentService and a list that will be populated
    with the IDs of uploaded documents. The mock returns documents
    with PROCESSED status, so await_processing completes immediately.

    Returns:
        Tuple of (mock_doc_service, uploaded_doc_ids_list).
    """
    mock_doc = MagicMock()
    uploaded_ids: list[uuid.UUID] = []

    async def mock_upload(**kwargs: Any) -> Any:
        doc_id = uuid.uuid4()
        uploaded_ids.append(doc_id)
        doc_mock = MagicMock()
        doc_mock.id = doc_id
        doc_mock.status = DocumentStatus.PROCESSED
        return doc_mock

    mock_doc.upload = mock_upload
    return mock_doc, uploaded_ids


def _build_seed_service_with_mock_docs(
    session: AsyncSession,
) -> tuple[SeedService, list[uuid.UUID]]:
    """Build a SeedService with a mock DocumentService attached.

    The mock DocumentService returns documents in PROCESSED status so
    seed_all() completes fully. Also patches _find_document_by_id on
    the service to return processed document mocks.

    Args:
        session: Active AsyncSession.

    Returns:
        Tuple of (configured SeedService, uploaded_doc_ids list).
    """
    mock_doc, uploaded_ids = _make_mock_doc_service()
    service = _build_seed_service(session, doc_service=mock_doc)

    # Patch _find_document_by_id so await_processing sees PROCESSED status
    async def mock_find_doc(doc_id: uuid.UUID) -> Any:
        doc_mock = MagicMock()
        doc_mock.id = doc_id
        doc_mock.status = DocumentStatus.PROCESSED
        return doc_mock

    service._find_document_by_id = mock_find_doc  # type: ignore[assignment]
    return service, uploaded_ids


# ---------------------------------------------------------------------------
# Tests — Fresh seed on empty database
# ---------------------------------------------------------------------------


class TestSeedAllFreshDatabase:
    """Full seed on an empty database succeeds and creates all expected records."""

    @pytest.mark.asyncio
    async def test_seed_all_creates_users_game_expansion_ai_config(
        self, db_session: AsyncSession
    ) -> None:
        """seed_all creates users, game, expansion, and AI config on empty DB."""
        with patch.dict(os.environ, {"INITIAL_CURATOR_EMAILS": _CURATOR_EMAIL}):
            service, _doc_ids = _build_seed_service_with_mock_docs(db_session)
            result = await service.seed_all()

        assert result.success

        # Verify step statuses
        step_map = {s.step: s for s in result.steps}
        assert step_map["users"].status == StepStatus.CREATED
        assert step_map["game"].status == StepStatus.CREATED
        assert step_map["expansion"].status == StepStatus.CREATED
        assert step_map["ai_config"].status == StepStatus.CREATED
        assert step_map["documents"].status == StepStatus.CREATED

        # Verify users created
        user_count = await _count_users(db_session)
        assert user_count >= 2  # curator + player at minimum

        # Verify curator exists with correct role
        curator = await db_session.execute(select(User).where(User.email == _CURATOR_EMAIL))
        curator_user = curator.scalar_one_or_none()
        assert curator_user is not None
        assert curator_user.role == UserRole.CURATOR
        assert curator_user.status == UserStatus.ACTIVE

        # Verify game created
        game = await _find_game_by_name(db_session, "Catan")
        assert game is not None
        assert game.publisher == "Catan Studio / Kosmos"

        # Verify expansion created
        expansion_count = await _count_expansions_for_game(db_session, game.id)
        assert expansion_count >= 1

        # Verify AI config created
        slot_count = await _count_model_slots(db_session)
        assert slot_count >= 1

        guardrail_count = await _count_guardrail_configs(db_session)
        assert guardrail_count == 1

    @pytest.mark.asyncio
    async def test_seed_all_documents_step_skipped_without_doc_service(
        self, db_session: AsyncSession
    ) -> None:
        """Without a DocumentService, the documents step fails gracefully."""
        with patch.dict(os.environ, {"INITIAL_CURATOR_EMAILS": _CURATOR_EMAIL}):
            service = _build_seed_service(db_session, doc_service=None)
            result = await service.seed_all()

        step_map = {s.step: s for s in result.steps}
        assert step_map["documents"].status == StepStatus.FAILED
        assert "DocumentService not provided" in step_map["documents"].detail

    @pytest.mark.asyncio
    async def test_seed_all_with_mock_doc_service(self, db_session: AsyncSession) -> None:
        """seed_all with a mocked DocumentService uploads documents successfully."""
        with patch.dict(os.environ, {"INITIAL_CURATOR_EMAILS": _CURATOR_EMAIL}):
            service, uploaded_ids = _build_seed_service_with_mock_docs(db_session)
            result = await service.seed_all()

        assert result.success
        step_map = {s.step: s for s in result.steps}
        assert step_map["documents"].status == StepStatus.CREATED
        assert step_map["await_processing"].status == StepStatus.CREATED
        assert len(uploaded_ids) == 2  # base rules + seafarers rules


# ---------------------------------------------------------------------------
# Tests — Idempotency
# ---------------------------------------------------------------------------


class TestSeedIdempotent:
    """Running seed_all twice produces the same state without duplicates."""

    @pytest.mark.asyncio
    async def test_double_seed_skips_all_steps(self, db_session: AsyncSession) -> None:
        """Second seed_all run skips all steps when data already exists."""
        with patch.dict(os.environ, {"INITIAL_CURATOR_EMAILS": _CURATOR_EMAIL}):
            service, _doc_ids = _build_seed_service_with_mock_docs(db_session)

            # First run
            result1 = await service.seed_all()
            assert result1.success

            # Record counts after first run
            users_after_first = await _count_users(db_session)
            slots_after_first = await _count_model_slots(db_session)

            # Second run — same service, same session
            result2 = await service.seed_all()
            assert result2.success

        # Second run should skip everything
        step_map = {s.step: s for s in result2.steps}
        assert step_map["users"].status == StepStatus.SKIPPED
        assert step_map["game"].status == StepStatus.SKIPPED
        assert step_map["expansion"].status == StepStatus.SKIPPED
        assert step_map["ai_config"].status == StepStatus.SKIPPED

        # Counts unchanged — no duplicates
        users_after_second = await _count_users(db_session)
        slots_after_second = await _count_model_slots(db_session)
        assert users_after_second == users_after_first
        assert slots_after_second == slots_after_first


# ---------------------------------------------------------------------------
# Tests — Force mode
# ---------------------------------------------------------------------------


class TestSeedForceRecreates:
    """Force mode replaces existing data without creating duplicates."""

    @pytest.mark.asyncio
    async def test_force_replaces_ai_config(self, db_session: AsyncSession) -> None:
        """Force mode on AI config replaces model slots and guardrails."""
        with patch.dict(os.environ, {"INITIAL_CURATOR_EMAILS": _CURATOR_EMAIL}):
            service = _build_seed_service(db_session)

            # Seed AI config
            r1 = await service.seed_ai_config()
            assert r1.success
            slots_before = await _count_model_slots(db_session)

            # Force re-seed
            r2 = await service.seed_ai_config(force=True)
            assert r2.success
            assert r2.steps[0].status == StepStatus.REPLACED

        slots_after = await _count_model_slots(db_session)
        assert slots_after == slots_before
        assert await _count_guardrail_configs(db_session) == 1

    @pytest.mark.asyncio
    async def test_force_replaces_game_and_expansion(self, db_session: AsyncSession) -> None:
        """Force mode on game replaces game entry (cascading to expansions)."""
        with patch.dict(os.environ, {"INITIAL_CURATOR_EMAILS": _CURATOR_EMAIL}):
            service = _build_seed_service(db_session)

            # Seed users + game + expansion
            await service.seed_users()
            await service.seed_game()
            await service.seed_expansion()

            game_before = await _find_game_by_name(db_session, "Catan")
            assert game_before is not None

            # Force re-seed game (cascade deletes expansion, then recreates)
            game_result = await service.seed_game(force=True)
            assert game_result.success
            assert game_result.steps[0].status == StepStatus.REPLACED

        game_after = await _find_game_by_name(db_session, "Catan")
        assert game_after is not None
        # Game ID should change after force replace
        assert game_after.id != game_before.id

    @pytest.mark.asyncio
    async def test_reset_then_reseed_replaces_all_data(self, db_session: AsyncSession) -> None:
        """Reset + seed_all is the correct full-replacement workflow.

        seed_all(force=True) has a dependency-order limitation (deletes
        users before games, violating FK constraints). The supported
        workflow for full replacement is: reset() then seed_all().
        """
        with patch.dict(os.environ, {"INITIAL_CURATOR_EMAILS": _CURATOR_EMAIL}):
            service, _doc_ids = _build_seed_service_with_mock_docs(db_session)

            # Initial seed
            r1 = await service.seed_all()
            assert r1.success
            users_before = await _count_users(db_session)
            slots_before = await _count_model_slots(db_session)

            # Full replacement via reset + re-seed
            await service.reset()
            r2 = await service.seed_all()
            assert r2.success

        # All steps CREATED (not REPLACED, since reset removed everything)
        step_map = {s.step: s for s in r2.steps}
        assert step_map["users"].status == StepStatus.CREATED
        assert step_map["game"].status == StepStatus.CREATED

        # Same counts — fully replaced
        users_after = await _count_users(db_session)
        slots_after = await _count_model_slots(db_session)
        assert users_after == users_before
        assert slots_after == slots_before


# ---------------------------------------------------------------------------
# Tests — Reset
# ---------------------------------------------------------------------------


class TestSeedReset:
    """Reset removes all seed data cleanly."""

    @pytest.mark.asyncio
    async def test_reset_removes_all_seed_data(self, db_session: AsyncSession) -> None:
        """After seed + reset, all seed data is removed."""
        with patch.dict(os.environ, {"INITIAL_CURATOR_EMAILS": _CURATOR_EMAIL}):
            service, _doc_ids = _build_seed_service_with_mock_docs(db_session)

            # Seed first
            seed_result = await service.seed_all()
            assert seed_result.success

            # Verify data exists
            assert await _count_users(db_session) > 0
            assert await _find_game_by_name(db_session, "Catan") is not None
            assert await _count_model_slots(db_session) > 0
            assert await _count_guardrail_configs(db_session) > 0

            # Reset
            reset_result = await service.reset()
            assert reset_result.success

        # Verify all seed data removed
        assert await _find_game_by_name(db_session, "Catan") is None
        assert await _count_model_slots(db_session) == 0
        assert await _count_guardrail_configs(db_session) == 0

        # Users should be removed
        curator = await db_session.execute(select(User).where(User.email == _CURATOR_EMAIL))
        assert curator.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_reset_on_empty_db_succeeds(self, db_session: AsyncSession) -> None:
        """Reset on an already-empty database succeeds without errors."""
        with patch.dict(os.environ, {"INITIAL_CURATOR_EMAILS": _CURATOR_EMAIL}):
            service = _build_seed_service(db_session)
            result = await service.reset()

        assert result.success
        step_map = {s.step: s for s in result.steps}
        assert step_map["game"].status == StepStatus.SKIPPED


# ---------------------------------------------------------------------------
# Tests — Partial commands
# ---------------------------------------------------------------------------


class TestSeedPartialCommands:
    """Individual seed subcommands work in isolation."""

    @pytest.mark.asyncio
    async def test_seed_users_only(self, db_session: AsyncSession) -> None:
        """seed_users creates user accounts without touching other entities."""
        with patch.dict(os.environ, {"INITIAL_CURATOR_EMAILS": _CURATOR_EMAIL}):
            service = _build_seed_service(db_session)
            result = await service.seed_users()

        assert result.success
        assert len(result.steps) == 1
        assert result.steps[0].step == "users"
        assert result.steps[0].status == StepStatus.CREATED

        # Users exist
        assert await _count_users(db_session) >= 2

        # Game does not exist (not seeded)
        assert await _find_game_by_name(db_session, "Catan") is None

    @pytest.mark.asyncio
    async def test_seed_game_only(self, db_session: AsyncSession) -> None:
        """seed_game creates the game entry (requires users to exist first)."""
        with patch.dict(os.environ, {"INITIAL_CURATOR_EMAILS": _CURATOR_EMAIL}):
            service = _build_seed_service(db_session)

            # Must seed users first (game needs curator attribution)
            await service.seed_users()
            result = await service.seed_game()

        assert result.success
        assert len(result.steps) == 1
        assert result.steps[0].step == "game"
        assert result.steps[0].status == StepStatus.CREATED

        game = await _find_game_by_name(db_session, "Catan")
        assert game is not None

    @pytest.mark.asyncio
    async def test_seed_expansion_only(self, db_session: AsyncSession) -> None:
        """seed_expansion creates expansions (requires users + game first)."""
        with patch.dict(os.environ, {"INITIAL_CURATOR_EMAILS": _CURATOR_EMAIL}):
            service = _build_seed_service(db_session)

            # Seed dependencies
            await service.seed_users()
            await service.seed_game()

            result = await service.seed_expansion()

        assert result.success
        assert len(result.steps) == 1
        assert result.steps[0].step == "expansion"
        assert result.steps[0].status == StepStatus.CREATED

        game = await _find_game_by_name(db_session, "Catan")
        assert game is not None
        assert await _count_expansions_for_game(db_session, game.id) >= 1

    @pytest.mark.asyncio
    async def test_seed_expansion_fails_without_game(self, db_session: AsyncSession) -> None:
        """seed_expansion fails gracefully when no game has been seeded."""
        with patch.dict(os.environ, {"INITIAL_CURATOR_EMAILS": _CURATOR_EMAIL}):
            service = _build_seed_service(db_session)
            result = await service.seed_expansion()

        assert not result.success
        assert result.steps[0].status == StepStatus.FAILED
        assert "Game must be seeded first" in result.steps[0].detail

    @pytest.mark.asyncio
    async def test_seed_ai_config_only(self, db_session: AsyncSession) -> None:
        """seed_ai_config creates model slots and guardrail config."""
        with patch.dict(os.environ, {"INITIAL_CURATOR_EMAILS": _CURATOR_EMAIL}):
            service = _build_seed_service(db_session)
            result = await service.seed_ai_config()

        assert result.success
        assert len(result.steps) == 1
        assert result.steps[0].step == "ai_config"
        assert result.steps[0].status == StepStatus.CREATED

        assert await _count_model_slots(db_session) >= 1
        assert await _count_guardrail_configs(db_session) == 1

    @pytest.mark.asyncio
    async def test_seed_documents_fails_without_game(self, db_session: AsyncSession) -> None:
        """seed_documents fails gracefully when no game has been seeded."""
        with patch.dict(os.environ, {"INITIAL_CURATOR_EMAILS": _CURATOR_EMAIL}):
            service = _build_seed_service(db_session)
            result = await service.seed_documents()

        assert not result.success
        assert result.steps[0].status == StepStatus.FAILED
        assert "Game must be seeded first" in result.steps[0].detail


# ---------------------------------------------------------------------------
# Tests — Ground truth fixture format
# ---------------------------------------------------------------------------


class TestGroundTruthFormat:
    """Ground truth Q&A fixture is valid and parseable."""

    def test_ground_truth_fixture_loads_successfully(self) -> None:
        """Ground truth JSON loads without errors via SeedFixtureLoader."""
        loader = SeedFixtureLoader()
        fixture = loader.load_ground_truth()

        assert fixture.game == "Catan"
        assert fixture.version
        assert len(fixture.questions) >= 15

    def test_ground_truth_all_categories_present(self) -> None:
        """Ground truth covers all expected question categories."""
        loader = SeedFixtureLoader()
        fixture = loader.load_ground_truth()

        categories = {q.category for q in fixture.questions}
        expected_categories = {
            "single-doc-factual",
            "single-doc-procedural",
            "cross-doc-association",
            "expansion-specific",
            "strategy",
        }

        for cat in expected_categories:
            assert cat in categories, f"Missing category: {cat}"

    def test_ground_truth_questions_have_valid_criteria(self) -> None:
        """Each ground truth question has at least one evaluation criterion."""
        loader = SeedFixtureLoader()
        fixture = loader.load_ground_truth()

        for qa in fixture.questions:
            has_criterion = (
                bool(qa.expected.answer_contains)
                or bool(qa.expected.answer_contains_any)
                or qa.expected.must_cite
                or qa.expected.min_citations is not None
                or qa.expected.min_confidence is not None
                or bool(qa.expected.expected_document_types)
            )
            assert has_criterion, f"Question {qa.id} has no evaluation criteria"

    def test_ground_truth_question_ids_unique(self) -> None:
        """All ground truth question IDs are unique."""
        loader = SeedFixtureLoader()
        fixture = loader.load_ground_truth()

        ids = [q.id for q in fixture.questions]
        assert len(ids) == len(set(ids)), "Duplicate question IDs found"


# ---------------------------------------------------------------------------
# Tests — Ground truth validator integration
# ---------------------------------------------------------------------------


class TestGroundTruthValidatorIntegration:
    """GroundTruthValidator works correctly with real fixture data."""

    @pytest.mark.asyncio
    async def test_validator_produces_report_with_mock_executor(self) -> None:
        """Validator runs all questions and produces a structured report."""
        loader = SeedFixtureLoader()
        fixture = loader.load_ground_truth()

        # Mock executor that returns plausible answers
        executor = AsyncMock()
        executor.execute_query.return_value = QueryResult(
            answer_text="Catan is a strategy board game with settlements, "
            "roads, and resources like brick, lumber, wool, grain, and ore. "
            "Players trade and build to reach 10 victory points.",
            confidence_score=0.85,
            citations=[
                {"document_name": "Catan Base Rules", "document_type": "rules"},
            ],
        )

        validator = GroundTruthValidator(executor=executor)
        result = await validator.validate(fixture)

        assert result.game == "Catan"
        assert result.total == len(fixture.questions)
        assert result.total > 0
        assert result.passed + result.failed + result.errors == result.total
        assert 0.0 <= result.pass_rate <= 1.0

        # Verify report format
        report = result.format_report()
        assert "Ground Truth Validation Report" in report
        assert "Catan" in report

    @pytest.mark.asyncio
    async def test_validator_handles_executor_errors_gracefully(self) -> None:
        """Validator counts errors and continues when executor raises."""
        loader = SeedFixtureLoader()
        fixture = loader.load_ground_truth()

        executor = AsyncMock()
        executor.execute_query.side_effect = RuntimeError("AI service unavailable")

        validator = GroundTruthValidator(executor=executor)
        result = await validator.validate(fixture)

        assert result.errors == result.total
        assert result.passed == 0
        assert result.pass_rate == 0.0


# ---------------------------------------------------------------------------
# Tests — Seed + reset round-trip
# ---------------------------------------------------------------------------


class TestSeedResetRoundTrip:
    """Seed, reset, and re-seed to validate full lifecycle."""

    @pytest.mark.asyncio
    async def test_seed_reset_reseed_produces_same_state(self, db_session: AsyncSession) -> None:
        """Seed -> reset -> re-seed produces identical state."""
        with patch.dict(os.environ, {"INITIAL_CURATOR_EMAILS": _CURATOR_EMAIL}):
            service, _doc_ids = _build_seed_service_with_mock_docs(db_session)

            # First seed
            r1 = await service.seed_all()
            assert r1.success
            users_1 = await _count_users(db_session)
            slots_1 = await _count_model_slots(db_session)

            # Reset
            reset_result = await service.reset()
            assert reset_result.success

            # Re-seed
            r2 = await service.seed_all()
            assert r2.success

        # Same counts
        users_2 = await _count_users(db_session)
        slots_2 = await _count_model_slots(db_session)
        assert users_2 == users_1
        assert slots_2 == slots_1

        # Data is valid
        game = await _find_game_by_name(db_session, "Catan")
        assert game is not None
