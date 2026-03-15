"""Unit tests for SeedService orchestrator.

Tests all seed methods with fully mocked dependencies: database session,
fixture loader, GameService, ExpansionService, ModelSlotRepository.
Covers idempotency, force mode, reset, individual step methods, and
error handling.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.models.enums import ModelCapability, UserRole, UserStatus
from tabletop_oracle.services.seed_fixture_loader import (
    DocumentMetaFixture,
    ExpansionFixture,
    GameFixture,
    GuardrailFixture,
    ModelSlotFixture,
    ModelSlotsFixture,
    UserFixture,
)
from tabletop_oracle.services.seed_service import (
    SeedResult,
    SeedService,
    StepStatus,
    _FixtureUploadFile,
)

# ---------------------------------------------------------------------------
# Factories — fixture data
# ---------------------------------------------------------------------------


def _user_fixture(
    email: str = "curator@test.dev",
    display_name: str = "Test Curator",
    role: str = "curator",
    **overrides: Any,
) -> UserFixture:
    """Build a UserFixture with sensible defaults."""
    defaults = {
        "email": email,
        "display_name": display_name,
        "oauth_provider": None,
        "oauth_subject_id": None,
        "role": role,
        "status": "active",
    }
    defaults.update(overrides)
    return UserFixture(**defaults)


def _game_fixture(**overrides: Any) -> GameFixture:
    """Build a GameFixture with sensible defaults."""
    defaults = {
        "name": "Catan",
        "publisher": "Catan Studio",
        "year_published": 1995,
        "edition": "5th Edition",
        "min_players": 3,
        "max_players": 4,
        "description": "Trade and build.",
        "cover_image_url": None,
        "complexity": "medium",
        "tags": ["strategy"],
    }
    defaults.update(overrides)
    return GameFixture(**defaults)


def _expansion_fixture(**overrides: Any) -> ExpansionFixture:
    """Build an ExpansionFixture with sensible defaults."""
    defaults = {
        "name": "Catan: Seafarers",
        "year_published": 1997,
        "description": "Explore the seas.",
    }
    defaults.update(overrides)
    return ExpansionFixture(**defaults)


def _model_slot_fixture(**overrides: Any) -> ModelSlotFixture:
    """Build a ModelSlotFixture with sensible defaults."""
    defaults = {
        "capability": "intent_analysis",
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "max_tokens_per_call": 4096,
        "temperature": 0.1,
        "fallback_provider": None,
        "fallback_model_id": None,
    }
    defaults.update(overrides)
    return ModelSlotFixture(**defaults)


def _guardrail_fixture(**overrides: Any) -> GuardrailFixture:
    """Build a GuardrailFixture with sensible defaults."""
    defaults = {
        "enforcement_enabled": True,
        "max_tokens_per_query": 8000,
        "max_model_calls_per_query": 5,
        "max_queries_per_session": 50,
        "daily_token_budget": 500000,
        "daily_query_budget": 1000,
        "per_document_ingestion_limit": 100000,
    }
    defaults.update(overrides)
    return GuardrailFixture(**defaults)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _mock_user(
    email: str = "curator@test.dev",
    role: UserRole = UserRole.CURATOR,
) -> MagicMock:
    """Build a mock User ORM entity."""
    user = MagicMock()
    user.id = uuid.uuid4()
    user.email = email
    user.role = role
    user.status = UserStatus.ACTIVE
    return user


def _mock_game(name: str = "Catan") -> MagicMock:
    """Build a mock Game ORM entity."""
    game = MagicMock()
    game.id = uuid.uuid4()
    game.name = name
    return game


def _mock_expansion(name: str = "Catan: Seafarers") -> MagicMock:
    """Build a mock Expansion ORM entity."""
    exp = MagicMock()
    exp.id = uuid.uuid4()
    exp.name = name
    return exp


def _mock_model_slot(capability: str = "intent_analysis") -> MagicMock:
    """Build a mock ModelSlot ORM entity."""
    slot = MagicMock()
    slot.id = uuid.uuid4()
    slot.capability = ModelCapability(capability)
    return slot


def _mock_guardrail_config() -> MagicMock:
    """Build a mock GuardrailConfig ORM entity."""
    config = MagicMock()
    config.id = uuid.uuid4()
    config.enforcement_enabled = True
    return config


class _DBResult:
    """Minimal stand-in for SQLAlchemy result."""

    def __init__(self, value: Any = None) -> None:
        self._value = value

    def scalar_one_or_none(self) -> Any:
        return self._value

    def scalars(self) -> _DBResult:
        return self

    def first(self) -> Any:
        return self._value


def _doc_meta_fixture(**overrides: Any) -> DocumentMetaFixture:
    """Build a DocumentMetaFixture with sensible defaults."""
    defaults = {
        "name": "Catan Base Game Rules",
        "type": "core_rules",
        "format": "pdf",
        "file": "catan-base-rules.pdf",
        "expansion_id": None,
        "expansion_association": None,
    }
    defaults.update(overrides)
    return DocumentMetaFixture(**defaults)


def _mock_document(
    name: str = "Catan Base Game Rules",
    status: str = "uploaded",
) -> MagicMock:
    """Build a mock Document ORM entity."""
    doc = MagicMock()
    doc.id = uuid.uuid4()
    doc.name = name
    doc.status = MagicMock()
    doc.status.value = status
    # Make status comparable with DocumentStatus enum values
    from tabletop_oracle.models.enums import DocumentStatus

    doc.status = DocumentStatus(status)
    return doc


def _build_service(
    db: AsyncMock | None = None,
    loader: MagicMock | None = None,
    game_service: AsyncMock | None = None,
    expansion_service: AsyncMock | None = None,
    slot_repo: AsyncMock | None = None,
    document_service: AsyncMock | None = None,
    wait_timeout: float = 300.0,
) -> SeedService:
    """Build a SeedService with mocked dependencies."""
    return SeedService(
        db=db or AsyncMock(),
        fixture_loader=loader or MagicMock(),
        game_service=game_service or AsyncMock(),
        expansion_service=expansion_service or AsyncMock(),
        model_slot_repo=slot_repo or AsyncMock(),
        document_service=document_service,
        wait_timeout=wait_timeout,
    )


# ---------------------------------------------------------------------------
# SeedResult dataclass tests
# ---------------------------------------------------------------------------


class TestSeedResult:
    """Tests for the SeedResult data structure."""

    def test_empty_result_is_success(self) -> None:
        result = SeedResult()
        assert result.success is True

    def test_success_with_created_steps(self) -> None:
        result = SeedResult()
        result.add("users", StepStatus.CREATED, "Created 2 users")
        result.add("game", StepStatus.SKIPPED, "Already exists")
        assert result.success is True

    def test_failure_makes_success_false(self) -> None:
        result = SeedResult()
        result.add("users", StepStatus.CREATED)
        result.add("game", StepStatus.FAILED, "Error")
        assert result.success is False

    def test_add_appends_step(self) -> None:
        result = SeedResult()
        result.add("users", StepStatus.CREATED, "detail")
        assert len(result.steps) == 1
        assert result.steps[0].step == "users"
        assert result.steps[0].status == StepStatus.CREATED
        assert result.steps[0].detail == "detail"


# ---------------------------------------------------------------------------
# seed_all — empty database
# ---------------------------------------------------------------------------


class TestSeedAllEmptyDB:
    """seed_all on an empty database creates all records."""

    @pytest.fixture()
    def db(self) -> AsyncMock:
        mock = AsyncMock()
        # Default: all lookups return None (empty DB)
        mock.execute = AsyncMock(return_value=_DBResult(None))
        mock.add = MagicMock()
        mock.flush = AsyncMock()
        mock.delete = AsyncMock()
        return mock

    @pytest.fixture()
    def loader(self) -> MagicMock:
        ldr = MagicMock()
        ldr.load_users.return_value = [
            _user_fixture(email="curator@test.dev", role="curator"),
            _user_fixture(email="player@test.dev", role="player", display_name="Player"),
        ]
        ldr.load_game.return_value = _game_fixture()
        ldr.load_expansions.return_value = [_expansion_fixture()]
        ldr.load_ai_config.return_value = (
            ModelSlotsFixture(slots=[_model_slot_fixture()]),
            _guardrail_fixture(),
        )
        ldr.load_document_metas.return_value = [_doc_meta_fixture()]
        return ldr

    @pytest.fixture()
    def game_service(self) -> AsyncMock:
        svc = AsyncMock()
        svc.create_game.return_value = _mock_game()
        return svc

    @pytest.fixture()
    def expansion_service(self) -> AsyncMock:
        svc = AsyncMock()
        svc.create_expansion.return_value = _mock_expansion()
        return svc

    @pytest.fixture()
    def slot_repo(self) -> AsyncMock:
        repo = AsyncMock()
        repo.get_by_capability.return_value = None
        return repo

    async def test_seed_all_creates_everything(
        self,
        db: AsyncMock,
        loader: MagicMock,
        game_service: AsyncMock,
        expansion_service: AsyncMock,
        slot_repo: AsyncMock,
    ) -> None:
        # The DB needs to return a curator when searching for curator user
        # after user creation. We use side_effect to sequence DB calls.
        curator = _mock_user()
        call_count = 0

        async def execute_side_effect(stmt: Any) -> _DBResult:
            nonlocal call_count
            call_count += 1
            return _DBResult(None)

        db.execute = AsyncMock(side_effect=execute_side_effect)

        # Mock DocumentService
        uploaded_doc = _mock_document(status="processed")
        doc_service = AsyncMock()
        doc_service.upload = AsyncMock(return_value=uploaded_doc)

        service = _build_service(
            db,
            loader,
            game_service,
            expansion_service,
            slot_repo,
            document_service=doc_service,
            wait_timeout=0.01,
        )

        # Override lookups to make them predictable
        service._find_user_by_email = AsyncMock(return_value=None)  # type: ignore[method-assign]
        service._get_curator_user = AsyncMock(return_value=curator)  # type: ignore[method-assign]
        service._find_game_by_name = AsyncMock(return_value=None)  # type: ignore[method-assign]
        service._find_expansion_by_name = AsyncMock(return_value=None)  # type: ignore[method-assign]
        service._find_guardrail_config = AsyncMock(return_value=None)  # type: ignore[method-assign]
        service._find_document_by_name = AsyncMock(return_value=None)  # type: ignore[method-assign]
        service._find_document_by_id = AsyncMock(return_value=uploaded_doc)  # type: ignore[method-assign]
        service._resolve_fixture_pdf = MagicMock(  # type: ignore[method-assign]
            return_value=MagicMock(name="fake.pdf", read_bytes=MagicMock(return_value=b"fake")),
        )

        result = await service.seed_all()

        assert result.success is True
        step_names = [s.step for s in result.steps]
        assert "users" in step_names
        assert "game" in step_names
        assert "expansion" in step_names
        assert "ai_config" in step_names
        assert "documents" in step_names
        assert "await_processing" in step_names

        # Verify services were called
        game_service.create_game.assert_called_once()
        expansion_service.create_expansion.assert_called_once()
        doc_service.upload.assert_called_once()


# ---------------------------------------------------------------------------
# seed_all — idempotent re-run
# ---------------------------------------------------------------------------


class TestSeedAllIdempotent:
    """seed_all re-run skips all steps when data exists."""

    async def test_seed_all_skips_existing(self) -> None:
        db = AsyncMock()
        loader = MagicMock()

        curator = _mock_user()
        existing_game = _mock_game()
        existing_expansion = _mock_expansion()
        existing_slot = _mock_model_slot()
        existing_config = _mock_guardrail_config()
        existing_doc = _mock_document()

        loader.load_users.return_value = [_user_fixture()]
        loader.load_game.return_value = _game_fixture()
        loader.load_expansions.return_value = [_expansion_fixture()]
        loader.load_ai_config.return_value = (
            ModelSlotsFixture(slots=[_model_slot_fixture()]),
            _guardrail_fixture(),
        )
        loader.load_document_metas.return_value = [_doc_meta_fixture()]

        game_service = AsyncMock()
        expansion_service = AsyncMock()
        doc_service = AsyncMock()
        slot_repo = AsyncMock()
        slot_repo.get_by_capability.return_value = existing_slot

        service = _build_service(
            db,
            loader,
            game_service,
            expansion_service,
            slot_repo,
            document_service=doc_service,
        )

        # All lookups find existing data
        service._find_user_by_email = AsyncMock(return_value=curator)  # type: ignore[method-assign]
        service._get_curator_user = AsyncMock(return_value=curator)  # type: ignore[method-assign]
        service._find_game_by_name = AsyncMock(return_value=existing_game)  # type: ignore[method-assign]
        service._find_expansion_by_name = AsyncMock(return_value=existing_expansion)  # type: ignore[method-assign]
        service._find_guardrail_config = AsyncMock(return_value=existing_config)  # type: ignore[method-assign]
        service._find_document_by_name = AsyncMock(return_value=existing_doc)  # type: ignore[method-assign]

        result = await service.seed_all()

        assert result.success is True
        # Check all data steps were skipped
        for step in result.steps:
            assert step.status == StepStatus.SKIPPED, f"{step.step} was {step.status}"

        # Services should NOT have been called
        game_service.create_game.assert_not_called()
        expansion_service.create_expansion.assert_not_called()
        doc_service.upload.assert_not_called()


# ---------------------------------------------------------------------------
# seed_all — force mode
# ---------------------------------------------------------------------------


class TestSeedAllForceMode:
    """Force mode replaces existing data."""

    async def test_force_replaces_existing_data(self) -> None:
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()
        db.delete = AsyncMock()

        loader = MagicMock()

        curator = _mock_user()
        existing_game = _mock_game()
        new_game = _mock_game()
        existing_expansion = _mock_expansion()
        existing_slot = _mock_model_slot()
        existing_config = _mock_guardrail_config()
        existing_doc = _mock_document()
        new_doc = _mock_document(status="processed")

        loader.load_users.return_value = [_user_fixture()]
        loader.load_game.return_value = _game_fixture()
        loader.load_expansions.return_value = [_expansion_fixture()]
        loader.load_ai_config.return_value = (
            ModelSlotsFixture(slots=[_model_slot_fixture()]),
            _guardrail_fixture(),
        )
        loader.load_document_metas.return_value = [_doc_meta_fixture()]

        game_service = AsyncMock()
        game_service.create_game.return_value = new_game
        expansion_service = AsyncMock()
        expansion_service.create_expansion.return_value = _mock_expansion()
        doc_service = AsyncMock()
        doc_service.upload = AsyncMock(return_value=new_doc)
        slot_repo = AsyncMock()
        slot_repo.get_by_capability.return_value = existing_slot

        service = _build_service(
            db,
            loader,
            game_service,
            expansion_service,
            slot_repo,
            document_service=doc_service,
            wait_timeout=0.01,
        )

        # All lookups find existing data (will be replaced)
        service._find_user_by_email = AsyncMock(return_value=curator)  # type: ignore[method-assign]
        service._get_curator_user = AsyncMock(return_value=curator)  # type: ignore[method-assign]
        service._find_game_by_name = AsyncMock(return_value=existing_game)  # type: ignore[method-assign]
        service._find_expansion_by_name = AsyncMock(return_value=existing_expansion)  # type: ignore[method-assign]
        service._find_guardrail_config = AsyncMock(return_value=existing_config)  # type: ignore[method-assign]
        service._find_document_by_name = AsyncMock(return_value=existing_doc)  # type: ignore[method-assign]
        service._find_document_by_id = AsyncMock(return_value=new_doc)  # type: ignore[method-assign]
        service._resolve_fixture_pdf = MagicMock(  # type: ignore[method-assign]
            return_value=MagicMock(name="fake.pdf", read_bytes=MagicMock(return_value=b"fake")),
        )

        result = await service.seed_all(force=True)

        assert result.success is True

        # Verify deletion happened (force mode deletes before re-creating)
        assert db.delete.call_count >= 1

        # Services should have been called to create new records
        game_service.create_game.assert_called_once()
        expansion_service.create_expansion.assert_called_once()
        doc_service.upload.assert_called_once()

        # Check statuses reflect replacement
        game_step = next(s for s in result.steps if s.step == "game")
        assert game_step.status == StepStatus.REPLACED
        doc_step = next(s for s in result.steps if s.step == "documents")
        assert doc_step.status == StepStatus.REPLACED


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    """Reset removes all seed data."""

    async def test_reset_deletes_everything(self) -> None:
        db = AsyncMock()
        db.execute = AsyncMock(return_value=_DBResult(None))
        db.delete = AsyncMock()
        db.flush = AsyncMock()

        loader = MagicMock()
        loader.load_game.return_value = _game_fixture()
        loader.load_users.return_value = [
            _user_fixture(email="curator@test.dev"),
            _user_fixture(email="player@test.dev", role="player"),
        ]

        existing_game = _mock_game()
        curator = _mock_user(email="curator@test.dev")
        player = _mock_user(email="player@test.dev", role=UserRole.PLAYER)

        service = _build_service(db, loader)
        service._find_game_by_name = AsyncMock(return_value=existing_game)  # type: ignore[method-assign]

        # Return different users for sequential calls
        service._find_user_by_email = AsyncMock(  # type: ignore[method-assign]
            side_effect=[curator, player],
        )

        result = await service.reset()

        assert result.success is True
        # Verify delete was called for bulk operations
        assert db.execute.call_count >= 2  # delete(GuardrailConfig) + delete(ModelSlot)
        # Verify individual deletes for game and users
        assert db.delete.call_count >= 3  # game + 2 users


# ---------------------------------------------------------------------------
# Individual step methods
# ---------------------------------------------------------------------------


class TestSeedUsersOnly:
    """seed_users works independently."""

    async def test_seed_users_creates_users(self) -> None:
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        loader = MagicMock()
        loader.load_users.return_value = [
            _user_fixture(email="curator@test.dev"),
            _user_fixture(email="player@test.dev", role="player"),
        ]

        service = _build_service(db, loader)
        service._find_user_by_email = AsyncMock(return_value=None)  # type: ignore[method-assign]

        result = await service.seed_users()

        assert result.success is True
        assert len(result.steps) == 1
        assert result.steps[0].status == StepStatus.CREATED
        assert db.add.call_count == 2


class TestSeedGameOnly:
    """seed_game works independently."""

    async def test_seed_game_creates_game(self) -> None:
        db = AsyncMock()
        loader = MagicMock()
        loader.load_game.return_value = _game_fixture()

        new_game = _mock_game()
        game_service = AsyncMock()
        game_service.create_game.return_value = new_game
        curator = _mock_user()

        service = _build_service(db, loader, game_service)
        service._find_game_by_name = AsyncMock(return_value=None)  # type: ignore[method-assign]
        service._get_curator_user = AsyncMock(return_value=curator)  # type: ignore[method-assign]

        result = await service.seed_game()

        assert result.success is True
        assert result.steps[0].status == StepStatus.CREATED
        game_service.create_game.assert_called_once()


class TestSeedExpansionOnly:
    """seed_expansion works independently."""

    async def test_seed_expansion_requires_game(self) -> None:
        db = AsyncMock()
        loader = MagicMock()
        loader.load_game.return_value = _game_fixture()

        service = _build_service(db, loader)
        service._find_game_by_name = AsyncMock(return_value=None)  # type: ignore[method-assign]

        result = await service.seed_expansion()

        assert result.success is False
        assert result.steps[0].status == StepStatus.FAILED
        assert "Game must be seeded first" in result.steps[0].detail

    async def test_seed_expansion_creates_expansion(self) -> None:
        db = AsyncMock()
        loader = MagicMock()
        loader.load_game.return_value = _game_fixture()
        loader.load_expansions.return_value = [_expansion_fixture()]

        existing_game = _mock_game()
        curator = _mock_user()
        expansion_service = AsyncMock()
        expansion_service.create_expansion.return_value = _mock_expansion()

        service = _build_service(db, loader, expansion_service=expansion_service)
        service._find_game_by_name = AsyncMock(return_value=existing_game)  # type: ignore[method-assign]
        service._get_curator_user = AsyncMock(return_value=curator)  # type: ignore[method-assign]
        service._find_expansion_by_name = AsyncMock(return_value=None)  # type: ignore[method-assign]

        result = await service.seed_expansion()

        assert result.success is True
        assert result.steps[0].status == StepStatus.CREATED
        expansion_service.create_expansion.assert_called_once()


class TestSeedAiConfigOnly:
    """seed_ai_config works independently."""

    async def test_seed_ai_config_creates_config(self) -> None:
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        loader = MagicMock()
        loader.load_ai_config.return_value = (
            ModelSlotsFixture(slots=[_model_slot_fixture()]),
            _guardrail_fixture(),
        )

        slot_repo = AsyncMock()
        slot_repo.get_by_capability.return_value = None

        service = _build_service(db, loader, slot_repo=slot_repo)
        service._find_guardrail_config = AsyncMock(return_value=None)  # type: ignore[method-assign]

        result = await service.seed_ai_config()

        assert result.success is True
        assert result.steps[0].status == StepStatus.CREATED
        assert db.add.call_count == 2  # 1 slot + 1 guardrail config


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Partial failures don't crash the entire seed process."""

    async def test_game_creation_error_is_captured(self) -> None:
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        loader = MagicMock()
        loader.load_users.return_value = [_user_fixture()]
        loader.load_game.return_value = _game_fixture()
        loader.load_expansions.return_value = [_expansion_fixture()]
        loader.load_ai_config.return_value = (
            ModelSlotsFixture(slots=[_model_slot_fixture()]),
            _guardrail_fixture(),
        )

        game_service = AsyncMock()
        game_service.create_game.side_effect = RuntimeError("DB connection failed")

        curator = _mock_user()
        slot_repo = AsyncMock()
        slot_repo.get_by_capability.return_value = None

        service = _build_service(db, loader, game_service, slot_repo=slot_repo)
        service._find_user_by_email = AsyncMock(return_value=None)  # type: ignore[method-assign]
        service._get_curator_user = AsyncMock(return_value=curator)  # type: ignore[method-assign]
        service._find_game_by_name = AsyncMock(return_value=None)  # type: ignore[method-assign]

        result = await service.seed_all()

        # Game step should fail, but users should succeed
        users_step = next(s for s in result.steps if s.step == "users")
        game_step = next(s for s in result.steps if s.step == "game")
        assert users_step.status == StepStatus.CREATED
        assert game_step.status == StepStatus.FAILED
        assert result.success is False

    async def test_no_curator_fails_game_step(self) -> None:
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        loader = MagicMock()
        loader.load_users.return_value = [_user_fixture()]
        loader.load_game.return_value = _game_fixture()

        game_service = AsyncMock()

        service = _build_service(db, loader, game_service)
        service._find_user_by_email = AsyncMock(return_value=None)  # type: ignore[method-assign]
        service._get_curator_user = AsyncMock(return_value=None)  # type: ignore[method-assign]
        service._find_game_by_name = AsyncMock(return_value=None)  # type: ignore[method-assign]

        result = await service.seed_all()

        game_step = next(s for s in result.steps if s.step == "game")
        assert game_step.status == StepStatus.FAILED
        assert "curator" in game_step.detail.lower()

    async def test_expansion_error_does_not_crash_seed(self) -> None:
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        loader = MagicMock()
        loader.load_users.return_value = [_user_fixture()]
        loader.load_game.return_value = _game_fixture()
        loader.load_expansions.return_value = [_expansion_fixture()]
        loader.load_ai_config.return_value = (
            ModelSlotsFixture(slots=[_model_slot_fixture()]),
            _guardrail_fixture(),
        )

        new_game = _mock_game()
        game_service = AsyncMock()
        game_service.create_game.return_value = new_game

        expansion_service = AsyncMock()
        expansion_service.create_expansion.side_effect = RuntimeError("Boom")

        curator = _mock_user()
        slot_repo = AsyncMock()
        slot_repo.get_by_capability.return_value = None

        service = _build_service(db, loader, game_service, expansion_service, slot_repo)
        service._find_user_by_email = AsyncMock(return_value=None)  # type: ignore[method-assign]
        service._get_curator_user = AsyncMock(return_value=curator)  # type: ignore[method-assign]
        service._find_game_by_name = AsyncMock(return_value=None)  # type: ignore[method-assign]
        service._find_expansion_by_name = AsyncMock(return_value=None)  # type: ignore[method-assign]
        service._find_guardrail_config = AsyncMock(return_value=None)  # type: ignore[method-assign]

        result = await service.seed_all()

        # Expansion fails but AI config still runs
        exp_step = next(s for s in result.steps if s.step == "expansion")
        ai_step = next(s for s in result.steps if s.step == "ai_config")
        assert exp_step.status == StepStatus.FAILED
        assert ai_step.status == StepStatus.CREATED


# ---------------------------------------------------------------------------
# Document seeding
# ---------------------------------------------------------------------------


class TestSeedDocuments:
    """Tests for _seed_documents and _await_processing."""

    async def test_seed_documents_no_service_fails(self) -> None:
        """Documents step fails when DocumentService is not provided."""
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        loader = MagicMock()
        loader.load_users.return_value = [_user_fixture()]
        loader.load_game.return_value = _game_fixture()
        loader.load_expansions.return_value = []
        loader.load_ai_config.return_value = (
            ModelSlotsFixture(slots=[]),
            _guardrail_fixture(),
        )

        new_game = _mock_game()
        game_service = AsyncMock()
        game_service.create_game.return_value = new_game

        curator = _mock_user()
        slot_repo = AsyncMock()

        service = _build_service(db, loader, game_service, slot_repo=slot_repo)
        service._find_user_by_email = AsyncMock(return_value=None)  # type: ignore[method-assign]
        service._get_curator_user = AsyncMock(return_value=curator)  # type: ignore[method-assign]
        service._find_game_by_name = AsyncMock(return_value=None)  # type: ignore[method-assign]
        service._find_guardrail_config = AsyncMock(return_value=None)  # type: ignore[method-assign]

        result = await service.seed_all()

        doc_step = next(s for s in result.steps if s.step == "documents")
        await_step = next(s for s in result.steps if s.step == "await_processing")
        assert doc_step.status == StepStatus.FAILED
        assert "DocumentService not provided" in doc_step.detail
        assert await_step.status == StepStatus.SKIPPED

    async def test_seed_documents_uploads_all(self) -> None:
        """Documents are uploaded via DocumentService when fixtures exist."""
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        doc1 = _mock_document(status="processed")
        doc2 = _mock_document(name="Seafarers Rules", status="processed")

        doc_service = AsyncMock()
        doc_service.upload = AsyncMock(side_effect=[doc1, doc2])

        loader = MagicMock()
        loader.load_document_metas.return_value = [
            _doc_meta_fixture(name="Catan Base Game Rules"),
            _doc_meta_fixture(
                name="Seafarers Rules",
                type="expansion_rules",
                file="catan-seafarers-rules.pdf",
                expansion_association="Catan: Seafarers",
            ),
        ]

        curator = _mock_user()
        expansion = _mock_expansion()
        game_id = uuid.uuid4()

        service = _build_service(db, loader, document_service=doc_service, wait_timeout=0.01)
        service._get_curator_user = AsyncMock(return_value=curator)  # type: ignore[method-assign]
        service._find_document_by_name = AsyncMock(return_value=None)  # type: ignore[method-assign]
        service._find_expansion_by_name = AsyncMock(return_value=expansion)  # type: ignore[method-assign]
        service._resolve_fixture_pdf = MagicMock(  # type: ignore[method-assign]
            return_value=MagicMock(name="fake.pdf", read_bytes=MagicMock(return_value=b"fake")),
        )
        service._find_document_by_id = AsyncMock(return_value=doc1)  # type: ignore[method-assign]

        from tabletop_oracle.services.seed_service import SeedResult

        result = SeedResult()
        doc_ids = await service._seed_documents(result, game_id, force=False)

        assert len(doc_ids) == 2
        assert doc_service.upload.call_count == 2
        doc_step = next(s for s in result.steps if s.step == "documents")
        assert doc_step.status == StepStatus.CREATED

    async def test_seed_documents_skips_existing(self) -> None:
        """Existing documents are skipped when force is False."""
        db = AsyncMock()
        doc_service = AsyncMock()

        existing_doc = _mock_document()

        loader = MagicMock()
        loader.load_document_metas.return_value = [_doc_meta_fixture()]

        curator = _mock_user()
        game_id = uuid.uuid4()

        service = _build_service(db, loader, document_service=doc_service)
        service._get_curator_user = AsyncMock(return_value=curator)  # type: ignore[method-assign]
        service._find_document_by_name = AsyncMock(return_value=existing_doc)  # type: ignore[method-assign]

        from tabletop_oracle.services.seed_service import SeedResult

        result = SeedResult()
        doc_ids = await service._seed_documents(result, game_id, force=False)

        assert len(doc_ids) == 0
        doc_service.upload.assert_not_called()
        doc_step = next(s for s in result.steps if s.step == "documents")
        assert doc_step.status == StepStatus.SKIPPED

    async def test_seed_documents_force_replaces(self) -> None:
        """Force mode deletes and re-uploads existing documents."""
        db = AsyncMock()
        db.delete = AsyncMock()
        db.flush = AsyncMock()

        existing_doc = _mock_document()
        new_doc = _mock_document(status="uploaded")

        doc_service = AsyncMock()
        doc_service.upload = AsyncMock(return_value=new_doc)

        loader = MagicMock()
        loader.load_document_metas.return_value = [_doc_meta_fixture()]

        curator = _mock_user()
        game_id = uuid.uuid4()

        service = _build_service(db, loader, document_service=doc_service, wait_timeout=0.01)
        service._get_curator_user = AsyncMock(return_value=curator)  # type: ignore[method-assign]
        service._find_document_by_name = AsyncMock(return_value=existing_doc)  # type: ignore[method-assign]
        service._resolve_fixture_pdf = MagicMock(  # type: ignore[method-assign]
            return_value=MagicMock(name="fake.pdf", read_bytes=MagicMock(return_value=b"fake")),
        )

        from tabletop_oracle.services.seed_service import SeedResult

        result = SeedResult()
        doc_ids = await service._seed_documents(result, game_id, force=True)

        assert len(doc_ids) == 1
        db.delete.assert_called_once_with(existing_doc)
        doc_service.upload.assert_called_once()
        doc_step = next(s for s in result.steps if s.step == "documents")
        assert doc_step.status == StepStatus.REPLACED

    async def test_seed_documents_handles_upload_error(self) -> None:
        """Upload errors are captured; other documents continue."""
        db = AsyncMock()
        db.delete = AsyncMock()
        db.flush = AsyncMock()

        good_doc = _mock_document(status="uploaded")
        doc_service = AsyncMock()
        doc_service.upload = AsyncMock(side_effect=[RuntimeError("blob store error"), good_doc])

        loader = MagicMock()
        loader.load_document_metas.return_value = [
            _doc_meta_fixture(name="Bad Doc", file="bad.pdf"),
            _doc_meta_fixture(name="Good Doc", file="good.pdf"),
        ]

        curator = _mock_user()
        game_id = uuid.uuid4()

        service = _build_service(db, loader, document_service=doc_service, wait_timeout=0.01)
        service._get_curator_user = AsyncMock(return_value=curator)  # type: ignore[method-assign]
        service._find_document_by_name = AsyncMock(return_value=None)  # type: ignore[method-assign]
        service._resolve_fixture_pdf = MagicMock(  # type: ignore[method-assign]
            return_value=MagicMock(name="fake.pdf", read_bytes=MagicMock(return_value=b"fake")),
        )

        from tabletop_oracle.services.seed_service import SeedResult

        result = SeedResult()
        doc_ids = await service._seed_documents(result, game_id, force=False)

        assert len(doc_ids) == 1
        doc_step = next(s for s in result.steps if s.step == "documents")
        assert doc_step.status == StepStatus.FAILED
        assert "errors 1" in doc_step.detail

    async def test_seed_documents_no_curator_fails(self) -> None:
        """Missing curator user fails the documents step."""
        db = AsyncMock()
        doc_service = AsyncMock()

        loader = MagicMock()
        loader.load_document_metas.return_value = [_doc_meta_fixture()]

        game_id = uuid.uuid4()

        service = _build_service(db, loader, document_service=doc_service)
        service._get_curator_user = AsyncMock(return_value=None)  # type: ignore[method-assign]

        from tabletop_oracle.services.seed_service import SeedResult

        result = SeedResult()
        doc_ids = await service._seed_documents(result, game_id, force=False)

        assert len(doc_ids) == 0
        doc_step = next(s for s in result.steps if s.step == "documents")
        assert doc_step.status == StepStatus.FAILED
        assert "curator" in doc_step.detail.lower()

    async def test_seed_documents_public_method(self) -> None:
        """seed_documents public method requires game to exist."""
        db = AsyncMock()
        loader = MagicMock()
        loader.load_game.return_value = _game_fixture()

        service = _build_service(db, loader)
        service._find_game_by_name = AsyncMock(return_value=None)  # type: ignore[method-assign]

        result = await service.seed_documents()

        assert result.success is False
        assert result.steps[0].status == StepStatus.FAILED
        assert "Game must be seeded first" in result.steps[0].detail


# ---------------------------------------------------------------------------
# Await processing
# ---------------------------------------------------------------------------


class TestAwaitProcessing:
    """Tests for _await_processing polling."""

    async def test_await_processing_all_processed(self) -> None:
        """Reports success when all documents reach processed status."""
        db = AsyncMock()
        doc1 = _mock_document(status="processed")
        doc2 = _mock_document(status="processed")

        service = _build_service(db, wait_timeout=10.0)
        service._find_document_by_id = AsyncMock(side_effect=[doc1, doc2])  # type: ignore[method-assign]

        from tabletop_oracle.services.seed_service import SeedResult

        result = SeedResult()
        await service._await_processing(result, [doc1.id, doc2.id])

        step = next(s for s in result.steps if s.step == "await_processing")
        assert step.status == StepStatus.CREATED
        assert "2 documents processed" in step.detail

    async def test_await_processing_with_errors(self) -> None:
        """Reports failure when some documents have errors."""
        db = AsyncMock()
        doc_ok = _mock_document(status="processed")
        doc_err = _mock_document(status="error")

        service = _build_service(db, wait_timeout=10.0)
        service._find_document_by_id = AsyncMock(  # type: ignore[method-assign]
            side_effect=[doc_ok, doc_err],
        )

        from tabletop_oracle.services.seed_service import SeedResult

        result = SeedResult()
        await service._await_processing(result, [doc_ok.id, doc_err.id])

        step = next(s for s in result.steps if s.step == "await_processing")
        assert step.status == StepStatus.FAILED
        assert "1 errors" in step.detail

    async def test_await_processing_timeout(self) -> None:
        """Reports timeout when documents stay pending beyond limit."""
        db = AsyncMock()
        pending_doc = _mock_document(status="uploaded")

        service = _build_service(db, wait_timeout=0.01)
        service._find_document_by_id = AsyncMock(return_value=pending_doc)  # type: ignore[method-assign]

        from tabletop_oracle.services.seed_service import SeedResult

        result = SeedResult()
        await service._await_processing(result, [pending_doc.id])

        step = next(s for s in result.steps if s.step == "await_processing")
        assert step.status == StepStatus.FAILED
        assert "Timeout" in step.detail

    async def test_await_processing_empty_list(self) -> None:
        """No documents to await is skipped."""
        db = AsyncMock()
        service = _build_service(db)

        from tabletop_oracle.services.seed_service import SeedResult

        result = SeedResult()
        await service._await_processing(result, [])

        step = next(s for s in result.steps if s.step == "await_processing")
        assert step.status == StepStatus.SKIPPED


# ---------------------------------------------------------------------------
# FixtureUploadFile adapter
# ---------------------------------------------------------------------------


class TestFixtureUploadFile:
    """Tests for the _FixtureUploadFile adapter."""

    async def test_read_returns_file_content(self, tmp_path: Any) -> None:
        """read() returns raw bytes of the fixture file."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 fake content")

        upload = _FixtureUploadFile(test_file)

        data = await upload.read()
        assert data == b"%PDF-1.4 fake content"
        assert upload.filename == "test.pdf"
        assert upload.content_type == "application/pdf"

    def test_custom_content_type(self, tmp_path: Any) -> None:
        """Content type can be overridden."""
        test_file = tmp_path / "test.md"
        test_file.write_bytes(b"# test")

        upload = _FixtureUploadFile(test_file, content_type="text/markdown")

        assert upload.content_type == "text/markdown"
