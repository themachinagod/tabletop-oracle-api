"""Orchestrates seed data provisioning for a fresh deployment.

Runs seeding steps in dependency order: users, game, expansion, AI config,
documents, await processing. Each step is idempotent — existing data is
skipped unless ``force=True``. Uses existing domain services where available
(GameService, ExpansionService, DocumentService, ModelSlotService) and direct
DB operations where no service exists (user upsert, guardrail config).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import delete, select

from tabletop_oracle.models.enums import (
    DocumentStatus,
    ModelCapability,
    OAuthProvider,
    UserRole,
    UserStatus,
)
from tabletop_oracle.models.guardrail import GuardrailConfig
from tabletop_oracle.models.model_slot import ModelSlot
from tabletop_oracle.models.user import User

if TYPE_CHECKING:
    import uuid
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

    from tabletop_oracle.models.document import Document
    from tabletop_oracle.models.expansion import Expansion
    from tabletop_oracle.models.game import Game
    from tabletop_oracle.services.document_service import DocumentService
    from tabletop_oracle.services.expansion_service import ExpansionService
    from tabletop_oracle.services.game_service import GameService
    from tabletop_oracle.services.model.slot_repository import ModelSlotRepository
    from tabletop_oracle.services.seed_fixture_loader import DocumentMetaFixture, SeedFixtureLoader

logger = logging.getLogger(__name__)

#: Default polling interval for document processing status (seconds).
_POLL_INTERVAL_SECONDS = 5.0

#: Default maximum wait time for document processing (seconds).
_DEFAULT_WAIT_TIMEOUT = 300.0


# ---------------------------------------------------------------------------
# UploadFile adapter for fixture PDFs
# ---------------------------------------------------------------------------


class _FixtureUploadFile:
    """Adapts a local fixture file to the ``UploadFile`` interface.

    ``DocumentService.upload()`` expects a ``fastapi.UploadFile``. This
    lightweight adapter reads a file from disk and exposes the same
    ``read()``, ``filename``, and ``content_type`` interface without
    pulling in the full Starlette/FastAPI dependency.

    Args:
        file_path: Absolute path to the fixture file.
        content_type: MIME type for the file.
    """

    def __init__(self, file_path: Path, content_type: str = "application/pdf") -> None:
        self.filename: str = file_path.name
        self.content_type: str = content_type
        self._data: bytes = file_path.read_bytes()

    async def read(self) -> bytes:
        """Return the full file content.

        Returns:
            Raw bytes of the fixture file.
        """
        return self._data


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class StepStatus(Enum):
    """Outcome of a single seed step."""

    CREATED = "created"
    SKIPPED = "skipped"
    REPLACED = "replaced"
    FAILED = "failed"
    STUBBED = "stubbed"


@dataclass
class StepResult:
    """Result of a single seed step.

    Attributes:
        step: Human-readable step name (e.g. "users", "game").
        status: Outcome of the step.
        detail: Optional detail message for logging / display.
    """

    step: str
    status: StepStatus
    detail: str = ""


@dataclass
class SeedResult:
    """Aggregate result of a seed operation.

    Attributes:
        steps: Ordered list of per-step results.
        success: True if no steps failed.
    """

    steps: list[StepResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True when no step has a FAILED status."""
        return all(s.status != StepStatus.FAILED for s in self.steps)

    def add(self, step: str, status: StepStatus, detail: str = "") -> None:
        """Append a step result.

        Args:
            step: Step name.
            status: Step outcome.
            detail: Optional detail message.
        """
        self.steps.append(StepResult(step=step, status=status, detail=detail))


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class SeedService:
    """Orchestrates seed data provisioning for a fresh deployment.

    Runs seeding steps in dependency order, delegating to domain services
    where they exist and falling back to direct DB operations where they
    do not.

    Args:
        db: SQLAlchemy async session (caller owns the transaction).
        fixture_loader: Loads typed fixture data from the seed directory.
        game_service: Service for game CRUD operations.
        expansion_service: Service for expansion CRUD operations.
        model_slot_repo: Repository for model slot data access.
        document_service: Service for document upload and retrieval.
        wait_timeout: Maximum seconds to wait for document processing
            (default 300).
    """

    def __init__(
        self,
        db: AsyncSession,
        fixture_loader: SeedFixtureLoader,
        game_service: GameService,
        expansion_service: ExpansionService,
        model_slot_repo: ModelSlotRepository,
        document_service: DocumentService | None = None,
        wait_timeout: float = _DEFAULT_WAIT_TIMEOUT,
    ) -> None:
        self._db = db
        self._loader = fixture_loader
        self._game_service = game_service
        self._expansion_service = expansion_service
        self._slot_repo = model_slot_repo
        self._doc_service = document_service
        self._wait_timeout = wait_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def seed_all(self, *, force: bool = False) -> SeedResult:
        """Run the full seed process in dependency order.

        Steps: users -> game -> expansion -> ai_config -> documents
        -> await_processing.

        Args:
            force: If True, replace existing data. If False, skip where
                data already exists.

        Returns:
            SeedResult with per-step outcomes.
        """
        result = SeedResult()

        await self._seed_users(result, force=force)

        game = await self._seed_game(result, force=force)
        game_id = game.id if game else await self._get_existing_game_id()
        if game_id is None:
            result.add("expansion", StepStatus.FAILED, "No game_id — cannot seed expansion")
            result.add("ai_config", StepStatus.FAILED, "Skipped due to prior failure")
            return result

        await self._seed_expansion(result, game_id, force=force)
        await self._seed_ai_config(result, force=force)
        doc_ids = await self._seed_documents(result, game_id, force=force)
        await self._await_processing(result, doc_ids)

        return result

    async def seed_users(self, *, force: bool = False) -> SeedResult:
        """Seed user accounts only.

        Args:
            force: If True, replace existing users.

        Returns:
            SeedResult with user step outcome.
        """
        result = SeedResult()
        await self._seed_users(result, force=force)
        return result

    async def seed_game(self, *, force: bool = False) -> SeedResult:
        """Seed the game entry only.

        Args:
            force: If True, replace existing game.

        Returns:
            SeedResult with game step outcome.
        """
        result = SeedResult()
        await self._seed_game(result, force=force)
        return result

    async def seed_expansion(self, *, force: bool = False) -> SeedResult:
        """Seed the expansion entry only.

        Requires the game to already exist.

        Args:
            force: If True, replace existing expansion.

        Returns:
            SeedResult with expansion step outcome.
        """
        result = SeedResult()
        game_id = await self._get_existing_game_id()
        if game_id is None:
            result.add("expansion", StepStatus.FAILED, "Game must be seeded first")
            return result
        await self._seed_expansion(result, game_id, force=force)
        return result

    async def seed_ai_config(self, *, force: bool = False) -> SeedResult:
        """Seed model slots and guardrail configuration only.

        Args:
            force: If True, replace existing config.

        Returns:
            SeedResult with ai_config step outcome.
        """
        result = SeedResult()
        await self._seed_ai_config(result, force=force)
        return result

    async def seed_documents(self, *, force: bool = False) -> SeedResult:
        """Seed documents only. Requires game and expansion to exist.

        Args:
            force: If True, replace existing documents.

        Returns:
            SeedResult with document and await_processing step outcomes.
        """
        result = SeedResult()
        game_id = await self._get_existing_game_id()
        if game_id is None:
            result.add("documents", StepStatus.FAILED, "Game must be seeded first")
            return result
        doc_ids = await self._seed_documents(result, game_id, force=force)
        await self._await_processing(result, doc_ids)
        return result

    async def reset(self) -> SeedResult:
        """Remove all seed data (development only).

        Deletes in reverse dependency order: guardrail config, model slots,
        expansions (via game cascade), games, seed users.

        Returns:
            SeedResult confirming removal steps.
        """
        result = SeedResult()

        # Guardrail config — delete all rows (single-row table)
        await self._db.execute(delete(GuardrailConfig))
        result.add("guardrail_config", StepStatus.CREATED, "Deleted all guardrail config")
        logger.info("Reset: deleted guardrail config")

        # Model slots — delete all
        await self._db.execute(delete(ModelSlot))
        result.add("model_slots", StepStatus.CREATED, "Deleted all model slots")
        logger.info("Reset: deleted model slots")

        # Game (cascade deletes expansions)
        fixture = self._loader.load_game()
        game = await self._find_game_by_name(fixture.name)
        if game:
            await self._db.delete(game)
            await self._db.flush()
            result.add("game", StepStatus.CREATED, f"Deleted game '{fixture.name}'")
            logger.info("Reset: deleted game '%s'", fixture.name)
        else:
            result.add("game", StepStatus.SKIPPED, "No game to delete")

        # Seed users
        user_fixtures = self._loader.load_users()
        for uf in user_fixtures:
            user = await self._find_user_by_email(uf.email)
            if user:
                await self._db.delete(user)
                await self._db.flush()
                logger.info("Reset: deleted user '%s'", uf.email)
        result.add("users", StepStatus.CREATED, f"Deleted {len(user_fixtures)} seed users")

        return result

    # ------------------------------------------------------------------
    # Internal step implementations
    # ------------------------------------------------------------------

    async def _seed_users(self, result: SeedResult, *, force: bool) -> None:
        """Seed curator and test player accounts.

        Uses direct DB operations because there is no user-creation
        API/service in V1. Idempotency: checks by email.

        Args:
            result: Accumulator for step outcomes.
            force: Replace existing users if True.
        """
        user_fixtures = self._loader.load_users()
        created_count = 0
        skipped_count = 0

        for uf in user_fixtures:
            existing = await self._find_user_by_email(uf.email)

            if existing and not force:
                logger.info("User '%s' already exists, skipping", uf.email)
                skipped_count += 1
                continue

            if existing and force:
                await self._db.delete(existing)
                await self._db.flush()
                logger.info("Force mode: deleted existing user '%s'", uf.email)

            user = User(
                email=uf.email,
                display_name=uf.display_name,
                oauth_provider=(
                    OAuthProvider(uf.oauth_provider) if uf.oauth_provider else OAuthProvider.GOOGLE
                ),
                oauth_subject_id=uf.oauth_subject_id,
                role=UserRole(uf.role),
                status=UserStatus(uf.status),
            )
            self._db.add(user)
            await self._db.flush()
            created_count += 1
            logger.info("Created seed user '%s' with role '%s'", uf.email, uf.role)

        if created_count > 0 and skipped_count > 0:
            status = StepStatus.CREATED
            detail = f"Created {created_count}, skipped {skipped_count}"
        elif created_count > 0:
            status = StepStatus.REPLACED if force else StepStatus.CREATED
            detail = f"Created {created_count} users"
        else:
            status = StepStatus.SKIPPED
            detail = f"All {skipped_count} users already exist"

        result.add("users", status, detail)

    async def _seed_game(
        self,
        result: SeedResult,
        *,
        force: bool,
    ) -> Game | None:
        """Seed the game entry via GameService.

        Idempotency: checks by game name. Returns the game model on
        success, or None if skipped.

        Args:
            result: Accumulator for step outcomes.
            force: Replace existing game if True.

        Returns:
            The created/existing Game, or None on failure.
        """
        from tabletop_oracle.schemas.game import GameCreate as GameCreateSchema

        fixture = self._loader.load_game()
        existing = await self._find_game_by_name(fixture.name)

        if existing and not force:
            logger.info("Game '%s' already exists, skipping", fixture.name)
            result.add("game", StepStatus.SKIPPED, f"Game '{fixture.name}' already exists")
            return existing

        if existing and force:
            await self._db.delete(existing)
            await self._db.flush()
            logger.info("Force mode: deleted existing game '%s'", fixture.name)

        # Need a curator user to attribute the game creation
        curator = await self._get_curator_user()
        if curator is None:
            result.add("game", StepStatus.FAILED, "No curator user found — seed users first")
            return None

        game_data = GameCreateSchema(
            name=fixture.name,
            publisher=fixture.publisher,
            year_published=fixture.year_published,
            edition=fixture.edition,
            min_players=fixture.min_players,
            max_players=fixture.max_players,
            description=fixture.description,
            cover_image_url=fixture.cover_image_url,
            complexity=fixture.complexity,
            tags=fixture.tags,
        )

        try:
            game = await self._game_service.create_game(game_data, curator.id)
        except Exception:
            logger.exception("Failed to create game '%s'", fixture.name)
            result.add("game", StepStatus.FAILED, f"Error creating game '{fixture.name}'")
            return None

        status = StepStatus.REPLACED if (force and existing) else StepStatus.CREATED
        result.add("game", status, f"Game '{fixture.name}' seeded")
        logger.info("Seeded game '%s' (id=%s)", fixture.name, game.id)
        return game

    async def _seed_expansion(
        self,
        result: SeedResult,
        game_id: uuid.UUID,
        *,
        force: bool,
    ) -> None:
        """Seed expansion entries via ExpansionService.

        Idempotency: checks by expansion name within the game.

        Args:
            result: Accumulator for step outcomes.
            game_id: Parent game UUID.
            force: Replace existing expansions if True.
        """
        from tabletop_oracle.schemas.expansion import ExpansionCreate as ExpansionCreateSchema

        fixtures = self._loader.load_expansions()
        created_count = 0
        skipped_count = 0

        curator = await self._get_curator_user()
        if curator is None:
            result.add("expansion", StepStatus.FAILED, "No curator user found — seed users first")
            return

        for ef in fixtures:
            existing = await self._find_expansion_by_name(game_id, ef.name)

            if existing and not force:
                logger.info("Expansion '%s' already exists, skipping", ef.name)
                skipped_count += 1
                continue

            if existing and force:
                await self._db.delete(existing)
                await self._db.flush()
                logger.info("Force mode: deleted existing expansion '%s'", ef.name)

            exp_data = ExpansionCreateSchema(
                name=ef.name,
                year_published=ef.year_published,
                description=ef.description,
            )

            try:
                await self._expansion_service.create_expansion(game_id, exp_data, curator.id)
                created_count += 1
                logger.info("Seeded expansion '%s'", ef.name)
            except Exception:
                logger.exception("Failed to create expansion '%s'", ef.name)
                result.add(
                    "expansion",
                    StepStatus.FAILED,
                    f"Error creating expansion '{ef.name}'",
                )
                return

        if created_count > 0:
            status = StepStatus.REPLACED if force else StepStatus.CREATED
            detail = f"Created {created_count} expansions"
        else:
            status = StepStatus.SKIPPED
            detail = f"All {skipped_count} expansions already exist"

        result.add("expansion", status, detail)

    async def _seed_ai_config(self, result: SeedResult, *, force: bool) -> None:
        """Seed model slots and guardrail configuration.

        Uses direct DB operations — no service-layer create for these
        entities in V1.

        Args:
            result: Accumulator for step outcomes.
            force: Replace existing config if True.
        """
        slots_fixture, guardrail_fixture = self._loader.load_ai_config()

        # -- Model slots -----------------------------------------------
        slots_created = 0
        slots_skipped = 0

        for sf in slots_fixture.slots:
            capability = ModelCapability(sf.capability)
            existing = await self._slot_repo.get_by_capability(capability)

            if existing and not force:
                logger.info("Model slot '%s' already exists, skipping", sf.capability)
                slots_skipped += 1
                continue

            if existing and force:
                await self._db.delete(existing)
                await self._db.flush()
                logger.info("Force mode: deleted existing model slot '%s'", sf.capability)

            slot = ModelSlot(
                capability=capability,
                provider=sf.provider,
                model_id=sf.model_id,
                max_tokens_per_call=sf.max_tokens_per_call,
                temperature=sf.temperature,
                fallback_provider=sf.fallback_provider,
                fallback_model_id=sf.fallback_model_id,
            )
            self._db.add(slot)
            await self._db.flush()
            slots_created += 1
            logger.info("Seeded model slot '%s'", sf.capability)

        # -- Guardrail config ------------------------------------------
        existing_config = await self._find_guardrail_config()
        guardrail_status: StepStatus

        if existing_config and not force:
            logger.info("Guardrail config already exists, skipping")
            guardrail_status = StepStatus.SKIPPED
        else:
            if existing_config and force:
                await self._db.delete(existing_config)
                await self._db.flush()
                logger.info("Force mode: deleted existing guardrail config")

            config = GuardrailConfig(
                enforcement_enabled=guardrail_fixture.enforcement_enabled,
                max_tokens_per_query=guardrail_fixture.max_tokens_per_query,
                max_model_calls_per_query=guardrail_fixture.max_model_calls_per_query,
                max_queries_per_session=guardrail_fixture.max_queries_per_session,
                daily_token_budget=guardrail_fixture.daily_token_budget,
                daily_query_budget=guardrail_fixture.daily_query_budget,
                per_document_ingestion_limit=guardrail_fixture.per_document_ingestion_limit,
            )
            self._db.add(config)
            await self._db.flush()
            guardrail_status = (
                StepStatus.REPLACED if (force and existing_config) else StepStatus.CREATED
            )
            logger.info("Seeded guardrail config")

        # Build combined detail
        slots_detail = (
            f"{slots_created} created, {slots_skipped} skipped"
            if slots_skipped
            else f"{slots_created} created"
        )
        combined_status = (
            StepStatus.CREATED
            if slots_created > 0 or guardrail_status == StepStatus.CREATED
            else StepStatus.SKIPPED
        )
        if force and (slots_created > 0 or guardrail_status == StepStatus.REPLACED):
            combined_status = StepStatus.REPLACED

        result.add(
            "ai_config",
            combined_status,
            f"Model slots: {slots_detail}. Guardrail config: {guardrail_status.value}",
        )

    async def _seed_documents(
        self,
        result: SeedResult,
        game_id: uuid.UUID,
        *,
        force: bool,
    ) -> list[uuid.UUID]:
        """Upload seed document fixtures via DocumentService.

        Loads document metadata from the manifest, resolves expansion
        associations by name, reads the corresponding PDF files from the
        fixtures directory, and uploads each via ``DocumentService.upload()``.
        The upload triggers the standard Celery ingestion pipeline.

        Args:
            result: Accumulator for step outcomes.
            game_id: Parent game UUID.
            force: If True, delete and re-upload existing documents.

        Returns:
            List of uploaded document UUIDs (for await_processing).
        """
        if self._doc_service is None:
            result.add(
                "documents",
                StepStatus.FAILED,
                "DocumentService not provided — cannot seed documents",
            )
            return []

        doc_metas = self._loader.load_document_metas()
        if not doc_metas:
            result.add("documents", StepStatus.SKIPPED, "No document fixtures found")
            return []

        curator = await self._get_curator_user()
        if curator is None:
            result.add("documents", StepStatus.FAILED, "No curator user — seed users first")
            return []

        uploaded_ids: list[uuid.UUID] = []
        skipped_count = 0
        error_count = 0

        for meta in doc_metas:
            try:
                doc_id = await self._upload_single_document(meta, game_id, curator.id, force=force)
                if doc_id is not None:
                    uploaded_ids.append(doc_id)
                else:
                    skipped_count += 1
            except Exception:
                logger.exception("Failed to upload document '%s'", meta.name)
                error_count += 1

        if error_count > 0:
            status = StepStatus.FAILED
            detail = f"Uploaded {len(uploaded_ids)}, skipped {skipped_count}, errors {error_count}"
        elif uploaded_ids:
            status = StepStatus.REPLACED if force else StepStatus.CREATED
            detail = f"Uploaded {len(uploaded_ids)} documents"
            if skipped_count:
                detail += f", skipped {skipped_count}"
        else:
            status = StepStatus.SKIPPED
            detail = f"All {skipped_count} documents already exist"

        result.add("documents", status, detail)
        return uploaded_ids

    async def _upload_single_document(
        self,
        meta: DocumentMetaFixture,
        game_id: uuid.UUID,
        curator_id: uuid.UUID,
        *,
        force: bool,
    ) -> uuid.UUID | None:
        """Upload a single document fixture via DocumentService.

        Checks for an existing document by name (idempotency). Resolves
        expansion association by name if specified. Creates a fixture
        file adapter and delegates to ``DocumentService.upload()``.

        Args:
            meta: Document metadata from the fixture .meta.json.
            game_id: Parent game UUID.
            curator_id: Curator user UUID for attribution.
            force: If True, delete existing document before re-upload.

        Returns:
            The created document UUID, or None if skipped.
        """
        from tabletop_oracle.schemas.document import DocumentTypeFilter

        assert self._doc_service is not None  # guarded by caller

        existing = await self._find_document_by_name(game_id, meta.name)

        if existing and not force:
            logger.info("Document '%s' already exists, skipping", meta.name)
            return None

        if existing and force:
            await self._db.delete(existing)
            await self._db.flush()
            logger.info("Force mode: deleted existing document '%s'", meta.name)

        # Resolve expansion association by name
        expansion_id: uuid.UUID | None = None
        if meta.expansion_association:
            expansion = await self._find_expansion_by_name(game_id, meta.expansion_association)
            if expansion is None:
                msg = (
                    f"Expansion '{meta.expansion_association}' not found for "
                    f"document '{meta.name}'"
                )
                raise RuntimeError(msg)
            expansion_id = expansion.id

        # Build a file adapter from the fixture PDF
        fixture_file = self._resolve_fixture_pdf(meta.file)
        upload_file = _FixtureUploadFile(fixture_file)

        doc_type = DocumentTypeFilter(meta.type)
        document = await self._doc_service.upload(
            game_id=game_id,
            file=upload_file,  # type: ignore[arg-type]
            doc_type=doc_type,
            user_id=curator_id,
            name=meta.name,
            expansion_id=expansion_id,
        )

        logger.info("Uploaded seed document '%s' (id=%s)", meta.name, document.id)
        return document.id

    def _resolve_fixture_pdf(self, filename: str) -> Path:
        """Resolve a PDF filename to its absolute fixture path.

        Searches the documents subdirectory tree within the fixtures
        directory for the named file.

        Args:
            filename: The PDF filename from the meta fixture.

        Returns:
            Absolute path to the fixture PDF.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        # The PDF lives alongside the .meta.json in the documents directory
        docs_dir = self._loader.fixtures_dir / "games"
        matches = list(docs_dir.rglob(filename))
        if not matches:
            msg = f"Fixture PDF not found: {filename} under {docs_dir}"
            raise FileNotFoundError(msg)
        return matches[0]

    async def _await_processing(
        self,
        result: SeedResult,
        doc_ids: list[uuid.UUID],
    ) -> None:
        """Poll document status until all reach a terminal state.

        Waits for all uploaded documents to reach ``processed`` or
        ``error`` status. Reports progress at each poll interval.
        Times out after ``wait_timeout`` seconds.

        Args:
            result: Accumulator for step outcomes.
            doc_ids: Document UUIDs to monitor.
        """
        if not doc_ids:
            result.add(
                "await_processing",
                StepStatus.SKIPPED,
                "No documents to await",
            )
            return

        elapsed = 0.0
        processed_count = 0
        error_count = 0

        while elapsed < self._wait_timeout:
            processed_count = 0
            error_count = 0
            pending_count = 0

            for doc_id in doc_ids:
                doc = await self._find_document_by_id(doc_id)
                if doc is None:
                    error_count += 1
                    continue
                if doc.status == DocumentStatus.PROCESSED:
                    processed_count += 1
                elif doc.status == DocumentStatus.ERROR:
                    error_count += 1
                else:
                    pending_count += 1

            logger.info(
                "Document processing: %d processed, %d errors, %d pending (%.0fs elapsed)",
                processed_count,
                error_count,
                pending_count,
                elapsed,
            )

            if pending_count == 0:
                break

            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            elapsed += _POLL_INTERVAL_SECONDS

        total = len(doc_ids)
        if processed_count == total:
            status = StepStatus.CREATED
            detail = f"All {total} documents processed successfully"
        elif processed_count + error_count == total:
            status = StepStatus.FAILED if error_count > 0 else StepStatus.CREATED
            detail = f"{processed_count} processed, {error_count} errors"
        else:
            status = StepStatus.FAILED
            remaining = total - processed_count - error_count
            detail = (
                f"Timeout after {self._wait_timeout}s: "
                f"{processed_count} processed, {error_count} errors, "
                f"{remaining} still pending"
            )

        result.add("await_processing", status, detail)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    async def _find_user_by_email(self, email: str) -> User | None:
        """Find a user by email address.

        Args:
            email: Email to search for.

        Returns:
            User or None.
        """
        stmt = select(User).where(User.email == email)
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def _find_game_by_name(self, name: str) -> Game | None:
        """Find a game by name.

        Args:
            name: Game name to search for.

        Returns:
            Game or None.
        """
        from tabletop_oracle.models.game import Game as GameModel

        stmt = select(GameModel).where(GameModel.name == name)
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def _find_expansion_by_name(
        self,
        game_id: uuid.UUID,
        name: str,
    ) -> Expansion | None:
        """Find an expansion by name within a game.

        Args:
            game_id: Parent game UUID.
            name: Expansion name to search for.

        Returns:
            Expansion or None.
        """
        from tabletop_oracle.models.expansion import Expansion as ExpansionModel

        stmt = select(ExpansionModel).where(
            ExpansionModel.game_id == game_id,
            ExpansionModel.name == name,
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def _find_document_by_name(
        self,
        game_id: uuid.UUID,
        name: str,
    ) -> Document | None:
        """Find a document by name within a game.

        Args:
            game_id: Parent game UUID.
            name: Document display name to search for.

        Returns:
            Document or None.
        """
        from tabletop_oracle.models.document import Document as DocumentModel

        stmt = select(DocumentModel).where(
            DocumentModel.game_id == game_id,
            DocumentModel.name == name,
            DocumentModel.archived_at.is_(None),
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def _find_document_by_id(self, doc_id: uuid.UUID) -> Document | None:
        """Find a document by ID for status polling.

        Args:
            doc_id: Document UUID.

        Returns:
            Document or None.
        """
        from tabletop_oracle.models.document import Document as DocumentModel

        stmt = select(DocumentModel).where(DocumentModel.id == doc_id)
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def _find_guardrail_config(self) -> GuardrailConfig | None:
        """Find the single-row guardrail configuration.

        Returns:
            GuardrailConfig or None if not configured.
        """
        stmt = select(GuardrailConfig).limit(1)
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_curator_user(self) -> User | None:
        """Find the first active curator user.

        Returns:
            The first curator User, or None if none exist.
        """
        stmt = select(User).where(
            User.role == UserRole.CURATOR,
            User.status == UserStatus.ACTIVE,
        )
        result = await self._db.execute(stmt)
        return result.scalars().first()

    async def _get_existing_game_id(self) -> uuid.UUID | None:
        """Look up the seeded game ID by fixture name.

        Returns:
            Game UUID if found, else None.
        """
        fixture = self._loader.load_game()
        game = await self._find_game_by_name(fixture.name)
        return game.id if game else None
