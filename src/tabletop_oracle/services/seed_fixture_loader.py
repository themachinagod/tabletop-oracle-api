"""Loads and validates seed fixture data from the bundled fixtures directory.

Reads JSON metadata files for users, games, expansions, documents,
AI configuration, and ground truth Q&A pairs. Returns typed Pydantic
models for each fixture type. Raises SeedFixtureError on missing files,
malformed JSON, or schema violations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Fixtures root directory
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "seed"


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class SeedFixtureError(Exception):
    """Raised when a seed fixture file is missing, malformed, or invalid.

    Args:
        message: Human-readable description of the fixture error.
        path: Path to the fixture file that caused the error (if applicable).
    """

    def __init__(self, message: str, *, path: Path | None = None) -> None:
        self.path = path
        super().__init__(message)


# ---------------------------------------------------------------------------
# Fixture models — Users
# ---------------------------------------------------------------------------


class UserFixture(BaseModel):
    """A seed user account fixture.

    The curator fixture uses ``email_source`` (e.g. ``"env:INITIAL_CURATOR_EMAILS"``)
    instead of a literal email — resolved at load time. Regular user fixtures
    use ``email`` directly.
    """

    email: str
    display_name: str
    oauth_provider: str | None = None
    oauth_subject_id: str | None = None
    role: str
    status: str
    notes: str | None = None


# ---------------------------------------------------------------------------
# Fixture models — Games
# ---------------------------------------------------------------------------


class GameFixture(BaseModel):
    """Seed game metadata fixture."""

    name: str
    publisher: str
    year_published: int
    edition: str | None = None
    min_players: int
    max_players: int
    description: str
    cover_image_url: str | None = None
    complexity: str | None = None
    tags: list[str] = Field(default_factory=list)


class ExpansionFixture(BaseModel):
    """Seed expansion metadata fixture."""

    name: str
    year_published: int
    description: str


# ---------------------------------------------------------------------------
# Fixture models — Documents
# ---------------------------------------------------------------------------


class DocumentMetaFixture(BaseModel):
    """Seed document metadata fixture (the .meta.json sidecar).

    Links a PDF file to its classification and parent game/expansion.
    """

    name: str
    type: str
    format: str
    file: str
    expansion_id: str | None = None
    expansion_association: str | None = None
    notes: str | None = None


# ---------------------------------------------------------------------------
# Fixture models — AI configuration
# ---------------------------------------------------------------------------


class ModelSlotFixture(BaseModel):
    """A single model slot assignment fixture."""

    capability: str
    provider: str
    model_id: str
    max_tokens_per_call: int
    temperature: float
    fallback_provider: str | None = None
    fallback_model_id: str | None = None
    notes: str | None = None

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        """Temperature must be between 0 and 2."""
        if not 0.0 <= v <= 2.0:
            msg = f"Temperature must be between 0 and 2, got {v}"
            raise ValueError(msg)
        return v


class ModelSlotsFixture(BaseModel):
    """Wrapper for the model-slots.json file containing all slot configs."""

    slots: list[ModelSlotFixture]


class GuardrailFixture(BaseModel):
    """Default guardrail configuration fixture."""

    enforcement_enabled: bool
    max_tokens_per_query: int
    max_model_calls_per_query: int
    max_queries_per_session: int
    daily_token_budget: int
    daily_query_budget: int
    per_document_ingestion_limit: int


# ---------------------------------------------------------------------------
# Fixture models — Ground truth
# ---------------------------------------------------------------------------


class GroundTruthExpected(BaseModel):
    """Expected answer criteria for a ground truth question."""

    answer_contains: list[str] = Field(default_factory=list)
    answer_contains_any: list[str] = Field(default_factory=list)
    must_cite: bool = False
    min_citations: int | None = None
    min_confidence: float | None = None
    expected_document_types: list[str] = Field(default_factory=list)


class GroundTruthQA(BaseModel):
    """A single ground truth question-answer pair."""

    id: str
    category: str
    question: str
    expected: GroundTruthExpected


class GroundTruthFixture(BaseModel):
    """Top-level ground truth Q&A set fixture."""

    version: str
    game: str
    description: str
    questions: list[GroundTruthQA]


# ---------------------------------------------------------------------------
# Fixture models — Manifest
# ---------------------------------------------------------------------------


class ManifestStep(BaseModel):
    """A single step in the seed manifest."""

    step: int
    type: str
    files: list[str]
    parent_game: str | None = None


class SeedManifest(BaseModel):
    """Seed manifest declaring all fixture content and ordering."""

    version: str
    description: str
    steps: list[ManifestStep]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class SeedFixtureLoader:
    """Loads and validates JSON fixture files from the seed fixtures directory.

    All paths are resolved relative to the bundled ``fixtures/seed/`` directory
    inside the package. Each ``load_*`` method reads JSON, validates it against
    the corresponding Pydantic model, and returns typed objects.

    Args:
        fixtures_dir: Override the default fixtures directory (for testing).
    """

    def __init__(self, fixtures_dir: Path | None = None) -> None:
        self._dir = fixtures_dir or _FIXTURES_DIR

    @property
    def fixtures_dir(self) -> Path:
        """The root directory for seed fixture files."""
        return self._dir

    # -- Manifest -----------------------------------------------------------

    def load_manifest(self) -> SeedManifest:
        """Load and validate the seed manifest.

        Returns:
            Parsed SeedManifest with ordered steps.

        Raises:
            SeedFixtureError: If manifest.json is missing or invalid.
        """
        data = self._read_json(self._dir / "manifest.json")
        return self._parse(SeedManifest, data, self._dir / "manifest.json")

    # -- Users --------------------------------------------------------------

    def load_users(self) -> list[UserFixture]:
        """Load all user fixture files declared in the manifest.

        Resolves ``email_source`` fields of the form ``"env:VAR_NAME"``
        by reading the named environment variable. If the variable is
        unset, raises SeedFixtureError.

        Returns:
            List of UserFixture objects with emails resolved.

        Raises:
            SeedFixtureError: If a file is missing, malformed, or an
                env-sourced email variable is not set.
        """
        manifest = self.load_manifest()
        user_step = self._find_step(manifest, "users")
        users: list[UserFixture] = []
        for rel_path in user_step.files:
            full_path = self._dir / rel_path
            data = self._read_json(full_path)
            resolved = self._resolve_email_source(data, full_path)
            users.append(self._parse(UserFixture, resolved, full_path))
        return users

    # -- Game ---------------------------------------------------------------

    def load_game(self) -> GameFixture:
        """Load the game fixture.

        Returns:
            Parsed GameFixture.

        Raises:
            SeedFixtureError: If the file is missing or invalid.
        """
        manifest = self.load_manifest()
        game_step = self._find_step(manifest, "game")
        if len(game_step.files) != 1:
            msg = f"Expected exactly 1 game file in manifest, got {len(game_step.files)}"
            raise SeedFixtureError(msg)
        full_path = self._dir / game_step.files[0]
        data = self._read_json(full_path)
        return self._parse(GameFixture, data, full_path)

    # -- Expansions ---------------------------------------------------------

    def load_expansions(self) -> list[ExpansionFixture]:
        """Load all expansion fixture files declared in the manifest.

        Returns:
            List of ExpansionFixture objects.

        Raises:
            SeedFixtureError: If a file is missing or invalid.
        """
        manifest = self.load_manifest()
        exp_step = self._find_step(manifest, "expansions")
        expansions: list[ExpansionFixture] = []
        for rel_path in exp_step.files:
            full_path = self._dir / rel_path
            data = self._read_json(full_path)
            expansions.append(self._parse(ExpansionFixture, data, full_path))
        return expansions

    # -- Document metadata --------------------------------------------------

    def load_document_metas(self) -> list[DocumentMetaFixture]:
        """Load all document metadata fixture files declared in the manifest.

        Returns:
            List of DocumentMetaFixture objects.

        Raises:
            SeedFixtureError: If a file is missing or invalid.
        """
        manifest = self.load_manifest()
        doc_step = self._find_step(manifest, "documents")
        metas: list[DocumentMetaFixture] = []
        for rel_path in doc_step.files:
            full_path = self._dir / rel_path
            data = self._read_json(full_path)
            metas.append(self._parse(DocumentMetaFixture, data, full_path))
        return metas

    # -- AI configuration ---------------------------------------------------

    def load_ai_config(self) -> tuple[ModelSlotsFixture, GuardrailFixture]:
        """Load model slots and guardrail configuration fixtures.

        Returns:
            Tuple of (ModelSlotsFixture, GuardrailFixture).

        Raises:
            SeedFixtureError: If files are missing or invalid.
        """
        manifest = self.load_manifest()
        ai_step = self._find_step(manifest, "ai-config")

        slots_path = self._find_file(ai_step.files, "model-slots.json")
        guardrails_path = self._find_file(ai_step.files, "guardrails.json")

        slots_data = self._read_json(self._dir / slots_path)
        guardrails_data = self._read_json(self._dir / guardrails_path)

        slots = self._parse(ModelSlotsFixture, slots_data, self._dir / slots_path)
        guardrails = self._parse(GuardrailFixture, guardrails_data, self._dir / guardrails_path)

        return slots, guardrails

    # -- Ground truth -------------------------------------------------------

    def load_ground_truth(self) -> GroundTruthFixture:
        """Load the ground truth Q&A fixture.

        The ground truth file is not referenced by the manifest (it is
        used by validation, not seeding). It is loaded directly by name.

        Returns:
            Parsed GroundTruthFixture.

        Raises:
            SeedFixtureError: If the file is missing or invalid.
        """
        gt_path = self._dir / "ground-truth" / "catan-qa.json"
        data = self._read_json(gt_path)
        return self._parse(GroundTruthFixture, data, gt_path)

    # -- Internal helpers ---------------------------------------------------

    def _read_json(self, path: Path) -> dict[str, Any]:
        """Read and parse a JSON file.

        Args:
            path: Absolute path to the JSON file.

        Returns:
            Parsed JSON as a dict.

        Raises:
            SeedFixtureError: If the file does not exist or contains
                invalid JSON.
        """
        if not path.is_file():
            raise SeedFixtureError(f"Fixture file not found: {path}", path=path)
        try:
            with path.open() as f:
                return json.load(f)  # type: ignore[no-any-return]
        except json.JSONDecodeError as exc:
            raise SeedFixtureError(
                f"Malformed JSON in fixture file: {path}: {exc}",
                path=path,
            ) from exc

    @staticmethod
    def _parse[T: BaseModel](model: type[T], data: dict[str, Any], path: Path) -> T:
        """Validate a dict against a Pydantic model.

        Args:
            model: The Pydantic model class.
            data: Raw dict from JSON.
            path: File path (for error context).

        Returns:
            Validated model instance.

        Raises:
            SeedFixtureError: If validation fails.
        """
        try:
            return model.model_validate(data)
        except Exception as exc:
            raise SeedFixtureError(
                f"Schema validation failed for {path}: {exc}",
                path=path,
            ) from exc

    @staticmethod
    def _find_step(manifest: SeedManifest, step_type: str) -> ManifestStep:
        """Find a manifest step by type.

        Args:
            manifest: The parsed seed manifest.
            step_type: The step type to look for (e.g. "users", "game").

        Returns:
            The matching ManifestStep.

        Raises:
            SeedFixtureError: If no step with the given type exists.
        """
        for step in manifest.steps:
            if step.type == step_type:
                return step
        raise SeedFixtureError(f"No step of type '{step_type}' found in manifest")

    @staticmethod
    def _find_file(files: list[str], filename: str) -> str:
        """Find a file in a list of relative paths by filename suffix.

        Args:
            files: List of relative paths from a manifest step.
            filename: The filename to match (e.g. "model-slots.json").

        Returns:
            The matching relative path.

        Raises:
            SeedFixtureError: If no matching file is found.
        """
        for f in files:
            if f.endswith(filename):
                return f
        raise SeedFixtureError(f"File '{filename}' not found in manifest step files: {files}")

    @staticmethod
    def _resolve_email_source(data: dict[str, Any], path: Path) -> dict[str, Any]:
        """Resolve ``email_source`` fields that reference environment variables.

        If the data contains ``"email_source": "env:VAR_NAME"``, reads the
        environment variable and replaces the field with ``"email": value``.
        If the variable is not set, raises SeedFixtureError.

        Args:
            data: Raw fixture dict.
            path: File path (for error context).

        Returns:
            The dict with ``email_source`` resolved to ``email``.
        """
        data = dict(data)  # shallow copy to avoid mutating the original
        email_source = data.pop("email_source", None)
        if email_source is None:
            return data

        if not isinstance(email_source, str) or not email_source.startswith("env:"):
            raise SeedFixtureError(
                f"Invalid email_source format in {path}: expected 'env:VAR_NAME', "
                f"got '{email_source}'",
                path=path,
            )

        var_name = email_source[4:]  # strip "env:" prefix
        value = os.environ.get(var_name)
        if not value:
            raise SeedFixtureError(
                f"Environment variable '{var_name}' required by {path} is not set",
                path=path,
            )

        # Take the first email if comma-separated
        data["email"] = value.split(",")[0].strip()
        return data
