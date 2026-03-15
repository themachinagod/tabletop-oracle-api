"""Validation tests for seed fixture files.

Verifies that all JSON fixture files parse correctly, the manifest
references all expected files, required fields are present per the
F001 schema, and the directory structure matches the design spec.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "tabletop_oracle" / "fixtures" / "seed"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict | list:
    """Load and parse a JSON fixture file."""
    with path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------


class TestDirectoryStructure:
    """Verify the fixture directory tree matches the design spec."""

    def test_fixtures_dir_exists(self) -> None:
        """Seed fixtures root directory exists."""
        assert FIXTURES_DIR.is_dir()

    @pytest.mark.parametrize(
        "subdir",
        [
            "users",
            "games/catan",
            "games/catan/expansions",
            "games/catan/documents",
            "ai-config",
            "ground-truth",
        ],
    )
    def test_subdirectory_exists(self, subdir: str) -> None:
        """Each expected subdirectory is present."""
        assert (FIXTURES_DIR / subdir).is_dir()

    @pytest.mark.parametrize(
        "filepath",
        [
            "manifest.json",
            "users/curator.json",
            "users/player.json",
            "games/catan/game.json",
            "games/catan/expansions/seafarers.json",
            "games/catan/documents/catan-base-rules.meta.json",
            "games/catan/documents/catan-seafarers-rules.meta.json",
            "ai-config/model-slots.json",
            "ai-config/guardrails.json",
            "ground-truth/catan-qa.json",
        ],
    )
    def test_fixture_file_exists(self, filepath: str) -> None:
        """Each expected fixture file is present."""
        assert (FIXTURES_DIR / filepath).is_file()


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


class TestJsonParsing:
    """Verify every JSON fixture file parses without errors."""

    @pytest.fixture
    def all_json_files(self) -> list[Path]:
        """Collect all .json files in the fixtures directory."""
        return sorted(FIXTURES_DIR.rglob("*.json"))

    def test_all_json_files_parse(self, all_json_files: list[Path]) -> None:
        """Every JSON file in the fixtures tree is valid JSON."""
        assert len(all_json_files) > 0, "No JSON files found"
        for path in all_json_files:
            try:
                _load_json(path)
            except json.JSONDecodeError as exc:
                pytest.fail(f"{path.relative_to(FIXTURES_DIR)} is invalid JSON: {exc}")


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


class TestManifest:
    """Verify manifest.json is well-formed and references all fixture files."""

    @pytest.fixture
    def manifest(self) -> dict:
        """Load the seed manifest."""
        return _load_json(FIXTURES_DIR / "manifest.json")

    def test_manifest_has_version(self, manifest: dict) -> None:
        """Manifest declares a semantic version."""
        assert "version" in manifest
        parts = manifest["version"].split(".")
        assert len(parts) == 3, "Version must be semver (MAJOR.MINOR.PATCH)"

    def test_manifest_has_description(self, manifest: dict) -> None:
        """Manifest has a human-readable description."""
        assert "description" in manifest
        assert len(manifest["description"]) > 0

    def test_manifest_has_steps(self, manifest: dict) -> None:
        """Manifest declares at least one step."""
        assert "steps" in manifest
        assert len(manifest["steps"]) > 0

    def test_steps_are_ordered(self, manifest: dict) -> None:
        """Steps have sequential step numbers starting at 1."""
        step_numbers = [s["step"] for s in manifest["steps"]]
        assert step_numbers == list(range(1, len(step_numbers) + 1))

    def test_each_step_has_required_fields(self, manifest: dict) -> None:
        """Every step has step number, type, and files list."""
        for step in manifest["steps"]:
            assert "step" in step, f"Step missing 'step' field: {step}"
            assert "type" in step, f"Step missing 'type' field: {step}"
            assert "files" in step, f"Step missing 'files' field: {step}"
            assert isinstance(step["files"], list), f"Step files must be a list: {step}"
            assert len(step["files"]) > 0, f"Step has no files: {step}"

    def test_all_referenced_files_exist(self, manifest: dict) -> None:
        """Every file referenced in the manifest exists on disk."""
        for step in manifest["steps"]:
            for filepath in step["files"]:
                full_path = FIXTURES_DIR / filepath
                assert full_path.is_file(), (
                    f"Manifest references '{filepath}' (step {step['step']}) "
                    f"but file does not exist"
                )

    def test_manifest_covers_expected_types(self, manifest: dict) -> None:
        """Manifest includes all required seed step types."""
        types = {s["type"] for s in manifest["steps"]}
        expected = {"users", "game", "expansions", "documents", "ai-config"}
        assert expected.issubset(types), f"Missing types: {expected - types}"


# ---------------------------------------------------------------------------
# User fixtures
# ---------------------------------------------------------------------------


class TestUserFixtures:
    """Verify user fixture files have required fields per F001 schema."""

    def test_curator_required_fields(self) -> None:
        """Curator fixture has all required fields."""
        data = _load_json(FIXTURES_DIR / "users" / "curator.json")
        assert "email_source" in data, "Curator must declare email_source"
        assert data["email_source"] == "env:INITIAL_CURATOR_EMAILS"
        assert data["display_name"]
        assert data["role"] == "curator"
        assert data["status"] == "active"

    def test_player_required_fields(self) -> None:
        """Player fixture has all required fields."""
        data = _load_json(FIXTURES_DIR / "users" / "player.json")
        assert "email" in data, "Player must have a fixed email"
        assert data["email"]
        assert data["display_name"]
        assert data["role"] == "player"
        assert data["status"] == "active"

    @pytest.mark.parametrize("filename", ["curator.json", "player.json"])
    def test_user_role_is_valid_enum(self, filename: str) -> None:
        """User role matches UserRole enum values."""
        data = _load_json(FIXTURES_DIR / "users" / filename)
        valid_roles = {"player", "curator"}
        assert data["role"] in valid_roles

    @pytest.mark.parametrize("filename", ["curator.json", "player.json"])
    def test_user_status_is_valid_enum(self, filename: str) -> None:
        """User status matches UserStatus enum values."""
        data = _load_json(FIXTURES_DIR / "users" / filename)
        valid_statuses = {"active", "disabled"}
        assert data["status"] in valid_statuses


# ---------------------------------------------------------------------------
# Game fixtures
# ---------------------------------------------------------------------------


class TestGameFixtures:
    """Verify game fixture has required fields per F001 schema."""

    @pytest.fixture
    def game(self) -> dict:
        """Load the Catan game fixture."""
        return _load_json(FIXTURES_DIR / "games" / "catan" / "game.json")

    def test_game_has_name(self, game: dict) -> None:
        """Game has a non-empty name."""
        assert game["name"]

    def test_game_has_description(self, game: dict) -> None:
        """Game has a description."""
        assert "description" in game

    def test_game_complexity_is_valid_enum(self, game: dict) -> None:
        """Game complexity matches GameComplexity enum values."""
        valid = {"light", "medium", "heavy", None}
        assert game.get("complexity") in valid

    def test_game_player_counts_are_positive(self, game: dict) -> None:
        """Player count fields are positive integers when present."""
        if game.get("min_players") is not None:
            assert game["min_players"] > 0
        if game.get("max_players") is not None:
            assert game["max_players"] > 0

    def test_game_min_not_greater_than_max(self, game: dict) -> None:
        """Min players does not exceed max players."""
        if game.get("min_players") and game.get("max_players"):
            assert game["min_players"] <= game["max_players"]

    def test_game_tags_are_strings(self, game: dict) -> None:
        """Tags list contains only non-empty strings."""
        assert "tags" in game
        assert isinstance(game["tags"], list)
        assert all(isinstance(t, str) and len(t) > 0 for t in game["tags"])


# ---------------------------------------------------------------------------
# Expansion fixtures
# ---------------------------------------------------------------------------


class TestExpansionFixtures:
    """Verify expansion fixture has required fields per F001 schema."""

    @pytest.fixture
    def expansion(self) -> dict:
        """Load the Seafarers expansion fixture."""
        return _load_json(FIXTURES_DIR / "games" / "catan" / "expansions" / "seafarers.json")

    def test_expansion_has_name(self, expansion: dict) -> None:
        """Expansion has a non-empty name."""
        assert expansion["name"]

    def test_expansion_has_description(self, expansion: dict) -> None:
        """Expansion has a description."""
        assert "description" in expansion


# ---------------------------------------------------------------------------
# Document meta fixtures
# ---------------------------------------------------------------------------


class TestDocumentMetaFixtures:
    """Verify document .meta.json files have required fields."""

    @pytest.mark.parametrize(
        "filename",
        [
            "catan-base-rules.meta.json",
            "catan-seafarers-rules.meta.json",
        ],
    )
    def test_document_meta_required_fields(self, filename: str) -> None:
        """Document meta has name, type, format, and file reference."""
        data = _load_json(FIXTURES_DIR / "games" / "catan" / "documents" / filename)
        assert data["name"], "Document must have a name"
        assert data["type"], "Document must have a type"
        assert data["format"], "Document must have a format"
        assert data["file"], "Document must reference a PDF file"

    @pytest.mark.parametrize(
        "filename",
        [
            "catan-base-rules.meta.json",
            "catan-seafarers-rules.meta.json",
        ],
    )
    def test_document_type_is_valid_enum(self, filename: str) -> None:
        """Document type matches DocumentType enum values."""
        data = _load_json(FIXTURES_DIR / "games" / "catan" / "documents" / filename)
        valid_types = {"core_rules", "faq", "errata", "expansion_rules", "strategy", "other"}
        assert data["type"] in valid_types

    @pytest.mark.parametrize(
        "filename",
        [
            "catan-base-rules.meta.json",
            "catan-seafarers-rules.meta.json",
        ],
    )
    def test_document_format_is_valid_enum(self, filename: str) -> None:
        """Document format matches DocumentFormat enum values."""
        data = _load_json(FIXTURES_DIR / "games" / "catan" / "documents" / filename)
        valid_formats = {"pdf", "markdown", "text", "html", "docx"}
        assert data["format"] in valid_formats

    def test_base_rules_is_core_rules(self) -> None:
        """Base rules document is classified as core_rules."""
        data = _load_json(
            FIXTURES_DIR / "games" / "catan" / "documents" / "catan-base-rules.meta.json"
        )
        assert data["type"] == "core_rules"
        assert data["expansion_id"] is None

    def test_seafarers_rules_is_expansion_rules(self) -> None:
        """Seafarers rules document is classified as expansion_rules."""
        data = _load_json(
            FIXTURES_DIR / "games" / "catan" / "documents" / "catan-seafarers-rules.meta.json"
        )
        assert data["type"] == "expansion_rules"
        assert "expansion_association" in data


# ---------------------------------------------------------------------------
# AI config fixtures
# ---------------------------------------------------------------------------


class TestModelSlotsFixture:
    """Verify model slots fixture covers all capabilities."""

    @pytest.fixture
    def model_slots(self) -> dict:
        """Load the model slots fixture."""
        return _load_json(FIXTURES_DIR / "ai-config" / "model-slots.json")

    def test_has_slots_array(self, model_slots: dict) -> None:
        """Model slots fixture has a 'slots' array."""
        assert "slots" in model_slots
        assert isinstance(model_slots["slots"], list)

    def test_all_capabilities_covered(self, model_slots: dict) -> None:
        """All ModelCapability enum values (except embedding) have a slot."""
        expected = {
            "intent_analysis",
            "retrieval_augmentation",
            "answer_synthesis",
            "clarification_generation",
            "concept_extraction",
            "vision_processing",
        }
        actual = {s["capability"] for s in model_slots["slots"]}
        assert expected == actual, f"Missing: {expected - actual}, Extra: {actual - expected}"

    def test_each_slot_has_required_fields(self, model_slots: dict) -> None:
        """Every slot declares capability, provider, and model_id."""
        for slot in model_slots["slots"]:
            assert "capability" in slot, f"Slot missing capability: {slot}"
            assert "provider" in slot, f"Slot missing provider: {slot}"
            assert "model_id" in slot, f"Slot missing model_id: {slot}"

    def test_no_duplicate_capabilities(self, model_slots: dict) -> None:
        """No two slots declare the same capability."""
        capabilities = [s["capability"] for s in model_slots["slots"]]
        assert len(capabilities) == len(set(capabilities))


class TestGuardrailsFixture:
    """Verify guardrails fixture has required fields."""

    @pytest.fixture
    def guardrails(self) -> dict:
        """Load the guardrails fixture."""
        return _load_json(FIXTURES_DIR / "ai-config" / "guardrails.json")

    def test_enforcement_enabled_is_bool(self, guardrails: dict) -> None:
        """enforcement_enabled is a boolean."""
        assert isinstance(guardrails["enforcement_enabled"], bool)

    @pytest.mark.parametrize(
        "field",
        [
            "max_tokens_per_query",
            "max_model_calls_per_query",
            "max_queries_per_session",
            "daily_token_budget",
            "daily_query_budget",
            "per_document_ingestion_limit",
        ],
    )
    def test_guardrail_limit_is_positive_int(self, guardrails: dict, field: str) -> None:
        """Each guardrail limit is a positive integer when present."""
        value = guardrails.get(field)
        if value is not None:
            assert isinstance(value, int), f"{field} must be an int"
            assert value > 0, f"{field} must be positive"


# ---------------------------------------------------------------------------
# Ground truth fixtures
# ---------------------------------------------------------------------------


class TestGroundTruthFixture:
    """Verify ground truth Q&A fixture structure."""

    @pytest.fixture
    def qa_set(self) -> dict:
        """Load the ground truth Q&A fixture."""
        return _load_json(FIXTURES_DIR / "ground-truth" / "catan-qa.json")

    def test_has_version(self, qa_set: dict) -> None:
        """Q&A set declares a version."""
        assert "version" in qa_set

    def test_has_game(self, qa_set: dict) -> None:
        """Q&A set declares the target game."""
        assert qa_set["game"] == "Catan"

    def test_has_questions(self, qa_set: dict) -> None:
        """Q&A set has at least one question."""
        assert "questions" in qa_set
        assert len(qa_set["questions"]) > 0

    def test_each_question_has_required_fields(self, qa_set: dict) -> None:
        """Every question has id, category, question text, and expected."""
        for q in qa_set["questions"]:
            assert "id" in q, f"Question missing id: {q}"
            assert "category" in q, f"Question missing category: {q}"
            assert "question" in q, f"Question missing question text: {q}"
            assert "expected" in q, f"Question missing expected: {q}"

    def test_question_ids_are_unique(self, qa_set: dict) -> None:
        """No duplicate question IDs."""
        ids = [q["id"] for q in qa_set["questions"]]
        assert len(ids) == len(set(ids))

    def test_expected_has_citation_requirements(self, qa_set: dict) -> None:
        """Each question's expected block declares citation requirements."""
        for q in qa_set["questions"]:
            expected = q["expected"]
            assert "must_cite" in expected, f"{q['id']} missing must_cite"
