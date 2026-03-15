"""Unit tests for SeedFixtureLoader.

Verifies that the loader correctly reads, parses, and validates each
fixture type from the seed directory. Tests cover happy paths, missing
files, malformed JSON, schema violations, and email_source resolution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tabletop_oracle.services.seed_fixture_loader import (
    DocumentMetaFixture,
    ExpansionFixture,
    GameFixture,
    GroundTruthFixture,
    GuardrailFixture,
    ModelSlotFixture,
    ModelSlotsFixture,
    SeedFixtureError,
    SeedFixtureLoader,
    SeedManifest,
    UserFixture,
)

# ---------------------------------------------------------------------------
# Real fixtures directory (for integration-style tests against actual files)
# ---------------------------------------------------------------------------

_REAL_FIXTURES_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "tabletop_oracle" / "fixtures" / "seed"
)


# ---------------------------------------------------------------------------
# Helpers — build minimal valid fixture data
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    """Write data as JSON to the given path, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f)


def _minimal_manifest(
    *,
    users: bool = True,
    game: bool = True,
    expansions: bool = True,
    documents: bool = True,
    ai_config: bool = True,
) -> dict[str, Any]:
    """Build a minimal manifest dict with controllable steps."""
    steps: list[dict[str, Any]] = []
    step_num = 1
    if users:
        steps.append(
            {
                "step": step_num,
                "type": "users",
                "files": ["users/curator.json", "users/player.json"],
            }
        )
        step_num += 1
    if game:
        steps.append(
            {
                "step": step_num,
                "type": "game",
                "files": ["games/catan/game.json"],
            }
        )
        step_num += 1
    if expansions:
        steps.append(
            {
                "step": step_num,
                "type": "expansions",
                "files": ["games/catan/expansions/seafarers.json"],
                "parent_game": "Catan",
            }
        )
        step_num += 1
    if documents:
        steps.append(
            {
                "step": step_num,
                "type": "documents",
                "files": ["games/catan/documents/catan-base-rules.meta.json"],
                "parent_game": "Catan",
            }
        )
        step_num += 1
    if ai_config:
        steps.append(
            {
                "step": step_num,
                "type": "ai-config",
                "files": [
                    "ai-config/model-slots.json",
                    "ai-config/guardrails.json",
                ],
            }
        )
    return {"version": "1.0.0", "description": "test", "steps": steps}


def _minimal_user(*, email: str = "a@b.com") -> dict[str, Any]:
    return {"email": email, "display_name": "Test", "role": "player", "status": "active"}


def _minimal_game() -> dict[str, Any]:
    return {
        "name": "TestGame",
        "publisher": "TestPub",
        "year_published": 2020,
        "min_players": 2,
        "max_players": 4,
        "description": "A test game.",
    }


def _minimal_expansion() -> dict[str, Any]:
    return {"name": "TestExp", "year_published": 2021, "description": "An expansion."}


def _minimal_doc_meta() -> dict[str, Any]:
    return {"name": "Test Doc", "type": "core_rules", "format": "pdf", "file": "test.pdf"}


def _minimal_model_slots() -> dict[str, Any]:
    return {
        "slots": [
            {
                "capability": "intent_analysis",
                "provider": "openai",
                "model_id": "gpt-4o-mini",
                "max_tokens_per_call": 1000,
                "temperature": 0.1,
            }
        ]
    }


def _minimal_guardrails() -> dict[str, Any]:
    return {
        "enforcement_enabled": True,
        "max_tokens_per_query": 8000,
        "max_model_calls_per_query": 6,
        "max_queries_per_session": 100,
        "daily_token_budget": 500000,
        "daily_query_budget": 1000,
        "per_document_ingestion_limit": 50000,
    }


def _minimal_ground_truth() -> dict[str, Any]:
    return {
        "version": "1.0.0",
        "game": "TestGame",
        "description": "test",
        "questions": [
            {
                "id": "GT-001",
                "category": "factual",
                "question": "How many resources?",
                "expected": {"answer_contains": ["five"]},
            }
        ],
    }


def _setup_all_fixtures(base: Path) -> None:
    """Write a full set of minimal fixtures to a temp directory."""
    _write_json(base / "manifest.json", _minimal_manifest())
    _write_json(base / "users" / "curator.json", _minimal_user(email="admin@test.com"))
    _write_json(base / "users" / "player.json", _minimal_user(email="player@test.com"))
    _write_json(base / "games" / "catan" / "game.json", _minimal_game())
    _write_json(base / "games" / "catan" / "expansions" / "seafarers.json", _minimal_expansion())
    _write_json(
        base / "games" / "catan" / "documents" / "catan-base-rules.meta.json",
        _minimal_doc_meta(),
    )
    _write_json(base / "ai-config" / "model-slots.json", _minimal_model_slots())
    _write_json(base / "ai-config" / "guardrails.json", _minimal_guardrails())
    _write_json(base / "ground-truth" / "catan-qa.json", _minimal_ground_truth())


# ---------------------------------------------------------------------------
# Tests — Manifest
# ---------------------------------------------------------------------------


class TestLoadManifest:
    """Tests for loading the seed manifest."""

    def test_load_manifest_from_real_fixtures(self) -> None:
        """Loads the actual manifest.json and returns a SeedManifest."""
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        manifest = loader.load_manifest()
        assert isinstance(manifest, SeedManifest)
        assert manifest.version == "1.0.0"
        assert len(manifest.steps) == 5

    def test_load_manifest_step_types(self) -> None:
        """Manifest contains the expected step types in order."""
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        manifest = loader.load_manifest()
        types = [s.type for s in manifest.steps]
        assert types == ["users", "game", "expansions", "documents", "ai-config"]

    def test_load_manifest_from_temp(self, tmp_path: Path) -> None:
        """Loads a manifest from a temp directory."""
        _write_json(tmp_path / "manifest.json", _minimal_manifest())
        loader = SeedFixtureLoader(tmp_path)
        manifest = loader.load_manifest()
        assert isinstance(manifest, SeedManifest)

    def test_load_manifest_missing_file(self, tmp_path: Path) -> None:
        """Raises SeedFixtureError when manifest.json is missing."""
        loader = SeedFixtureLoader(tmp_path)
        with pytest.raises(SeedFixtureError, match="not found"):
            loader.load_manifest()

    def test_load_manifest_malformed_json(self, tmp_path: Path) -> None:
        """Raises SeedFixtureError for invalid JSON."""
        (tmp_path / "manifest.json").write_text("{bad json!!}")
        loader = SeedFixtureLoader(tmp_path)
        with pytest.raises(SeedFixtureError, match="Malformed JSON"):
            loader.load_manifest()

    def test_load_manifest_schema_violation(self, tmp_path: Path) -> None:
        """Raises SeedFixtureError when required fields are missing."""
        _write_json(tmp_path / "manifest.json", {"not": "a manifest"})
        loader = SeedFixtureLoader(tmp_path)
        with pytest.raises(SeedFixtureError, match="Schema validation failed"):
            loader.load_manifest()


# ---------------------------------------------------------------------------
# Tests — Users
# ---------------------------------------------------------------------------


class TestLoadUsers:
    """Tests for loading user fixtures."""

    def test_load_users_from_real_fixtures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Loads actual user fixtures (curator needs env var)."""
        monkeypatch.setenv("INITIAL_CURATOR_EMAILS", "admin@example.com")
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        users = loader.load_users()
        assert len(users) == 2
        assert all(isinstance(u, UserFixture) for u in users)

    def test_curator_email_resolved_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Curator email is resolved from INITIAL_CURATOR_EMAILS env var."""
        monkeypatch.setenv("INITIAL_CURATOR_EMAILS", "curator@test.dev")
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        users = loader.load_users()
        curator = next(u for u in users if u.role == "curator")
        assert curator.email == "curator@test.dev"

    def test_curator_email_takes_first_when_comma_separated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When INITIAL_CURATOR_EMAILS has multiple emails, takes the first."""
        monkeypatch.setenv("INITIAL_CURATOR_EMAILS", "first@test.dev, second@test.dev")
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        users = loader.load_users()
        curator = next(u for u in users if u.role == "curator")
        assert curator.email == "first@test.dev"

    def test_curator_email_env_var_missing_raises(self) -> None:
        """Raises SeedFixtureError when the env var is not set."""
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        # Don't set the env var — the real curator.json requires it
        with pytest.raises(SeedFixtureError, match="INITIAL_CURATOR_EMAILS"):
            loader.load_users()

    def test_player_has_fixed_email(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Player fixture has a fixed email address."""
        monkeypatch.setenv("INITIAL_CURATOR_EMAILS", "admin@test.dev")
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        users = loader.load_users()
        player = next(u for u in users if u.role == "player")
        assert player.email == "test-player@tabletop-oracle.dev"

    def test_load_users_from_temp(self, tmp_path: Path) -> None:
        """Loads users from a temp directory."""
        _setup_all_fixtures(tmp_path)
        loader = SeedFixtureLoader(tmp_path)
        users = loader.load_users()
        assert len(users) == 2

    def test_load_users_missing_file(self, tmp_path: Path) -> None:
        """Raises SeedFixtureError when a user file is missing."""
        _write_json(tmp_path / "manifest.json", _minimal_manifest())
        _write_json(tmp_path / "users" / "curator.json", _minimal_user())
        # player.json is missing
        loader = SeedFixtureLoader(tmp_path)
        with pytest.raises(SeedFixtureError, match="not found"):
            loader.load_users()

    def test_load_users_schema_violation(self, tmp_path: Path) -> None:
        """Raises SeedFixtureError when user data is invalid."""
        _write_json(tmp_path / "manifest.json", _minimal_manifest())
        _write_json(tmp_path / "users" / "curator.json", {"bad": "data"})
        _write_json(tmp_path / "users" / "player.json", _minimal_user())
        loader = SeedFixtureLoader(tmp_path)
        with pytest.raises(SeedFixtureError, match="Schema validation failed"):
            loader.load_users()

    def test_invalid_email_source_format(self, tmp_path: Path) -> None:
        """Raises SeedFixtureError for an unrecognised email_source format."""
        _write_json(tmp_path / "manifest.json", _minimal_manifest())
        _write_json(
            tmp_path / "users" / "curator.json",
            {
                "email_source": "file:/etc/passwd",
                "display_name": "X",
                "role": "curator",
                "status": "active",
            },
        )
        _write_json(tmp_path / "users" / "player.json", _minimal_user())
        loader = SeedFixtureLoader(tmp_path)
        with pytest.raises(SeedFixtureError, match="Invalid email_source format"):
            loader.load_users()


# ---------------------------------------------------------------------------
# Tests — Game
# ---------------------------------------------------------------------------


class TestLoadGame:
    """Tests for loading the game fixture."""

    def test_load_game_from_real_fixtures(self) -> None:
        """Loads the actual Catan game fixture."""
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        game = loader.load_game()
        assert isinstance(game, GameFixture)
        assert game.name == "Catan"
        assert game.min_players == 3
        assert game.max_players == 4
        assert game.year_published == 1995

    def test_game_has_tags(self) -> None:
        """Catan game fixture has expected tags."""
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        game = loader.load_game()
        assert "strategy" in game.tags
        assert len(game.tags) == 5

    def test_load_game_from_temp(self, tmp_path: Path) -> None:
        """Loads a game from temp fixtures."""
        _setup_all_fixtures(tmp_path)
        loader = SeedFixtureLoader(tmp_path)
        game = loader.load_game()
        assert game.name == "TestGame"

    def test_load_game_missing_file(self, tmp_path: Path) -> None:
        """Raises SeedFixtureError when game file is missing."""
        _write_json(tmp_path / "manifest.json", _minimal_manifest())
        loader = SeedFixtureLoader(tmp_path)
        with pytest.raises(SeedFixtureError, match="not found"):
            loader.load_game()


# ---------------------------------------------------------------------------
# Tests — Expansions
# ---------------------------------------------------------------------------


class TestLoadExpansions:
    """Tests for loading expansion fixtures."""

    def test_load_expansions_from_real_fixtures(self) -> None:
        """Loads the actual Seafarers expansion fixture."""
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        expansions = loader.load_expansions()
        assert len(expansions) == 1
        assert expansions[0].name == "Catan: Seafarers"
        assert expansions[0].year_published == 1997

    def test_load_expansions_from_temp(self, tmp_path: Path) -> None:
        """Loads expansions from temp fixtures."""
        _setup_all_fixtures(tmp_path)
        loader = SeedFixtureLoader(tmp_path)
        expansions = loader.load_expansions()
        assert len(expansions) == 1
        assert isinstance(expansions[0], ExpansionFixture)

    def test_load_expansions_missing_file(self, tmp_path: Path) -> None:
        """Raises SeedFixtureError when expansion file is missing."""
        _write_json(tmp_path / "manifest.json", _minimal_manifest())
        loader = SeedFixtureLoader(tmp_path)
        with pytest.raises(SeedFixtureError, match="not found"):
            loader.load_expansions()


# ---------------------------------------------------------------------------
# Tests — Document metadata
# ---------------------------------------------------------------------------


class TestLoadDocumentMetas:
    """Tests for loading document metadata fixtures."""

    def test_load_document_metas_from_real_fixtures(self) -> None:
        """Loads actual document metadata fixtures."""
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        metas = loader.load_document_metas()
        assert len(metas) == 2
        assert all(isinstance(m, DocumentMetaFixture) for m in metas)

    def test_base_rules_meta_fields(self) -> None:
        """Base rules meta has correct fields."""
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        metas = loader.load_document_metas()
        base = next(m for m in metas if m.type == "core_rules")
        assert base.name == "Catan Base Game Rules"
        assert base.format == "pdf"
        assert base.file == "catan-base-rules.pdf"
        assert base.expansion_id is None

    def test_seafarers_rules_meta_fields(self) -> None:
        """Seafarers rules meta has correct fields."""
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        metas = loader.load_document_metas()
        seafarers = next(m for m in metas if m.type == "expansion_rules")
        assert seafarers.name == "Catan: Seafarers Rules"
        assert seafarers.expansion_association == "Catan: Seafarers"

    def test_load_document_metas_from_temp(self, tmp_path: Path) -> None:
        """Loads document metas from temp fixtures."""
        _setup_all_fixtures(tmp_path)
        loader = SeedFixtureLoader(tmp_path)
        metas = loader.load_document_metas()
        assert len(metas) == 1

    def test_load_document_metas_missing_file(self, tmp_path: Path) -> None:
        """Raises SeedFixtureError when a doc meta file is missing."""
        _write_json(tmp_path / "manifest.json", _minimal_manifest())
        loader = SeedFixtureLoader(tmp_path)
        with pytest.raises(SeedFixtureError, match="not found"):
            loader.load_document_metas()


# ---------------------------------------------------------------------------
# Tests — AI configuration
# ---------------------------------------------------------------------------


class TestLoadAiConfig:
    """Tests for loading AI configuration fixtures."""

    def test_load_ai_config_from_real_fixtures(self) -> None:
        """Loads actual model slots and guardrails fixtures."""
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        slots, guardrails = loader.load_ai_config()
        assert isinstance(slots, ModelSlotsFixture)
        assert isinstance(guardrails, GuardrailFixture)

    def test_model_slots_count(self) -> None:
        """Model slots fixture contains all 6 capabilities."""
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        slots, _ = loader.load_ai_config()
        assert len(slots.slots) == 6

    def test_model_slot_capabilities(self) -> None:
        """Model slots cover all expected capabilities."""
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        slots, _ = loader.load_ai_config()
        capabilities = {s.capability for s in slots.slots}
        expected = {
            "intent_analysis",
            "answer_synthesis",
            "clarification_generation",
            "concept_extraction",
            "retrieval_augmentation",
            "vision_processing",
        }
        assert capabilities == expected

    def test_model_slot_fields(self) -> None:
        """Individual model slot has expected fields."""
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        slots, _ = loader.load_ai_config()
        intent = next(s for s in slots.slots if s.capability == "intent_analysis")
        assert isinstance(intent, ModelSlotFixture)
        assert intent.provider == "openai"
        assert intent.model_id == "gpt-4o-mini"
        assert intent.max_tokens_per_call == 1000
        assert intent.temperature == pytest.approx(0.1)

    def test_guardrail_fields(self) -> None:
        """Guardrail fixture has all expected fields."""
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        _, guardrails = loader.load_ai_config()
        assert guardrails.enforcement_enabled is True
        assert guardrails.max_tokens_per_query == 8000
        assert guardrails.daily_token_budget == 500000

    def test_load_ai_config_from_temp(self, tmp_path: Path) -> None:
        """Loads AI config from temp fixtures."""
        _setup_all_fixtures(tmp_path)
        loader = SeedFixtureLoader(tmp_path)
        slots, guardrails = loader.load_ai_config()
        assert len(slots.slots) == 1
        assert guardrails.enforcement_enabled is True

    def test_load_ai_config_missing_model_slots(self, tmp_path: Path) -> None:
        """Raises SeedFixtureError when model-slots.json is missing."""
        _write_json(tmp_path / "manifest.json", _minimal_manifest())
        _write_json(tmp_path / "ai-config" / "guardrails.json", _minimal_guardrails())
        loader = SeedFixtureLoader(tmp_path)
        with pytest.raises(SeedFixtureError, match="not found"):
            loader.load_ai_config()

    def test_load_ai_config_missing_guardrails(self, tmp_path: Path) -> None:
        """Raises SeedFixtureError when guardrails.json is missing."""
        _write_json(tmp_path / "manifest.json", _minimal_manifest())
        _write_json(tmp_path / "ai-config" / "model-slots.json", _minimal_model_slots())
        loader = SeedFixtureLoader(tmp_path)
        with pytest.raises(SeedFixtureError, match="not found"):
            loader.load_ai_config()


# ---------------------------------------------------------------------------
# Tests — Ground truth
# ---------------------------------------------------------------------------


class TestLoadGroundTruth:
    """Tests for loading ground truth Q&A fixtures."""

    def test_load_ground_truth_from_real_fixtures(self) -> None:
        """Loads actual ground truth fixture."""
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        gt = loader.load_ground_truth()
        assert isinstance(gt, GroundTruthFixture)
        assert gt.game == "Catan"
        assert len(gt.questions) == 5

    def test_ground_truth_question_fields(self) -> None:
        """Ground truth questions have expected fields."""
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        gt = loader.load_ground_truth()
        q1 = gt.questions[0]
        assert q1.id == "GT-001"
        assert q1.category == "single-doc-factual"
        assert "resource" in q1.question.lower()
        assert q1.expected.must_cite is True

    def test_ground_truth_expected_criteria(self) -> None:
        """Expected criteria are parsed correctly."""
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        gt = loader.load_ground_truth()
        q3 = next(q for q in gt.questions if q.id == "GT-003")
        assert q3.expected.min_citations == 2
        assert "core_rules" in q3.expected.expected_document_types
        assert "expansion_rules" in q3.expected.expected_document_types

    def test_load_ground_truth_from_temp(self, tmp_path: Path) -> None:
        """Loads ground truth from temp fixtures."""
        _setup_all_fixtures(tmp_path)
        loader = SeedFixtureLoader(tmp_path)
        gt = loader.load_ground_truth()
        assert len(gt.questions) == 1

    def test_load_ground_truth_missing_file(self, tmp_path: Path) -> None:
        """Raises SeedFixtureError when ground truth file is missing."""
        loader = SeedFixtureLoader(tmp_path)
        with pytest.raises(SeedFixtureError, match="not found"):
            loader.load_ground_truth()


# ---------------------------------------------------------------------------
# Tests — SeedFixtureError
# ---------------------------------------------------------------------------


class TestSeedFixtureError:
    """Tests for the SeedFixtureError exception."""

    def test_error_message(self) -> None:
        """Error carries the message."""
        err = SeedFixtureError("something broke")
        assert str(err) == "something broke"

    def test_error_path(self) -> None:
        """Error carries the optional path."""
        p = Path("/some/file.json")
        err = SeedFixtureError("bad file", path=p)
        assert err.path == p

    def test_error_path_default_none(self) -> None:
        """Path defaults to None."""
        err = SeedFixtureError("oops")
        assert err.path is None


# ---------------------------------------------------------------------------
# Tests — Edge cases and validation
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and validation boundary tests."""

    def test_temperature_too_high(self, tmp_path: Path) -> None:
        """Model slot with temperature > 2 fails validation."""
        _setup_all_fixtures(tmp_path)
        bad_slots = {
            "slots": [
                {
                    "capability": "test",
                    "provider": "openai",
                    "model_id": "gpt-4o",
                    "max_tokens_per_call": 1000,
                    "temperature": 2.5,
                }
            ]
        }
        _write_json(tmp_path / "ai-config" / "model-slots.json", bad_slots)
        loader = SeedFixtureLoader(tmp_path)
        with pytest.raises(SeedFixtureError, match="Schema validation failed"):
            loader.load_ai_config()

    def test_temperature_negative(self, tmp_path: Path) -> None:
        """Model slot with negative temperature fails validation."""
        _setup_all_fixtures(tmp_path)
        bad_slots = {
            "slots": [
                {
                    "capability": "test",
                    "provider": "openai",
                    "model_id": "gpt-4o",
                    "max_tokens_per_call": 1000,
                    "temperature": -0.1,
                }
            ]
        }
        _write_json(tmp_path / "ai-config" / "model-slots.json", bad_slots)
        loader = SeedFixtureLoader(tmp_path)
        with pytest.raises(SeedFixtureError, match="Schema validation failed"):
            loader.load_ai_config()

    def test_manifest_missing_step_type(self, tmp_path: Path) -> None:
        """Raises SeedFixtureError when requested step type is not in manifest."""
        _write_json(tmp_path / "manifest.json", _minimal_manifest(expansions=False))
        loader = SeedFixtureLoader(tmp_path)
        with pytest.raises(SeedFixtureError, match="No step of type 'expansions'"):
            loader.load_expansions()

    def test_fixtures_dir_property(self, tmp_path: Path) -> None:
        """The fixtures_dir property returns the configured directory."""
        loader = SeedFixtureLoader(tmp_path)
        assert loader.fixtures_dir == tmp_path

    def test_default_fixtures_dir(self) -> None:
        """Default fixtures_dir points to the package fixtures/seed directory."""
        loader = SeedFixtureLoader()
        assert loader.fixtures_dir.name == "seed"
        assert "fixtures" in str(loader.fixtures_dir)

    def test_game_step_with_multiple_files_raises(self, tmp_path: Path) -> None:
        """Raises SeedFixtureError when game step has multiple files."""
        manifest = _minimal_manifest()
        game_step = next(s for s in manifest["steps"] if s["type"] == "game")
        game_step["files"] = ["games/a.json", "games/b.json"]
        _write_json(tmp_path / "manifest.json", manifest)
        loader = SeedFixtureLoader(tmp_path)
        with pytest.raises(SeedFixtureError, match="Expected exactly 1 game file"):
            loader.load_game()

    def test_user_fixture_optional_fields(self, tmp_path: Path) -> None:
        """User fixtures with optional fields omitted parse correctly."""
        _setup_all_fixtures(tmp_path)
        loader = SeedFixtureLoader(tmp_path)
        users = loader.load_users()
        for user in users:
            assert user.oauth_provider is None
            assert user.oauth_subject_id is None

    def test_document_meta_optional_expansion_fields(self, tmp_path: Path) -> None:
        """Document meta with expansion fields omitted parses correctly."""
        _setup_all_fixtures(tmp_path)
        loader = SeedFixtureLoader(tmp_path)
        metas = loader.load_document_metas()
        assert metas[0].expansion_id is None
        assert metas[0].expansion_association is None

    def test_ground_truth_optional_expected_fields(self, tmp_path: Path) -> None:
        """Ground truth expected with only required fields parses correctly."""
        _setup_all_fixtures(tmp_path)
        loader = SeedFixtureLoader(tmp_path)
        gt = loader.load_ground_truth()
        q = gt.questions[0]
        assert q.expected.answer_contains_any == []
        assert q.expected.min_citations is None
        assert q.expected.min_confidence is None

    def test_model_slot_fallback_fields(self) -> None:
        """Model slots with fallback providers are parsed correctly."""
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        slots, _ = loader.load_ai_config()
        synthesis = next(s for s in slots.slots if s.capability == "answer_synthesis")
        assert synthesis.fallback_provider == "openai"
        assert synthesis.fallback_model_id == "gpt-4o-mini"

    def test_model_slot_no_fallback(self) -> None:
        """Model slots without fallback have None fallback fields."""
        loader = SeedFixtureLoader(_REAL_FIXTURES_DIR)
        slots, _ = loader.load_ai_config()
        intent = next(s for s in slots.slots if s.capability == "intent_analysis")
        assert intent.fallback_provider is None
        assert intent.fallback_model_id is None
