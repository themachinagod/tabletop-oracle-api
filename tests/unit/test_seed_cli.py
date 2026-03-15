"""Unit tests for the seed CLI entry point.

Tests argument parsing, subcommand dispatch, output formatting, exit codes,
and flag propagation. All SeedService interactions are mocked — no database
or fixture files required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tabletop_oracle.cli.seed import (
    _STATUS_SYMBOLS,
    _format_result,
    build_parser,
    main,
)
from tabletop_oracle.services.seed_service import SeedResult, StepStatus

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_result(*steps: tuple[str, StepStatus, str]) -> SeedResult:
    """Build a SeedResult with the given step tuples."""
    result = SeedResult()
    for step_name, status, detail in steps:
        result.add(step_name, status, detail)
    return result


def _mock_db_context():
    """Create mock session factory and session for patching.

    Returns a tuple of (factory_func, mock_session) where factory_func
    returns an async context manager yielding mock_session.
    """
    mock_session = AsyncMock()
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    def factory():
        return mock_cm

    return factory, mock_session


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestBuildParser:
    """Tests for argument parser construction."""

    def test_default_command_is_none(self) -> None:
        """No subcommand defaults to None (main() maps to 'all')."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_all_subcommand(self) -> None:
        """'all' subcommand is recognized."""
        parser = build_parser()
        args = parser.parse_args(["all"])
        assert args.command == "all"

    def test_users_subcommand(self) -> None:
        """'users' subcommand is recognized."""
        parser = build_parser()
        args = parser.parse_args(["users"])
        assert args.command == "users"

    def test_game_subcommand(self) -> None:
        """'game' subcommand is recognized."""
        parser = build_parser()
        args = parser.parse_args(["game"])
        assert args.command == "game"

    def test_ai_config_subcommand(self) -> None:
        """'ai-config' subcommand is recognized."""
        parser = build_parser()
        args = parser.parse_args(["ai-config"])
        assert args.command == "ai-config"

    def test_validate_subcommand(self) -> None:
        """'validate' subcommand is recognized."""
        parser = build_parser()
        args = parser.parse_args(["validate"])
        assert args.command == "validate"

    def test_reset_subcommand(self) -> None:
        """'reset' subcommand is recognized."""
        parser = build_parser()
        args = parser.parse_args(["reset"])
        assert args.command == "reset"

    def test_force_flag_default_false(self) -> None:
        """--force defaults to False."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.force is False

    def test_force_flag_set(self) -> None:
        """--force sets force to True."""
        parser = build_parser()
        args = parser.parse_args(["--force"])
        assert args.force is True

    def test_skip_validation_default_false(self) -> None:
        """--skip-validation defaults to False."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.skip_validation is False

    def test_skip_validation_flag_set(self) -> None:
        """--skip-validation sets skip_validation to True."""
        parser = build_parser()
        args = parser.parse_args(["--skip-validation"])
        assert args.skip_validation is True

    def test_wait_timeout_default(self) -> None:
        """--wait-timeout defaults to 300."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.wait_timeout == 300

    def test_wait_timeout_custom(self) -> None:
        """--wait-timeout accepts a custom integer."""
        parser = build_parser()
        args = parser.parse_args(["--wait-timeout", "600"])
        assert args.wait_timeout == 600

    def test_force_with_subcommand(self) -> None:
        """--force can be combined with a subcommand."""
        parser = build_parser()
        args = parser.parse_args(["--force", "users"])
        assert args.force is True
        assert args.command == "users"

    def test_all_options_combined(self) -> None:
        """All options can be specified together."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "--force",
                "--skip-validation",
                "--wait-timeout",
                "120",
                "all",
            ]
        )
        assert args.force is True
        assert args.skip_validation is True
        assert args.wait_timeout == 120
        assert args.command == "all"


# ---------------------------------------------------------------------------
# Output formatting tests
# ---------------------------------------------------------------------------


class TestFormatResult:
    """Tests for _format_result output."""

    def test_success_message(self) -> None:
        """Successful result ends with success message."""
        result = _make_result(("users", StepStatus.CREATED, "Created 2 users"))
        output = _format_result(result)
        assert "Seed completed successfully." in output

    def test_failure_message(self) -> None:
        """Failed result ends with error message."""
        result = _make_result(("users", StepStatus.FAILED, "No curator"))
        output = _format_result(result)
        assert "Seed completed with errors." in output

    def test_created_symbol(self) -> None:
        """Created steps show [+] symbol."""
        result = _make_result(("users", StepStatus.CREATED, "Created 2 users"))
        output = _format_result(result)
        assert "[+] users" in output

    def test_skipped_symbol(self) -> None:
        """Skipped steps show [-] symbol."""
        result = _make_result(("users", StepStatus.SKIPPED, "Already exists"))
        output = _format_result(result)
        assert "[-] users" in output

    def test_replaced_symbol(self) -> None:
        """Replaced steps show [~] symbol."""
        result = _make_result(("users", StepStatus.REPLACED, "Replaced"))
        output = _format_result(result)
        assert "[~] users" in output

    def test_failed_symbol(self) -> None:
        """Failed steps show [!] symbol."""
        result = _make_result(("users", StepStatus.FAILED, "Error"))
        output = _format_result(result)
        assert "[!] users" in output

    def test_stubbed_symbol(self) -> None:
        """Stubbed steps show [.] symbol."""
        result = _make_result(("documents", StepStatus.STUBBED, "Not yet"))
        output = _format_result(result)
        assert "[.] documents" in output

    def test_detail_included(self) -> None:
        """Step detail appears after the step name."""
        result = _make_result(("users", StepStatus.CREATED, "Created 2 users"))
        output = _format_result(result)
        assert "Created 2 users" in output

    def test_multiple_steps(self) -> None:
        """Multiple steps are all shown in output."""
        result = _make_result(
            ("users", StepStatus.CREATED, "Created 2"),
            ("game", StepStatus.SKIPPED, "Exists"),
            ("ai_config", StepStatus.CREATED, "3 slots"),
        )
        output = _format_result(result)
        assert "[+] users" in output
        assert "[-] game" in output
        assert "[+] ai_config" in output

    def test_empty_detail_no_separator(self) -> None:
        """Steps with no detail omit the dash separator."""
        result = _make_result(("users", StepStatus.CREATED, ""))
        output = _format_result(result)
        assert "—" not in output.split("\n")[0]

    def test_all_status_symbols_defined(self) -> None:
        """Every StepStatus value has a corresponding symbol."""
        for status in StepStatus:
            assert status.value in _STATUS_SYMBOLS


# ---------------------------------------------------------------------------
# Help output tests
# ---------------------------------------------------------------------------


class TestHelpOutput:
    """Tests that --help produces output without errors."""

    def test_help_flag(self) -> None:
        """--help causes SystemExit with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_subcommand_help(self) -> None:
        """Subcommand --help causes SystemExit with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["all", "--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Subcommand dispatch and exit code tests
# ---------------------------------------------------------------------------


class TestSubcommandAll:
    """Tests for the 'all' subcommand."""

    @patch("tabletop_oracle.cli.seed._build_seed_service")
    @patch("tabletop_oracle.cli.seed._get_session_factory")
    def test_success_exit_zero(self, mock_get_factory, mock_build) -> None:
        """Returns 0 on successful seed."""
        factory, _session = _mock_db_context()
        mock_get_factory.return_value = factory

        mock_service = AsyncMock()
        mock_service.seed_all.return_value = _make_result(
            ("users", StepStatus.CREATED, "ok"),
        )
        mock_build.return_value = mock_service

        exit_code = main(["all"])
        assert exit_code == 0
        mock_service.seed_all.assert_called_once_with(force=False)

    @patch("tabletop_oracle.cli.seed._build_seed_service")
    @patch("tabletop_oracle.cli.seed._get_session_factory")
    def test_failure_exit_one(self, mock_get_factory, mock_build) -> None:
        """Returns 1 when a step fails."""
        factory, _session = _mock_db_context()
        mock_get_factory.return_value = factory

        mock_service = AsyncMock()
        mock_service.seed_all.return_value = _make_result(
            ("users", StepStatus.FAILED, "error"),
        )
        mock_build.return_value = mock_service

        exit_code = main(["all"])
        assert exit_code == 1

    @patch("tabletop_oracle.cli.seed._build_seed_service")
    @patch("tabletop_oracle.cli.seed._get_session_factory")
    def test_force_flag_propagated(self, mock_get_factory, mock_build) -> None:
        """--force flag is passed through to seed_all."""
        factory, _session = _mock_db_context()
        mock_get_factory.return_value = factory

        mock_service = AsyncMock()
        mock_service.seed_all.return_value = _make_result(
            ("users", StepStatus.REPLACED, "replaced"),
        )
        mock_build.return_value = mock_service

        exit_code = main(["--force", "all"])
        assert exit_code == 0
        mock_service.seed_all.assert_called_once_with(force=True)

    @patch("tabletop_oracle.cli.seed._build_seed_service")
    @patch("tabletop_oracle.cli.seed._get_session_factory")
    def test_session_committed(self, mock_get_factory, mock_build) -> None:
        """Database session is committed after seed."""
        factory, mock_session = _mock_db_context()
        mock_get_factory.return_value = factory

        mock_service = AsyncMock()
        mock_service.seed_all.return_value = _make_result(
            ("users", StepStatus.CREATED, "ok"),
        )
        mock_build.return_value = mock_service

        main(["all"])
        mock_session.commit.assert_called_once()


class TestSubcommandUsers:
    """Tests for the 'users' subcommand."""

    @patch("tabletop_oracle.cli.seed._build_seed_service")
    @patch("tabletop_oracle.cli.seed._get_session_factory")
    def test_calls_seed_users(self, mock_get_factory, mock_build) -> None:
        """Delegates to seed_users on SeedService."""
        factory, _ = _mock_db_context()
        mock_get_factory.return_value = factory

        mock_service = AsyncMock()
        mock_service.seed_users.return_value = _make_result(
            ("users", StepStatus.CREATED, "ok"),
        )
        mock_build.return_value = mock_service

        exit_code = main(["users"])
        assert exit_code == 0
        mock_service.seed_users.assert_called_once_with(force=False)

    @patch("tabletop_oracle.cli.seed._build_seed_service")
    @patch("tabletop_oracle.cli.seed._get_session_factory")
    def test_force_propagated(self, mock_get_factory, mock_build) -> None:
        """--force is propagated to seed_users."""
        factory, _ = _mock_db_context()
        mock_get_factory.return_value = factory

        mock_service = AsyncMock()
        mock_service.seed_users.return_value = _make_result(
            ("users", StepStatus.REPLACED, "replaced"),
        )
        mock_build.return_value = mock_service

        exit_code = main(["--force", "users"])
        assert exit_code == 0
        mock_service.seed_users.assert_called_once_with(force=True)


class TestSubcommandGame:
    """Tests for the 'game' subcommand."""

    @patch("tabletop_oracle.cli.seed._build_seed_service")
    @patch("tabletop_oracle.cli.seed._get_session_factory")
    def test_calls_seed_game_and_expansion(self, mock_get_factory, mock_build) -> None:
        """Delegates to seed_game and seed_expansion."""
        factory, _ = _mock_db_context()
        mock_get_factory.return_value = factory

        mock_service = AsyncMock()
        mock_service.seed_game.return_value = _make_result(
            ("game", StepStatus.CREATED, "ok"),
        )
        mock_service.seed_expansion.return_value = _make_result(
            ("expansion", StepStatus.CREATED, "ok"),
        )
        mock_build.return_value = mock_service

        exit_code = main(["game"])
        assert exit_code == 0
        mock_service.seed_game.assert_called_once_with(force=False)
        mock_service.seed_expansion.assert_called_once_with(force=False)

    @patch("tabletop_oracle.cli.seed._build_seed_service")
    @patch("tabletop_oracle.cli.seed._get_session_factory")
    def test_game_failure_exit_one(self, mock_get_factory, mock_build) -> None:
        """Returns 1 when game step fails."""
        factory, _ = _mock_db_context()
        mock_get_factory.return_value = factory

        mock_service = AsyncMock()
        mock_service.seed_game.return_value = _make_result(
            ("game", StepStatus.FAILED, "no curator"),
        )
        mock_service.seed_expansion.return_value = _make_result(
            ("expansion", StepStatus.SKIPPED, "ok"),
        )
        mock_build.return_value = mock_service

        exit_code = main(["game"])
        assert exit_code == 1


class TestSubcommandAiConfig:
    """Tests for the 'ai-config' subcommand."""

    @patch("tabletop_oracle.cli.seed._build_seed_service")
    @patch("tabletop_oracle.cli.seed._get_session_factory")
    def test_calls_seed_ai_config(self, mock_get_factory, mock_build) -> None:
        """Delegates to seed_ai_config."""
        factory, _ = _mock_db_context()
        mock_get_factory.return_value = factory

        mock_service = AsyncMock()
        mock_service.seed_ai_config.return_value = _make_result(
            ("ai_config", StepStatus.CREATED, "ok"),
        )
        mock_build.return_value = mock_service

        exit_code = main(["ai-config"])
        assert exit_code == 0
        mock_service.seed_ai_config.assert_called_once_with(force=False)


class TestSubcommandValidate:
    """Tests for the 'validate' subcommand."""

    def test_returns_zero_stub(self) -> None:
        """Validate stub returns 0."""
        exit_code = main(["validate"])
        assert exit_code == 0


class TestSubcommandReset:
    """Tests for the 'reset' subcommand."""

    @patch("tabletop_oracle.cli.seed._build_seed_service")
    @patch("tabletop_oracle.cli.seed._get_session_factory")
    def test_calls_reset(self, mock_get_factory, mock_build) -> None:
        """Delegates to reset on SeedService."""
        factory, _ = _mock_db_context()
        mock_get_factory.return_value = factory

        mock_service = AsyncMock()
        mock_service.reset.return_value = _make_result(
            ("users", StepStatus.CREATED, "Deleted"),
        )
        mock_build.return_value = mock_service

        exit_code = main(["reset"])
        assert exit_code == 0
        mock_service.reset.assert_called_once()


class TestDefaultCommand:
    """Tests for default command behavior."""

    @patch("tabletop_oracle.cli.seed._build_seed_service")
    @patch("tabletop_oracle.cli.seed._get_session_factory")
    def test_no_subcommand_runs_all(self, mock_get_factory, mock_build) -> None:
        """No subcommand defaults to running 'all'."""
        factory, _ = _mock_db_context()
        mock_get_factory.return_value = factory

        mock_service = AsyncMock()
        mock_service.seed_all.return_value = _make_result(
            ("users", StepStatus.CREATED, "ok"),
        )
        mock_build.return_value = mock_service

        exit_code = main([])
        assert exit_code == 0
        mock_service.seed_all.assert_called_once_with(force=False)


class TestErrorHandling:
    """Tests for error handling in main()."""

    @patch("tabletop_oracle.cli.seed._build_seed_service")
    @patch("tabletop_oracle.cli.seed._get_session_factory")
    def test_unexpected_exception_returns_one(
        self,
        mock_get_factory,
        mock_build,
    ) -> None:
        """Unexpected exception returns exit code 1."""
        mock_get_factory.side_effect = RuntimeError("connection refused")

        exit_code = main(["all"])
        assert exit_code == 1


class TestOutputFormat:
    """Tests for CLI output content."""

    @patch("tabletop_oracle.cli.seed._build_seed_service")
    @patch("tabletop_oracle.cli.seed._get_session_factory")
    def test_success_output_contains_success_message(
        self,
        mock_get_factory,
        mock_build,
        capsys,
    ) -> None:
        """Successful seed prints success message to stdout."""
        factory, _ = _mock_db_context()
        mock_get_factory.return_value = factory

        mock_service = AsyncMock()
        mock_service.seed_all.return_value = _make_result(
            ("users", StepStatus.CREATED, "Created 2 users"),
            ("game", StepStatus.CREATED, "Catan seeded"),
        )
        mock_build.return_value = mock_service

        main(["all"])
        captured = capsys.readouterr()
        assert "Seed completed successfully." in captured.out
        assert "[+] users" in captured.out
        assert "[+] game" in captured.out

    @patch("tabletop_oracle.cli.seed._build_seed_service")
    @patch("tabletop_oracle.cli.seed._get_session_factory")
    def test_failure_output_contains_error_message(
        self,
        mock_get_factory,
        mock_build,
        capsys,
    ) -> None:
        """Failed seed prints error message to stdout."""
        factory, _ = _mock_db_context()
        mock_get_factory.return_value = factory

        mock_service = AsyncMock()
        mock_service.seed_all.return_value = _make_result(
            ("users", StepStatus.FAILED, "Database error"),
        )
        mock_build.return_value = mock_service

        main(["all"])
        captured = capsys.readouterr()
        assert "Seed completed with errors." in captured.out
        assert "[!] users" in captured.out
