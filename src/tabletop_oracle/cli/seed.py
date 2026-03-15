"""Seed data management command.

Provides a CLI for seeding a fresh Tabletop Oracle deployment with baseline
data: user accounts, pilot game (Catan), AI model configuration, and
documents. Each subcommand delegates to SeedService, which handles
idempotency, dependency ordering, and error recovery.

Usage::

    python -m tabletop_oracle.cli.seed [command] [options]

Commands:
    all           Run the full seed process (default)
    users         Seed user accounts only
    game          Seed pilot game and documents only
    ai-config     Seed model slots and guardrails only
    validate      Run ground truth validation only
    reset         Remove all seed data (dev only)

Options:
    --force             Re-seed even if data exists
    --skip-validation   Skip ground truth validation after seeding
    --wait-timeout N    Max seconds to wait for ingestion (default: 300)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from tabletop_oracle.services.seed_service import SeedResult, SeedService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Status symbols for terminal output
# ---------------------------------------------------------------------------

_STATUS_SYMBOLS = {
    "created": "[+]",
    "replaced": "[~]",
    "skipped": "[-]",
    "failed": "[!]",
    "stubbed": "[.]",
}


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _format_result(result: SeedResult) -> str:
    """Format a SeedResult into human-readable terminal output.

    Each step is shown with a status symbol, name, and detail message.
    A summary line follows with the overall outcome.

    Args:
        result: The SeedResult to format.

    Returns:
        Multi-line string suitable for printing to stdout/stderr.
    """
    lines: list[str] = []
    for step in result.steps:
        symbol = _STATUS_SYMBOLS.get(step.status.value, "[?]")
        detail = f" — {step.detail}" if step.detail else ""
        lines.append(f"  {symbol} {step.step}{detail}")

    if result.success:
        lines.append("\nSeed completed successfully.")
    else:
        lines.append("\nSeed completed with errors.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session and service factories
# ---------------------------------------------------------------------------


def _get_session_factory():
    """Import and return the async session factory.

    Deferred import avoids loading the database module at CLI parse time,
    which would fail if the database URL is not configured. Also provides
    a single patch point for tests.

    Returns:
        The async_sessionmaker callable from tabletop_oracle.database.
    """
    from tabletop_oracle.database import async_session_factory

    return async_session_factory


def _build_seed_service(session: AsyncSession) -> SeedService:
    """Construct a SeedService with all required dependencies.

    Wires up the fixture loader, game service, expansion service, and
    model slot repository using the provided database session.

    Args:
        session: Active AsyncSession for database operations.

    Returns:
        Configured SeedService instance.
    """
    from tabletop_oracle.repositories.expansion_repository import ExpansionRepository
    from tabletop_oracle.repositories.game_repository import GameRepository
    from tabletop_oracle.services.expansion_service import ExpansionService
    from tabletop_oracle.services.game_service import GameService
    from tabletop_oracle.services.model.slot_repository import ModelSlotRepository
    from tabletop_oracle.services.seed_fixture_loader import SeedFixtureLoader
    from tabletop_oracle.services.seed_service import SeedService as SeedServiceCls

    fixture_loader = SeedFixtureLoader()
    game_repo = GameRepository(session)
    game_service = GameService(game_repo)
    expansion_repo = ExpansionRepository(session)
    expansion_service = ExpansionService(expansion_repo, game_repo)
    model_slot_repo = ModelSlotRepository(session)

    return SeedServiceCls(
        db=session,
        fixture_loader=fixture_loader,
        game_service=game_service,
        expansion_service=expansion_service,
        model_slot_repo=model_slot_repo,
    )


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


async def _run_all(args: argparse.Namespace) -> int:
    """Execute the full seed process.

    Args:
        args: Parsed CLI arguments (force, skip_validation, wait_timeout).

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    factory = _get_session_factory()
    async with factory() as session:
        service = _build_seed_service(session)
        result = await service.seed_all(force=args.force)
        await session.commit()

    print(_format_result(result))
    return 0 if result.success else 1


async def _run_users(args: argparse.Namespace) -> int:
    """Seed user accounts only.

    Args:
        args: Parsed CLI arguments (force).

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    factory = _get_session_factory()
    async with factory() as session:
        service = _build_seed_service(session)
        result = await service.seed_users(force=args.force)
        await session.commit()

    print(_format_result(result))
    return 0 if result.success else 1


async def _run_game(args: argparse.Namespace) -> int:
    """Seed pilot game, expansion, and documents only.

    Runs the game and expansion seed steps in order, since expansion
    depends on the game existing.

    Args:
        args: Parsed CLI arguments (force).

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    from tabletop_oracle.services.seed_service import SeedResult as SeedResultCls

    factory = _get_session_factory()
    async with factory() as session:
        service = _build_seed_service(session)

        game_result = await service.seed_game(force=args.force)
        expansion_result = await service.seed_expansion(force=args.force)

        combined = SeedResultCls()
        combined.steps.extend(game_result.steps)
        combined.steps.extend(expansion_result.steps)

        await session.commit()

    print(_format_result(combined))
    return 0 if combined.success else 1


async def _run_ai_config(args: argparse.Namespace) -> int:
    """Seed model slots and guardrail configuration only.

    Args:
        args: Parsed CLI arguments (force).

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    factory = _get_session_factory()
    async with factory() as session:
        service = _build_seed_service(session)
        result = await service.seed_ai_config(force=args.force)
        await session.commit()

    print(_format_result(result))
    return 0 if result.success else 1


async def _run_validate(args: argparse.Namespace) -> int:
    """Run ground truth validation only.

    Stubbed — ground truth validation will be implemented in task #87.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code: 0 (stub always succeeds).
    """
    print("  [.] validate — ground truth validation not yet implemented (#87)")
    print("\nValidation skipped (stub).")
    return 0


async def _run_reset(args: argparse.Namespace) -> int:
    """Remove all seed data (development only).

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    factory = _get_session_factory()
    async with factory() as session:
        service = _build_seed_service(session)
        result = await service.reset()
        await session.commit()

    print(_format_result(result))
    return 0 if result.success else 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the seed CLI.

    Returns:
        Configured ArgumentParser with subcommands and global options.
    """
    parser = argparse.ArgumentParser(
        prog="python -m tabletop_oracle.cli.seed",
        description="Seed data management for Tabletop Oracle",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-seed even if data exists (resets to baseline)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        default=False,
        help="Skip ground truth validation after seeding",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=300,
        help="Max seconds to wait for ingestion (default: 300)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Seed subcommand")

    subparsers.add_parser("all", help="Run the full seed process (default)")
    subparsers.add_parser("users", help="Seed user accounts only")
    subparsers.add_parser("game", help="Seed pilot game and documents only")
    subparsers.add_parser("ai-config", help="Seed model slots and guardrails only")
    subparsers.add_parser("validate", help="Run ground truth validation only")
    subparsers.add_parser("reset", help="Remove all seed data (dev only)")

    return parser


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------

_COMMAND_MAP = {
    "all": _run_all,
    "users": _run_users,
    "game": _run_game,
    "ai-config": _run_ai_config,
    "ai_config": _run_ai_config,
    "validate": _run_validate,
    "reset": _run_reset,
}


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate subcommand.

    Defaults to ``all`` when no subcommand is specified. Configures
    logging before running the async handler.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Default to "all" when no subcommand given
    command = args.command or "all"

    handler = _COMMAND_MAP.get(command)
    if handler is None:
        parser.print_help()
        return 1

    # Configure logging for CLI context
    log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        return asyncio.run(handler(args))
    except KeyboardInterrupt:
        print("\nSeed operation cancelled.")
        return 1
    except Exception:
        logger.exception("Seed operation failed with an unexpected error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
