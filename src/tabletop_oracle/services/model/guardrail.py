"""Protocol for guardrail enforcement.

Defines the interface that ``ModelClient`` depends on for pre-call
guardrail checking. The concrete implementation (task #58) reads
limits from the ``guardrail_config`` table and checks current usage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from uuid import UUID

    from tabletop_oracle.services.model.types import GuardrailCheckResult


@runtime_checkable
class GuardrailServiceProtocol(Protocol):
    """Enforces token and query guardrails.

    Reads limits from the ``guardrail_config`` table. Checks current
    usage against limits before each model call.
    """

    async def check_pre_call(
        self,
        session_id: UUID,
        message_id: UUID,
    ) -> GuardrailCheckResult:
        """Check per-query guardrails before an individual model call.

        Checks per-query token total and per-query model call count.

        Args:
            session_id: The session context.
            message_id: The message context.

        Returns:
            GuardrailCheckResult with allowed flag and denial message.
        """
        ...
