from typing import Optional


class TokenPoliceBlockedError(Exception):
    """Raised when an LLM call is blocked by TokenPolice firewall policy.

    Covers budget exhaustion, loop-detection, and any other firewall rule that
    blocks a call. This is the ONLY exception the SDK ever propagates into
    customer code, and only when the firewall is in enforce mode on an explicit
    block decision.

    Structured attributes let callers branch on the block without parsing the
    message string (all Optional — populated best-effort from what the server /
    local decision provided):

        reason: Human-readable block reason from the server.
        rule_id: ID of the firewall rule that blocked the call.
        kind: Block-type discriminator — the loop detector that fired
                  (e.g. "HASH_CYCLE", "SKELETON", "GROWTH", "CAP",
                  "SPAN_NAME_CYCLE") for a loop block, or "budget" for a server
                  budget block.
        trace_id: Trace ID associated with the block.
    """

    def __init__(
        self,
        message: str = "",
        *,
        reason: Optional[str] = None,
        rule_id: Optional[str] = None,
        kind: Optional[str] = None,
        trace_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.reason = reason
        self.rule_id = rule_id
        self.kind = kind
        self.trace_id = trace_id
