import functools
import logging
from typing import Callable, Any

logger = logging.getLogger("token_police")

def fail_safe(func: Callable) -> Callable:
    """
    Fail-open guard: any exception raised by the decorated SDK-internal method
    is swallowed (e.g. timeout, connection error, internal parsing error) so
    the customer's LLM call proceeds. TokenPoliceBlockedError is re-raised — a
    deliberate block is not an error.

    On a swallowed exception the wrapper returns None — only decorate
    functions whose callers treat None as a safe no-op.
    """
    from .exceptions import TokenPoliceBlockedError

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except TokenPoliceBlockedError:
            raise
        except Exception as e:
            # We swallow the error so the LLM call proceeds. The best-effort
            # reporting below is itself fully guarded: this handler is the
            # SDK's outermost fail-open boundary, so a raise from the client
            # lookup, a hostile exception __str__ (evaluated by the f-string),
            # or customer logging config (filters/record factories propagate
            # out of logger.error) must never escape into the caller's LLM
            # call. Import stays deferred inside the guard (import cycle).
            try:
                from .state import get_client
                tp = get_client()
                if tp and tp.log_errors:
                    logger.error(f"TokenPolice instrumentation failed (fail-open): {e}")
            except Exception:
                pass

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        try:
            return await func(*args, **kwargs)
        except TokenPoliceBlockedError:
            raise
        except Exception as e:
            # We swallow the error so the LLM call proceeds. Reporting is
            # best-effort and fully guarded — see sync_wrapper above.
            try:
                from .state import get_client
                tp = get_client()
                if tp and tp.log_errors:
                    logger.error(f"TokenPolice async instrumentation failed (fail-open): {e}")
            except Exception:
                pass

    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
