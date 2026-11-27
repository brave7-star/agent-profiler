"""Read a Claude CLI envelope even when the process exits unsuccessfully."""

from __future__ import annotations

import json


def read_result(done) -> tuple[dict | None, dict | None]:
    """Return (envelope, failure), keeping API diagnostics and billed failures.

    The CLI can put an API error in stdout while leaving stderr empty. Its subtype
    may even say success, so the explicit is_error flag must take precedence.
    """
    stdout = getattr(done, "stdout", "") or ""
    stderr = getattr(done, "stderr", "") or ""
    try:
        envelope = json.loads(stdout)
    except (ValueError, TypeError):
        envelope = None
    if not isinstance(envelope, dict):
        envelope = None
    rejected = envelope is not None and (
        envelope.get("is_error") is True
        or str(envelope.get("subtype") or "").startswith("error"))
    if done.returncode == 0 and envelope is not None and not rejected:
        return envelope, None

    details = [stderr] if stderr else []
    if envelope is not None:
        result = envelope.get("result")
        if isinstance(result, str) and result:
            details.append(result)
        errors = envelope.get("errors")
        if isinstance(errors, list):
            details.extend(e for e in errors if isinstance(e, str))
        elif isinstance(errors, str):
            details.append(errors)
    elif stdout:
        details.append(stdout)
    # Reuse the existing credential redactor before shortening the diagnostic.
    from .judge import _redact

    detail = " ".join(_redact("; ".join(details))[0].split())[:500]
    label = (f"exit {done.returncode}" if done.returncode else
             "CLI reported an error" if rejected else "non-JSON envelope")
    cost = (envelope or {}).get("total_cost_usd")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool):
        cost = None
    failure = {"ok": False, "error": label + (f": {detail}" if detail else ""),
               "cost_usd": cost}
    if envelope and envelope.get("api_error_status") is not None:
        failure["api_error_status"] = envelope["api_error_status"]
    return envelope, failure
