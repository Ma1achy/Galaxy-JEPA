"""Authenticate to SciServer with an auto-refreshing token (no chat paste, no reprints).

The pull scripts import :func:`authenticate` instead of touching the token directly. It:

  1. loads the gitignored ``.env`` (``SCISERVER_TOKEN`` + optional ``SCISERVER_USERNAME`` /
     ``SCISERVER_PASSWORD``);
  2. uses the cached token if it still validates;
  3. otherwise — if username + password are present — logs in, mints a fresh token, sets it
     on the session, and rewrites the ``SCISERVER_TOKEN=`` line in ``.env`` in place.

Secrets are never printed or logged: validation/login failures are scrubbed to a generic
message (the SciServer client otherwise echoes the token in its 401 text). ``.env`` is
gitignored, so nothing is committed.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from SciServer import Authentication

ENV = Path(__file__).resolve().parent.parent / ".env"


def _persist_token(token: str) -> None:
    """Rewrite the SCISERVER_TOKEN line in .env in place (append if absent)."""
    lines = ENV.read_text().splitlines() if ENV.exists() else []
    out, found = [], False
    for line in lines:
        if line.strip().startswith("SCISERVER_TOKEN="):
            out.append(f"SCISERVER_TOKEN={token}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"SCISERVER_TOKEN={token}")
    ENV.write_text("\n".join(out) + "\n")


def _valid(token: str) -> str | None:
    """Return the username if the token validates, else None — never leak the token."""
    if not token:
        return None
    try:
        return str(Authentication.getKeystoneUserWithToken(token).userName)
    except Exception:  # noqa: BLE001 — scrub: the raw error embeds the token
        return None


def authenticate(*, verbose: bool = True) -> str:
    """Return a valid SciServer token, refreshing via login if the cached one is stale."""
    load_dotenv(ENV)
    token = os.environ.get("SCISERVER_TOKEN", "").strip()

    user = _valid(token)
    if user:
        Authentication.setToken(token)
        if verbose:
            print(f"auth: cached token OK (user {user})")
        return token

    # This account logs into SciServer via institutional SSO (Microsoft), so the
    # username/password login API cannot mint a token (it 401s for federated accounts).
    # The only path is a portal token: log into SciServer in the browser, open a Compute
    # container, run `from SciServer import Authentication; print(Authentication.getToken())`,
    # and paste it into the SCISERVER_TOKEN line in .env.
    raise SystemExit(
        "SciServer token in .env is missing or expired. This is an SSO account, so it must "
        "be refreshed manually: log into SciServer, open a Compute container, run "
        "`from SciServer import Authentication; print(Authentication.getToken())`, and paste "
        "the value into SCISERVER_TOKEN in .env."
    )


if __name__ == "__main__":
    authenticate()
