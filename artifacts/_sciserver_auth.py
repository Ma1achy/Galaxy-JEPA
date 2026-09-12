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


def _login(name: str, secret: str) -> str | None:
    """Mint a token from SciServer credentials, or None — never leak the password."""
    try:
        token = str(Authentication.login(name, secret)).strip()
    except Exception:  # noqa: BLE001 — scrub: the raw error can echo the credentials
        return None
    return token if token and _valid(token) else None


def authenticate(*, verbose: bool = True) -> str:
    """Return a valid SciServer token, refreshing via login if the cached one is stale."""
    # override=True matters: without it a second call is a no-op, because python-dotenv
    # refuses to replace a variable already in os.environ. A long-running driver could then
    # never pick up a refreshed token -- it would re-read the stale one forever and have to be
    # killed and relaunched. Re-reading .env is the whole point of calling this again.
    load_dotenv(ENV, override=True)
    token = os.environ.get("SCISERVER_TOKEN", "").strip()

    user = _valid(token)
    if user:
        Authentication.setToken(token)
        if verbose:
            print(f"auth: cached token OK (user {user})")
        return token

    # Token lifetime is NOT predictable: measured at 24 h once (Sun 19:40 -> Mon 19:39) and
    # 7.5 h the next time (Mon 22:03 -> Tue 05:30). Do not build a clock on it -- probe.
    #
    # Minting from credentials would make a long pull unattended, and SciServer's docs are
    # clear that SSO links to a native account rather than replacing one, so credentials do
    # exist. Measured 2026-09-07: they sign in fine at the web portal but this endpoint
    # (login-portal/keystone/v3/tokens) returns 401 for the same pair -- the portal and the
    # Keystone API are different backends. The branch stays wired because it costs nothing
    # and would start working if that is ever fixed; today it always falls through.
    name = os.environ.get("SCISERVER_USERNAME", "").strip()
    secret = os.environ.get("SCISERVER_PASSWORD", "").strip()
    if name and secret:
        fresh = _login(name, secret)
        if fresh:
            Authentication.setToken(fresh)
            _persist_token(fresh)
            if verbose:
                print(f"auth: minted a fresh token by login (user {_valid(fresh) or name})")
            return fresh
        if verbose:
            print("auth: login with SCISERVER_USERNAME/PASSWORD was refused; falling back")

    raise SystemExit(
        "SciServer token in .env is missing or expired.\n"
        "  Refresh it: sign in at apps.sciserver.org, then read the `portalCookie` cookie --\n"
        "  that IS the token (32 hex). No Compute container or JupyterLab wait is needed.\n"
        "  Put the value on the SCISERVER_TOKEN line in .env.\n"
        "  Credentials do not help: the Keystone login API 401s even for the username and\n"
        "  password the web portal accepts."
    )


if __name__ == "__main__":
    authenticate()
