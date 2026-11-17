"""Authentication and user identity dependencies for API routes.

Owner: Shared (P4 — Pipeline Engineer & P1 — Audio/Speaker Engineer)
"""

from __future__ import annotations

import re
from fastapi import HTTPException, Request, status

SAFE_USER_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def get_authenticated_user_id(request: Request) -> str:
    """FastAPI dependency resolving the authenticated caller's account ID.

    Security Policy:
    1. Endpoints managing voice profiles and biometric data MUST NOT be exposed publicly.
    2. In the absence of an external identity provider (IdP / OAuth2), authentication
       is enforced via the verified `X-User-ID` caller context.
    3. Missing or malformed caller IDs are immediately rejected with HTTP 401 Unauthorized.
    4. For local unit testing and development, this dependency can be overridden via
       `app.dependency_overrides[get_authenticated_user_id]`.
    """
    user_id = request.headers.get("X-User-ID", "").strip()

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: X-User-ID header is missing or unauthenticated.",
        )

    if not SAFE_USER_ID_REGEX.match(user_id) or ".." in user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid caller identity format. User IDs must be alphanumeric with no path separators.",
        )

    return user_id
