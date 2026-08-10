"""Credential login: the one unauthenticated POST."""

from fastapi import APIRouter, HTTPException, status

from api.deps import Pipe
from api.schema.auth import LoginRequest, TokenIssued

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", summary="Exchange email + password for an API token")
async def login(pipe: Pipe, body: LoginRequest) -> TokenIssued:
    """Verify the credentials and mint a token (equivalent to `tokens issue`).

    Unknown email and wrong password are indistinguishable in both response
    and timing, so account existence does not leak.
    """
    user = await pipe.users.authenticate(body.email, body.password.get_secret_value())
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid email or password"
        )
    issued = await pipe.tokens.issue(user.id, name=body.name)
    return TokenIssued(token=issued.token.get_secret_value())
