"""The authenticated caller."""

from fastapi import APIRouter

from api.deps import CurrentUser, Pipe
from api.schema.users import MeRead, UsageRead, UserRead

router = APIRouter(tags=["me"])


@router.get("/me", summary="Who this token belongs to")
async def read_me(user: CurrentUser, pipe: Pipe) -> MeRead:
    """Identity plus stored-audio totals."""
    usage = await pipe.segments.usage_for_user(user.id)
    return MeRead(user=UserRead.from_model(user), usage=UsageRead.from_model(usage))
