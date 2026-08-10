"""Invariants of the database layer that no HTTP endpoint exposes."""

from datetime import timedelta
from uuid import uuid4

import pytest

from database.exceptions import DuplicateEmailError, NotFoundError
from database.pipe import DatabasePipe
from database.schema.sessions import SessionCreate
from database.schema.users import UserCreate
from tests.e2e.conftest import Account, ingest, make_session
from tests.wav import wav_bytes


async def test_email_uniqueness_is_case_insensitive_and_survivable(account: Account) -> None:
    """The savepoint matters: the caller must be able to continue afterwards."""
    async with DatabasePipe() as pipe:
        with pytest.raises(DuplicateEmailError):
            await pipe.users.create(
                UserCreate(email=account.user.email.upper(), password="whatever")
            )

        # Same transaction, still usable.
        assert await pipe.users.get_by_email(account.user.email) is not None


async def test_email_lookup_ignores_case_and_padding(account: Account) -> None:
    async with DatabasePipe() as pipe:
        found = await pipe.users.get_by_email(f"  {account.user.email.upper()} ")
    assert found is not None and found.id == account.user.id


async def test_an_escaping_exception_rolls_the_block_back(account: Account) -> None:
    with pytest.raises(RuntimeError):
        async with DatabasePipe() as pipe:
            await pipe.sessions.create(SessionCreate(user_id=account.user.id, label="doomed"))
            raise RuntimeError("boom")

    async with DatabasePipe() as pipe:
        assert await pipe.sessions.list_for_user(account.user.id) == []


async def test_updated_at_trigger_fires(account: Account) -> None:
    session = await make_session(account)
    async with DatabasePipe() as pipe:
        ended = await pipe.sessions.end(session.id)
    assert ended.updated_at > ended.created_at


async def test_password_authentication(account: Account) -> None:
    async with DatabasePipe() as pipe:
        assert await pipe.users.authenticate(account.user.email, "s3cret") is not None
        assert await pipe.users.authenticate(account.user.email, "wrong") is None
        assert await pipe.users.authenticate("nobody@example.com", "s3cret") is None


async def test_deactivated_user_cannot_authenticate(account: Account) -> None:
    async with DatabasePipe() as pipe:
        await pipe.users.set_active(account.user.id, False)
        assert await pipe.users.authenticate(account.user.email, "s3cret") is None


async def test_sequences_increment_within_a_session(account: Account) -> None:
    session = await make_session(account)
    segments = [await ingest(session.id, wav_bytes(freq=freq)) for freq in (440, 660, 880)]
    assert [segment.sequence for segment in segments] == [0, 1, 2]


async def test_sequences_are_independent_per_session(account: Account) -> None:
    one = await make_session(account)
    two = await make_session(account)
    assert (await ingest(one.id, wav_bytes())).sequence == 0
    assert (await ingest(two.id, wav_bytes())).sequence == 0


async def test_deleting_a_user_cascades_to_everything(account: Account) -> None:
    session = await make_session(account)
    await ingest(session.id, wav_bytes())

    async with DatabasePipe() as pipe:
        assert await pipe.users.delete(account.user.id) is True
        assert await pipe.sessions.get(session.id) is None
        assert await pipe.segments.list_for_user(account.user.id) == []
        assert await pipe.tokens.list_for_user(account.user.id) == []


async def test_token_lifecycle(account: Account) -> None:
    async with DatabasePipe() as pipe:
        issued = await pipe.tokens.issue(account.user.id, name="second")
        secret = issued.token.get_secret_value()

        assert (await pipe.tokens.authenticate(secret)).id == account.user.id
        assert await pipe.tokens.revoke(issued.record.id) is True
        assert await pipe.tokens.revoke(issued.record.id) is False, "already revoked"
        assert await pipe.tokens.authenticate(secret) is None
        assert await pipe.tokens.resolve(secret) is None


async def test_expired_tokens_are_purged(account: Account) -> None:
    async with DatabasePipe() as pipe:
        await pipe.tokens.issue(account.user.id, ttl=timedelta(seconds=-1))
        assert await pipe.tokens.purge_expired() == 1
        assert len(await pipe.tokens.list_for_user(account.user.id)) == 1, "live token untouched"


async def test_issuing_for_an_unknown_user_is_not_found() -> None:
    async with DatabasePipe() as pipe:
        with pytest.raises(NotFoundError):
            await pipe.tokens.issue(uuid4())


async def test_ending_an_unknown_session_is_not_found() -> None:
    async with DatabasePipe() as pipe:
        with pytest.raises(NotFoundError):
            await pipe.sessions.end(uuid4())
