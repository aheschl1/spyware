"""The published contract."""

import httpx

EXPECTED_METHODS = {
    "/health": {"get"},
    "/health/ready": {"get"},
    "/v1/me": {"get"},
    "/v1/sessions": {"get", "post"},
    "/v1/sessions/{session_id}": {"get"},
    "/v1/sessions/{session_id}/end": {"post"},
    "/v1/sessions/{session_id}/segments": {"get"},
    "/v1/segments": {"get"},
    "/v1/segments/{segment_id}": {"get"},
    "/v1/segments/{segment_id}/audio": {"get"},
}


async def test_openapi_publishes_every_route(client: httpx.AsyncClient) -> None:
    spec = (await client.get("/openapi.json")).json()
    assert {path: set(methods) for path, methods in spec["paths"].items()} == EXPECTED_METHODS
    # The streaming websocket cannot appear in OpenAPI; its contract lives in
    # docs/streaming-protocol.md and is exercised by test_stream.py.
    assert "/v1/sessions/{session_id}/stream" not in spec["paths"]


async def test_segment_schema_hides_storage_layout(client: httpx.AsyncClient) -> None:
    """Guards the omission: a re-added column would fail here, not in review."""
    spec = (await client.get("/openapi.json")).json()
    properties = spec["components"]["schemas"]["SegmentRead"]["properties"]

    assert "bucket" not in properties
    assert "object_key" not in properties
    assert "user_id" not in properties
    assert "checksum_sha256" in properties


async def test_paged_schemas_are_generated_per_item_type(client: httpx.AsyncClient) -> None:
    spec = (await client.get("/openapi.json")).json()
    assert {"Page_SegmentRead_", "Page_SessionRead_"} <= set(spec["components"]["schemas"])


async def test_docs_render(client: httpx.AsyncClient) -> None:
    assert (await client.get("/docs")).status_code == 200
