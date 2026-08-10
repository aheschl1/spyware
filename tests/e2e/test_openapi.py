"""The published contract."""

import httpx

EXPECTED_PATHS = {
    "/health",
    "/health/ready",
    "/v1/me",
    "/v1/sessions",
    "/v1/sessions/{session_id}",
    "/v1/sessions/{session_id}/segments",
    "/v1/segments",
    "/v1/segments/{segment_id}",
    "/v1/segments/{segment_id}/audio",
}


async def test_openapi_publishes_every_route(client: httpx.AsyncClient) -> None:
    spec = (await client.get("/openapi.json")).json()
    assert set(spec["paths"]) == EXPECTED_PATHS
    assert all(set(methods) == {"get"} for methods in spec["paths"].values()), "reads only"


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
