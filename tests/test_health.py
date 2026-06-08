import pytest


@pytest.mark.asyncio
async def test_health_endpoint(async_client):
    """Test that the health endpoint returns ok status."""
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "postgresql"
