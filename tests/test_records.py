import pytest


@pytest.mark.asyncio
async def test_records_count_empty(async_client):
    """Test records count endpoint returns 0 when no records exist."""
    response = await async_client.get("/records/count")
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 0


@pytest.mark.asyncio
async def test_records_count_with_records(async_client):
    """Test records count endpoint after ingesting records."""
    # Ingest some records
    for hour in range(5):
        payload = {
            "date": "2026-06-07",
            "hour": hour,
            "demand": 20000.0 + hour * 100,
        }
        await async_client.post("/ingest", json=payload)

    response = await async_client.get("/records/count")
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 5


@pytest.mark.asyncio
async def test_records_endpoint_empty(async_client):
    """Test records endpoint returns empty list when no records exist."""
    response = await async_client.get("/records")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


@pytest.mark.asyncio
async def test_records_endpoint_with_data(async_client):
    """Test records endpoint returns ingested records."""
    # Ingest records
    records_data = [
        {"date": "2026-06-07", "hour": 0, "demand": 20000.0},
        {"date": "2026-06-07", "hour": 1, "demand": 20100.0},
        {"date": "2026-06-07", "hour": 2, "demand": 20200.0},
    ]
    for record in records_data:
        await async_client.post("/ingest", json=record)

    response = await async_client.get("/records")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    assert data[0]["Hour"] == 0
    assert data[0]["Ontario Demand"] == 20000.0
    assert data[2]["Hour"] == 2


@pytest.mark.asyncio
async def test_records_endpoint_with_limit(async_client):
    """Test records endpoint with limit parameter."""
    # Ingest 10 records
    for hour in range(10):
        payload = {
            "date": "2026-06-07",
            "hour": hour % 24,
            "demand": 20000.0 + hour * 100,
        }
        await async_client.post("/ingest", json=payload)

    response = await async_client.get("/records?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5


@pytest.mark.asyncio
async def test_records_endpoint_with_after_id(async_client):
    """Test records endpoint with after_id parameter."""
    # Ingest records
    for hour in range(5):
        payload = {
            "date": "2026-06-07",
            "hour": hour,
            "demand": 20000.0 + hour * 100,
        }
        await async_client.post("/ingest", json=payload)

    # Get all records to find an ID
    response = await async_client.get("/records")
    all_records = response.json()
    first_id = all_records[0]["id"]

    # Get records after first ID
    response = await async_client.get(f"/records?after_id={first_id}")
    assert response.status_code == 200
    data = response.json()
    # Should return records with ID > first_id
    assert all(record["id"] > first_id for record in data)
