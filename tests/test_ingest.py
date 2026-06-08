import pytest


@pytest.mark.asyncio
async def test_ingest_valid_record(async_client):
    """Test ingesting a valid demand record."""
    payload = {
        "date": "2026-06-07",
        "hour": 12,
        "demand": 25000.5,
    }
    response = await async_client.post("/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "saved"
    assert data["id"] is not None


@pytest.mark.asyncio
async def test_ingest_duplicate_record(async_client):
    """Test that duplicate records are skipped."""
    payload = {
        "date": "2026-06-07",
        "hour": 12,
        "demand": 25000.5,
    }
    # Ingest first time
    response1 = await async_client.post("/ingest", json=payload)
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["status"] == "saved"
    first_id = data1["id"]

    # Ingest duplicate
    response2 = await async_client.post("/ingest", json=payload)
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["status"] == "skipped"
    assert data2["reason"] == "duplicate"
    assert data2["id"] == first_id


@pytest.mark.asyncio
async def test_ingest_invalid_hour(async_client):
    """Test that invalid hours are rejected."""
    payload = {
        "date": "2026-06-07",
        "hour": 25,  # Invalid hour (0-23)
        "demand": 25000.5,
    }
    response = await async_client.post("/ingest", json=payload)
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_ingest_bulk_records(async_client):
    """Test bulk ingest with multiple records."""
    payload = {
        "rows": [
            {"date": "2026-06-07", "hour": 0, "demand": 20000.0},
            {"date": "2026-06-07", "hour": 1, "demand": 19500.0},
            {"date": "2026-06-07", "hour": 2, "demand": 19000.0},
        ]
    }
    response = await async_client.post("/ingest/bulk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "saved"
    assert data["received"] == 3
    assert data["valid"] == 3
    assert data["saved"] == 3
    assert data["skipped"] == 0


@pytest.mark.asyncio
async def test_ingest_bulk_with_duplicates(async_client):
    """Test bulk ingest with some duplicate records."""
    # Ingest first record
    first_payload = {
        "date": "2026-06-07",
        "hour": 12,
        "demand": 25000.5,
    }
    await async_client.post("/ingest", json=first_payload)

    # Bulk ingest with one duplicate
    bulk_payload = {
        "rows": [
            {"date": "2026-06-07", "hour": 12, "demand": 25000.5},  # Duplicate
            {"date": "2026-06-07", "hour": 13, "demand": 25100.0},  # New
            {"date": "2026-06-07", "hour": 14, "demand": 25200.0},  # New
        ]
    }
    response = await async_client.post("/ingest/bulk", json=bulk_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "saved"
    assert data["saved"] == 2  # Only 2 new records saved
    assert data["skipped"] == 1  # 1 duplicate skipped


@pytest.mark.asyncio
async def test_ingest_bulk_exceeds_limit(async_client):
    """Test that bulk ingest rejects requests exceeding row limit."""
    # Create a payload with more than MAX_BULK_INGEST_ROWS (5000)
    rows = [
        {"date": "2026-06-07", "hour": i % 24, "demand": 20000.0 + i}
        for i in range(5001)
    ]
    payload = {"rows": rows}
    response = await async_client.post("/ingest/bulk", json=payload)
    assert response.status_code == 413
    data = response.json()
    assert "Bulk ingest accepts at most" in data["detail"]
