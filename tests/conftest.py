import asyncio
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from server.app import app
from server.database import Base
from server.models import Demand, ForecastCache, Weather


TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def async_engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def async_session_factory(async_engine):
    """Create an async session factory for testing."""
    factory = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    return factory


@pytest.fixture
async def async_db_session(async_session_factory):
    """Create a database session for each test."""
    async with async_session_factory() as session:
        yield session


@pytest.fixture
async def async_client(async_session_factory):
    """Create an async test client with a test database session."""
    async def override_get_db():
        async with async_session_factory() as session:
            yield session

    from server.database import get_db
    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
