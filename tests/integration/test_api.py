import asyncio
from api.main import health_check


def test_health_direct_call():
    """Call the health_check coroutine directly to avoid importing heavy model startup in CI."""
    result = asyncio.run(health_check())
    assert result.status == "healthy"
    assert hasattr(result, "model_loaded")
    assert hasattr(result, "timestamp")
