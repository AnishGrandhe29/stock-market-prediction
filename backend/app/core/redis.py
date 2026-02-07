"""
Redis cache client for session management and data caching.
"""
from typing import Optional
import redis.asyncio as redis

from app.config import settings

# Redis client instance
redis_client: Optional[redis.Redis] = None


async def init_redis():
    """Initialize Redis connection."""
    global redis_client
    try:
        redis_client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        await redis_client.ping()
        return True
    except Exception:
        redis_client = None
        return False


async def close_redis():
    """Close Redis connection."""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None


async def get_cache(key: str) -> Optional[str]:
    """Get value from cache."""
    if redis_client:
        return await redis_client.get(key)
    return None


async def set_cache(key: str, value: str, expire: int = 300):
    """Set value in cache with expiration (default 5 minutes)."""
    if redis_client:
        await redis_client.setex(key, expire, value)


async def delete_cache(key: str):
    """Delete value from cache."""
    if redis_client:
        await redis_client.delete(key)
