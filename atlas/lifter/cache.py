"""
Caching layer for compiled binaries using Redis.
"""

import pickle
from typing import Optional, Any

from atlas.config import get_config


class IRCache:
    """
    Redis-based cache for compiled LLVM IR / machine code.
    
    Gracefully degrades to no-op if Redis is unavailable.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        self._client = None
        self._enabled = True
        
        config = get_config()
        url = redis_url or config.lifter.redis_url
        
        if not config.lifter.cache_enabled:
            self._enabled = False
            return
        
        try:
            import redis
            self._client = redis.from_url(url)
            # Test connection
            self._client.ping()
        except Exception:
            # Redis not available, disable caching
            self._client = None
            self._enabled = False
    
    @property
    def is_available(self) -> bool:
        """Check if caching is available."""
        return self._enabled and self._client is not None
    
    def _make_key(self, ir_hash: str, suffix: str = "") -> str:
        """Create a Redis key from IR hash."""
        return f"atlas:{ir_hash}{':' + suffix if suffix else ''}"
    
    def get_cached_binary(self, ir_hash: str) -> Optional[bytes]:
        """
        Retrieve compiled binary from cache.
        
        Args:
            ir_hash: Hash of the original LLVM IR
            
        Returns:
            Compiled binary bytes or None if not cached
        """
        if not self.is_available:
            return None
        
        try:
            key = self._make_key(ir_hash, "binary")
            data = self._client.get(key)
            return data
        except Exception:
            return None
    
    def cache_binary(self, ir_hash: str, binary: bytes, ttl: Optional[int] = None) -> bool:
        """
        Store compiled binary in cache.
        
        Args:
            ir_hash: Hash of the original LLVM IR
            binary: Compiled binary bytes
            ttl: Time-to-live in seconds (uses config default if not specified)
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.is_available:
            return False
        
        try:
            config = get_config()
            ttl = ttl or config.lifter.cache_ttl
            key = self._make_key(ir_hash, "binary")
            self._client.setex(key, ttl, binary)
            return True
        except Exception:
            return False
    
    def get_cached_optimized_ir(self, ir_hash: str) -> Optional[str]:
        """Retrieve cached optimized IR."""
        if not self.is_available:
            return None
        
        try:
            key = self._make_key(ir_hash, "optimized_ir")
            data = self._client.get(key)
            return data.decode() if data else None
        except Exception:
            return None
    
    def cache_optimized_ir(self, ir_hash: str, optimized_ir: str, ttl: Optional[int] = None) -> bool:
        """Store optimized IR in cache."""
        if not self.is_available:
            return False
        
        try:
            config = get_config()
            ttl = ttl or config.lifter.cache_ttl
            key = self._make_key(ir_hash, "optimized_ir")
            self._client.setex(key, ttl, optimized_ir.encode())
            return True
        except Exception:
            return False
    
    def invalidate(self, ir_hash: str) -> bool:
        """Invalidate all cached data for an IR hash."""
        if not self.is_available:
            return False
        
        try:
            keys = [
                self._make_key(ir_hash, "binary"),
                self._make_key(ir_hash, "optimized_ir"),
            ]
            self._client.delete(*keys)
            return True
        except Exception:
            return False
    
    def clear_all(self) -> bool:
        """Clear all Atlas cache entries."""
        if not self.is_available:
            return False
        
        try:
            pattern = "atlas:*"
            keys = list(self._client.scan_iter(match=pattern))
            if keys:
                self._client.delete(*keys)
            return True
        except Exception:
            return False


# In-memory fallback cache for when Redis is unavailable
class InMemoryCache:
    """Simple in-memory cache as fallback."""
    
    def __init__(self, max_size: int = 100):
        self._cache: dict = {}
        self._max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        if len(self._cache) >= self._max_size:
            # Simple eviction: remove first item
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        self._cache[key] = value
    
    def delete(self, key: str) -> None:
        self._cache.pop(key, None)
    
    def clear(self) -> None:
        self._cache.clear()
