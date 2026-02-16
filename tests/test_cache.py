from models.repositories.cache import LRUCache


class TestLRUCache:
    """Unit tests for LRUCache behavior."""

    def test_get_returns_none_for_missing_key(self) -> None:
        """Return None when a key is not present in cache."""
        cache = LRUCache(capacity=2)

        assert cache.get("missing") is None

    def test_get_returns_value_for_existing_key(self) -> None:
        """Return stored value for an existing key."""
        cache = LRUCache(capacity=2)
        cache.put("schema-a", "value-a")

        assert cache.get("schema-a") == "value-a"

    def test_put_evicts_least_recently_used_item(self) -> None:
        """Evict least recently used item when capacity is exceeded."""
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")
        cache.put("c", 3)

        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
