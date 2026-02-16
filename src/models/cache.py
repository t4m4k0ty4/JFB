from typing import Any


class LRUCache:
    def __init__(self, capacity: int) -> None:
        self.__validate_capacity(capacity)
        self.capacity = capacity
        self.cache = {}

    def get(self, key: str) -> Any:
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None

    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
        self.cache[key] = value

    def __validate_capacity(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
