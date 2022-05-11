"""Cache for queries and responses."""
import json
import logging
from typing import Callable, Dict, Tuple

from sqlitedict import SqliteDict

logging.getLogger("sqlitedict").setLevel(logging.WARNING)


def request_to_key(request: Dict) -> str:
    """Normalize a `request` into a `key`."""
    return json.dumps(request, sort_keys=True)


def key_to_request(key: str) -> Dict:
    """Convert the normalized version to the request."""
    return json.loads(key)


class Cache(object):
    """A cache for request/response pairs."""

    def __init__(self, cache_path: str):
        """Init."""
        self.cache_path = cache_path

    def get(
        self, request: Dict, overwrite_cache: bool, compute: Callable[[], Dict]
    ) -> Tuple[Dict, bool]:
        """Get the result of `request` (by calling `compute` as needed)."""
        key = request_to_key(request)

        with SqliteDict(self.cache_path) as cache:
            response = cache.get(key)
            if response and not overwrite_cache:
                cached = True
            else:
                response = compute()
                # Commit the request and response to SQLite
                cache[key] = response
                cache.commit()
                cached = False
        return response, cached
