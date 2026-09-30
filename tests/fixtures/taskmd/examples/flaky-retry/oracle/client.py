"""A small HTTP client that retries requests that time out."""

import random
import time

JITTER = 0.2  # each delay is stretched by up to this fraction


class RetryError(Exception):
    """Raised when every attempt at a request timed out."""


class HttpClient:
    def __init__(self, transport, retries=3, base_delay=0.5):
        self.transport = transport
        self.retries = retries
        self.base_delay = base_delay

    def get(self, path):
        last = None
        for attempt in range(self.retries + 1):
            try:
                return self.transport.send(("GET", path))
            except TimeoutError as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(self.base_delay * 2**attempt * (1 + random.uniform(0, JITTER)))
        raise RetryError(f"GET {path} timed out after {self.retries + 1} attempts") from last
