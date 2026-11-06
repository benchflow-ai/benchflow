"""A small HTTP client that retries requests that time out."""

import time


class RetryError(Exception):
    """Raised when every attempt at a request timed out."""


class HttpClient:
    def __init__(self, transport, retries=3, base_delay=0.5):
        self.transport = transport
        self.retries = retries
        self.base_delay = base_delay

    def get(self, path):
        for attempt in range(self.retries):
            try:
                return self.transport.send(("GET", path))
            except TimeoutError:
                time.sleep(self.base_delay)
        raise RetryError(f"GET {path} failed after {self.retries} attempts")
