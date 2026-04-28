from __future__ import annotations

import asyncio
import atexit
import logging
from typing import Any

logger = logging.getLogger(__name__)


class DaytonaClientManager:
    """Singleton manager for a shared AsyncDaytona client."""

    _instance: DaytonaClientManager | None = None
    _lock = asyncio.Lock()

    def __init__(self) -> None:
        self._client: Any | None = None
        self._client_lock = asyncio.Lock()
        self._cleanup_registered = False

    @classmethod
    async def get_instance(cls) -> DaytonaClientManager:
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        assert cls._instance is not None
        return cls._instance

    async def get_client(self) -> Any:
        async with self._client_lock:
            if self._client is None:
                from benchflow.sandbox._sdk_ops import import_daytona_sdk

                sdk = import_daytona_sdk()
                logger.debug("Creating new AsyncDaytona client")
                self._client = sdk.AsyncDaytona()
                if not self._cleanup_registered:
                    atexit.register(self._cleanup_sync)
                    self._cleanup_registered = True
            return self._client

    def _cleanup_sync(self) -> None:
        try:
            asyncio.run(self._cleanup())
        except Exception as exc:
            print(f"Error during Daytona client cleanup: {exc}")

    async def _cleanup(self) -> None:
        async with self._client_lock:
            if self._client is not None:
                try:
                    logger.debug("Closing AsyncDaytona client at program exit")
                    await self._client.close()
                except Exception:
                    logger.exception("Error closing AsyncDaytona client")
                finally:
                    self._client = None
