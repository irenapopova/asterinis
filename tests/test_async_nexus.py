import asyncio
import time

import pytest

from asterinis import Nexus
from asterinis.providers import Provider
from asterinis.timeouts import (
    AsterinisTimeoutError,
)


class EchoProvider(Provider):
    name = "echo"

    def invoke(
        self,
        text: str,
        **kwargs,
    ):
        return {
            "text": text,
            "metadata": kwargs.get(
                "metadata",
                {},
            ),
        }


class SlowProvider(Provider):
    name = "slow"

    def invoke(
        self,
        text: str,
        **kwargs,
    ):
        time.sleep(0.1)

        return {
            "text": text,
        }


def test_async_nexus_processes_provider() -> None:
    async def run() -> None:
        nexus = Nexus()

        nexus.register_provider(
            "llm",
            EchoProvider(),
        )

        result = await nexus.process_async(
            "hello",
        )

        assert result.route == "llm"
        assert result.provider == "echo"
        assert result.output["text"] == "hello"

    asyncio.run(run())


def test_async_nexus_passes_metadata_to_provider() -> None:
    async def run() -> None:
        nexus = Nexus()

        nexus.register_provider(
            "llm",
            EchoProvider(),
        )

        result = await nexus.process_async(
            "hello",
            metadata={
                "request_id": "123",
                "language": "en",
            },
        )

        assert (
            result.output["metadata"]["request_id"]
            == "123"
        )

        assert (
            result.output["metadata"]["language"]
            == "en"
        )

    asyncio.run(run())


def test_async_nexus_returns_unregistered_route_result() -> None:
    async def run() -> None:
        nexus = Nexus()

        result = await nexus.process_async(
            "hello"
        )

        assert result.route == "llm"
        assert result.provider is None
        assert result.output is None

        assert (
            "No provider registered"
            in result.metadata["message"]
        )

    asyncio.run(run())


def test_async_nexus_rejects_empty_input() -> None:
    async def run() -> None:
        nexus = Nexus()

        with pytest.raises(
            ValueError,
            match="empty",
        ):
            await nexus.process_async(
                "   "
            )

    asyncio.run(run())


def test_async_nexus_rejects_non_string_input() -> None:
    async def run() -> None:
        nexus = Nexus()

        with pytest.raises(TypeError):
            await nexus.process_async(
                123  # type: ignore[arg-type]
            )

    asyncio.run(run())


def test_async_nexus_enforces_provider_timeout() -> None:
    async def run() -> None:
        nexus = Nexus()

        nexus.register_provider(
            "llm",
            SlowProvider(),
        )

        with pytest.raises(
            AsterinisTimeoutError
        ):
            await nexus.process_async(
                "slow request",
                timeout_seconds=0.01,
            )

    asyncio.run(run())


def test_async_nexus_does_not_block_event_loop() -> None:
    async def run() -> None:
        nexus = Nexus()

        nexus.register_provider(
            "llm",
            SlowProvider(),
        )

        marker = {
            "completed": False,
        }

        async def lightweight_task() -> None:
            await asyncio.sleep(0.01)
            marker["completed"] = True

        provider_task = asyncio.create_task(
            nexus.process_async(
                "slow request"
            )
        )

        lightweight = asyncio.create_task(
            lightweight_task()
        )

        await lightweight

        assert marker["completed"]

        result = await provider_task

        assert result.provider == "slow"

    asyncio.run(run())