"""Shared pytest configuration.

Every test uses StubEmbedder: no downloads, no network, fully deterministic.
The one test that exercises the real model is marked `network` and deselected
by default via `addopts` in pyproject.toml.
"""

import pytest


class ScriptedAsker:
    """A stand-in for a person (docs/decisions/0005). `confirms` and `choices`
    are consumed in order; `on_ask(message)` runs before each answer, so a
    test can change the vault while the "person" is thinking."""

    def __init__(self, confirms=(), choices=(), can_ask=True, on_ask=None):
        self.confirms = list(confirms)
        self.choices = list(choices)
        self._can_ask = can_ask
        self.on_ask = on_ask
        self.messages: list[str] = []

    def can_ask(self) -> bool:
        return self._can_ask

    async def confirm(self, message: str) -> bool:
        self.messages.append(message)
        if self.on_ask:
            self.on_ask(message)
        return self.confirms.pop(0)

    async def choose(self, message: str):
        self.messages.append(message)
        if self.on_ask:
            self.on_ask(message)
        return self.choices.pop(0)


@pytest.fixture
def scripted():
    return ScriptedAsker
