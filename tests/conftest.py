"""Shared pytest configuration.

Every test uses StubEmbedder: no downloads, no network, fully deterministic.
The one test that exercises the real model is marked `network` and deselected
by default via `addopts` in pyproject.toml.
"""
