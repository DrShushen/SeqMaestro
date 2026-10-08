"""The feature finder sends every batch of requests on one event loop.

LiteLLM keeps an HTTP client for each event loop it is called from, and closes none of them. With a
fresh loop for each batch, every batch left its connections open, and a run failed once the process
reached its limit on open files. The test that counts connections runs against a server on the
loopback interface, so it needs no credentials and no network.

All offline.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import dill
import pytest

from deft.features.feature_finder import LLMFeatureFinder
from deft.llm.engines.litellm_engine import LiteLLMEngine


class LoopRecorder:
    """An engine that answers nothing, and notes the event loop each batch ran on."""

    def __init__(self):
        self.loops = []

    async def query_llm(self, list_prompts, n_generations_per_prompt):
        self.loops.append(asyncio.get_running_loop())
        return [None] * len(list_prompts)


def test_every_batch_runs_on_one_event_loop_until_the_finder_is_closed():
    engine = LoopRecorder()
    finder = LLMFeatureFinder(llm_engine=engine, random_state=0)
    finder._ask(["first"], 1)
    finder._ask(["second"], 1)
    first, second = engine.loops
    assert first is second and not first.is_closed()

    finder.close()
    assert first.is_closed()
    finder._ask(["after closing"], 1)
    assert engine.loops[-1] is not first, "a request after closing opens a new loop"
    finder.close()


class Silent:
    """An engine that answers nothing and keeps nothing."""

    async def query_llm(self, list_prompts, n_generations_per_prompt):
        return [None] * len(list_prompts)


def test_a_saved_finder_holds_no_event_loop():
    """The finder is saved with the fitted tree, and an event loop cannot be."""
    finder = LLMFeatureFinder(llm_engine=Silent(), random_state=0)
    finder._ask(["a prompt"], 1)
    restored = dill.loads(dill.dumps(finder))
    assert restored._runner is None
    finder.close()


class Completions(BaseHTTPRequestHandler):
    """Answers every request with as many choices as it asks for, and keeps its connection open,
    as the provider does."""

    protocol_version = "HTTP/1.1"

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        n = int(body.get("n", 1))
        reply = json.dumps({
            "id": "test", "object": "chat.completion", "created": 0, "model": "test",
            "choices": [{"index": i, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "Feature: gc"}}
                        for i in range(n)],
            "usage": {"prompt_tokens": 1, "completion_tokens": n, "total_tokens": 1 + n},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


@pytest.fixture
def endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Completions)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def open_sockets() -> int:
    """The sockets this process holds, the server's side of each connection included."""
    count = 0
    for fd in os.listdir("/proc/self/fd"):
        # The descriptor `listdir` itself held is closed by now.
        with contextlib.suppress(FileNotFoundError):
            count += os.readlink(f"/proc/self/fd/{fd}").startswith("socket:")
    return count


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="counts sockets through /proc")
def test_batch_after_batch_the_open_connections_stay_those_of_one_batch(endpoint):
    engine = LiteLLMEngine(content="system", num_retries=0, config={
        "model": "azure/test", "api_base": endpoint, "api_key": "test",
        "api_version": "2024-06-01"})
    finder = LLMFeatureFinder(llm_engine=engine, random_state=0)
    prompts = [f"prompt {index}" for index in range(10)]
    try:
        finder._ask(prompts, 1)
        after_one = open_sockets()
        for _ in range(10):
            responses = finder._ask(prompts, 1)
        assert all(response is not None for response in responses)
        # A loop per batch left ten more connections open with each batch, a hundred here.
        assert open_sockets() <= after_one + 5
    finally:
        finder.close()
