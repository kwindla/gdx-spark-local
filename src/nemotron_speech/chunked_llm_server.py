#!/usr/bin/env python3
"""
Chunked LLM Inference Server

WebSocket-based streaming LLM server for voice agents where LLM and TTS share GPU.
Client controls pause/resume with explicit pause configs per request.

Key design:
- True token streaming to client (optional, enabled with stream_tokens=True)
- Always disable reasoning (no <think> tokens)
- 250ms cooldown between requests (enforced server-side)

API:
  start_stream(stream_id, messages, pause?, stream_tokens?) - Start generation
  continue_stream(stream_id, pause?) - Continue an existing stream
  end_stream(stream_id) - End and cleanup

  pause options:
    { "max_tokens": N } - Generate N tokens then pause
    { "sentence_boundary": true } - Generate until sentence boundary then pause
    (omit pause to generate until EOS)

Protocol (stream_tokens=True):
  Client → Server:
    {"action": "start_stream", "stream_id": "123", "messages": [...],
     "pause": {"sentence_boundary": true}, "stream_tokens": true}
    {"action": "continue_stream", "stream_id": "123", "pause": {"sentence_boundary": true}}
    {"action": "end_stream", "stream_id": "123"}

  Server → Client (streaming):
    {"type": "token", "content": "Hello"}
    {"type": "token", "content": "!"}
    {"type": "paused", "reason": "sentence_boundary", "text": "Hello!", "tokens": 2, "ttft_ms": 175}
    or
    {"type": "done", "reason": "eos", "text": "...", "tokens": 70, "ttft_ms": 175}

Protocol (stream_tokens=False, default - backward compatible):
  Server → Client (buffered):
    {"stream_id": "123", "text": "Hello!", "tokens": 2, "paused": true, "reason": "sentence_boundary"}
"""

import argparse
import asyncio
import json
import logging
import time
import re
from dataclasses import dataclass
from typing import Optional
import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sentence boundary pattern - ends with .!? followed by space/newline or end
SENTENCE_END_PATTERN = re.compile(r'[.!?](?:\s|$)')


def ends_at_sentence_boundary(text: str) -> bool:
    """Check if text ends at a sentence boundary (not just contains one)."""
    if not text:
        return False
    # Check last few characters for sentence-ending punctuation
    text = text.rstrip()
    if not text:
        return False
    # Ends with .!? (possibly followed by closing quotes/parens)
    return bool(re.search(r'[.!?]["\'\)]*$', text))

# Minimum cooldown between requests to same slot (ms)
# Note: 100ms wasn't enough to prevent llama.cpp slot assertion crashes
SLOT_COOLDOWN_MS = 250


@dataclass
class PauseConfig:
    """Configuration for when to pause generation."""
    max_tokens: Optional[int] = None
    sentence_boundary: bool = False

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "PauseConfig":
        if not d:
            return cls()
        return cls(
            max_tokens=d.get("max_tokens"),
            sentence_boundary=d.get("sentence_boundary", False)
        )


class LLMStream:
    """
    Maintains state for a single LLM generation stream.

    Always uses streaming mode for token-by-token control.
    Always disables reasoning for voice agent use case.
    Enforces 250ms cooldown between requests to prevent llama.cpp crashes.
    """

    # Class-level cooldown tracking per slot
    _slot_last_request: dict[int, float] = {}
    _slot_lock: asyncio.Lock = None

    def __init__(
        self,
        stream_id: str,
        slot_id: int,
        llama_url: str,
        temperature: float = 0.7
    ):
        self.stream_id = stream_id
        self.slot_id = slot_id
        self.llama_url = llama_url
        self.temperature = temperature

        # State
        self.prompt: str = ""
        self.generated_text: str = ""
        self.generated_tokens: int = 0
        self.is_started: bool = False
        self.is_done: bool = False
        self._received_eos: bool = False  # Guard: true after EOS or empty response

        # HTTP client with longer timeout for streaming
        self._client = httpx.AsyncClient(timeout=300.0)

        # Initialize class lock if needed
        if LLMStream._slot_lock is None:
            LLMStream._slot_lock = asyncio.Lock()

    async def _wait_for_cooldown(self):
        """Wait if needed to respect slot cooldown."""
        async with LLMStream._slot_lock:
            last_time = LLMStream._slot_last_request.get(self.slot_id, 0)
            elapsed_ms = (time.time() - last_time) * 1000

            if elapsed_ms < SLOT_COOLDOWN_MS:
                wait_ms = SLOT_COOLDOWN_MS - elapsed_ms
                logger.debug(
                    f"Stream {self.stream_id}: Waiting {wait_ms:.0f}ms for slot cooldown"
                )
                await asyncio.sleep(wait_ms / 1000)

    def _mark_request_complete(self):
        """Mark that a request to this slot just completed."""
        LLMStream._slot_last_request[self.slot_id] = time.time()

    async def start(
        self,
        messages: list[dict],
        pause: Optional[PauseConfig] = None
    ) -> dict:
        """
        Start generation stream.

        Returns dict with: text, tokens, paused, reason, done
        """
        if self.is_started:
            raise RuntimeError("Stream already started")

        # Format prompt (always disable thinking for voice agents)
        self.prompt = self._format_messages(messages)
        self.is_started = True

        logger.info(f"Stream {self.stream_id}: Starting on slot {self.slot_id}")

        return await self._generate(pause or PauseConfig())

    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages using ChatML template. Always disables thinking."""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        # Always disable thinking for voice agents
        prompt_parts.append("<|im_start|>assistant\n<think></think>")

        return "\n".join(prompt_parts)

    async def continue_stream(
        self,
        pause: Optional[PauseConfig] = None
    ) -> dict:
        """
        Continue generation stream.

        Returns dict with: text, tokens, paused, reason, done
        """
        if not self.is_started:
            raise RuntimeError("Stream not started")
        if self.is_done:
            return {
                "text": "",
                "tokens": 0,
                "paused": False,
                "reason": "already_done",
                "done": True
            }

        return await self._generate(pause or PauseConfig())

    async def _generate(self, pause: PauseConfig) -> dict:
        """
        Internal streaming generation with pause handling.

        Always uses streaming mode for token-by-token control.
        """
        # Wait for slot cooldown if needed
        await self._wait_for_cooldown()

        # Determine max tokens
        if pause.max_tokens:
            max_tokens = pause.max_tokens
        elif pause.sentence_boundary:
            max_tokens = 200  # Reasonable limit for a sentence
        else:
            max_tokens = 500  # Default for no pause (generate until EOS)

        # Build full prompt including generated text so far
        full_prompt = self.prompt + self.generated_text

        logger.info(
            f"Stream {self.stream_id}: Generating "
            f"(max={max_tokens}, sentence_boundary={pause.sentence_boundary})"
        )
        start_time = time.time()

        # Always use streaming
        result = await self._stream_generate(
            full_prompt,
            max_tokens,
            pause.sentence_boundary
        )

        # Mark request complete for cooldown tracking
        self._mark_request_complete()

        elapsed = (time.time() - start_time) * 1000
        tps = result["tokens"] / (elapsed / 1000) if elapsed > 0 else 0

        # Update state
        self.generated_text += result["text"]
        self.generated_tokens += result["tokens"]
        if result["done"]:
            self.is_done = True

        logger.info(
            f"Stream {self.stream_id}: Generated {result['tokens']} tokens "
            f"in {elapsed:.0f}ms ({tps:.1f} tok/s), TTFT={result['ttft_ms']:.0f}ms, "
            f"paused={result['paused']}, reason={result['reason']}"
        )

        return result

    async def _stream_generate(
        self,
        full_prompt: str,
        max_tokens: int,
        stop_at_sentence: bool
    ) -> dict:
        """
        Stream tokens from llama.cpp, stopping at pause condition.

        Key fix: After stopping at sentence boundary, peek at the next token
        to check if the model is actually done (hit stop token). This prevents
        continuation requests that would crash llama.cpp.
        """
        payload = {
            "prompt": full_prompt,
            "n_predict": max_tokens,
            "id_slot": self.slot_id,
            "cache_prompt": True,
            "temperature": self.temperature,
            "top_p": 0.95,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "stream": True,
            "stop": ["<|im_end|>"],  # Explicit stop token for ChatML
        }

        collected_text = ""
        collected_tokens = 0
        hit_eos = False
        stop_reason = None
        ttft_ms = None  # Time to first token
        stream_start = time.time()
        hit_sentence_boundary = False

        try:
            async with self._client.stream(
                "POST",
                f"{self.llama_url}/completion",
                json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        if hit_sentence_boundary:
                            hit_eos = True
                            stop_reason = "sentence_boundary_eos"
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    token_text = data.get("content", "")

                    # Check for EOS/stop FIRST
                    if data.get("stop"):
                        stop_type = data.get("stop_type", "")
                        if stop_type in ("eos", "word"):
                            if token_text:
                                collected_text += token_text
                                collected_tokens += 1
                            hit_eos = True
                            stop_reason = "eos" if stop_type == "eos" else "stop_word"
                            break

                    # If we already hit sentence boundary, we're peeking for stop signal
                    if hit_sentence_boundary:
                        collected_text += token_text
                        collected_tokens += 1
                        if ends_at_sentence_boundary(collected_text):
                            continue
                        else:
                            stop_reason = "sentence_boundary"
                            break

                    collected_text += token_text
                    collected_tokens += 1

                    # Track time to first token
                    if ttft_ms is None:
                        ttft_ms = (time.time() - stream_start) * 1000

                    # Check for sentence boundary - peek at next token
                    if stop_at_sentence and ends_at_sentence_boundary(collected_text):
                        hit_sentence_boundary = True
                        continue

                    # Check token limit
                    if collected_tokens >= max_tokens:
                        stop_reason = "max_tokens"
                        break

        except httpx.RemoteProtocolError as e:
            logger.warning(f"Stream {self.stream_id}: Connection issue: {e}")
            if not stop_reason:
                stop_reason = "connection_error"

        if hit_sentence_boundary and not stop_reason:
            hit_eos = True
            stop_reason = "sentence_boundary_eos"

        if not stop_reason:
            stop_reason = "max_tokens"

        # Mark stream as done if we hit EOS
        if hit_eos:
            self._received_eos = True

        return {
            "text": collected_text,
            "tokens": collected_tokens,
            "paused": not hit_eos,
            "reason": stop_reason,
            "done": hit_eos,
            "ttft_ms": ttft_ms or 0
        }

    async def stream_to_client(
        self,
        websocket: WebSocket,
        pause: PauseConfig,
        is_start: bool = False,
        messages: Optional[list[dict]] = None
    ) -> dict:
        """
        Stream tokens directly to client WebSocket.

        Sends:
          {"type": "token", "content": "..."} for each token
          {"type": "paused", ...} when pause condition met
          {"type": "done", ...} when EOS reached

        Returns final result dict for state tracking.
        """
        if is_start:
            if self.is_started:
                raise RuntimeError("Stream already started")
            self.prompt = self._format_messages(messages or [])
            self.is_started = True
            logger.info(f"Stream {self.stream_id}: Starting (streaming) on slot {self.slot_id}")
        else:
            if not self.is_started:
                raise RuntimeError("Stream not started")
            # Guard: don't make requests after EOS/empty - llama.cpp crashes
            if self.is_done or self._received_eos:
                logger.info(f"Stream {self.stream_id}: Already complete, not requesting more")
                await websocket.send_json({
                    "type": "done",
                    "reason": "already_done",
                    "text": "",
                    "tokens": 0,
                    "ttft_ms": 0
                })
                return {"done": True, "tokens": 0, "text": ""}

        # Wait for slot cooldown
        await self._wait_for_cooldown()

        # Determine max tokens
        if pause.max_tokens:
            max_tokens = pause.max_tokens
        elif pause.sentence_boundary:
            max_tokens = 200
        else:
            max_tokens = 500

        full_prompt = self.prompt + self.generated_text

        logger.info(
            f"Stream {self.stream_id}: Streaming "
            f"(max={max_tokens}, sentence_boundary={pause.sentence_boundary})"
        )

        # Stream tokens to client
        result = await self._stream_to_websocket(
            websocket,
            full_prompt,
            max_tokens,
            pause.sentence_boundary
        )

        # Mark request complete
        self._mark_request_complete()

        # Update state
        self.generated_text += result["text"]
        self.generated_tokens += result["tokens"]
        if result["done"]:
            self.is_done = True

        elapsed = result.get("elapsed_ms", 0)
        tps = result["tokens"] / (elapsed / 1000) if elapsed > 0 else 0

        logger.info(
            f"Stream {self.stream_id}: Streamed {result['tokens']} tokens "
            f"in {elapsed:.0f}ms ({tps:.1f} tok/s), TTFT={result['ttft_ms']:.0f}ms, "
            f"done={result['done']}, reason={result['reason']}"
        )

        return result

    async def _stream_to_websocket(
        self,
        websocket: WebSocket,
        full_prompt: str,
        max_tokens: int,
        stop_at_sentence: bool
    ) -> dict:
        """Stream tokens to WebSocket, send paused/done message at end.

        Key fix: After stopping at sentence boundary, peek at the next token
        to check if the model is actually done (hit stop token). This prevents
        continuation requests that would crash llama.cpp.
        """
        payload = {
            "prompt": full_prompt,
            "n_predict": max_tokens,
            "id_slot": self.slot_id,
            "cache_prompt": True,
            "temperature": self.temperature,
            "top_p": 0.95,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "stream": True,
            "stop": ["<|im_end|>"],  # Explicit stop token for ChatML
        }

        collected_text = ""
        collected_tokens = 0
        hit_eos = False
        stop_reason = None
        ttft_ms = None
        stream_start = time.time()
        hit_sentence_boundary = False  # Track if we hit sentence boundary

        try:
            async with self._client.stream(
                "POST",
                f"{self.llama_url}/completion",
                json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        # Stream ended - if we were waiting to check for stop, model is done
                        if hit_sentence_boundary:
                            hit_eos = True
                            stop_reason = "sentence_boundary_eos"
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    token_text = data.get("content", "")

                    # Check for EOS/stop FIRST (before adding to collected)
                    # Handle both EOS and stop word (like <|im_end|>)
                    if data.get("stop"):
                        stop_type = data.get("stop_type", "")
                        if stop_type in ("eos", "word"):
                            # Include any final content
                            if token_text:
                                collected_text += token_text
                                collected_tokens += 1
                                await websocket.send_json({
                                    "type": "token",
                                    "content": token_text
                                })
                            hit_eos = True
                            stop_reason = "eos" if stop_type == "eos" else "stop_word"
                            break

                    # If we already hit sentence boundary, we're peeking for stop signal
                    if hit_sentence_boundary:
                        # Got more content - model isn't done, just had a sentence boundary
                        # Include this token and check if we're still at a boundary
                        collected_text += token_text
                        collected_tokens += 1
                        await websocket.send_json({
                            "type": "token",
                            "content": token_text
                        })
                        # Check if text still ENDS at a sentence boundary
                        if ends_at_sentence_boundary(collected_text):
                            # Still at a sentence boundary, keep peeking
                            continue
                        else:
                            # Model has more to say, stop here
                            stop_reason = "sentence_boundary"
                            break

                    collected_text += token_text
                    collected_tokens += 1

                    # Track TTFT
                    if ttft_ms is None:
                        ttft_ms = (time.time() - stream_start) * 1000

                    # Stream token to client immediately
                    await websocket.send_json({
                        "type": "token",
                        "content": token_text
                    })

                    # Check for sentence boundary - don't break yet, peek at next token
                    if stop_at_sentence and ends_at_sentence_boundary(collected_text):
                        hit_sentence_boundary = True
                        # Continue to next iteration to peek at next token
                        continue

                    # Check token limit
                    if collected_tokens >= max_tokens:
                        stop_reason = "max_tokens"
                        break

        except httpx.RemoteProtocolError as e:
            logger.warning(f"Stream {self.stream_id}: Connection issue: {e}")
            if not stop_reason:
                stop_reason = "connection_error"

        # If we hit sentence boundary and didn't get more tokens, model is done
        if hit_sentence_boundary and not stop_reason:
            hit_eos = True
            stop_reason = "sentence_boundary_eos"

        if not stop_reason:
            stop_reason = "max_tokens"

        elapsed_ms = (time.time() - stream_start) * 1000

        # If we got 0 tokens or hit EOS, mark as complete and set guard
        if collected_tokens == 0:
            hit_eos = True
            stop_reason = "empty_response"
            self._received_eos = True

        if hit_eos:
            self._received_eos = True

        # Send final message (paused or done)
        final_msg = {
            "type": "done" if hit_eos else "paused",
            "reason": stop_reason,
            "text": collected_text,
            "tokens": collected_tokens,
            "ttft_ms": ttft_ms or 0,
            "elapsed_ms": elapsed_ms
        }
        await websocket.send_json(final_msg)

        return {
            "text": collected_text,
            "tokens": collected_tokens,
            "done": hit_eos or collected_tokens == 0,
            "reason": stop_reason,
            "ttft_ms": ttft_ms or 0,
            "elapsed_ms": elapsed_ms
        }

    async def end(self):
        """End the stream and cleanup."""
        self.is_done = True
        await self._client.aclose()
        logger.info(
            f"Stream {self.stream_id}: Ended. "
            f"Total: {self.generated_tokens} tokens."
        )


class StreamManager:
    """Manages LLM streams with slot allocation."""

    def __init__(self, llama_url: str, num_slots: int = 1):
        self.llama_url = llama_url
        self.num_slots = num_slots
        self.streams: dict[str, LLMStream] = {}
        self._slot_in_use: dict[int, str] = {}

    def _allocate_slot(self) -> Optional[int]:
        """Find an available slot."""
        for slot_id in range(self.num_slots):
            if slot_id not in self._slot_in_use:
                return slot_id
        return None

    def create_stream(
        self,
        stream_id: str,
        temperature: float = 0.7
    ) -> LLMStream:
        """Create a new stream with an allocated slot."""
        if stream_id in self.streams:
            raise RuntimeError(f"Stream {stream_id} already exists")

        slot_id = self._allocate_slot()
        if slot_id is None:
            raise RuntimeError(f"No slots available (max {self.num_slots})")

        stream = LLMStream(
            stream_id=stream_id,
            slot_id=slot_id,
            llama_url=self.llama_url,
            temperature=temperature,
        )
        self.streams[stream_id] = stream
        self._slot_in_use[slot_id] = stream_id

        logger.info(f"Created stream {stream_id} on slot {slot_id}")
        return stream

    def get_stream(self, stream_id: str) -> Optional[LLMStream]:
        return self.streams.get(stream_id)

    async def remove_stream(self, stream_id: str):
        if stream_id in self.streams:
            stream = self.streams[stream_id]
            await stream.end()
            if stream.slot_id in self._slot_in_use:
                del self._slot_in_use[stream.slot_id]
            del self.streams[stream_id]
            logger.info(f"Removed stream {stream_id}")


# FastAPI app
app = FastAPI(title="Chunked LLM Inference Server")
_stream_manager: Optional[StreamManager] = None
_llama_url: str = "http://localhost:8000"


@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    """WebSocket endpoint for stream-based generation."""
    await websocket.accept()
    active_streams: set[str] = set()

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "start_stream":
                stream_id = data.get("stream_id")
                messages = data.get("messages", [])
                pause_dict = data.get("pause")
                temperature = data.get("temperature", 0.7)
                stream_tokens = data.get("stream_tokens", False)

                if not stream_id:
                    await websocket.send_json({"error": "stream_id required"})
                    continue

                try:
                    pause = PauseConfig.from_dict(pause_dict)
                    stream = _stream_manager.create_stream(stream_id, temperature)
                    active_streams.add(stream_id)
                    # Track if this stream uses token streaming
                    stream._stream_tokens = stream_tokens

                    if stream_tokens:
                        # True token streaming - tokens sent as they arrive
                        await stream.stream_to_client(
                            websocket, pause, is_start=True, messages=messages
                        )
                    else:
                        # Buffered mode (backward compatible)
                        result = await stream.start(messages, pause)
                        await websocket.send_json({
                            "stream_id": stream_id,
                            "status": "started",
                            **result,
                            "full_text": stream.generated_text,
                        })

                except Exception as e:
                    logger.exception(f"Error starting stream: {e}")
                    await websocket.send_json({
                        "stream_id": stream_id,
                        "error": str(e)
                    })

            elif action == "continue_stream":
                stream_id = data.get("stream_id")
                pause_dict = data.get("pause")

                if not stream_id:
                    await websocket.send_json({"error": "stream_id required"})
                    continue

                stream = _stream_manager.get_stream(stream_id)
                if not stream:
                    await websocket.send_json({
                        "stream_id": stream_id,
                        "error": "Stream not found"
                    })
                    continue

                try:
                    pause = PauseConfig.from_dict(pause_dict)

                    # Check if this stream uses token streaming
                    if getattr(stream, '_stream_tokens', False):
                        # True token streaming
                        await stream.stream_to_client(websocket, pause)
                    else:
                        # Buffered mode
                        result = await stream.continue_stream(pause)
                        await websocket.send_json({
                            "stream_id": stream_id,
                            **result,
                            "full_text": stream.generated_text,
                        })

                except Exception as e:
                    logger.exception(f"Error continuing stream: {e}")
                    await websocket.send_json({
                        "stream_id": stream_id,
                        "error": str(e)
                    })

            elif action == "end_stream":
                stream_id = data.get("stream_id")
                if stream_id:
                    await _stream_manager.remove_stream(stream_id)
                    active_streams.discard(stream_id)
                    await websocket.send_json({
                        "stream_id": stream_id,
                        "status": "ended"
                    })

            elif action == "ping":
                await websocket.send_json({"status": "pong"})

            else:
                await websocket.send_json({"error": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        logger.info("Client disconnected")
        for stream_id in active_streams:
            await _stream_manager.remove_stream(stream_id)

    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        for stream_id in active_streams:
            await _stream_manager.remove_stream(stream_id)


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{_llama_url}/health", timeout=5.0)
            llama_healthy = response.status_code == 200
    except:
        llama_healthy = False

    return {
        "status": "ok" if llama_healthy else "degraded",
        "llama_server": "healthy" if llama_healthy else "unreachable",
        "llama_url": _llama_url,
        "active_streams": len(_stream_manager.streams) if _stream_manager else 0,
        "slot_cooldown_ms": SLOT_COOLDOWN_MS,
    }


@app.get("/")
async def root():
    return {
        "name": "Chunked LLM Inference Server",
        "description": "Streaming LLM server with pause/resume for voice agents",
        "features": [
            "Always streaming (token-by-token)",
            "Always disables reasoning",
            f"{SLOT_COOLDOWN_MS}ms cooldown enforced between requests"
        ],
        "websocket": "/ws",
        "protocol": {
            "start_stream": {
                "action": "start_stream",
                "stream_id": "unique-id",
                "messages": [{"role": "user", "content": "Hello"}],
                "pause": {"max_tokens": 30}
            },
            "continue_stream": {
                "action": "continue_stream",
                "stream_id": "unique-id",
                "pause": {"sentence_boundary": True}
            },
            "end_stream": {
                "action": "end_stream",
                "stream_id": "unique-id"
            }
        }
    }


def main():
    global _stream_manager, _llama_url

    parser = argparse.ArgumentParser(description="Chunked LLM Inference Server")
    parser.add_argument("--llama-url", default="http://localhost:8000")
    parser.add_argument("--num-slots", type=int, default=1)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=8002)

    args = parser.parse_args()

    _llama_url = args.llama_url
    _stream_manager = StreamManager(_llama_url, args.num_slots)

    logger.info(f"Chunked LLM Server starting")
    logger.info(f"  llama-server: {_llama_url}")
    logger.info(f"  WebSocket: ws://{args.host}:{args.port}/ws")
    logger.info(f"  Slot cooldown: {SLOT_COOLDOWN_MS}ms")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
