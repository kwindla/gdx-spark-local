"""Direct llama.cpp chunked LLM service for Pipecat.

Connects directly to llama.cpp's HTTP API without an intermediary WebSocket server.
Generates LLM responses in sentence-boundary chunks for optimal TTS integration.

Architecture:
    ┌─────────────────────────┐         ┌─────────────────┐
    │ LlamaCppChunkedLLMService│◄──HTTP──►│   llama.cpp     │
    │                         │   SSE    │   (llama-server)│
    │  - Sentence chunking    │         │   Port 8000     │
    │  - KV cache management  │         │                 │
    │  - TTS synchronization  │         │                 │
    └─────────────────────────┘         └─────────────────┘

Key features:
- Direct HTTP streaming to llama.cpp /completion endpoint
- Sentence boundary detection for natural chunk breaks
- First chunk optimization (min/max tokens) for low TTFB
- Two-slot alternation to avoid cancel race conditions (requires --parallel 2)
- KV cache reuse via cache_prompt with partial hits across slots
- TTS synchronization via ChunkedLLMContinueGenerationFrame
- Proper interruption handling with slot reuse guard

Usage:
    llm = LlamaCppChunkedLLMService(llama_url="http://localhost:8000")
    # In pipeline: ... -> context_aggregator.user() -> llm -> tts -> ...
"""

import asyncio
import json
import re
import time
from typing import Optional

import httpx
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    StartFrame,
    SystemFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService

# Import context frame types (handle different Pipecat versions)
try:
    from pipecat.frames.frames import LLMContextFrame
except ImportError:
    LLMContextFrame = None

try:
    from pipecat.frames.frames import LLMMessagesFrame
except ImportError:
    LLMMessagesFrame = None


# Two-slot configuration to avoid llama.cpp cancel race conditions
# When we close a connection mid-stream, llama.cpp's async cleanup can leave
# the slot in an inconsistent state. By alternating between two slots and
# enforcing a minimum reuse delay, we ensure each slot has time to clean up.

# Number of slots to alternate between (requires --parallel N on server)
DEFAULT_NUM_SLOTS = 2

# Minimum time (seconds) before reusing the same slot
# This ensures any async cleanup from cancelled requests is complete
# and late-arriving TCP close signals don't trigger GGML_ASSERT crashes.
# 2 seconds provides buffer for network buffering delays.
DEFAULT_MIN_SLOT_REUSE_DELAY_S = 2.0


class ChunkedLLMContinueGenerationFrame(SystemFrame):
    """Signal frame sent upstream by TTS when a segment completes.

    This frame tells the LLM service that TTS has finished processing
    the current chunk and generation can continue to the next chunk.
    """
    pass


class LLMSlotMetricsFrame(SystemFrame):
    """Frame containing LLM slot usage and cache metrics.

    Pushed after LLMFullResponseEndFrame to report slot reuse and cache performance.
    """

    def __init__(
        self,
        slot_id: int,
        slot_reused: bool,
        total_chunks: int,
        total_time_ms: float,
        tokens_cached: int = 0,
        tokens_evaluated: int = 0,
    ):
        super().__init__()
        self.slot_id = slot_id
        self.slot_reused = slot_reused
        self.total_chunks = total_chunks
        self.total_time_ms = total_time_ms
        self.tokens_cached = tokens_cached
        self.tokens_evaluated = tokens_evaluated

    @property
    def cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio (0.0-1.0)."""
        total = self.tokens_cached + self.tokens_evaluated
        if total == 0:
            return 0.0
        return self.tokens_cached / total

    def __str__(self) -> str:
        """String representation for logging."""
        return (
            f"LLMSlotMetrics(slot={self.slot_id}, reused={self.slot_reused}, "
            f"chunks={self.total_chunks}, time={self.total_time_ms:.0f}ms, "
            f"cached={self.tokens_cached}, eval={self.tokens_evaluated}, "
            f"hit={self.cache_hit_ratio:.1%})"
        )

    def __repr__(self) -> str:
        return self.__str__()


def ends_at_sentence_boundary(text: str) -> bool:
    """Check if text ends at a sentence boundary.

    Matches text ending with .!? optionally followed by closing quotes/parens.
    """
    if not text:
        return False
    text = text.rstrip()
    if not text:
        return False
    return bool(re.search(r'[.!?]["\'\)]*$', text))


def is_word_boundary(text: str) -> bool:
    """Check if text ends at a word boundary (whitespace or punctuation)."""
    if not text:
        return True
    last_char = text[-1]
    return last_char.isspace() or last_char in '.,!?;:"\')]-'


class LlamaCppChunkedLLMService(AIService):
    """LLM service that generates responses in sentence-boundary chunks.

    Connects directly to llama.cpp's HTTP API and generates complete chunks
    (sentences) rather than individual tokens. Each chunk is emitted as an
    LLMTextFrame, enabling TTS to process natural sentence units.

    Key behaviors:
    - First chunk: Uses min_tokens/max_tokens bounds for quick first response
    - Subsequent chunks: Pure sentence boundary detection
    - Waits for TTS completion signal before generating next chunk
    - Handles interruptions by cancelling HTTP stream immediately
    - Manages KV cache via cache_prompt and id_slot pinning
    """

    class InputParams(BaseModel):
        """Configuration parameters for LlamaCppChunkedLLMService."""
        # First chunk bounds (for low TTFB)
        first_chunk_min_tokens: int = 10
        first_chunk_max_tokens: int = 24
        # LLM generation parameters
        # temperature=0.0 for deterministic output - prevents punctuation inconsistency
        temperature: float = 0.0
        top_p: float = 0.95
        top_k: int = 40
        repeat_penalty: float = 1.1
        # Two-slot management (requires llama-server --parallel 2)
        num_slots: int = DEFAULT_NUM_SLOTS
        min_slot_reuse_delay_s: float = DEFAULT_MIN_SLOT_REUSE_DELAY_S

    def __init__(
        self,
        *,
        llama_url: str = "http://localhost:8000",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize LlamaCppChunkedLLMService.

        Args:
            llama_url: Base URL for llama.cpp server (e.g., "http://localhost:8000")
            params: Configuration parameters
        """
        super().__init__(**kwargs)

        self._params = params or self.InputParams()
        self._llama_url = llama_url.rstrip("/")

        # HTTP client (created in start())
        self._client: Optional[httpx.AsyncClient] = None

        # Generation state
        self._prompt: str = ""  # Formatted prompt (without generated text)
        self._generated_text: str = ""  # Full response so far (for cache)
        self._pending_token: Optional[str] = None  # Token saved from peeking
        self._cancelled: bool = False
        self._generating: bool = False
        self._is_first_chunk: bool = True

        # TTS synchronization
        self._continue_event: Optional[asyncio.Event] = None

        # Two-slot management: alternate between slots to avoid cancel race conditions
        self._num_slots = self._params.num_slots
        self._current_slot: int = 0
        self._slot_last_used: dict[int, float] = {}  # slot_id -> timestamp
        self._last_used_slot: Optional[int] = None  # Track last slot for reuse optimization

        # Metrics for current generation
        self._generation_start_time: Optional[float] = None
        self._generation_slot_id: int = 0
        self._generation_slot_reused: bool = False
        self._generation_tokens_cached: int = 0
        self._generation_tokens_evaluated: int = 0
        self._generation_tokens_predicted: int = 0  # Completion tokens

        self.set_model_name("llama-cpp-chunked")

        logger.info(
            f"LlamaCppChunkedLLMService initialized: url={self._llama_url}, "
            f"slots={self._num_slots} (reuse_delay={self._params.min_slot_reuse_delay_s}s), "
            f"first_chunk=({self._params.first_chunk_min_tokens}, "
            f"{self._params.first_chunk_max_tokens})"
        )

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Handle StartFrame - create HTTP client."""
        await super().start(frame)
        # Workaround for llama.cpp server crash (2025-12-31):
        # When using connection pooling, HTTP connections return to the pool after
        # a request completes. When the pool later closes a connection, llama.cpp
        # interprets the TCP close as a cancel signal. If a new task has started
        # on the same slot, the cancel handler sees slot.is_processing()=true and
        # hits GGML_ASSERT(!slot.is_processing()) in server-context.cpp:1011.
        # Disabling pooling ensures connections close immediately after each request,
        # so any cancel signal arrives while the slot is still idle.
        self._client = httpx.AsyncClient(
            timeout=300.0,
            limits=httpx.Limits(max_keepalive_connections=0),
        )
        logger.debug("LlamaCppChunkedLLM: HTTP client created")

    async def stop(self, frame: EndFrame):
        """Handle EndFrame - close HTTP client."""
        await super().stop(frame)
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.debug("LlamaCppChunkedLLM: HTTP client closed")

    async def cancel(self, frame: CancelFrame):
        """Handle CancelFrame - cancel generation and close client."""
        await super().cancel(frame)
        self._cancelled = True
        if self._continue_event:
            self._continue_event.set()
        if self._client:
            await self._client.aclose()
            self._client = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames."""
        await super().process_frame(frame, direction)

        context = None

        # Handle TTS continue signal (upstream)
        if isinstance(frame, ChunkedLLMContinueGenerationFrame):
            if self._continue_event:
                self._continue_event.set()
            return  # Don't propagate

        # Handle context frames
        if LLMContextFrame and isinstance(frame, LLMContextFrame):
            context = frame.context
        elif LLMMessagesFrame and isinstance(frame, LLMMessagesFrame):
            context = LLMContext(frame.messages)
        elif isinstance(frame, InterruptionFrame):
            await self._handle_interruption(frame)
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)

    async def _handle_interruption(self, frame: InterruptionFrame):
        """Handle interruption by cancelling current generation."""
        self._cancelled = True
        self._generating = False
        self._is_first_chunk = True
        if self._continue_event:
            self._continue_event.set()

    async def _get_next_slot(self) -> tuple[int, bool]:
        """Get next slot to use, preferring slot reuse for KV cache hits.

        Optimization: If the last used slot has been idle for at least
        min_slot_reuse_delay_s seconds, reuse it for better KV cache hits.
        Otherwise, alternate between slots to avoid the GGML_ASSERT crash.

        Returns:
            Tuple of (slot_id, slot_reused) where:
            - slot_id: The slot ID to use for the next request
            - slot_reused: True if we're reusing the same slot as last time
        """
        min_delay = self._params.min_slot_reuse_delay_s

        # Try to reuse the last slot if it's been idle long enough (better KV cache)
        if self._last_used_slot is not None:
            last_used = self._slot_last_used.get(self._last_used_slot, 0)
            elapsed = time.time() - last_used
            if elapsed >= min_delay:
                logger.info(
                    f"LlamaCppChunkedLLM: Reusing slot {self._last_used_slot} "
                    f"for KV cache (idle {elapsed:.1f}s)"
                )
                return self._last_used_slot, True

        # Otherwise, use the next slot in rotation
        slot = self._current_slot
        self._current_slot = (self._current_slot + 1) % self._num_slots
        if self._last_used_slot is None:
            logger.info(f"LlamaCppChunkedLLM: Using slot {slot} (first request)")
        else:
            last_used = self._slot_last_used.get(self._last_used_slot, 0)
            elapsed = time.time() - last_used
            logger.info(
                f"LlamaCppChunkedLLM: Rotating to slot {slot} "
                f"(slot {self._last_used_slot} only idle {elapsed:.1f}s < {min_delay}s)"
            )

        # Check reuse guard for this slot
        last_used = self._slot_last_used.get(slot, 0)
        elapsed = time.time() - last_used

        if elapsed < min_delay:
            wait_time = min_delay - elapsed
            logger.warning(
                f"LlamaCppChunkedLLM: Slot {slot} reuse guard triggered, "
                f"waiting {wait_time:.2f}s"
            )
            # Wait in small increments so we can respond to cancellation
            wait_end = time.time() + wait_time
            while time.time() < wait_end and not self._cancelled:
                await asyncio.sleep(0.1)  # 100ms increments

        return slot, False

    def _mark_slot_used(self, slot: int):
        """Record when a slot's request completed or was cancelled.

        This timestamp is used by _get_next_slot() to enforce the reuse guard
        and enable slot reuse for KV cache optimization.
        """
        self._slot_last_used[slot] = time.time()
        self._last_used_slot = slot

    def _format_messages(self, messages: list) -> str:
        """Format messages as ChatML prompt with thinking disabled.

        Always disables thinking (<think></think>) for voice agent use case
        where we want direct responses without internal reasoning.
        """
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        # Always disable thinking for voice agents
        prompt_parts.append("<|im_start|>assistant\n<think></think>")
        return "\n".join(prompt_parts)

    def _get_messages_for_logging(self, messages: list) -> list:
        """Truncate message content for logging."""
        result = []
        for msg in messages:
            content = msg.get("content", "")
            if len(content) > 100:
                content = content[:100] + "..."
            result.append({"role": msg.get("role"), "content": content})
        return result

    async def _process_context(self, context: LLMContext):
        """Process LLM context and generate response in chunks."""
        # Ensure HTTP client exists
        if not self._client:
            self._client = httpx.AsyncClient(
                timeout=300.0,
                limits=httpx.Limits(max_keepalive_connections=0),
            )

        # Cancel any existing generation and wait for it to exit
        if self._generating:
            logger.info("LlamaCppChunkedLLM: Cancelling previous generation")
            self._cancelled = True
            if self._continue_event:
                self._continue_event.set()
            # Wait for previous generation to actually exit (up to 500ms)
            # This prevents race where we reset _cancelled while previous is still running
            wait_start = time.time()
            while self._generating and (time.time() - wait_start) < 0.5:
                await asyncio.sleep(0.01)
            if self._generating:
                logger.warning("LlamaCppChunkedLLM: Previous generation didn't exit in time")

        # Reset state for new generation
        self._generated_text = ""
        self._pending_token = None
        self._cancelled = False
        self._generating = True
        self._is_first_chunk = True
        self._continue_event = asyncio.Event()

        # Reset metrics for this generation
        self._generation_slot_id = 0
        self._generation_slot_reused = False
        self._generation_tokens_cached = 0
        self._generation_tokens_evaluated = 0
        self._generation_tokens_predicted = 0

        messages = context.get_messages()
        if not messages:
            logger.warning("LlamaCppChunkedLLM: No messages in context")
            return

        logger.debug(f"{self}: Generating: {self._get_messages_for_logging(messages)}")

        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
            self._generation_start_time = time.time()

            # Format prompt once (generated_text is appended for each chunk)
            self._prompt = self._format_messages(messages)
            chunk_num = 0

            while self._generating and not self._cancelled:
                # Generate one chunk
                chunk_text, is_done = await self._generate_chunk()

                if self._cancelled:
                    break

                if not chunk_text and not is_done:
                    # Empty chunk but not done - might be an error or timeout
                    logger.warning("LlamaCppChunkedLLM: Empty chunk received, stopping")
                    break

                if chunk_text:
                    chunk_num += 1
                    # Stop TTFB metrics on first chunk
                    if self._is_first_chunk:
                        await self.stop_ttfb_metrics()
                    await self.push_frame(LLMTextFrame(text=chunk_text))
                    self._is_first_chunk = False

                if is_done:
                    break

                # Wait for TTS to finish before generating next chunk
                try:
                    await asyncio.wait_for(self._continue_event.wait(), timeout=30.0)
                except asyncio.TimeoutError:
                    logger.warning(f"LlamaCppChunkedLLM: Timeout waiting for TTS segment {chunk_num}")

                if self._cancelled:
                    break

                self._continue_event.clear()

            # Log completion metrics
            elapsed_ms = 0.0
            if self._generation_start_time:
                elapsed_ms = (time.time() - self._generation_start_time) * 1000
                logger.info(
                    f"LlamaCppChunkedLLM: Complete in {elapsed_ms:.0f}ms, "
                    f"{chunk_num} chunk{'s' if chunk_num != 1 else ''}"
                )

            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())

            # Push slot metrics frame after LLMFullResponseEndFrame
            slot_metrics_frame = LLMSlotMetricsFrame(
                slot_id=self._generation_slot_id,
                slot_reused=self._generation_slot_reused,
                total_chunks=chunk_num,
                total_time_ms=elapsed_ms,
                tokens_cached=self._generation_tokens_cached,
                tokens_evaluated=self._generation_tokens_evaluated,
            )
            logger.info(f"LlamaCppChunkedLLM: {slot_metrics_frame}")
            await self.push_frame(slot_metrics_frame)

            # Emit LLM token usage metrics for Pipecat Playground
            prompt_tokens = self._generation_tokens_cached + self._generation_tokens_evaluated
            completion_tokens = self._generation_tokens_predicted
            total_tokens = prompt_tokens + completion_tokens
            if total_tokens > 0:
                token_usage = LLMTokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cache_read_input_tokens=self._generation_tokens_cached,
                )
                usage_metrics_frame = MetricsFrame(
                    data=[
                        LLMUsageMetricsData(
                            processor=self.name,
                            value=token_usage,
                        )
                    ]
                )
                logger.info(
                    f"LlamaCppChunkedLLM: Token usage - "
                    f"prompt={prompt_tokens}, completion={completion_tokens}, "
                    f"total={total_tokens}, cached={self._generation_tokens_cached}"
                )
                await self.push_frame(usage_metrics_frame)

        except Exception as e:
            logger.error(f"LlamaCppChunkedLLM error: {e}")
            await self.stop_processing_metrics()
            await self.push_frame(ErrorFrame(error=str(e)))
            await self.push_frame(LLMFullResponseEndFrame())
        finally:
            self._generating = False
            self._continue_event = None

    async def _generate_chunk(self) -> tuple[str, bool]:
        """Generate a single chunk via HTTP streaming to llama.cpp.

        Handles:
        - Slot reuse optimization for KV cache hits
        - Two-slot alternation with reuse guard
        - Pending token prepending (from previous peek)
        - Sentence boundary detection with token peeking
        - First chunk min/max token bounds
        - Cache state tracking (_generated_text)
        - Cache metrics collection from llama.cpp response

        Returns:
            (chunk_text, is_done) where:
            - chunk_text: Text to emit to client (includes prepended pending token)
            - is_done: True if generation is complete (EOS or empty response)
        """
        # Get next slot (prefers reuse for KV cache, falls back to alternation)
        slot_id, slot_reused = await self._get_next_slot()

        # Track slot info on first chunk for metrics
        if self._is_first_chunk:
            self._generation_slot_id = slot_id
            self._generation_slot_reused = slot_reused

        if self._cancelled:
            return "", True

        # Build full prompt including generated text for cache hit
        full_prompt = self._prompt + self._generated_text

        # Determine token bounds based on chunk position
        if self._is_first_chunk:
            min_tokens = self._params.first_chunk_min_tokens
            max_tokens = self._params.first_chunk_max_tokens
            request_max = max_tokens + 20  # Buffer for word boundary completion
        else:
            min_tokens = None
            max_tokens = None
            request_max = 200  # Reasonable sentence limit

        payload = {
            "prompt": full_prompt,
            "n_predict": request_max,
            "id_slot": slot_id,
            "cache_prompt": True,
            "temperature": self._params.temperature,
            "top_p": self._params.top_p,
            "top_k": self._params.top_k,
            "repeat_penalty": self._params.repeat_penalty,
            "stream": True,
            "stop": ["<|im_end|>"],
        }

        collected_text = ""
        collected_tokens = 0
        hit_eos = False
        hit_sentence_boundary = False
        hit_max_tokens = False

        # Track prepended pending token (for cache state correction)
        prepended_from_pending = ""
        if self._pending_token:
            prepended_from_pending = self._pending_token
            collected_text = self._pending_token
            collected_tokens = 1
            self._pending_token = None

        try:
            async with self._client.stream(
                "POST",
                f"{self._llama_url}/completion",
                json=payload
            ) as response:
                async for line in response.aiter_lines():
                    # Check for cancellation
                    if self._cancelled:
                        logger.info("LlamaCppChunkedLLM: Cancelled during streaming")
                        break

                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        # Stream ended
                        if hit_sentence_boundary:
                            hit_eos = True
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    token_text = data.get("content", "")

                    # Check for EOS/stop token
                    if data.get("stop"):
                        stop_type = data.get("stop_type", "")
                        if stop_type in ("eos", "word"):
                            if token_text:
                                collected_text += token_text
                                collected_tokens += 1
                            # Capture token metrics from the stop event
                            # llama.cpp returns:
                            #   tokens_predicted = completion tokens generated
                            #   timings.cache_n = tokens reused from cache (cache HIT)
                            #   timings.prompt_n = tokens that needed evaluation
                            tokens_predicted = data.get("tokens_predicted", 0)
                            self._generation_tokens_predicted += tokens_predicted
                            timings = data.get("timings", {})
                            if timings:
                                cache_hit = timings.get("cache_n", 0)
                                prompt_eval = timings.get("prompt_n", 0)
                                self._generation_tokens_cached += cache_hit
                                self._generation_tokens_evaluated += prompt_eval
                                logger.info(
                                    f"LlamaCppChunkedLLM: Chunk cache: "
                                    f"{cache_hit}/{cache_hit + prompt_eval} tokens from cache "
                                    f"({100*cache_hit/(cache_hit + prompt_eval) if (cache_hit + prompt_eval) > 0 else 0:.0f}%), "
                                    f"predicted={tokens_predicted}, "
                                    f"prompt_ms={timings.get('prompt_ms', 0):.0f}, "
                                    f"predicted_ms={timings.get('predicted_ms', 0):.0f}"
                                )
                            hit_eos = True
                            break

                    # If we hit max_tokens, seek word boundary
                    if hit_max_tokens:
                        collected_text += token_text
                        collected_tokens += 1
                        if is_word_boundary(collected_text):
                            break
                        # Safety limit
                        if max_tokens and collected_tokens >= max_tokens + 20:
                            break
                        continue

                    # If we hit sentence boundary, peek for EOS or continuation
                    if hit_sentence_boundary:
                        peek_text = collected_text + token_text
                        if ends_at_sentence_boundary(peek_text):
                            # Token is part of sentence ending (e.g., closing quote)
                            collected_text = peek_text
                            collected_tokens += 1
                            continue
                        else:
                            # Token starts next sentence - save for next chunk
                            self._pending_token = token_text
                            break

                    collected_text += token_text
                    collected_tokens += 1

                    # Check for sentence boundary
                    if ends_at_sentence_boundary(collected_text):
                        if min_tokens and collected_tokens < min_tokens:
                            pass  # Keep going until min_tokens
                        else:
                            # Hit boundary with sufficient tokens, peek next
                            hit_sentence_boundary = True
                            continue

                    # Check max_tokens limit
                    if max_tokens and collected_tokens >= max_tokens:
                        if is_word_boundary(collected_text):
                            break
                        else:
                            hit_max_tokens = True
                            continue

        except httpx.RemoteProtocolError as e:
            logger.warning(f"LlamaCppChunkedLLM: Connection issue: {e}")
        except Exception as e:
            logger.error(f"LlamaCppChunkedLLM: HTTP streaming error: {e}")
            raise
        finally:
            # Mark slot as used so the reuse guard knows when it was last accessed
            self._mark_slot_used(slot_id)

        # If we hit sentence boundary without peeking more tokens, model is done
        if hit_sentence_boundary and not self._pending_token:
            hit_eos = True

        # Handle empty response
        if collected_tokens == 0:
            hit_eos = True

        # Update cache state correctly:
        # - Only add NEW tokens (exclude prepended pending token)
        # - Add any new pending token (was consumed from llama.cpp but saved for next chunk)
        new_tokens = collected_text[len(prepended_from_pending):]
        self._generated_text += new_tokens
        if self._pending_token:
            self._generated_text += self._pending_token

        return collected_text, hit_eos
