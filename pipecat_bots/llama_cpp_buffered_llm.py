"""Buffered chunked LLM service for Pipecat.

Connects directly to llama.cpp's HTTP API with a buffered approach that:
1. Runs LLM generations to completion (no mid-stream cancellation)
2. Uses a SentenceBuffer to extract text at sentence boundaries
3. Achieves ~100% KV cache reuse via single-slot operation

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                      LlamaCppBufferedLLMService                             │
    │                                                                             │
    │  ┌────────────────┐      ┌─────────────────┐      ┌───────────────────┐    │
    │  │  LLM Generator │      │  SentenceBuffer │      │   TTS Emitter     │    │
    │  │                │      │                 │      │                   │    │
    │  │  - Single slot │─────►│  - Accumulates  │─────►│  - Emits complete │    │
    │  │  - max_tokens  │      │  - Extracts at  │      │    sentences      │    │
    │  │  - Runs to     │      │    boundaries   │      │  - Waits for      │    │
    │  │    completion  │      │  - Keeps tail   │      │    continue       │    │
    │  └────────────────┘      └─────────────────┘      └───────────────────┘    │
    └─────────────────────────────────────────────────────────────────────────────┘

Key benefits vs sentence-cancel approach:
- 100% KV cache reuse (single slot, no mid-stream cancel)
- No GGML_ASSERT crashes from cancel race conditions
- Simpler code (no two-slot alternation, reuse guards)
- Lower GPU memory (--parallel 1 vs --parallel 2)

Usage:
    llm = LlamaCppBufferedLLMService(llama_url="http://localhost:8000")
    # In pipeline: ... -> context_aggregator.user() -> llm -> tts -> ...
"""

import asyncio
import json
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

from sentence_buffer import SentenceBuffer

# Import context frame types (handle different Pipecat versions)
try:
    from pipecat.frames.frames import LLMContextFrame
except ImportError:
    LLMContextFrame = None

try:
    from pipecat.frames.frames import LLMMessagesFrame
except ImportError:
    LLMMessagesFrame = None

# Import continue frame from shared module
from frames import ChunkedLLMContinueGenerationFrame


class LLMSlotMetricsFrame(SystemFrame):
    """Frame containing LLM slot usage and cache metrics.

    Enhanced with first-segment cache tracking for TTFB analysis.
    """

    def __init__(
        self,
        slot_id: int,
        total_chunks: int,
        total_time_ms: float,
        tokens_cached: int = 0,
        tokens_evaluated: int = 0,
        tokens_predicted: int = 0,
        first_segment_tokens_cached: int = 0,
        first_segment_tokens_evaluated: int = 0,
    ):
        super().__init__()
        self.slot_id = slot_id
        self.total_chunks = total_chunks
        self.total_time_ms = total_time_ms
        self.tokens_cached = tokens_cached
        self.tokens_evaluated = tokens_evaluated
        self.tokens_predicted = tokens_predicted
        self.first_segment_tokens_cached = first_segment_tokens_cached
        self.first_segment_tokens_evaluated = first_segment_tokens_evaluated

    @property
    def cache_hit_ratio(self) -> float:
        """Overall cache hit ratio across all generations."""
        total = self.tokens_cached + self.tokens_evaluated
        return self.tokens_cached / total if total > 0 else 0.0

    @property
    def first_segment_cache_hit_ratio(self) -> float:
        """Cache hit ratio for first segment (affects TTFB).

        High ratio (>90%) = System prompt + history cached = Fast TTFB
        Low ratio (<50%) = Context shifted or cold start = Slower TTFB
        """
        total = self.first_segment_tokens_cached + self.first_segment_tokens_evaluated
        return self.first_segment_tokens_cached / total if total > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"LLMSlotMetrics(slot={self.slot_id}, chunks={self.total_chunks}, "
            f"time={self.total_time_ms:.0f}ms, cached={self.tokens_cached}, "
            f"eval={self.tokens_evaluated}, predicted={self.tokens_predicted}, "
            f"hit={self.cache_hit_ratio:.1%}, first_hit={self.first_segment_cache_hit_ratio:.1%})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class LlamaCppBufferedLLMService(AIService):
    """LLM service using buffered approach for optimal KV cache utilization.

    Generates LLM responses by running generations to completion (no mid-stream
    cancel), accumulating output in a SentenceBuffer, and emitting at sentence
    boundaries for TTS.

    Key behaviors:
    - First segment: Uses first_segment_max_tokens for quick TTFB
    - Subsequent segments: Uses segment_max_tokens with hard_max for accumulation
    - Waits for TTS completion signal before emitting next chunk
    - Single slot operation for 100% KV cache reuse
    """

    class InputParams(BaseModel):
        """Configuration parameters for LlamaCppBufferedLLMService."""
        # First segment: quick TTFB, single generation then emit
        first_segment_max_tokens: int = 24
        first_segment_hard_max_tokens: int = 24

        # Subsequent segments: allow accumulation for complete sentences
        segment_max_tokens: int = 32
        segment_hard_max_tokens: int = 96

        # LLM generation parameters
        # Note: With temperature=0.0, top_p/top_k have no effect (greedy decoding)
        # repeat_penalty=1.0 matches NVIDIA's defaults for Nemotron 3 Nano
        temperature: float = 0.0
        repeat_penalty: float = 1.0

        # Single slot (no alternation needed with buffered approach)
        slot_id: int = 0

        # Context management (queried from server on startup)
        max_context_tokens: int = 16384
        context_reserve_tokens: int = 2048

    def __init__(
        self,
        *,
        llama_url: str = "http://localhost:8000",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize LlamaCppBufferedLLMService.

        Args:
            llama_url: Base URL for llama.cpp server (e.g., "http://localhost:8000")
            params: Configuration parameters
        """
        super().__init__(**kwargs)

        self._params = params or self.InputParams()
        self._llama_url = llama_url.rstrip("/")

        # HTTP client (created in start())
        self._client: Optional[httpx.AsyncClient] = None

        # Context management
        self._max_context_tokens = self._params.max_context_tokens

        # Generation state
        self._prompt: str = ""  # Formatted prompt (without generated text)
        self._generated_text: str = ""  # Full response so far (for cache)
        self._buffer: Optional[SentenceBuffer] = None
        self._cancelled: bool = False
        self._generating: bool = False
        self._generation_id: int = 0  # Incremented on each new generation/interrupt

        # TTS synchronization
        self._continue_event: Optional[asyncio.Event] = None

        # Metrics tracking
        self._generation_start_time: Optional[float] = None
        self._is_first_generation: bool = True
        self._first_segment_tokens_cached: int = 0
        self._first_segment_tokens_evaluated: int = 0
        self._total_tokens_cached: int = 0
        self._total_tokens_evaluated: int = 0
        self._total_tokens_predicted: int = 0

        self.set_model_name("llama-cpp-buffered")

        logger.info(
            f"LlamaCppBufferedLLMService initialized: url={self._llama_url}, "
            f"slot={self._params.slot_id}, "
            f"first_segment=({self._params.first_segment_max_tokens}, "
            f"{self._params.first_segment_hard_max_tokens}), "
            f"segment=({self._params.segment_max_tokens}, "
            f"{self._params.segment_hard_max_tokens})"
        )

    def can_generate_metrics(self) -> bool:
        return True

    async def _query_server_props(self) -> dict:
        """Query llama.cpp server for configuration properties.

        Returns:
            Server properties including n_ctx (context size per slot).
        """
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self._llama_url}/props")
            response.raise_for_status()
            return response.json()

    async def start(self, frame: StartFrame):
        """Handle StartFrame - create HTTP client and query server config."""
        await super().start(frame)

        # Query actual context size from server
        try:
            props = await self._query_server_props()
            logger.debug(f"LlamaCppBufferedLLM: Server props: {props}")

            n_ctx = props.get("default_generation_settings", {}).get("n_ctx", 16384)
            total_slots = props.get("total_slots", 1)

            # With single-slot mode, we get the full context
            self._max_context_tokens = n_ctx
            logger.info(
                f"LlamaCppBufferedLLM: Server config - n_ctx={n_ctx}, "
                f"slots={total_slots}"
            )
        except Exception as e:
            logger.warning(f"Failed to query server props, using default: {e}")
            self._max_context_tokens = self._params.max_context_tokens

        # Create HTTP client with connection pooling disabled
        # This avoids issues with connection reuse after interruptions
        self._client = httpx.AsyncClient(
            timeout=300.0,
            limits=httpx.Limits(max_keepalive_connections=0)
        )
        logger.debug("LlamaCppBufferedLLM: HTTP client created")

    async def stop(self, frame: EndFrame):
        """Handle EndFrame - close HTTP client."""
        await super().stop(frame)
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.debug("LlamaCppBufferedLLM: HTTP client closed")

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
        """Handle user interruption."""
        # Increment generation ID to invalidate in-flight results
        self._generation_id += 1

        # Clear buffer
        if self._buffer:
            self._buffer.clear()

        # Reset state
        self._cancelled = True
        self._generating = False

        # Signal any waiting coroutines
        if self._continue_event:
            self._continue_event.set()

    def _format_messages(self, messages: list) -> str:
        """Format messages as ChatML prompt with thinking disabled."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        # Always disable thinking for voice agents
        prompt_parts.append("<|im_start|>assistant\n<think></think>")
        return "\n".join(prompt_parts)

    def _estimate_tokens(self, msg: dict) -> int:
        """Estimate token count for a message (~4 chars per token)."""
        content = msg.get("content", "")
        return len(content) // 4 + 10  # +10 for role/formatting overhead

    def _trim_messages_to_fit_context(self, messages: list) -> list:
        """Drop oldest messages if conversation exceeds context limit.

        Preserves:
        - System message (always first)
        - Most recent messages up to context limit

        Returns:
            Trimmed message list that fits within context window.
        """
        max_tokens = self._max_context_tokens - self._params.context_reserve_tokens

        # Always keep system message
        if messages and messages[0].get("role") == "system":
            system_msg = messages[0]
            other_msgs = messages[1:]
            system_tokens = self._estimate_tokens(system_msg)
        else:
            system_msg = None
            other_msgs = messages
            system_tokens = 0

        # Calculate tokens for remaining messages (newest first)
        available = max_tokens - system_tokens
        kept_msgs = []
        for msg in reversed(other_msgs):
            msg_tokens = self._estimate_tokens(msg)
            if available >= msg_tokens:
                kept_msgs.insert(0, msg)
                available -= msg_tokens
            else:
                break  # No more room

        if system_msg:
            kept_msgs.insert(0, system_msg)

        if len(kept_msgs) < len(messages):
            dropped = len(messages) - len(kept_msgs)
            logger.warning(f"Context limit: dropped {dropped} oldest messages")

        return kept_msgs

    async def _process_context(self, context: LLMContext):
        """Process LLM context and generate response using buffered approach."""
        context_received_time = time.time()
        logger.debug(f"LLM: Context received at {context_received_time:.3f}")

        # Ensure HTTP client exists
        if not self._client:
            self._client = httpx.AsyncClient(
                timeout=300.0,
                limits=httpx.Limits(max_keepalive_connections=0),
            )

        # Cancel any existing generation and wait for it to exit
        if self._generating:
            logger.info("LlamaCppBufferedLLM: Cancelling previous generation")
            self._cancelled = True
            if self._continue_event:
                self._continue_event.set()
            # Wait for previous generation to actually exit
            wait_start = time.time()
            while self._generating and (time.time() - wait_start) < 0.5:
                await asyncio.sleep(0.01)
            gen_wait_ms = (time.time() - wait_start) * 1000
            if gen_wait_ms > 1:
                logger.debug(f"LLM: Generation cancel wait took {gen_wait_ms:.0f}ms")

        # Initialize state for new generation
        self._buffer = SentenceBuffer()
        self._generated_text = ""
        self._generation_id += 1
        my_generation_id = self._generation_id
        self._cancelled = False
        self._generating = True
        self._continue_event = asyncio.Event()

        # Initialize metrics for this response
        self._is_first_generation = True
        self._first_segment_tokens_cached = 0
        self._first_segment_tokens_evaluated = 0
        self._total_tokens_cached = 0
        self._total_tokens_evaluated = 0
        self._total_tokens_predicted = 0

        messages = context.get_messages()
        if not messages:
            logger.warning("LlamaCppBufferedLLM: No messages in context")
            return

        # Trim messages to fit context window, then format
        messages = self._trim_messages_to_fit_context(messages)
        self._prompt = self._format_messages(messages)

        # Segment limits - first segment uses equal max/hard_max
        max_tokens = self._params.first_segment_max_tokens
        hard_max_tokens = self._params.first_segment_hard_max_tokens

        chunk_num = 0
        hit_eos = False

        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
            self._generation_start_time = time.time()

            while not self._cancelled:
                # Step 1: Generate tokens (runs to completion, no mid-stream cancel)
                new_text, new_tokens, hit_eos = await self._generate(
                    max_tokens, my_generation_id
                )

                if self._cancelled or self._generation_id != my_generation_id:
                    break  # Interrupted, discard results

                self._buffer.add(new_text, new_tokens)

                # Step 2: Check buffer and decide action
                # Priority: sentences > hard_max > generate more > EOS

                sentences = self._buffer.extract_complete_sentences()
                if sentences:
                    # Found complete sentences - emit all of them
                    chunk_num += 1
                    if chunk_num == 1:
                        await self.stop_ttfb_metrics()
                    await self._emit_and_wait(sentences)

                    if hit_eos:
                        # EOS reached - emit any remainder and finish
                        # Don't try to generate more - the LLM has completed its response
                        if self._buffer.has_content():
                            chunk_num += 1
                            await self._emit_and_wait(self._buffer.text.strip())
                            self._buffer.clear()
                        break

                    # Switch to subsequent segment limits for next iteration
                    max_tokens = self._params.segment_max_tokens
                    hard_max_tokens = self._params.segment_hard_max_tokens
                    continue

                if self._buffer.token_count >= hard_max_tokens:
                    # Hit hard limit without sentence - emit at best boundary
                    text = self._buffer.extract_at_boundary()
                    if text:
                        chunk_num += 1
                        if chunk_num == 1:
                            await self.stop_ttfb_metrics()
                        await self._emit_and_wait(text)

                    if hit_eos:
                        # EOS reached - emit any remainder and finish
                        if self._buffer.has_content():
                            chunk_num += 1
                            await self._emit_and_wait(self._buffer.text.strip())
                            self._buffer.clear()
                        break

                    # Switch to subsequent segment limits
                    max_tokens = self._params.segment_max_tokens
                    hard_max_tokens = self._params.segment_hard_max_tokens
                    continue

                if hit_eos:
                    # EOS reached - emit whatever remains and finish
                    if self._buffer.has_content():
                        chunk_num += 1
                        if chunk_num == 1:
                            await self.stop_ttfb_metrics()
                        await self._emit_and_wait(self._buffer.text.strip())
                        self._buffer.clear()
                    break

                # No sentence, under hard_max, not EOS - generate more tokens
                # (Loop continues with same limits until we emit something)

            # Log completion metrics
            elapsed_ms = 0.0
            if self._generation_start_time:
                elapsed_ms = (time.time() - self._generation_start_time) * 1000
                logger.info(
                    f"LlamaCppBufferedLLM: Complete in {elapsed_ms:.0f}ms, "
                    f"{chunk_num} chunk{'s' if chunk_num != 1 else ''}"
                )

            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())

            # Push slot metrics frame
            slot_metrics_frame = LLMSlotMetricsFrame(
                slot_id=self._params.slot_id,
                total_chunks=chunk_num,
                total_time_ms=elapsed_ms,
                tokens_cached=self._total_tokens_cached,
                tokens_evaluated=self._total_tokens_evaluated,
                tokens_predicted=self._total_tokens_predicted,
                first_segment_tokens_cached=self._first_segment_tokens_cached,
                first_segment_tokens_evaluated=self._first_segment_tokens_evaluated,
            )
            logger.info(f"LlamaCppBufferedLLM: {slot_metrics_frame}")
            await self.push_frame(slot_metrics_frame)

            # Emit LLM token usage metrics for Pipecat Playground
            prompt_tokens = self._total_tokens_cached + self._total_tokens_evaluated
            completion_tokens = self._total_tokens_predicted
            total_tokens = prompt_tokens + completion_tokens
            if total_tokens > 0:
                token_usage = LLMTokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cache_read_input_tokens=self._total_tokens_cached,
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
                    f"LlamaCppBufferedLLM: Token usage - "
                    f"prompt={prompt_tokens}, completion={completion_tokens}, "
                    f"total={total_tokens}, cached={self._total_tokens_cached}"
                )
                await self.push_frame(usage_metrics_frame)

        except Exception as e:
            logger.error(f"LlamaCppBufferedLLM error: {e}")
            await self.stop_processing_metrics()
            await self.push_frame(ErrorFrame(error=str(e)))
            await self.push_frame(LLMFullResponseEndFrame())
        finally:
            self._generating = False
            self._continue_event = None

    async def _emit_and_wait(self, text: str):
        """Emit text to TTS and wait for completion signal."""
        await self.push_frame(LLMTextFrame(text=text))
        self._buffer.reset_token_count()  # Reset after emit, regardless of remainder

        try:
            await asyncio.wait_for(self._continue_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for TTS continue signal")
        self._continue_event.clear()

    async def _generate(
        self, max_tokens: int, expected_gen_id: int
    ) -> tuple[str, int, bool]:
        """Generate tokens via llama.cpp HTTP API.

        Args:
            max_tokens: Maximum tokens to generate in this request
            expected_gen_id: Generation ID to check for staleness

        Returns:
            (generated_text, tokens_generated, hit_eos)

        Note: hit_eos is True only for natural completion (EOS token or stop word).
        Hitting the n_predict limit returns hit_eos=False.
        """
        full_prompt = self._prompt + self._generated_text

        payload = {
            "prompt": full_prompt,
            "n_predict": max_tokens,
            "id_slot": self._params.slot_id,
            "cache_prompt": True,
            "temperature": self._params.temperature,
            "repeat_penalty": self._params.repeat_penalty,
            "stream": True,
            "stop": ["<|im_end|>"],
        }

        collected_text = ""
        tokens_generated = 0
        hit_eos = False

        try:
            async with self._client.stream(
                "POST", f"{self._llama_url}/completion", json=payload
            ) as response:
                async for line in response.aiter_lines():
                    # Check for stale generation
                    if self._generation_id != expected_gen_id:
                        return "", 0, False

                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if data.get("stop"):
                        # Generation stopped - check why
                        stop_type = data.get("stop_type", "")
                        token_text = data.get("content", "")
                        if token_text:
                            collected_text += token_text

                        # Capture metrics from response
                        tokens_generated = data.get("tokens_predicted", 0)
                        timings = data.get("timings", {})

                        # Update aggregated metrics
                        self._total_tokens_cached += timings.get("cache_n", 0)
                        self._total_tokens_evaluated += timings.get("prompt_n", 0)
                        self._total_tokens_predicted += tokens_generated

                        # Capture first segment metrics
                        if self._is_first_generation:
                            self._first_segment_tokens_cached = timings.get("cache_n", 0)
                            self._first_segment_tokens_evaluated = timings.get("prompt_n", 0)
                            self._is_first_generation = False

                        # Only set hit_eos for natural completion, NOT for hitting n_predict
                        if stop_type in ("eos", "word"):
                            hit_eos = True

                        break

                    collected_text += data.get("content", "")

        except httpx.RemoteProtocolError as e:
            logger.warning(f"LlamaCppBufferedLLM: Connection issue: {e}")
        except Exception as e:
            logger.error(f"LlamaCppBufferedLLM: HTTP streaming error: {e}")
            raise

        # Handle empty generation (model immediately hit EOS)
        if not collected_text and tokens_generated == 0:
            hit_eos = True

        # Update cache state for next generation
        self._generated_text += collected_text

        return collected_text, tokens_generated, hit_eos
