#!/usr/bin/env python3
"""WebRTC test client for multi-turn voice agent testing.

Connects to the bot via WebRTC, maintains a persistent connection,
and supports sending multiple audio turns with proper timing.

Usage:
    # Multi-turn test (primary use case)
    uv run scripts/run_20_turn_test.py

    # Single turn via this script
    uv run scripts/voice_agent_test_client.py --text "Hello, how are you?"
"""

import asyncio
import json
import os
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import aiohttp
    from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
    from aiortc.contrib.media import MediaRecorder
    from av import AudioFrame
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install aiortc aiohttp numpy")
    exit(1)


@dataclass
class TurnMetrics:
    """Metrics collected for a single conversation turn."""
    turn_number: int
    utterance_text: str
    utterance_duration_ms: float
    audio_sent_time: float  # Timestamp when audio finished sending
    bot_started_speaking_time: Optional[float] = None
    bot_stopped_speaking_time: Optional[float] = None
    events: list = field(default_factory=list)

    @property
    def time_to_response_ms(self) -> Optional[float]:
        """Time from audio sent to bot starting to speak."""
        if self.bot_started_speaking_time and self.audio_sent_time:
            return (self.bot_started_speaking_time - self.audio_sent_time) * 1000
        return None

    @property
    def response_duration_ms(self) -> Optional[float]:
        """Duration of bot's response."""
        if self.bot_started_speaking_time and self.bot_stopped_speaking_time:
            return (self.bot_stopped_speaking_time - self.bot_started_speaking_time) * 1000
        return None


def load_audio_file(path: str, target_sample_rate: int = 16000) -> np.ndarray:
    """Load audio from WAV or raw PCM file and resample to target rate."""
    # Try to open as WAV first
    try:
        with wave.open(path, "rb") as wf:
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            raw_data = wf.readframes(wf.getnframes())
    except wave.Error:
        # Not a WAV file - assume raw PCM (Magpie format: 22kHz mono s16le)
        with open(path, "rb") as f:
            raw_data = f.read()
        nchannels = 1
        sampwidth = 2
        framerate = 22000

    # Convert to numpy array
    if sampwidth == 2:
        samples = np.frombuffer(raw_data, dtype=np.int16)
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    # Convert stereo to mono
    if nchannels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)

    # Resample if needed
    if framerate != target_sample_rate:
        ratio = target_sample_rate / framerate
        new_length = int(len(samples) * ratio)
        indices = np.linspace(0, len(samples) - 1, new_length)
        samples = np.interp(indices, np.arange(len(samples)), samples).astype(np.int16)

    return samples


class MultiTurnAudioTrack(MediaStreamTrack):
    """Audio track that supports sending multiple audio segments across conversation turns.

    Sends silence when idle, real audio when queued. Maintains realtime pacing.
    """

    kind = "audio"

    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self._sample_rate = sample_rate
        self._frame_duration = 0.02  # 20ms frames
        self._samples_per_frame = int(sample_rate * self._frame_duration)

        # Audio state
        self._current_audio: Optional[np.ndarray] = None
        self._audio_position = 0
        self._audio_complete = asyncio.Event()
        self._audio_complete.set()  # Initially complete (no audio pending)

        # Timing
        self._start_time: Optional[float] = None
        self._frame_count = 0

    def queue_audio(self, samples: np.ndarray):
        """Queue audio samples to be sent. Clears any previous audio."""
        self._current_audio = samples
        self._audio_position = 0
        self._audio_complete.clear()

    async def wait_for_completion(self) -> float:
        """Wait until queued audio has been fully sent. Returns completion timestamp."""
        await self._audio_complete.wait()
        return time.time()

    def is_sending(self) -> bool:
        """Check if currently sending non-silence audio."""
        return self._current_audio is not None and self._audio_position < len(self._current_audio)

    async def recv(self) -> AudioFrame:
        """Generate next audio frame at realtime rate."""
        # Initialize timing on first call
        if self._start_time is None:
            self._start_time = time.time()

        # Calculate expected time for this frame
        expected_time = self._start_time + self._frame_count * self._frame_duration
        now = time.time()
        if expected_time > now:
            await asyncio.sleep(expected_time - now)

        # Get samples for this frame
        if self._current_audio is not None and self._audio_position < len(self._current_audio):
            # Send real audio
            start = self._audio_position
            end = min(start + self._samples_per_frame, len(self._current_audio))
            samples = self._current_audio[start:end]

            # Pad if needed
            if len(samples) < self._samples_per_frame:
                samples = np.pad(samples, (0, self._samples_per_frame - len(samples)))

            self._audio_position = end

            # Check if audio is complete
            if self._audio_position >= len(self._current_audio):
                self._audio_complete.set()
        else:
            # Send silence
            samples = np.zeros(self._samples_per_frame, dtype=np.int16)

        # Create AudioFrame
        frame = AudioFrame(format="s16", layout="mono", samples=self._samples_per_frame)
        frame.sample_rate = self._sample_rate
        frame.pts = self._frame_count * self._samples_per_frame
        frame.planes[0].update(samples.tobytes())

        self._frame_count += 1
        return frame


class MultiTurnVoiceAgentClient:
    """WebRTC client for multi-turn voice agent testing.

    Maintains a persistent connection and supports sending multiple turns.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:7860",
        tts_url: str = "http://localhost:8001",
        output_dir: Optional[str] = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.tts_url = tts_url.rstrip("/")
        self.output_dir = Path(output_dir) if output_dir else Path("/tmp/voice_agent_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pc: Optional[RTCPeerConnection] = None
        self.audio_track: Optional[MultiTurnAudioTrack] = None
        self.data_channel = None

        # Connection state
        self._connection_ready = asyncio.Event()
        self._bot_stopped_speaking = asyncio.Event()

        # Event tracking
        self.all_events: list = []
        self._turn_events: list = []  # Events for current turn
        self._bot_stopped_count = 0
        self._current_turn = 0
        self.connection_start_time: Optional[float] = None

        # Turn timing
        self._bot_started_time: Optional[float] = None
        self._bot_stopped_time: Optional[float] = None

    async def connect(self) -> bool:
        """Establish WebRTC connection to voice agent."""
        self.pc = RTCPeerConnection()
        self.connection_start_time = time.time()

        # Create audio track that starts with silence
        self.audio_track = MultiTurnAudioTrack()
        self.pc.addTrack(self.audio_track)

        # Create data channel for RTVI messages
        self.data_channel = self.pc.createDataChannel("rtvi-ai", ordered=True)

        @self.data_channel.on("open")
        def on_dc_open():
            print("Data channel opened, sending client-ready")
            ready_msg = json.dumps({
                "label": "rtvi-ai",
                "type": "client-ready",
                "id": "test-client-1",
                "data": {}
            })
            self.data_channel.send(ready_msg)

        @self.data_channel.on("message")
        def on_dc_message(message):
            self._handle_rtvi_message(message)

        # Handle incoming audio track from bot
        @self.pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                print("Receiving bot audio track")
                # Could add recording here if needed

        # Create and send offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        # Exchange SDP with server
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.server_url}/api/offer",
                    json={"sdp": self.pc.localDescription.sdp, "type": self.pc.localDescription.type},
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status != 200:
                        print(f"Failed to connect: {response.status}")
                        text = await response.text()
                        print(f"Response: {text[:200]}")
                        return False
                    answer = await response.json()
                    await self.pc.setRemoteDescription(
                        RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
                    )
                    print("WebRTC connection established")

                    # Wait for bot-ready
                    try:
                        await asyncio.wait_for(self._connection_ready.wait(), timeout=10.0)
                        print("Bot is ready")
                    except asyncio.TimeoutError:
                        print("Warning: Timeout waiting for bot-ready")

                    return True
            except aiohttp.ClientError as e:
                print(f"Connection error: {e}")
                return False

    def _handle_rtvi_message(self, message: str):
        """Handle incoming RTVI message."""
        try:
            event = json.loads(message)
            event_time = time.time()

            # Store event
            event_record = {"time": event_time, "event": event}
            self.all_events.append(event_record)
            self._turn_events.append(event_record)

            event_type = event.get("type", "unknown")

            if event_type == "bot-ready":
                self._connection_ready.set()

            elif event_type == "bot-started-speaking":
                self._bot_started_time = event_time

            elif event_type == "bot-stopped-speaking":
                self._bot_stopped_count += 1
                self._bot_stopped_time = event_time
                self._bot_stopped_speaking.set()

        except json.JSONDecodeError:
            pass

    async def wait_for_greeting(self, timeout: float = 30.0) -> bool:
        """Wait for bot's initial greeting to complete."""
        print("Waiting for bot greeting to complete...")
        try:
            await asyncio.wait_for(self._bot_stopped_speaking.wait(), timeout=timeout)
            print("Bot greeting complete")
            self._bot_stopped_speaking.clear()
            return True
        except asyncio.TimeoutError:
            print("Timeout waiting for greeting")
            return False

    async def synthesize_audio(self, text: str) -> str:
        """Synthesize audio from text using TTS service."""
        output_path = f"/tmp/test_audio_{int(time.time() * 1000)}.pcm"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.tts_url}/v1/audio/speech",
                json={"input": text, "voice": "aria", "model": "magpie"},
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"TTS failed: {response.status}")
                audio_data = await response.read()

        with open(output_path, "wb") as f:
            f.write(audio_data)

        return output_path

    async def send_turn(self, text: str, timeout: float = 90.0) -> TurnMetrics:
        """Send a conversation turn and wait for bot response.

        Args:
            text: The utterance text to synthesize and send
            timeout: Maximum time to wait for bot response

        Returns:
            TurnMetrics with timing and event data
        """
        self._current_turn += 1
        self._turn_events = []
        self._bot_stopped_speaking.clear()
        self._bot_started_time = None
        self._bot_stopped_time = None

        # Synthesize audio (with retry on transient failures)
        for attempt in range(3):
            try:
                audio_path = await self.synthesize_audio(text)
                break
            except RuntimeError as e:
                if attempt < 2:
                    print(f"  TTS failed, retrying... ({e})")
                    await asyncio.sleep(0.5)
                else:
                    raise

        samples = load_audio_file(audio_path)
        utterance_duration_ms = len(samples) / self.audio_track._sample_rate * 1000

        # Queue audio and wait for it to be sent at realtime pace.
        # During long utterances, natural speech pauses may trigger false starts
        # where the bot briefly responds then gets interrupted when speech resumes.
        self.audio_track.queue_audio(samples)
        audio_sent_time = await self.audio_track.wait_for_completion()

        # NOW clear timing state - ignore all false starts that occurred during audio.
        # The real response will be the first bot speaking cycle AFTER audio is done.
        self._bot_stopped_speaking.clear()
        self._bot_started_time = None
        self._bot_stopped_time = None

        # Wait for the real bot response (after user truly finishes speaking)
        try:
            await asyncio.wait_for(self._bot_stopped_speaking.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"  Timeout waiting for response on turn {self._current_turn}")

        # Clean up temp file
        try:
            os.unlink(audio_path)
        except OSError:
            pass

        return TurnMetrics(
            turn_number=self._current_turn,
            utterance_text=text,
            utterance_duration_ms=utterance_duration_ms,
            audio_sent_time=audio_sent_time,
            bot_started_speaking_time=self._bot_started_time,
            bot_stopped_speaking_time=self._bot_stopped_time,
            events=list(self._turn_events),
        )

    async def close(self):
        """Close the WebRTC connection."""
        if self.pc:
            await self.pc.close()
            print("Connection closed")

        # Save all events
        events_file = self.output_dir / "all_events.json"
        with open(events_file, "w") as f:
            json.dump(self.all_events, f, indent=2, default=str)


async def synthesize_audio(text: str, tts_url: str = "http://localhost:8001") -> str:
    """Synthesize audio from text using Magpie TTS (standalone function for compatibility)."""
    output_path = f"/tmp/test_audio_{int(time.time() * 1000)}.pcm"

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{tts_url}/v1/audio/speech",
            json={"input": text, "voice": "aria", "model": "magpie"},
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"TTS failed: {response.status}")
            audio_data = await response.read()

    with open(output_path, "wb") as f:
        f.write(audio_data)

    return output_path


async def main():
    """Simple single-turn test for standalone usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Voice agent test client")
    parser.add_argument("--text", required=True, help="Text to synthesize and send")
    parser.add_argument("--server-url", default="http://localhost:7860", help="Bot server URL")
    parser.add_argument("--tts-url", default="http://localhost:8001", help="TTS server URL")
    parser.add_argument("--output-dir", default="/tmp/voice_agent_test", help="Output directory")
    parser.add_argument("--timeout", type=float, default=30.0, help="Response timeout")
    args = parser.parse_args()

    client = MultiTurnVoiceAgentClient(
        server_url=args.server_url,
        tts_url=args.tts_url,
        output_dir=args.output_dir,
    )

    try:
        if not await client.connect():
            return

        await client.wait_for_greeting()

        metrics = await client.send_turn(args.text, timeout=args.timeout)

        print(f"\nResults:")
        print(f"  Utterance: {metrics.utterance_text[:50]}...")
        print(f"  Utterance duration: {metrics.utterance_duration_ms:.0f}ms")
        if metrics.time_to_response_ms:
            print(f"  Time to response: {metrics.time_to_response_ms:.0f}ms")
        if metrics.response_duration_ms:
            print(f"  Response duration: {metrics.response_duration_ms:.0f}ms")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
