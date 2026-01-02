#!/usr/bin/env python3
"""WebRTC test client for voice agent testing.

Connects to the bot via WebRTC, sends audio at realtime rate,
and captures response metrics and audio.

Usage:
    uv run scripts/voice_agent_test_client.py --audio-file /tmp/test.wav
    uv run scripts/voice_agent_test_client.py --text "Hello, how are you?"
"""

import argparse
import asyncio
import json
import os
import struct
import time
import wave
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


class AudioFileTrack(MediaStreamTrack):
    """Audio track that plays from a WAV file at realtime rate."""

    kind = "audio"

    def __init__(self, path: str, sample_rate: int = 16000, start_event: Optional[asyncio.Event] = None):
        super().__init__()
        self._path = path
        self._sample_rate = sample_rate
        self._samples: Optional[np.ndarray] = None
        self._position = 0
        self._start_time: Optional[float] = None
        self._frame_duration = 0.02  # 20ms frames
        self._samples_per_frame = int(sample_rate * self._frame_duration)
        self._start_event = start_event  # Wait for this before sending real audio
        self._started = False
        self._load_audio()

    def _load_audio(self):
        """Load audio from WAV or raw PCM file."""
        # Try to open as WAV first
        try:
            with wave.open(self._path, "rb") as wf:
                nchannels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                raw_data = wf.readframes(wf.getnframes())
        except wave.Error:
            # Not a WAV file - assume raw PCM (Magpie format: 22kHz mono s16le)
            print("Detected raw PCM audio, assuming Magpie TTS format (22kHz mono s16le)")
            with open(self._path, "rb") as f:
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

        # Resample if needed (simple decimation/interpolation)
        if framerate != self._sample_rate:
            ratio = self._sample_rate / framerate
            new_length = int(len(samples) * ratio)
            indices = np.linspace(0, len(samples) - 1, new_length)
            samples = np.interp(indices, np.arange(len(samples)), samples).astype(
                np.int16
            )

        self._samples = samples
        print(f"Loaded audio: {len(self._samples)} samples, {len(self._samples) / self._sample_rate:.2f}s")

    async def recv(self) -> AudioFrame:
        """Receive next audio frame at realtime rate.

        Waits for start_event before sending real audio (sends silence while waiting).
        Continues sending silence frames after audio ends to maintain connection.
        """
        # Track frame timing from first call
        if self._start_time is None:
            self._start_time = time.time()

        frame_num = self._position // self._samples_per_frame

        # Wait for frame timing
        expected_time = self._start_time + frame_num * self._frame_duration
        now = time.time()
        if expected_time > now:
            await asyncio.sleep(expected_time - now)

        # Check if we should start sending real audio
        if not self._started:
            if self._start_event is None or self._start_event.is_set():
                self._started = True
                self._position = 0  # Reset position when starting
                print("Starting to send user audio")
            else:
                # Send silence while waiting for start event
                samples = np.zeros(self._samples_per_frame, dtype=np.int16)
                frame = AudioFrame(format="s16", layout="mono", samples=self._samples_per_frame)
                frame.sample_rate = self._sample_rate
                frame.pts = frame_num * self._samples_per_frame
                frame.planes[0].update(samples.tobytes())
                self._position += self._samples_per_frame
                return frame

        # Get samples for this frame
        start = self._position
        end = min(start + self._samples_per_frame, len(self._samples))

        if start >= len(self._samples):
            # End of audio - continue sending silence to maintain connection
            samples = np.zeros(self._samples_per_frame, dtype=np.int16)
            self._position += self._samples_per_frame
        else:
            samples = self._samples[start:end]
            if len(samples) < self._samples_per_frame:
                samples = np.pad(samples, (0, self._samples_per_frame - len(samples)))
            self._position = end

        # Create AudioFrame
        frame = AudioFrame(format="s16", layout="mono", samples=self._samples_per_frame)
        frame.sample_rate = self._sample_rate
        frame.pts = frame_num * self._samples_per_frame
        frame.planes[0].update(samples.tobytes())

        return frame


class VoiceAgentTestClient:
    """WebRTC client for testing voice agents."""

    def __init__(
        self,
        server_url: str = "http://localhost:7860",
        output_dir: Optional[str] = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.output_dir = Path(output_dir) if output_dir else Path("/tmp/voice_agent_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pc: Optional[RTCPeerConnection] = None
        self.recorder: Optional[MediaRecorder] = None
        self.events: list = []
        self.start_time: Optional[float] = None
        self.first_audio_time: Optional[float] = None
        self.data_channel = None

    async def connect(self, audio_path: str) -> bool:
        """Connect to voice agent via WebRTC with audio track ready."""
        self.pc = RTCPeerConnection()
        self.start_time = time.time()
        self._connection_ready = asyncio.Event()
        self._audio_start_event = asyncio.Event()  # Triggered when bot finishes greeting

        # Load and add audio track BEFORE creating offer
        # Audio track waits for _audio_start_event before sending real audio
        self.audio_track = AudioFileTrack(audio_path, start_event=self._audio_start_event)
        self.pc.addTrack(self.audio_track)
        print(f"Added audio track: {self.audio_track._samples.shape[0] / self.audio_track._sample_rate:.2f}s")

        # Create data channel for RTVI messages (client creates it)
        self.data_channel = self.pc.createDataChannel("rtvi-ai", ordered=True)
        print("Created RTVI data channel")

        @self.data_channel.on("open")
        def on_dc_open():
            print("Data channel opened, sending client-ready")
            # Send RTVI client-ready message in proper format
            ready_msg = json.dumps({
                "label": "rtvi-ai",
                "type": "client-ready",
                "id": "test-client-1",
                "data": {}
            })
            self.data_channel.send(ready_msg)

        @self.data_channel.on("message")
        def on_dc_message(message):
            try:
                event = json.loads(message)
                self.events.append({
                    "time": time.time() - self.start_time,
                    "event": event,
                })
                event_type = event.get("type", "unknown")
                print(f"RTVI event: {event_type}")

                # Mark connection ready when bot is ready
                if event_type == "bot-ready":
                    self._connection_ready.set()

                # Start sending user audio when bot stops speaking (greeting finished)
                if event_type == "bot-stopped-speaking":
                    if not self._audio_start_event.is_set():
                        print("Bot finished greeting, starting user audio")
                        self._audio_start_event.set()
            except json.JSONDecodeError:
                pass

        # Handle incoming audio track from bot
        @self.pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                print("Receiving bot audio track")
                if self.first_audio_time is None:
                    self.first_audio_time = time.time()
                # Record received audio
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.recorder = MediaRecorder(str(self.output_dir / f"bot_response_{timestamp}.wav"))
                self.recorder.addTrack(track)
                asyncio.create_task(self.recorder.start())

        # Create offer with audio track already added
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

                    # Wait for bot-ready with timeout
                    try:
                        await asyncio.wait_for(self._connection_ready.wait(), timeout=10.0)
                        print("Bot is ready")
                    except asyncio.TimeoutError:
                        print("Warning: Timeout waiting for bot-ready, continuing anyway")

                    return True
            except aiohttp.ClientError as e:
                print(f"Connection error: {e}")
                return False

    async def wait_for_audio_sent(self):
        """Wait for audio track to finish sending."""
        if not hasattr(self, 'audio_track'):
            return

        # Wait for audio to finish
        duration = len(self.audio_track._samples) / self.audio_track._sample_rate
        print(f"Sending audio: {duration:.2f}s")
        await asyncio.sleep(duration + 0.5)  # Extra time for processing

    async def wait_for_response(self, timeout: float = 30.0):
        """Wait for bot response to complete.

        Waits for 2nd bot-stopped-speaking event (greeting + response).
        """
        print(f"Waiting up to {timeout}s for response...")
        start = time.time()
        last_event_count = 0

        # Wait for response audio
        while time.time() - start < timeout:
            await asyncio.sleep(0.1)  # Check frequently

            # Count bot-stopped-speaking events
            stopped_count = sum(
                1 for e in self.events
                if e["event"].get("type") == "bot-stopped-speaking"
            )

            # Log progress
            if len(self.events) > last_event_count:
                for e in self.events[last_event_count:]:
                    if e["event"].get("type") == "bot-started-speaking":
                        print("Bot started speaking")
                    elif e["event"].get("type") == "bot-stopped-speaking":
                        print(f"Bot stopped speaking ({stopped_count} total)")
                last_event_count = len(self.events)

            # Done when we've seen 2 bot-stopped-speaking (greeting + response)
            if stopped_count >= 2:
                print("Bot response complete")
                return True

        print("Timeout waiting for response")
        return False

    async def close(self):
        """Close connection and save results."""
        if self.recorder:
            await self.recorder.stop()

        if self.pc:
            await self.pc.close()

        # Save events log
        events_file = self.output_dir / "events.json"
        with open(events_file, "w") as f:
            json.dump(self.events, f, indent=2)
        print(f"Events saved to: {events_file}")

        # Print summary
        if self.first_audio_time and self.start_time:
            ttfb = (self.first_audio_time - self.start_time) * 1000
            print(f"Time to first audio: {ttfb:.0f}ms")

    def get_metrics(self) -> dict:
        """Get collected metrics."""
        return {
            "events": self.events,
            "first_audio_ms": (
                (self.first_audio_time - self.start_time) * 1000
                if self.first_audio_time and self.start_time
                else None
            ),
        }


async def synthesize_audio(text: str, tts_url: str = "http://localhost:8001") -> str:
    """Synthesize audio from text using Magpie TTS (OpenAI-compatible API)."""
    output_path = f"/tmp/test_audio_{int(time.time())}.pcm"

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

    print(f"Synthesized audio: {output_path} ({len(audio_data)} bytes)")
    return output_path


async def main():
    parser = argparse.ArgumentParser(description="Voice agent test client")
    parser.add_argument("--audio-file", help="Path to audio file to send")
    parser.add_argument("--text", help="Text to synthesize and send")
    parser.add_argument("--server-url", default="http://localhost:7860", help="Bot server URL")
    parser.add_argument("--tts-url", default="http://localhost:8001", help="TTS server URL")
    parser.add_argument("--output-dir", default="/tmp/voice_agent_test", help="Output directory")
    parser.add_argument("--timeout", type=float, default=30.0, help="Response timeout")
    args = parser.parse_args()

    # Get audio file
    audio_path = args.audio_file
    if args.text and not audio_path:
        audio_path = await synthesize_audio(args.text, args.tts_url)
    elif not audio_path:
        print("Error: Provide --audio-file or --text")
        return

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return

    # Run test
    client = VoiceAgentTestClient(
        server_url=args.server_url,
        output_dir=args.output_dir,
    )

    try:
        if not await client.connect(audio_path):
            return

        await client.wait_for_audio_sent()
        await client.wait_for_response(timeout=args.timeout)

        metrics = client.get_metrics()
        print(f"\nResults:")
        print(f"  Events: {len(metrics['events'])}")
        if metrics["first_audio_ms"]:
            print(f"  Time to first audio: {metrics['first_audio_ms']:.0f}ms")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
