#!/usr/bin/env python3
"""Run 20-turn voice agent test with persistent WebRTC connection.

This test maintains a single WebRTC connection across all 20 turns,
enabling proper LLM KV cache reuse between turns.

Usage:
    uv run scripts/run_20_turn_test.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from voice_agent_test_client import MultiTurnVoiceAgentClient, TurnMetrics

TEST_UTTERANCES = [
    # Short inputs (1 sentence)
    "Hello there.",
    "What time is it?",
    "Tell me a joke.",
    "How are you?",
    "What is two plus two?",
    "Goodbye.",
    "Thanks very much!",
    "Can you help me?",
    "What do you think?",
    "That sounds good.",
    # Long inputs (2-3 sentences)
    "I have been thinking about taking a vacation this summer. What are some good destinations you would recommend for a family with young children?",
    "Can you explain how machine learning works? I am particularly interested in understanding the difference between supervised and unsupervised learning approaches.",
    "I just finished reading a fascinating book about the history of computing. It discussed how early computers filled entire rooms. Tell me more about the evolution of computer hardware.",
    "My company is considering migrating our infrastructure to the cloud. What are the main factors we should consider when choosing between different cloud providers?",
    "I have been trying to learn a new programming language. What advice would you give to someone who wants to become proficient in Python for data science applications?",
    "The weather has been quite unpredictable lately. I was planning an outdoor event next weekend. What factors should I consider when making contingency plans?",
    "I recently started a new exercise routine and I am curious about nutrition. Can you explain the relationship between protein intake and muscle recovery after workouts?",
    "My daughter is interested in pursuing a career in technology. What skills and education path would you recommend for someone entering the field today?",
    "I have been learning about renewable energy sources. Can you compare the advantages and disadvantages of solar power versus wind power for residential use?",
    "Our team has been discussing different project management methodologies. What are the key differences between agile and waterfall approaches, and when should each be used?",
]

OUTPUT_DIR = "/tmp/20turn_test"
SERVER_URL = "http://localhost:7860"
TTS_URL = "http://localhost:8001"
INTER_TURN_PAUSE = 1.0  # seconds


def print_summary(results: list[TurnMetrics]):
    """Print summary statistics for the test run."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Per-turn results
    print("\n| Turn | Utterance (first 30 chars) | Time to Response |")
    print("|------|----------------------------|------------------|")
    for m in results:
        text_preview = m.utterance_text[:30] + "..." if len(m.utterance_text) > 30 else m.utterance_text
        ttr = f"{m.time_to_response_ms:.0f}ms" if m.time_to_response_ms else "N/A"
        print(f"| {m.turn_number:4} | {text_preview:28} | {ttr:>16} |")

    # Statistics
    valid_ttrs = [m.time_to_response_ms for m in results if m.time_to_response_ms]
    if valid_ttrs:
        print(f"\nTime to Response Statistics (n={len(valid_ttrs)}):")
        print(f"  Min: {min(valid_ttrs):.0f}ms")
        print(f"  Max: {max(valid_ttrs):.0f}ms")
        print(f"  Avg: {sum(valid_ttrs) / len(valid_ttrs):.0f}ms")

        # Percentiles
        sorted_ttrs = sorted(valid_ttrs)
        p50_idx = len(sorted_ttrs) // 2
        p90_idx = int(len(sorted_ttrs) * 0.9)
        print(f"  P50: {sorted_ttrs[p50_idx]:.0f}ms")
        print(f"  P90: {sorted_ttrs[p90_idx]:.0f}ms")


async def main():
    print(f"Running 20-turn test with persistent connection")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    client = MultiTurnVoiceAgentClient(
        server_url=SERVER_URL,
        tts_url=TTS_URL,
        output_dir=OUTPUT_DIR,
    )

    results: list[TurnMetrics] = []

    try:
        # Connect to bot
        print("\nConnecting to bot...")
        if not await client.connect():
            print("Failed to connect")
            return

        # Wait for greeting to complete
        if not await client.wait_for_greeting():
            print("Failed to receive greeting")
            return

        print("\n" + "-" * 70)

        # Run through all utterances
        for i, utterance in enumerate(TEST_UTTERANCES):
            turn_num = i + 1
            print(f"\n[{turn_num}/{len(TEST_UTTERANCES)}] {utterance[:50]}{'...' if len(utterance) > 50 else ''}")

            # Send turn and wait for response
            metrics = await client.send_turn(utterance)
            results.append(metrics)

            # Print turn metrics
            if metrics.time_to_response_ms:
                print(f"  Time to response: {metrics.time_to_response_ms:.0f}ms")
            if metrics.response_duration_ms:
                print(f"  Response duration: {metrics.response_duration_ms:.0f}ms")

            # Inter-turn pause (except after last turn)
            if turn_num < len(TEST_UTTERANCES):
                await asyncio.sleep(INTER_TURN_PAUSE)

        print("\n" + "-" * 70)
        print("All turns complete!")

    finally:
        await client.close()

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
