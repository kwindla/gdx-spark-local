#!/usr/bin/env python3
"""Run 20-turn voice agent test."""
import subprocess
import time
import os

TEST_UTTERANCES = [
    # Short inputs (1 sentence)
    "Hello there.",
    "What time is it?",
    "Tell me a joke.",
    "How are you?",
    "What is two plus two?",
    "Goodbye.",
    "Thanks!",
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
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Running 20-turn test, output to {OUTPUT_DIR}")
print("=" * 60)

for i, text in enumerate(TEST_UTTERANCES, 1):
    print(f"\n[{i}/20] {text[:50]}{'...' if len(text) > 50 else ''}")
    
    result = subprocess.run(
        [
            "uv", "run", "scripts/voice_agent_test_client.py",
            "--text", text,
            "--output-dir", OUTPUT_DIR,
            "--timeout", "45"
        ],
        capture_output=True,
        text=True,
        timeout=90
    )
    
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-200:] if result.stderr else 'Unknown error'}")
    else:
        # Extract key metrics from output
        for line in result.stdout.split('\n'):
            if 'V2V' in line or 'TTFB' in line or 'complete' in line.lower():
                print(f"  {line.strip()}")
    
    time.sleep(2)  # Brief pause between turns

print("\n" + "=" * 60)
print("Test complete!")
