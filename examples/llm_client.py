#!/usr/bin/env python3
"""
Example client for Nemotron-3-Nano LLM inference via vLLM.

Usage:
    python examples/llm_client.py "What is the capital of France?"
    python examples/llm_client.py --stream "Write a haiku about GPUs"
    python examples/llm_client.py --reasoning "What is 15 * 23?"

Requirements:
    pip install openai
"""

import argparse
import sys

from openai import OpenAI


def chat_completion(
    client: OpenAI,
    model: str,
    message: str,
    stream: bool = False,
    enable_thinking: bool = False,
    max_tokens: int = 256,
):
    """Send a chat completion request to the vLLM server."""
    extra_body = {}
    if not enable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    messages = [{"role": "user", "content": message}]

    if stream:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
            extra_body=extra_body if extra_body else None,
        )

        print("Response: ", end="", flush=True)
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        print()  # newline after streaming
        return full_response
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=False,
            extra_body=extra_body if extra_body else None,
        )

        message = response.choices[0].message
        content = message.content

        # Check for reasoning content (when enable_thinking=True)
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            print(f"Reasoning: {reasoning}\n")

        print(f"Response: {content}")
        return content


def main():
    parser = argparse.ArgumentParser(description="Nemotron LLM Client")
    parser.add_argument("prompt", nargs="?", default="Hello, who are you?",
                        help="The prompt to send to the model")
    parser.add_argument("--url", default="http://localhost:8000/v1",
                        help="vLLM server URL (default: http://localhost:8000/v1)")
    parser.add_argument("--model", default="/workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                        help="Model name/path")
    parser.add_argument("--stream", action="store_true",
                        help="Enable streaming output")
    parser.add_argument("--reasoning", action="store_true",
                        help="Enable chain-of-thought reasoning")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    args = parser.parse_args()

    client = OpenAI(base_url=args.url, api_key="not-needed")

    print(f"Server: {args.url}")
    print(f"Model: {args.model}")
    print(f"Reasoning: {'enabled' if args.reasoning else 'disabled'}")
    print(f"Streaming: {'enabled' if args.stream else 'disabled'}")
    print(f"Prompt: {args.prompt}\n")

    try:
        chat_completion(
            client=client,
            model=args.model,
            message=args.prompt,
            stream=args.stream,
            enable_thinking=args.reasoning,
            max_tokens=args.max_tokens,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
