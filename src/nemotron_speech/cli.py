"""CLI for Nemotron-Speech-ASR."""

import argparse
import sys


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Nemotron-Speech-ASR: Streaming STT service for Pipecat"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/nemotron_speech_asr.nemo",
        help="Path to the NeMo model checkpoint",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick test to verify model loading",
    )

    args = parser.parse_args()

    if args.test:
        return run_test(args.model)

    parser.print_help()
    return 0


def run_test(model_path: str) -> int:
    """Run a quick test to verify model loading."""
    print(f"Testing model loading: {model_path}")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")

        import nemo.collections.asr as nemo_asr

        print("NeMo ASR module loaded successfully")

        # Try to load the model if it exists
        import os

        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            model = nemo_asr.models.ASRModel.restore_from(model_path)
            print(f"Model type: {type(model).__name__}")
            print("Model loaded successfully!")
        else:
            print(f"Model file not found at {model_path}")
            print("Run the setup script to download the model.")

        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
