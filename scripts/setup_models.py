"""
One-time helper to pre-download model weights so the Flask app starts
fast and can also be used offline afterwards.

Usage:
    python scripts/setup_models.py            # downloads Swin encoder + T5 decoder
    python scripts/setup_models.py --llm       # also downloads LLaMA-3.1-8B-Instruct
                                                # (needs HF_TOKEN with access approved
                                                #  on huggingface.co for that model)
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="Also download the LLaMA-3.1 chat model")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    token = Config.HF_TOKEN or os.getenv("HF_TOKEN")

    print(f"Downloading Swin encoder: {Config.SWIN_ENCODER_ID}")
    snapshot_download(repo_id=Config.SWIN_ENCODER_ID, token=token)

    print(f"Downloading T5 decoder: {Config.T5_DECODER_ID}")
    snapshot_download(repo_id=Config.T5_DECODER_ID, token=token)

    if args.llm:
        if not token:
            print(
                "No HF_TOKEN set. meta-llama/Meta-Llama-3.1-8B-Instruct is a "
                "gated model - request access on huggingface.co, then set "
                "HF_TOKEN in your .env and re-run this script."
            )
            return
        print(f"Downloading LLM: {Config.LLM_MODEL_ID} (this is large, ~16GB)")
        snapshot_download(repo_id=Config.LLM_MODEL_ID, token=token)

    print("Done.")


if __name__ == "__main__":
    main()
