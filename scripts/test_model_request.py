"""Send a test chat-completions request to a configured SwissAI model.

Usage:
    uv run python scripts/test_model_request.py
    uv run python scripts/test_model_request.py --prompt "Explain RSA in one paragraph."
    uv run python scripts/test_model_request.py --model jminder/jZ7aJGjetjNN
"""

import argparse
import json
import os
import sys
from pathlib import Path

import dotenv
import openai

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import load_config

DEFAULT_MODEL = "jkminder/lAiFTaihSJ"
DEFAULT_PROMPT = (
    "Reply with exactly one short sentence confirming the request reached you."
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--system",
        default="You are a concise assistant. Answer the user's request directly.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode (separate_reasoning + enable_thinking)",
    )
    return parser.parse_args()


def main() -> None:
    """Send one request and print the response."""
    args = parse_args()
    dotenv.load_dotenv()

    api_key = os.environ.get("SWISS_AI_API_KEY")
    assert api_key, "SWISS_AI_API_KEY not set in environment"

    cfg = load_config()
    client = openai.OpenAI(api_key=api_key, base_url=cfg.phase2.endpoint)

    extra_body = None
    if args.thinking:
        extra_body = {
            "separate_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": True},
        }

    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        extra_body=extra_body,
    )
    assert response.choices, f"No choices returned for model={args.model}"

    message = response.choices[0].message
    content = message.content
    assert content is not None, f"No content returned for model={args.model}"

    reasoning = getattr(message, "reasoning_content", None)
    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None) or {}
    if isinstance(details, dict):
        reasoning_tokens = details.get("reasoning_tokens", 0) or 0
    else:
        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

    print(f"endpoint: {cfg.phase2.endpoint}")
    print(f"model: {args.model}")
    print(f"input_tokens: {getattr(usage, 'prompt_tokens', 0) or 0}")
    print(f"output_tokens: {getattr(usage, 'completion_tokens', 0) or 0}")
    print(
        f"reasoning_tokens: {getattr(usage, 'reasoning_tokens', 0) or reasoning_tokens}"
    )
    print("\ncontent:\n")
    print(content.strip())
    if reasoning is not None:
        print("\nreasoning:\n")
        if isinstance(reasoning, str):
            print(reasoning.strip())
        else:
            print(json.dumps(reasoning, indent=2))


if __name__ == "__main__":
    main()
