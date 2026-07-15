"""
Usage:
    python -m orchestrator.cli --task "Write a function that checks if a string is a palindrome, ignoring case and spaces"
"""
from __future__ import annotations

import argparse
import os

from .orchestrator import Orchestrator, format_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent code orchestrator")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--max-revisions", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GROQ_API_KEY before running.")

    orchestrator = Orchestrator(api_key, max_revisions=args.max_revisions)

    print(f"Task: {args.task}\n")
    print("Running Planner -> Coder -> Critic loop...\n")

    result = orchestrator.run(args.task)

    print("=" * 70)
    print("AGENT TRACE")
    print("=" * 70)
    print(format_trace(result.trace))

    print("\n" + "=" * 70)
    if result.success:
        print(f"✅ APPROVED after {result.revisions_used} attempt(s)")
    else:
        print(f"❌ NOT APPROVED after {result.revisions_used} attempt(s)")
    print("=" * 70)
    print("\nFinal code:\n")
    print(result.final_code)


if __name__ == "__main__":
    main()
