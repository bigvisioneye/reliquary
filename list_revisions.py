#!/usr/bin/env python3
"""
List available revisions of a HuggingFace model repo: branches, tags, and
recent commit SHAs. Any of these strings is a valid `revision=` value.

Usage:
    pip install huggingface_hub
    python3 list_revisions.py <repo_id>
    python3 list_revisions.py someorg/reliquary-qwen3.5-4b
    python3 list_revisions.py <repo_id> --token hf_xxx   # if the repo is gated/private

Note: for the Reliquary harness you normally want the revision the VALIDATOR
is currently serving (from GET /state -> checkpoint_revision), not just any
revision listed here. Use this to inspect history, not to pick the live one.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id")
    ap.add_argument("--token", default=None, help="HF token if the repo is gated/private")
    ap.add_argument("--commits", type=int, default=10, help="how many recent commits to show")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Install first: pip install huggingface_hub", file=sys.stderr)
        return 1

    api = HfApi()

    # Branches and tags
    try:
        refs = api.list_repo_refs(args.repo_id, token=args.token)
    except Exception as e:
        print(f"Failed to list refs for {args.repo_id}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"\n=== Repo: {args.repo_id} ===\n")
    print("Branches (use the name as revision=):")
    for b in refs.branches:
        print(f"  {b.name:<20} -> {b.target_commit}")
    if not refs.branches:
        print("  (none)")

    print("\nTags (use the name as revision=):")
    for t in refs.tags:
        print(f"  {t.name:<20} -> {t.target_commit}")
    if not refs.tags:
        print("  (none)")

    # Recent commits (each SHA is a valid revision=)
    try:
        commits = api.list_repo_commits(args.repo_id, token=args.token)
        print(f"\nRecent commits (newest first, use the SHA as revision=):")
        for c in commits[: args.commits]:
            title = (c.title or "").strip().replace("\n", " ")
            print(f"  {c.commit_id[:12]}  {c.created_at:%Y-%m-%d %H:%M}  {title[:60]}")
    except Exception as e:
        print(f"\n(could not list commits: {type(e).__name__}: {e})")

    print("\nTip: 'main' is the tip of the default branch and moves as the "
          "validator pushes. Pin an exact SHA for reproducible offline runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())