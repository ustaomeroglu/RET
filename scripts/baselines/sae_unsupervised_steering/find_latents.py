#!/usr/bin/env python3
"""Quick keyword search over the SAE latent-names file.

Examples
--------
    python scripts/sae_steering/find_latents.py "derivative" "differentiate"
    python scripts/sae_steering/find_latents.py --all "modulo" "remainder"
    python scripts/sae_steering/find_latents.py --case-sensitive "Euler"
"""
from __future__ import annotations

import argparse
import sys

from evaluation.baselines.sae_unsupervised_steering.latent_search import (
    load_sae_latent_names,
    pretty_print_latent,
    search_latents_by_keyword,
)


DEFAULT_NAMES_FILE = (
    "outputs/GptOss20B/numina_math/baselines/sae_l11/"
    "interp_score/sae_latent_names.json"
)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Search SAE latent names by keyword(s).",
    )
    parser.add_argument(
        "keywords", nargs="+",
        help="Substring(s) to search for in the top-activating tokens.",
    )
    parser.add_argument(
        "--names-file", default=DEFAULT_NAMES_FILE,
        help=f"Path to sae_latent_names.json (default: {DEFAULT_NAMES_FILE})",
    )
    parser.add_argument(
        "--all", dest="require_all", action="store_true",
        help="Require ALL keywords to match (default: any).",
    )
    parser.add_argument(
        "--case-sensitive", action="store_true",
        help="Case-sensitive match (default: case-insensitive).",
    )
    parser.add_argument(
        "--max", type=int, default=50,
        help="Max number of results to print (default 50).",
    )
    args = parser.parse_args(argv)

    names = load_sae_latent_names(args.names_file)
    matches = search_latents_by_keyword(
        names,
        args.keywords,
        case_sensitive=args.case_sensitive,
        require_all=args.require_all,
        max_results=args.max,
    )

    if not matches:
        print(f"No latents matched {args.keywords!r}", file=sys.stderr)
        sys.exit(1)

    print(f"# {len(matches)} match(es) for {args.keywords!r} "
          f"(require_all={args.require_all}, case_sensitive={args.case_sensitive})")
    for lid, toks in matches:
        print(pretty_print_latent(lid, toks))


if __name__ == "__main__":
    main()
