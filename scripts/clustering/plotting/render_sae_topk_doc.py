#!/usr/bin/env python
"""Render a doc with each token colored by its argmax SAE latent.

Uses the andyrdt/saes-gpt-oss-20b SAE at layer 11. Token text and line
breaks are pulled from an existing interactive_doc_*.html so the figure
aligns visually with the other paper-ready figures.

Usage:
    python scripts/clustering/render_sae_topk_doc.py \\
        <html_path> <hidden_states_dir> <doc_idx> <output_base>
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.baselines.sae import _load_sae_from_state_dict
from render_doc_paper_ready import parse_html, render


GOLDEN_RATIO_CONJ = 0.6180339887498949


def assign_latent_colors(latent_seq) -> dict[int, tuple[int, int, int]]:
    """Assign a stable color to each unique latent ID in encounter order,
    stepping the hue by the golden ratio so that consecutive (different)
    latents are maximally separated on the color wheel.

    Lightness and saturation are also varied with low-frequency offsets so
    that latents that happen to land near the same hue still differ in
    brightness or vividness.
    """
    palette: dict[int, tuple[int, int, int]] = {}
    for lid in latent_seq:
        lid = int(lid)
        if lid in palette:
            continue
        i = len(palette)
        hue = (i * GOLDEN_RATIO_CONJ) % 1.0
        # Cycle lightness/saturation slowly so neighbouring entries differ
        # in more than just hue.
        sat = 0.62 + 0.22 * ((i * 0.27) % 1.0)
        light = 0.40 + 0.18 * ((i * 0.41) % 1.0)
        r, g, b = colorsys.hls_to_rgb(hue, light, sat)
        palette[lid] = (int(r * 255), int(g * 255), int(b * 255))
    return palette


def load_doc_hidden_states(cache_dir: Path, doc_idx: int) -> np.ndarray:
    metadata_path = cache_dir / 'metadata.json'
    import json
    md = json.loads(metadata_path.read_text())
    hidden_dim = md['hidden_dim']
    dtype = np.float16 if md['dtype'] == 'float16' else np.float32
    lengths = np.load(cache_dir / 'lengths.npy')
    offsets = np.load(cache_dir / 'offsets.npy')
    n_toks = int(lengths[doc_idx])
    offset = int(offsets[doc_idx])
    arr = np.memmap(cache_dir / 'hidden_states.bin', dtype=dtype, mode='r')
    flat = arr[offset * hidden_dim:(offset + n_toks) * hidden_dim]
    return np.asarray(flat).reshape(n_toks, hidden_dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('html_path', type=Path)
    ap.add_argument('hidden_states_dir', type=Path,
                    help='e.g. data/numina_math_64k_samples/GptOss20B/layer11/test')
    ap.add_argument('doc_idx', type=int)
    ap.add_argument('out_base', type=Path)
    ap.add_argument('--sae-repo', default='andyrdt/saes-gpt-oss-20b')
    ap.add_argument('--sae-folder', default='resid_post_layer_11/trainer_0')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--font-size', type=int, default=22)
    ap.add_argument('--width', type=int, default=2200)
    ap.add_argument('--scale', type=int, default=4)
    args = ap.parse_args()

    print(f'Parsing tokens from {args.html_path}')
    tokens = parse_html(args.html_path)
    print(f'  {len(tokens)} tokens')

    print(f'Loading doc {args.doc_idx} hidden states from {args.hidden_states_dir}')
    h_np = load_doc_hidden_states(args.hidden_states_dir, args.doc_idx)
    print(f'  shape={h_np.shape}, dtype={h_np.dtype}')

    if h_np.shape[0] != len(tokens):
        print(f'WARNING: token count mismatch (tokens={len(tokens)}, '
              f'hidden_states={h_np.shape[0]})')

    print(f'Loading SAE {args.sae_repo}/{args.sae_folder}')
    ae_path = hf_hub_download(args.sae_repo, f'{args.sae_folder}/ae.pt')
    state = torch.load(ae_path, map_location='cpu', weights_only=False)
    sae = _load_sae_from_state_dict(state, args.device)
    print(f'  d_in={sae.d_in}, num_latents={sae.d_sae}, k={sae.k}')

    h_t = torch.from_numpy(h_np.astype(np.float32)).to(args.device)
    with torch.no_grad():
        acts = sae.encode(h_t)  # (n_toks, num_latents)
        top_latent = acts.argmax(dim=-1).cpu().numpy().astype(np.int64)
    print(f'  unique argmax latents in doc: {len(np.unique(top_latent))}')

    n = min(len(tokens), len(top_latent))
    # Walk the token sequence in order so colors are assigned in encounter
    # order (golden-ratio stepping → adjacent unique latents are far apart
    # on the color wheel).
    encounter_seq = [int(top_latent[i if i < n else n - 1]) for i in range(len(tokens))]
    palette = assign_latent_colors(encounter_seq)

    group_base_rgb: dict[int, tuple[int, int, int]] = {}
    for i, t in enumerate(tokens):
        lid = encounter_seq[i]
        color = palette[lid]
        t['cluster'] = lid
        t['group'] = lid
        t['base_rgb'] = color
        group_base_rgb[lid] = color

    png, pdf = render(
        tokens, {}, group_base_rgb, args.out_base,
        width=args.width, scale=args.scale,
        head_ellipsis=False, tail_ellipsis=False,
        font_size=args.font_size, no_legend=True,
    )
    print(f'Wrote {png} and {pdf}')


if __name__ == '__main__':
    main()
