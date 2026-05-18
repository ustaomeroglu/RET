#!/usr/bin/env python
"""Compose a paper-ready figure of N stacked trajectories from
`interactive_trajectory.html` files produced by the unsupervised-
steering runs.

- Solid per-group colors (no margin-based shading), matching the style
  of `render_doc_paper_ready.py`.
- Steered/avoid-gated tokens are marked with a 4px red underline
  (auto-extracted from each HTML's JS block).
- A single shared legend is overlaid in the top-right corner of the
  first panel (Figure 1 style). Its group set is the union of groups
  present in any panel, with very-rare groups (singletons by default)
  dropped.

Usage:
    python render_steering_figure.py \\
        --panel <html> "Title 1" \\
        --panel <html> "Title 2" \\
        [--panel <html> "Title 3" ...] \\
        --naming-txt <naming.txt> \\
        --out-base <path-without-extension>

Or the legacy two-panel form:
    --baseline-html <html> --steered-html <html>
    [--baseline-title "..."] [--steered-title "..."]
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent.parent / 'clustering' / 'plotting'))
from render_doc_paper_ready import (  # noqa: E402
    parse_html,
    parse_group_names,
    find_token_index_for_substring,
    humanize,
    load_font,
    render,
)


STEERED_IDS_RE = re.compile(r'var ids = \[([0-9,\s]+)\]')


def parse_steered_ids(html_path: Path) -> set[int]:
    text = html_path.read_text(encoding='utf-8')
    m = STEERED_IDS_RE.search(text)
    if not m:
        return set()
    return {int(x) for x in m.group(1).split(',') if x.strip()}


def merge_group_palette(token_lists) -> dict[int, tuple[int, int, int]]:
    palette: dict[int, tuple[int, int, int]] = {}
    for tokens in token_lists:
        for t in tokens:
            g = t['group']
            if g not in palette:
                palette[g] = t['base_rgb']
    return palette


def group_token_counts(token_lists) -> Counter:
    c: Counter = Counter()
    for tokens in token_lists:
        for t in tokens:
            c[t['group']] += 1
    return c


def make_title_strip(
    text: str,
    width: int,
    font_size: int,
    pad_top: int = 6,
    pad_bottom: int = 2,
    bg=(255, 255, 255),
    fg=(20, 20, 20),
    left_pad: int = 16,
) -> Image.Image:
    font = load_font('DejaVuSans-Bold.ttf', font_size)
    bbox = font.getbbox('Mg')
    text_h = bbox[3] - bbox[1]
    h = text_h + pad_top + pad_bottom
    img = Image.new('RGB', (width, h), bg)
    draw = ImageDraw.Draw(img)
    draw.text((left_pad, pad_top - bbox[1]), text, fill=fg, font=font)
    return img


def stack_with_gaps(images: list[Image.Image], gaps: list[int],
                    bg=(255, 255, 255)) -> Image.Image:
    assert len(gaps) == len(images) - 1
    width = max(im.size[0] for im in images)
    total_h = sum(im.size[1] for im in images) + sum(gaps)
    out = Image.new('RGB', (width, total_h), bg)
    y = 0
    for i, im in enumerate(images):
        out.paste(im, (0, y))
        y += im.size[1]
        if i < len(gaps):
            y += gaps[i]
    return out


def hstack_with_gaps(images: list[Image.Image], gaps: list[int],
                     bg=(255, 255, 255)) -> Image.Image:
    assert len(gaps) == len(images) - 1
    height = max(im.size[1] for im in images)
    total_w = sum(im.size[0] for im in images) + sum(gaps)
    out = Image.new('RGB', (total_w, height), bg)
    x = 0
    for i, im in enumerate(images):
        out.paste(im, (x, 0))
        x += im.size[0]
        if i < len(gaps):
            x += gaps[i]
    return out


def render_legend_row(
    gids: list[int],
    palette: dict[int, tuple[int, int, int]],
    group_names: dict,
    width: int,
    scale: int = 4,
    font_size: int = 13,
    n_rows: int = 1,
    bg=(250, 250, 250),
    border=(215, 215, 215),
    min_inter_cell_gap: int = 4,
    row_gap: int = 6,
) -> Image.Image:
    """Multi-row standalone legend image at fixed `width`. Groups are
    distributed roughly evenly across `n_rows`. Auto-shrinks the font
    until the widest row fits."""
    s = scale
    W = width * s
    pad = 12 * s
    swatch_gap = 6 * s
    radius = 8 * s
    sep_chars = '  '
    row_gap_px = row_gap * s

    n = len(gids)
    n_rows = max(1, n_rows)
    per_row = (n + n_rows - 1) // n_rows  # ceil
    rows_idx: list[list[int]] = []
    for r in range(n_rows):
        chunk = gids[r * per_row: (r + 1) * per_row]
        if chunk:
            rows_idx.append(chunk)

    def lay_out(fs: int):
        font_tag = load_font('DejaVuSans-Bold.ttf', fs * s)
        font_name = load_font('DejaVuSans.ttf', fs * s)
        sw = (fs + 5) * s
        asc, dsc = font_tag.getmetrics()
        text_h = asc + dsc
        ch = max(sw, text_h)
        sep = font_tag.getbbox(sep_chars)[2]
        labs: dict[int, tuple[str, str, int, int, int]] = {}
        for g in gids:
            tag = f'G{g}'
            name = humanize(group_names[g][0]) if g in group_names else ''
            tag_w = font_tag.getbbox(tag)[2]
            name_w = font_name.getbbox(name)[2]
            cell_w = sw + swatch_gap + tag_w + sep + name_w
            labs[g] = (tag, name, tag_w, name_w, cell_w)
        # widest row determines fit
        max_row_used = max(
            sum(labs[g][4] for g in row) for row in rows_idx
        )
        return font_tag, font_name, sw, ch, text_h, sep, labs, max_row_used

    fs = font_size
    inner_w = W - 2 * pad
    while fs >= 8:
        (font_tag, font_name, swatch, cell_h, leg_text_h,
         sep_w, labels, max_row_used) = lay_out(fs)
        biggest_row_n = max(len(r) for r in rows_idx)
        min_total_gap = max(0, biggest_row_n - 1) * min_inter_cell_gap * s
        if max_row_used + min_total_gap <= inner_w:
            break
        fs -= 1
    print(f'  legend row using font_size={fs}')

    H = (
        len(rows_idx) * cell_h
        + max(0, len(rows_idx) - 1) * row_gap_px
        + 2 * pad
    )
    img = Image.new('RGB', (W, H), bg)
    draw = ImageDraw.Draw(img)
    try:
        draw.rounded_rectangle(
            [0, 0, W - 1, H - 1], radius=radius, fill=bg,
            outline=border, width=max(1, s // 2),
        )
    except AttributeError:
        draw.rectangle(
            [0, 0, W - 1, H - 1], fill=bg, outline=border,
        )

    y = pad
    for row in rows_idx:
        used = sum(labels[g][4] for g in row)
        extra = max(0, inner_w - used)
        col_gap = (extra // (len(row) - 1)) if len(row) > 1 else 0
        x = pad
        for gid in row:
            color = palette.get(gid, (160, 160, 160))
            tag, name, tag_w, _name_w, cell_w = labels[gid]
            sw_y = y + (cell_h - swatch) // 2
            try:
                draw.rounded_rectangle(
                    [x, sw_y, x + swatch, sw_y + swatch],
                    radius=3 * s, fill=color,
                )
            except AttributeError:
                draw.rectangle(
                    [x, sw_y, x + swatch, sw_y + swatch], fill=color,
                )
            cx = x + swatch + swatch_gap
            ty = y + (cell_h - leg_text_h) // 2
            draw.text((cx, ty), tag, fill=(35, 35, 35),
                      font=font_tag)
            if name:
                draw.text((cx + tag_w + sep_w, ty), name,
                          fill=(35, 35, 35), font=font_name)
            x += cell_w + col_gap
        y += cell_h + row_gap_px

    if s > 1:
        img = img.resize((img.size[0] // s, img.size[1] // s), Image.LANCZOS)
    return img


def render_legend_column(
    gids: list[int],
    palette: dict[int, tuple[int, int, int]],
    group_names: dict,
    scale: int = 4,
    font_size: int = 13,
    bg=(250, 250, 250),
    border=(215, 215, 215),
) -> Image.Image:
    """Single-column standalone legend image: each row shows
    [swatch] G{N}  Group Name. Returns post-downscale PIL image."""
    s = scale
    pad = 12 * s
    swatch = (font_size + 5) * s
    swatch_gap = 8 * s
    row_gap = 6 * s
    radius = 8 * s

    font_tag = load_font('DejaVuSans-Bold.ttf', font_size * s)
    font_name = load_font('DejaVuSans.ttf', font_size * s)

    asc, dsc = font_tag.getmetrics()
    leg_text_h = asc + dsc
    cell_h = max(swatch, leg_text_h)
    sep_w = font_tag.getbbox('  ')[2]

    labels: dict[int, tuple[str, str]] = {}
    max_row_w = 0
    for gid in gids:
        tag = f'G{gid}'
        name = humanize(group_names[gid][0]) if gid in group_names else ''
        labels[gid] = (tag, name)
        tag_w = font_tag.getbbox(tag)[2]
        name_w = font_name.getbbox(name)[2]
        row_w = swatch + swatch_gap + tag_w + sep_w + name_w
        max_row_w = max(max_row_w, row_w)

    legend_w = max_row_w + 2 * pad
    n_rows = max(1, len(gids))
    legend_h = n_rows * cell_h + max(0, n_rows - 1) * row_gap + 2 * pad

    img = Image.new('RGB', (legend_w, legend_h), bg)
    draw = ImageDraw.Draw(img)
    try:
        draw.rounded_rectangle(
            [0, 0, legend_w - 1, legend_h - 1],
            radius=radius, fill=bg, outline=border,
            width=max(1, s // 2),
        )
    except AttributeError:
        draw.rectangle(
            [0, 0, legend_w - 1, legend_h - 1],
            fill=bg, outline=border,
        )

    y = pad
    for gid in gids:
        color = palette.get(gid, (160, 160, 160))
        sw_y = y + (cell_h - swatch) // 2
        try:
            draw.rounded_rectangle(
                [pad, sw_y, pad + swatch, sw_y + swatch],
                radius=3 * s, fill=color,
            )
        except AttributeError:
            draw.rectangle(
                [pad, sw_y, pad + swatch, sw_y + swatch], fill=color,
            )
        cx = pad + swatch + swatch_gap
        ty = y + (cell_h - leg_text_h) // 2
        tag, name = labels[gid]
        draw.text((cx, ty), tag, fill=(35, 35, 35), font=font_tag)
        tag_w = font_tag.getbbox(tag)[2]
        if name:
            draw.text((cx + tag_w + sep_w, ty), name,
                      fill=(35, 35, 35), font=font_name)
        y += cell_h + row_gap

    if s > 1:
        img = img.resize((img.size[0] // s, img.size[1] // s), Image.LANCZOS)
    return img


def collect_panel_specs(args) -> list[tuple[Path, str]]:
    """Return ordered list of (html_path, title)."""
    if args.panel:
        return [(Path(p), t) for p, t in args.panel]
    # legacy two-panel form
    if not (args.baseline_html and args.steered_html):
        raise SystemExit(
            'Provide either repeated --panel <html> <title> args, '
            'or both --baseline-html and --steered-html.'
        )
    return [
        (args.baseline_html, args.baseline_title),
        (args.steered_html, args.steered_title),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', nargs=2, action='append',
                    metavar=('HTML', 'TITLE'),
                    help='add a panel; first --panel is the top one. '
                         'Repeat for multiple panels.')
    ap.add_argument('--baseline-html', type=Path)
    ap.add_argument('--steered-html', type=Path)
    ap.add_argument('--baseline-title', default='No steering')
    ap.add_argument('--steered-title', default='Steered')
    ap.add_argument('--naming-txt', type=Path, required=True)
    ap.add_argument('--out-base', type=Path, required=True)
    ap.add_argument('--width', type=int, default=2200)
    ap.add_argument('--scale', type=int, default=4)
    ap.add_argument('--font-size', type=int, default=18)
    ap.add_argument('--legend-font-size', type=int, default=13)
    ap.add_argument('--title-font-size', type=int, default=18)
    ap.add_argument('--legend-rows', type=int, default=3)
    ap.add_argument('--title-panel-gap', type=int, default=2)
    ap.add_argument('--inter-panel-gap', type=int, default=10)
    ap.add_argument('--legend-min-count', type=int, default=2,
                    help='groups with fewer tokens than this in the '
                         'union of trajectories are dropped from the '
                         'legend (default: 2 — drops singletons)')
    ap.add_argument('--head-marker', default='',
                    help='if non-empty, trim each panel to start at '
                         'the first occurrence of this substring and '
                         'prepend "... " to indicate truncation')
    ap.add_argument('--layout', choices=['vertical', 'horizontal'],
                    default='vertical',
                    help='vertical: stacked panels. '
                         'horizontal: panels side-by-side. '
                         'In both cases the legend can be placed '
                         'either in the first panel\'s top-right '
                         'corner or as its own right column '
                         '(see --legend-position).')
    ap.add_argument('--legend-position',
                    choices=['corner', 'right', 'top'],
                    default='corner',
                    help='corner: overlay on first panel\'s '
                         'top-right (Figure 1 style). '
                         'right: separate single-column legend on '
                         'the right of the figure. '
                         'top: single-row legend strip across the '
                         'top of the figure.')
    ap.add_argument('--legend-width', type=int, default=400,
                    help='legend column width (px) for horizontal layout')
    ap.add_argument('--horizontal-gap', type=int, default=20,
                    help='horizontal gap between panels and legend (px)')
    ap.add_argument('--dpi-scale', type=float, default=1.0,
                    help='multiplies every output dimension and font '
                         'size while keeping the visual layout '
                         'identical. e.g. 2.0 = double-resolution '
                         'raster.')
    args = ap.parse_args()

    panels_in = collect_panel_specs(args)
    print(f'Panels: {len(panels_in)}')
    for p, t in panels_in:
        print(f'  {t!r}: {p}')

    ds = args.dpi_scale
    if ds != 1.0:
        print(f'dpi-scale = {ds}: multiplying all dimensions')
    args.width = int(args.width * ds)
    args.legend_width = int(args.legend_width * ds)
    args.font_size = max(1, int(round(args.font_size * ds)))
    args.legend_font_size = max(1, int(round(args.legend_font_size * ds)))
    args.title_font_size = max(1, int(round(args.title_font_size * ds)))
    args.title_panel_gap = max(0, int(round(args.title_panel_gap * ds)))
    args.inter_panel_gap = max(0, int(round(args.inter_panel_gap * ds)))
    args.horizontal_gap = max(0, int(round(args.horizontal_gap * ds)))

    panel_tokens = [parse_html(p) for p, _ in panels_in]
    panel_steered = [parse_steered_ids(p) for p, _ in panels_in]

    if args.head_marker:
        trimmed_tokens, trimmed_steered, panel_head_ellipsis = [], [], []
        for (p, t), tokens, sid in zip(
                panels_in, panel_tokens, panel_steered):
            idx = find_token_index_for_substring(tokens, args.head_marker)
            if idx <= 0:
                if idx == -1:
                    print(f'Warning: head marker '
                          f'{args.head_marker!r} not found in '
                          f'panel {t!r}; rendering full trace')
                trimmed_tokens.append(tokens)
                trimmed_steered.append(sid)
                panel_head_ellipsis.append(False)
            else:
                trimmed_tokens.append(tokens[idx:])
                trimmed_steered.append({i - idx for i in sid if i >= idx})
                panel_head_ellipsis.append(True)
                print(f'Panel {t!r}: trimmed first {idx} tokens '
                      f'before {args.head_marker!r}')
        panel_tokens = trimmed_tokens
        panel_steered = trimmed_steered
    else:
        panel_head_ellipsis = [False] * len(panel_tokens)

    for (p, t), tok, sid in zip(panels_in, panel_tokens, panel_steered):
        rng = (
            f'tok-{min(sid)}..tok-{max(sid)} ({len(sid)})'
            if sid else 'none'
        )
        print(f'  {t!r}: {len(tok)} tokens, steered={rng}')

    group_names = parse_group_names(args.naming_txt)
    palette = merge_group_palette(panel_tokens)
    counts = group_token_counts(panel_tokens)
    legend_gids = sorted(g for g, n in counts.items()
                         if n >= args.legend_min_count)
    dropped = sorted(g for g, n in counts.items()
                     if n < args.legend_min_count)
    print(f'Legend groups: {legend_gids}')
    if dropped:
        print(f'Dropped (count < {args.legend_min_count}): '
              + ', '.join(f'G{g}({counts[g]})' for g in dropped))

    n_cols = max(1, math.ceil(len(legend_gids) / args.legend_rows))
    print(f'Legend layout: {args.legend_rows} rows x {n_cols} cols')

    legend_on_right = (args.legend_position == 'right')
    legend_on_top = (args.legend_position == 'top')
    legend_in_corner = (args.legend_position == 'corner')
    n = len(panels_in)
    gap_h = args.horizontal_gap

    if args.layout == 'horizontal':
        if legend_on_right:
            panel_w = (args.width - args.legend_width
                       - n * gap_h) // n
        else:
            panel_w = (args.width - max(0, n - 1) * gap_h) // n
        print(f'Horizontal layout: {n} panels @ {panel_w}px wide '
              f'each, legend on {args.legend_position}')
    else:
        if legend_on_right:
            panel_w = args.width - args.legend_width - gap_h
        else:
            panel_w = args.width
        print(f'Vertical layout: panels @ {panel_w}px wide, '
              f'legend on {args.legend_position}')

    panel_imgs: list[Image.Image] = []
    for i, (tokens, sid, head_ell) in enumerate(zip(
            panel_tokens, panel_steered, panel_head_ellipsis)):
        is_first = (i == 0)
        kwargs = dict(
            width=panel_w, scale=args.scale,
            font_size=args.font_size,
            head_ellipsis=head_ell, tail_ellipsis=False,
            ellipsis_color=(0, 0, 0),
            return_image=True,
        )
        if sid:
            kwargs['steered_indices'] = sid
        if is_first and legend_in_corner:
            kwargs.update(
                no_legend=False,
                legend_groups=legend_gids,
                legend_font_size=args.legend_font_size,
                n_legend_cols=n_cols,
            )
        else:
            kwargs['no_legend'] = True
        panel_imgs.append(render(tokens, group_names, palette,
                                 out_base=None, **kwargs))

    # Combine each (title, panel) pair into a column.
    panel_columns: list[Image.Image] = []
    for (_, title), img in zip(panels_in, panel_imgs):
        title_img = make_title_strip(
            title, img.size[0], args.title_font_size,
        )
        panel_columns.append(stack_with_gaps(
            [title_img, img], gaps=[args.title_panel_gap],
        ))

    if args.layout == 'horizontal':
        if len(panel_columns) > 1:
            panels_block = hstack_with_gaps(
                panel_columns, gaps=[gap_h] * (len(panel_columns) - 1),
            )
        else:
            panels_block = panel_columns[0]
    else:
        # vertical: stack columns top-to-bottom with inter-panel gap.
        seq: list[Image.Image] = []
        gaps: list[int] = []
        for i, col in enumerate(panel_columns):
            if i > 0:
                gaps.append(args.inter_panel_gap)
            seq.append(col)
        if len(seq) > 1:
            panels_block = stack_with_gaps(seq, gaps=gaps)
        else:
            panels_block = seq[0]

    if legend_on_right:
        legend_img = render_legend_column(
            legend_gids, palette, group_names,
            scale=args.scale, font_size=args.legend_font_size,
        )
        print(f'Legend column: {legend_img.size[0]}x'
              f'{legend_img.size[1]}')
        final = hstack_with_gaps(
            [panels_block, legend_img], gaps=[gap_h],
        )
    elif legend_on_top:
        legend_img = render_legend_row(
            legend_gids, palette, group_names,
            width=panels_block.size[0],
            scale=args.scale, font_size=args.legend_font_size,
            n_rows=args.legend_rows,
        )
        print(f'Legend row: {legend_img.size[0]}x'
              f'{legend_img.size[1]}')
        final = stack_with_gaps(
            [legend_img, panels_block],
            gaps=[args.inter_panel_gap],
        )
    else:
        final = panels_block

    args.out_base.parent.mkdir(parents=True, exist_ok=True)
    png = args.out_base.with_suffix('.png')
    pdf = args.out_base.with_suffix('.pdf')
    final.save(png, 'PNG', optimize=True)
    final.save(pdf, 'PDF', resolution=300.0)
    print(f'Wrote {png} ({final.size[0]}x{final.size[1]}) and {pdf}')


if __name__ == '__main__':
    main()
