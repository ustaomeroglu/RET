#!/usr/bin/env python
"""Render a paper-ready single-figure version of an interactive_doc HTML.

- Truncate text to a focused window (head + tail snipped, shown as '...').
- Solid per-cluster colors (no margin-based shading).
- Inline group badges (e.g. 'G10') at the start of major group runs.
- No external legend.
- High-DPI raster (4x supersampling).

Usage:
    python scripts/clustering/render_doc_paper_ready.py \
        <html_path> <naming_txt> <output_base> \
        [--head "marker"] [--tail "marker"] \
        [--width 2200] [--scale 4] [--min-run 10]
"""
from __future__ import annotations

import argparse
import colorsys
import html as htmllib
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TOKEN_RE = re.compile(
    r'<span class="token"[^>]*?data-cluster="(\d+)"[^>]*?data-group="(\d+)"'
    r'[^>]*?style="background:rgba\((\d+),(\d+),(\d+),([\d.]+)\);color:(#[0-9a-fA-F]{3,6});"'
    r'[^>]*?>(.*?)</span>(<br>)?',
    re.DOTALL,
)

GROUP_RE = re.compile(r'^\s*G(\d+)\s*:\s*(.+?)\s*\|\s*(.+?)\s*$')


def parse_group_names(naming_path: Path | None) -> dict[int, tuple[str, str]]:
    if naming_path is None or not naming_path.exists():
        return {}
    names: dict[int, tuple[str, str]] = {}
    for line in naming_path.read_text(encoding='utf-8').splitlines():
        m = GROUP_RE.match(line)
        if m:
            names[int(m.group(1))] = (m.group(2).strip(), m.group(3).strip())
    return names


def humanize(name: str) -> str:
    return ' '.join(w.capitalize() for w in name.replace('_', ' ').split())


def parse_html(path: Path):
    text = path.read_text(encoding='utf-8')
    tokens = []
    for m in TOKEN_RE.finditer(text):
        cid = int(m.group(1))
        gid = int(m.group(2))
        r, g, b = int(m.group(3)), int(m.group(4)), int(m.group(5))
        raw = m.group(8)
        has_br = m.group(9) is not None
        raw_no_tags = re.sub(r'<br\s*/?>', '', raw)
        txt = htmllib.unescape(raw_no_tags).replace('↵', '')
        tokens.append({
            'cluster': cid, 'group': gid,
            'base_rgb': (r, g, b),
            'text': txt, 'br': has_br,
        })
    return tokens


def find_token_index_for_substring(tokens, substring: str, after: int = 0) -> int:
    """Return the smallest token index >= `after` whose text starts the
    substring's containing match, scanning forward."""
    flat = ''
    char_to_tok: list[int] = []
    for i, t in enumerate(tokens):
        if i < after:
            flat += '\x00' * len(t['text'])
        else:
            flat += t['text']
        char_to_tok.extend([i] * len(t['text']))
    pos = flat.find(substring)
    if pos == -1:
        return -1
    return char_to_tok[pos]


def find_token_index_end(tokens, substring: str, after: int = 0) -> int:
    flat = ''
    char_to_tok: list[int] = []
    for i, t in enumerate(tokens):
        if i < after:
            flat += '\x00' * len(t['text'])
        else:
            flat += t['text']
        char_to_tok.extend([i] * len(t['text']))
    pos = flat.find(substring)
    if pos == -1:
        return -1
    end_char = pos + len(substring) - 1
    return char_to_tok[end_char]


def group_base_rgb_from_tokens(tokens) -> dict[int, tuple[int, int, int]]:
    out: dict[int, tuple[int, int, int]] = {}
    for t in tokens:
        g = t['group']
        if g not in out:
            out[g] = t['base_rgb']
    return out


_GOLDEN_RATIO_CONJ = 0.6180339887498949


def golden_ratio_group_colors(tokens) -> dict[int, tuple[int, int, int]]:
    """Assign each group a stable color in encounter order, stepping the
    hue by the golden ratio so neighbouring groups are maximally separated
    on the color wheel. Lightness/saturation also cycle slowly so groups
    that land near the same hue still differ in brightness."""
    palette: dict[int, tuple[int, int, int]] = {}
    for t in tokens:
        g = int(t['group'])
        if g in palette:
            continue
        i = len(palette)
        hue = (i * _GOLDEN_RATIO_CONJ) % 1.0
        sat = 0.62 + 0.22 * ((i * 0.27) % 1.0)
        light = 0.40 + 0.18 * ((i * 0.41) % 1.0)
        r, gg, b = colorsys.hls_to_rgb(hue, light, sat)
        palette[g] = (int(r * 255), int(gg * 255), int(b * 255))
    return palette


def cluster_color_map(tokens, group_base_rgb) -> dict[int, tuple[int, int, int]]:
    """Solid color per cluster: same hue as the group, lightness varies
    across the clusters that belong to that group (so adjacent ones are
    visually distinguishable)."""
    group_clusters: dict[int, list[int]] = {}
    for t in tokens:
        g = t['group']
        c = t['cluster']
        group_clusters.setdefault(g, [])
        if c not in group_clusters[g]:
            group_clusters[g].append(c)

    cluster_rgb: dict[int, tuple[int, int, int]] = {}
    for g, clusters in group_clusters.items():
        base = group_base_rgb.get(g, (160, 160, 160))
        h, _l, s = colorsys.rgb_to_hls(base[0] / 255, base[1] / 255, base[2] / 255)
        # bump saturation a bit for paper print
        s_use = min(1.0, max(0.55, s))
        n = len(clusters)
        for i, c in enumerate(sorted(clusters)):
            if n == 1:
                lightness = 0.50
            else:
                t01 = i / (n - 1)
                lightness = 0.32 + 0.42 * t01  # 0.32..0.74
            r, gg, b = colorsys.hls_to_rgb(h, lightness, s_use)
            cluster_rgb[c] = (int(r * 255), int(gg * 255), int(b * 255))
    return cluster_rgb


def text_color_for(bg_rgb) -> tuple[int, int, int]:
    r, g, b = bg_rgb
    luma = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return (15, 15, 15) if luma > 0.58 else (255, 255, 255)


def find_group_runs(tokens, min_visible: int = 10):
    """Return list of (start_idx, end_idx_excl, group_id) where the run has
    at least `min_visible` tokens with non-whitespace text."""
    if not tokens:
        return []
    out = []
    cur_g = tokens[0]['group']
    cur_start = 0

    def visible_count(s: int, e: int) -> int:
        return sum(1 for t in tokens[s:e] if t['text'].strip())

    for i in range(1, len(tokens)):
        if tokens[i]['group'] != cur_g:
            if visible_count(cur_start, i) >= min_visible:
                out.append((cur_start, i, cur_g))
            cur_g = tokens[i]['group']
            cur_start = i
    if visible_count(cur_start, len(tokens)) >= min_visible:
        out.append((cur_start, len(tokens), cur_g))
    return out


def macro_group_transitions(tokens, dominant_min: int = 12,
                            region_min: int = 60):
    """Return positions where the *dominant* group changes.

    A group becomes dominant when we see an uninterrupted run of at least
    `dominant_min` visible tokens of that group.  Then a second pass drops
    any transition whose resulting region is shorter than `region_min`
    visible tokens — those are "blips" inside a longer dominant region and
    would clutter the figure.

    Returns a list of (start_token_idx, group_id) for placing a badge at
    the first token of each surviving dominant region.
    """
    if not tokens:
        return []
    runs = []
    cur_g = tokens[0]['group']
    cur_start = 0

    def vis(s, e):
        return sum(1 for t in tokens[s:e] if t['text'].strip())

    for i in range(1, len(tokens)):
        if tokens[i]['group'] != cur_g:
            runs.append((cur_start, i, cur_g, vis(cur_start, i)))
            cur_g = tokens[i]['group']
            cur_start = i
    runs.append((cur_start, len(tokens), cur_g, vis(cur_start, len(tokens))))

    raw = []
    dominant = None
    for s, _e, g, n_vis in runs:
        if n_vis >= dominant_min and g != dominant:
            raw.append((s, g))
            dominant = g

    # Filter blips: a transition's region runs from its start to the next
    # transition's start (or end of tokens). Drop if too short.
    filtered = []
    for i, (start, gid) in enumerate(raw):
        end = raw[i + 1][0] if i + 1 < len(raw) else len(tokens)
        n_vis = vis(start, end)
        if n_vis >= region_min:
            filtered.append((start, gid))

    # Collapse adjacent same-group transitions that survived after blips
    # were dropped (e.g. G10 -> [dropped G7 blip] -> G10 should not double-
    # badge G10).
    deduped = []
    last_g = None
    for start, gid in filtered:
        if gid != last_g:
            deduped.append((start, gid))
            last_g = gid
    return deduped


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def render(
    tokens,
    group_names,
    group_base_rgb,
    out_base: Path | None,
    width: int = 2200,
    scale: int = 4,
    n_legend_cols: int = 2,
    head_ellipsis: bool = True,
    tail_ellipsis: bool = True,
    font_size: int = 15,
    no_legend: bool = False,
    steered_indices: set[int] | None = None,
    return_image: bool = False,
    legend_groups: list[int] | None = None,
    legend_valign: str = 'top',
    external_legend_cutout: tuple[int, int, int] | None = None,
    legend_font_size: int | None = None,
    ellipsis_color: tuple[int, int, int] = (80, 80, 80),
):
    s = scale
    W = width * s
    pad = 30 * s

    font_text = load_font('DejaVuSansMono.ttf', font_size * s)
    leg_fs = legend_font_size if legend_font_size is not None else font_size
    font_leg_tag = load_font('DejaVuSans-Bold.ttf', leg_fs * s)
    font_leg_name = load_font('DejaVuSans.ttf', leg_fs * s)
    font_ellipsis = load_font('DejaVuSans-Bold.ttf', (font_size + 2) * s)

    bbox = font_text.getbbox('M')
    char_w = bbox[2] - bbox[0]
    asc, dsc = font_text.getmetrics()
    line_h = asc + dsc + 4 * s

    lt_asc, lt_dsc = font_leg_tag.getmetrics()
    leg_text_h = lt_asc + lt_dsc

    # ---- compact legend laid out at top-right ----
    if no_legend or external_legend_cutout is not None:
        gids = []
    elif legend_groups is not None:
        gids = sorted(legend_groups)
    else:
        gids = sorted(set(t['group'] for t in tokens))
    swatch = (leg_fs + 5) * s
    swatch_gap = 8 * s
    leg_pad = 12 * s
    col_gap = 20 * s
    row_gap = 6 * s

    if gids:
        n_cols = max(1, min(n_legend_cols, len(gids)))
        n_rows = (len(gids) + n_cols - 1) // n_cols
        # Per-cell width: longest entry within a column wins.
        col_widths = [0] * n_cols
        labels: dict[int, tuple[str, str]] = {}
        for k, gid in enumerate(gids):
            tag = f'G{gid}'
            name = humanize(group_names[gid][0]) if gid in group_names else ''
            labels[gid] = (tag, name)
            tag_w = font_leg_tag.getbbox(tag)[2]
            sep_w = font_leg_tag.getbbox('  ')[2]
            name_w = font_leg_name.getbbox(name)[2]
            cell_w_k = swatch + swatch_gap + tag_w + sep_w + name_w
            col = k % n_cols
            col_widths[col] = max(col_widths[col], cell_w_k)
        cell_h = max(swatch, leg_text_h)
        legend_w = sum(col_widths) + col_gap * (n_cols - 1) + leg_pad * 2
        legend_h = (
            n_rows * cell_h + row_gap * (n_rows - 1) + leg_pad * 2
        )
    else:
        n_cols = n_rows = 0
        col_widths = []
        labels = {}
        legend_w = legend_h = 0

    leg_x1 = W - pad
    leg_y0 = pad
    leg_x0 = leg_x1 - legend_w
    leg_y1 = leg_y0 + legend_h

    # Reserve right-margin cutout for the lines the legend covers.
    body_top = pad
    legend_gap = 12 * s

    # ---- build item stream ----
    items = []  # ('tok', token, idx) | ('br',) | ('ell', text)
    if head_ellipsis:
        items.append(('ell', '... '))
    for idx, t in enumerate(tokens):
        items.append(('tok', t, idx))
        if t['br']:
            items.append(('br',))
    if tail_ellipsis:
        items.append(('ell', ' ...'))

    def layout_with_legend(ly0: int, ly1: int, lx_left: int = leg_x0,
                           cutout_active: bool | None = None):
        if cutout_active is None:
            cutout_active = bool(gids)

        def line_max_w(line_top_y: int) -> int:
            if (cutout_active and line_top_y < ly1
                    and line_top_y + line_h > ly0):
                return (lx_left - legend_gap) - pad
            return W - pad * 2

        out_lines: list = []
        cur_line: list = []
        cur_x = 0

        def cur_top_y():
            return body_top + len(out_lines) * line_h

        def cur_max_w():
            return line_max_w(cur_top_y())

        def push_line():
            nonlocal cur_line, cur_x
            out_lines.append(cur_line)
            cur_line = []
            cur_x = 0

        for item in items:
            kind = item[0]
            if kind == 'br':
                push_line()
            elif kind == 'ell':
                text = item[1]
                tw = font_ellipsis.getbbox(text)[2]
                if cur_x + tw > cur_max_w() and cur_line:
                    push_line()
                cur_line.append(('ell', text, cur_x, tw))
                cur_x += tw
            else:
                t = item[1]
                t_idx = item[2]
                txt = t['text']
                if not txt:
                    continue
                part = txt
                while part:
                    max_w_now = cur_max_w()
                    avail_chars = (max_w_now - cur_x) // char_w
                    if avail_chars <= 0:
                        push_line()
                        max_w_now = cur_max_w()
                        avail_chars = max_w_now // char_w
                    take = part[: int(avail_chars)]
                    if not take:
                        push_line()
                        continue
                    w_px = char_w * len(take)
                    cur_line.append(
                        ('tok', t, t_idx, take, cur_x, w_px)
                    )
                    cur_x += w_px
                    part = part[len(take):]

        if cur_line:
            push_line()

        while out_lines and not out_lines[0]:
            out_lines.pop(0)
        while out_lines and not out_lines[-1]:
            out_lines.pop()
        return out_lines, len(out_lines) * line_h

    if external_legend_cutout is not None:
        ext_y0, ext_y1, ext_xl = (
            int(v * s) for v in external_legend_cutout
        )
        lines, text_h = layout_with_legend(
            ext_y0, ext_y1, ext_xl, cutout_active=True,
        )
        total_h = body_top + text_h + pad
    else:
        lines, text_h = layout_with_legend(leg_y0, leg_y1)

        if gids and legend_valign == 'center':
            body_h_pass1 = max(text_h, legend_h)
            leg_y0 = body_top + max(0, (body_h_pass1 - legend_h) // 2)
            leg_y1 = leg_y0 + legend_h
            lines, text_h = layout_with_legend(leg_y0, leg_y1)

        total_h = body_top + max(text_h, legend_h) + pad

    img = Image.new('RGB', (W, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # ---- draw text body ----
    sep_w = max(1, s // 2)
    bar_h = 4 * s
    red_bar = (224, 0, 0)
    bar_outline_h = max(1, s // 2)
    bar_outline = (25, 25, 25)
    y = body_top
    for line in lines:
        prev_cluster = None
        for elem in line:
            kind = elem[0]
            if kind == 'tok':
                _, t, t_idx, take, x, w = elem
                fill_rgb = group_base_rgb.get(t['group'], (200, 200, 200))
                fg = text_color_for(fill_rgb)
                x0 = pad + x
                x1 = x0 + w
                draw.rectangle([x0, y, x1, y + line_h], fill=fill_rgb)
                if prev_cluster is not None and prev_cluster != t['cluster']:
                    draw.rectangle(
                        [x0, y, x0 + sep_w, y + line_h], fill=(255, 255, 255)
                    )
                draw.text((x0, y + 2 * s), take, fill=fg, font=font_text)
                if steered_indices is not None and t_idx in steered_indices:
                    draw.rectangle(
                        [x0, y + line_h - bar_h, x1, y + line_h],
                        fill=red_bar,
                    )
                    draw.rectangle(
                        [x0, y + line_h - bar_h,
                         x1, y + line_h - bar_h + bar_outline_h],
                        fill=bar_outline,
                    )
                prev_cluster = t['cluster']
                continue
            prev_cluster = None
            if kind == 'ell':
                _, text, x, _w = elem
                tx = pad + x
                ty = y + (line_h - leg_text_h) // 2
                draw.text(
                    (tx, ty), text, fill=ellipsis_color,
                    font=font_ellipsis,
                )
        y += line_h

    # ---- draw legend card on top of body in top-right corner ----
    if gids:
        try:
            draw.rounded_rectangle(
                [leg_x0, leg_y0, leg_x1, leg_y1],
                radius=8 * s, fill=(250, 250, 250),
                outline=(215, 215, 215), width=max(1, s // 2),
            )
        except AttributeError:
            draw.rectangle(
                [leg_x0, leg_y0, leg_x1, leg_y1],
                fill=(250, 250, 250), outline=(215, 215, 215),
            )

        col_x_offsets = [0]
        for cw in col_widths[:-1]:
            col_x_offsets.append(col_x_offsets[-1] + cw + col_gap)

        for k, gid in enumerate(gids):
            r = k // n_cols
            c = k % n_cols
            cell_x = leg_x0 + leg_pad + col_x_offsets[c]
            cell_y = leg_y0 + leg_pad + r * (cell_h + row_gap)
            sw_color = group_base_rgb.get(gid, (160, 160, 160))
            sy = cell_y + (cell_h - swatch) // 2
            try:
                draw.rounded_rectangle(
                    [cell_x, sy, cell_x + swatch, sy + swatch],
                    radius=3 * s, fill=sw_color,
                )
            except AttributeError:
                draw.rectangle(
                    [cell_x, sy, cell_x + swatch, sy + swatch],
                    fill=sw_color,
                )
            tx = cell_x + swatch + swatch_gap
            ty = cell_y + (cell_h - leg_text_h) // 2
            tag, name = labels[gid]
            draw.text((tx, ty), tag, fill=(35, 35, 35), font=font_leg_tag)
            tag_w = font_leg_tag.getbbox(tag)[2]
            sep_px = font_leg_tag.getbbox('  ')[2]
            draw.text(
                (tx + tag_w + sep_px, ty), name, fill=(35, 35, 35),
                font=font_leg_name,
            )

    if s > 1:
        img = img.resize(
            (img.size[0] // s, img.size[1] // s), Image.LANCZOS
        )

    if return_image:
        return img

    out_base.parent.mkdir(parents=True, exist_ok=True)
    png = out_base.with_suffix('.png')
    pdf = out_base.with_suffix('.pdf')
    img.save(png, 'PNG', optimize=True)
    img.save(pdf, 'PDF', resolution=300.0)
    return png, pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('html_path', type=Path)
    ap.add_argument('naming_txt', type=Path,
                    help='cluster_naming/naming.txt; pass any non-existent '
                         'path to fall back to generic "Group N" labels')
    ap.add_argument('out_base', type=Path)
    ap.add_argument('--head', default='You are a helpful assistant')
    ap.add_argument('--tail', default='one polynomial')
    ap.add_argument('--width', type=int, default=2200)
    ap.add_argument('--scale', type=int, default=4)
    ap.add_argument('--n-legend-cols', type=int, default=2)
    ap.add_argument('--font-size', type=int, default=15)
    ap.add_argument('--full-trace', action='store_true',
                    help='render all tokens, ignore --head/--tail markers '
                         'and skip leading/trailing ellipsis')
    ap.add_argument('--distinct-colors', action='store_true',
                    help='override HTML colors and assign each group a '
                         'golden-ratio-spaced hue in encounter order so '
                         'adjacent groups always look distinct')
    ap.add_argument('--no-legend', action='store_true',
                    help='omit the floating legend in the top-right corner')
    args = ap.parse_args()

    tokens = parse_html(args.html_path)
    print(f'Parsed {len(tokens)} tokens.')

    if args.full_trace:
        window = tokens
        head_ellipsis = tail_ellipsis = False
        print(f'Rendering full trace ({len(tokens)} tokens)')
    else:
        head_idx = find_token_index_for_substring(tokens, args.head)
        tail_idx = find_token_index_end(tokens, args.tail)
        if head_idx == -1 or tail_idx == -1:
            raise SystemExit(
                f'Could not find truncation markers '
                f'(head={head_idx}, tail={tail_idx})'
            )
        print(f'Head token #{head_idx}, tail token #{tail_idx} '
              f'({tail_idx - head_idx + 1} kept)')
        window = tokens[head_idx: tail_idx + 1]
        head_ellipsis = tail_ellipsis = True

    group_names = parse_group_names(args.naming_txt)
    if args.distinct_colors:
        gbr = golden_ratio_group_colors(window)
    else:
        gbr = group_base_rgb_from_tokens(window)

    png, pdf = render(
        window, group_names, gbr, args.out_base,
        width=args.width, scale=args.scale,
        n_legend_cols=args.n_legend_cols,
        font_size=args.font_size,
        head_ellipsis=head_ellipsis, tail_ellipsis=tail_ellipsis,
        no_legend=args.no_legend,
    )
    print(f'Wrote {png} and {pdf}')


if __name__ == '__main__':
    main()
