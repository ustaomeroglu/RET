#!/usr/bin/env python
"""Render interactive_doc_*.html files as horizontal figures with a
group-legend row on top and the colored token trace below.

Usage:
    python scripts/clustering/render_doc_horizontal.py \
        <html_dir> <naming_txt> <output_dir> [--width 2400]
"""
from __future__ import annotations

import argparse
import html as htmllib
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TOKEN_RE = re.compile(
    r'<span class="token"[^>]*?data-cluster="(\d+)"[^>]*?data-group="(\d+)"'
    r'[^>]*?style="background:rgba\((\d+),(\d+),(\d+),([\d.]+)\);color:(#[0-9a-fA-F]{3,6});"'
    r'[^>]*?>(.*?)</span>(<br>)?',
    re.DOTALL,
)

GROUP_BORDER_RE = re.compile(
    r'<span class="legend-group"\s+data-group="(\d+)"[^>]*?border:\s*2px solid ([^;"]+)'
)

GROUP_RE = re.compile(r'^\s*G(\d+)\s*:\s*(.+?)\s*\|\s*(.+?)\s*$')


def parse_group_names(naming_path: Path) -> dict[int, tuple[str, str]]:
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
        a = float(m.group(6))
        fg_hex = m.group(7)
        raw = m.group(8)
        has_br = m.group(9) is not None
        raw_no_tags = re.sub(r'<br\s*/?>', '', raw)
        txt = htmllib.unescape(raw_no_tags).replace('↵', '')
        tokens.append({
            'cluster': cid, 'group': gid,
            'bg': (r, g, b, a), 'fg': fg_hex,
            'text': txt, 'br': has_br,
        })
    group_colors = {int(gid): col.strip()
                    for gid, col in GROUP_BORDER_RE.findall(text)}
    group_base_rgb: dict[int, tuple[int, int, int]] = {}
    for t in tokens:
        g = t['group']
        if g not in group_base_rgb:
            group_base_rgb[g] = t['bg'][:3]
    return tokens, group_colors, group_base_rgb


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip('#')
    if len(h) == 3:
        h = ''.join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def blend(bg_rgba, white=(255, 255, 255)):
    r, g, b, a = bg_rgba
    return (int(r * a + white[0] * (1 - a)),
            int(g * a + white[1] * (1 - a)),
            int(b * a + white[2] * (1 - a)))


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def wrap_text(text: str, max_w: int, font) -> list[str]:
    out: list[str] = []
    cur = ''
    for w in text.split():
        trial = w if not cur else cur + ' ' + w
        if font.getbbox(trial)[2] <= max_w:
            cur = trial
        else:
            if cur:
                out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def render_doc(tokens, group_colors, group_base_rgb, group_names,
               out_base: Path, width: int = 2400, n_cols: int = 4,
               scale: int = 2):
    s = scale
    W = width * s

    font_text = load_font('DejaVuSansMono.ttf', 13 * s)
    font_tag = load_font('DejaVuSans-Bold.ttf', 15 * s)
    font_label = load_font('DejaVuSans.ttf', 15 * s)
    font_desc = load_font('DejaVuSans.ttf', 12 * s)

    t_bbox = font_text.getbbox('M')
    char_w = t_bbox[2] - t_bbox[0]
    t_asc, t_dsc = font_text.getmetrics()
    line_h = t_asc + t_dsc + 4 * s

    n_asc, n_dsc = font_label.getmetrics()
    name_line_h = n_asc + n_dsc + 4 * s
    d_asc, d_dsc = font_desc.getmetrics()
    desc_line_h = d_asc + d_dsc + 2 * s

    pad = 28 * s
    content_w = W - pad * 2

    # --- Top legend: grid of N cols ---
    ordered_gids = sorted(
        group_names.keys() | group_colors.keys() | group_base_rgb.keys()
    )
    n_rows = (len(ordered_gids) + n_cols - 1) // n_cols
    col_gap = 24 * s
    row_gap = 14 * s
    cell_w = (content_w - col_gap * (n_cols - 1)) // n_cols
    swatch = 20 * s
    swatch_gap = 12 * s
    text_col_w = cell_w - swatch - swatch_gap - 4 * s

    def label_for(gid: int) -> tuple[str, str, str]:
        entry = group_names.get(gid)
        if entry is None:
            return f'G{gid}', f'Group {gid}', ''
        raw_name, desc = entry
        return f'G{gid}', humanize(raw_name), desc

    # Pre-wrap descriptions to compute per-row heights
    cells: list[tuple] = []  # (gid, tag, name, desc_lines)
    for gid in ordered_gids:
        tag, name, desc = label_for(gid)
        desc_lines = wrap_text(desc, text_col_w, font_desc) if desc else []
        cells.append((gid, tag, name, desc_lines))

    def cell_height(desc_lines) -> int:
        h = name_line_h
        if desc_lines:
            h += 4 * s + desc_line_h * len(desc_lines)
        return h

    row_heights: list[int] = []
    for r in range(n_rows):
        in_row = cells[r * n_cols:(r + 1) * n_cols]
        row_heights.append(
            max(cell_height(c[3]) for c in in_row) if in_row else 0
        )
    legend_h = sum(row_heights) + row_gap * (n_rows - 1) + pad * 2

    # --- Token area ---
    text_w = content_w
    max_chars = text_w // char_w
    lines = []
    cur_line = []
    cur_col = 0
    for t in tokens:
        tx = t['text']
        if not tx and not t['br']:
            continue
        part = tx
        while part:
            remaining = max_chars - cur_col
            if remaining <= 0:
                lines.append(cur_line)
                cur_line = []
                cur_col = 0
                remaining = max_chars
            take = part[:remaining]
            cur_line.append((take, blend(t['bg']), hex_to_rgb(t['fg'])))
            cur_col += len(take)
            part = part[remaining:]
        if t['br']:
            lines.append(cur_line)
            cur_line = []
            cur_col = 0
    if cur_line:
        lines.append(cur_line)

    text_h = len(lines) * line_h

    total_h = legend_h + pad + text_h + pad
    img = Image.new('RGB', (W, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Legend background card
    try:
        draw.rounded_rectangle(
            [pad // 2, pad // 2, W - pad // 2, legend_h - pad // 2],
            radius=10 * s, fill=(250, 250, 250), outline=(225, 225, 225),
            width=max(1, s // 2),
        )
    except AttributeError:
        pass

    # Draw legend cells
    y_cursor = pad
    swatch_radius = 4 * s
    for r in range(n_rows):
        row_cells = cells[r * n_cols:(r + 1) * n_cols]
        for c, (gid, tag, name, desc_lines) in enumerate(row_cells):
            cell_x = pad + c * (cell_w + col_gap)
            gcol_rgb = group_base_rgb.get(gid)
            if gcol_rgb is None:
                try:
                    gcol_rgb = hex_to_rgb(group_colors.get(gid, '#888888'))
                except ValueError:
                    gcol_rgb = (136, 136, 136)
            sy = y_cursor + (name_line_h - swatch) // 2
            try:
                draw.rounded_rectangle(
                    [cell_x, sy, cell_x + swatch, sy + swatch],
                    radius=swatch_radius, fill=gcol_rgb,
                )
            except AttributeError:
                draw.rectangle([cell_x, sy, cell_x + swatch, sy + swatch],
                               fill=gcol_rgb)
            tag_x = cell_x + swatch + swatch_gap
            text_y = y_cursor + 2 * s
            draw.text((tag_x, text_y), tag, fill=(60, 60, 60), font=font_tag)
            tag_w = font_tag.getbbox(tag + '  ')[2]
            draw.text((tag_x + tag_w, text_y), name,
                      fill=(25, 25, 25), font=font_label)
            dy = y_cursor + name_line_h + 4 * s
            for dl in desc_lines:
                draw.text((tag_x, dy), dl, fill=(95, 95, 95), font=font_desc)
                dy += desc_line_h
        y_cursor += row_heights[r] + row_gap

    # Token text area
    y = legend_h + pad
    for line in lines:
        x = pad
        for seg_text, bg, fg in line:
            w = char_w * len(seg_text)
            draw.rectangle([x, y, x + w, y + line_h], fill=bg)
            draw.text((x, y + 2 * s), seg_text, fill=fg, font=font_text)
            x += w
        y += line_h

    if s > 1:
        img = img.resize((img.size[0] // s, img.size[1] // s), Image.LANCZOS)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    png = out_base.with_suffix('.png')
    pdf = out_base.with_suffix('.pdf')
    img.save(png, 'PNG', optimize=True)
    img.save(pdf, 'PDF', resolution=200.0)
    return png, pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('html_dir', type=Path)
    ap.add_argument('naming_txt', type=Path)
    ap.add_argument('out_dir', type=Path)
    ap.add_argument('--width', type=int, default=2400)
    ap.add_argument('--n-cols', type=int, default=4)
    args = ap.parse_args()

    group_names = parse_group_names(args.naming_txt)
    print(f'Parsed {len(group_names)} group names')
    html_files = sorted(args.html_dir.glob('interactive_doc_*.html'))
    if not html_files:
        html_files = sorted(args.html_dir.glob('doc_*.html'))
    if not html_files:
        print(f'No (interactive_)doc_*.html found in {args.html_dir}')
        return
    for hp in html_files:
        tokens, gc, gb = parse_html(hp)
        out_name = hp.stem.replace('interactive_doc_', 'doc_')
        out_base = args.out_dir / out_name
        png, pdf = render_doc(
            tokens, gc, gb, group_names, out_base,
            width=args.width, n_cols=args.n_cols,
        )
        html_copy = out_base.with_suffix('.html')
        if hp.resolve() != html_copy.resolve():
            shutil.copy2(hp, html_copy)
        print(f'  {hp.name}: {len(tokens)} tokens -> '
              f'{png.name}, {html_copy.name}')


if __name__ == '__main__':
    main()
