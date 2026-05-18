#!/usr/bin/env python
"""Render interactive_doc_*.html files to PNG+PDF with a group legend.

Usage:
    python scripts/clustering/render_doc_images.py \
        <html_dir> <naming_txt> <output_dir>
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
            'cluster': cid,
            'group': gid,
            'bg': (r, g, b, a),
            'fg': fg_hex,
            'text': txt,
            'br': has_br,
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


def text_color_for(bg_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = bg_rgb
    luma = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return (15, 15, 15) if luma > 0.58 else (255, 255, 255)


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


def render(tokens, group_colors, group_base_rgb, group_names, out_base: Path,
           scale: int = 2, no_shading: bool = False, font_size: int = 13):
    # Logical sizes (scaled up by `scale` internally for crisp raster / PDF)
    s = scale
    font_text = load_font('DejaVuSansMono.ttf', font_size * s)
    font_tag = load_font('DejaVuSans-Bold.ttf', (font_size + 4) * s)
    font_label = load_font('DejaVuSans.ttf', (font_size + 4) * s)

    font_desc = load_font('DejaVuSans.ttf', (font_size + 1) * s)

    bbox = font_text.getbbox('M')
    char_w = bbox[2] - bbox[0]
    ascent, descent = font_text.getmetrics()
    line_h = ascent + descent + 4 * s
    name_ascent, name_descent = font_label.getmetrics()
    name_line_h = name_ascent + name_descent + 4 * s
    desc_ascent, desc_descent = font_desc.getmetrics()
    desc_line_h = desc_ascent + desc_descent + 2 * s

    pad = 22 * s
    text_w = 1100 * s
    max_chars = text_w // char_w

    # Layout text tokens
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
            if no_shading:
                blended = group_base_rgb.get(t['group'], blend(t['bg']))
            else:
                blended = blend(t['bg'])
            fg = text_color_for(blended) if no_shading else hex_to_rgb(t['fg'])
            cur_line.append((
                take, blended, fg, t['cluster'],
            ))
            cur_col += len(take)
            part = part[remaining:]
        if t['br']:
            lines.append(cur_line)
            cur_line = []
            cur_col = 0
    if cur_line:
        lines.append(cur_line)

    text_h = pad * 2 + len(lines) * line_h

    # Legend dimensions — fixed width gives room for wrapped descriptions
    ordered_gids = sorted(
        group_names.keys() | group_colors.keys() | group_base_rgb.keys()
    )
    swatch = 22 * s
    swatch_gap = 14 * s
    panel_gap = 32 * s

    def label_for(gid: int) -> tuple[str, str, str]:
        entry = group_names.get(gid)
        if entry is None:
            return f'G{gid}', f'Group {gid}', ''
        raw_name, desc = entry
        return f'G{gid}', humanize(raw_name), desc

    # Fixed inner width for text column (name line + wrapped description)
    text_col_w = 560 * s
    legend_w = pad + swatch + swatch_gap + text_col_w + pad

    def wrap_text(text: str, max_w: int, font) -> list[str]:
        words = text.split()
        out: list[str] = []
        line = ''
        for w in words:
            trial = w if not line else line + ' ' + w
            if font.getbbox(trial)[2] <= max_w:
                line = trial
            else:
                if line:
                    out.append(line)
                line = w
        if line:
            out.append(line)
        return out

    # Pre-compute wrapped descriptions per group to size rows
    row_blocks: list[tuple[int, str, str, list[str]]] = []
    row_gap = 14 * s
    for gid in ordered_gids:
        tag, name, desc = label_for(gid)
        desc_lines = wrap_text(desc, text_col_w, font_desc) if desc else []
        row_blocks.append((gid, tag, name, desc_lines))

    def row_height(desc_lines: list[str]) -> int:
        h = name_line_h  # tag + name line
        if desc_lines:
            h += 4 * s + desc_line_h * len(desc_lines)
        return h

    legend_h = pad * 2 + sum(
        row_height(b[3]) for b in row_blocks
    ) + row_gap * (len(row_blocks) - 1)

    total_w = pad + text_w + panel_gap + legend_w + pad
    total_h = max(text_h, legend_h)

    img = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Text panel (flush left, no title, no border)
    sep_w = max(1, s // 2)
    y = pad
    for line in lines:
        x = pad
        prev_cluster = None
        for seg_text, bg, fg, cluster in line:
            w = char_w * len(seg_text)
            draw.rectangle([x, y, x + w, y + line_h], fill=bg)
            if no_shading and prev_cluster is not None and prev_cluster != cluster:
                draw.rectangle(
                    [x, y, x + sep_w, y + line_h], fill=(255, 255, 255)
                )
            draw.text((x, y + 2 * s), seg_text, fill=fg, font=font_text)
            x += w
            prev_cluster = cluster
        y += line_h

    # Subtle vertical separator between text and legend
    sep_x = pad + text_w + panel_gap // 2
    draw.line([(sep_x, pad), (sep_x, total_h - pad)],
              fill=(220, 220, 220), width=max(1, s // 2))

    # Legend panel: soft rounded card
    lx0 = pad + text_w + panel_gap
    ly0 = pad
    lx1 = lx0 + legend_w - pad * 0  # card spans legend_w
    ly1 = ly0 + legend_h
    # Center legend vertically if text is taller
    if text_h > legend_h:
        ly0 = (total_h - legend_h) // 2
        ly1 = ly0 + legend_h
    try:
        draw.rounded_rectangle(
            [lx0, ly0, lx1, ly1], radius=10 * s,
            fill=(250, 250, 250), outline=(225, 225, 225),
            width=max(1, s // 2),
        )
    except AttributeError:
        draw.rectangle([lx0, ly0, lx1, ly1], fill=(250, 250, 250),
                       outline=(225, 225, 225))

    # Legend rows
    row_y = ly0 + pad
    row_x = lx0 + pad
    swatch_radius = 4 * s
    for i, (gid, tag, name, desc_lines) in enumerate(row_blocks):
        gcol_rgb = group_base_rgb.get(gid)
        if gcol_rgb is None:
            try:
                gcol_rgb = hex_to_rgb(group_colors.get(gid, '#888888'))
            except ValueError:
                gcol_rgb = (136, 136, 136)
        # Swatch aligned with the first (name) line
        sy = row_y + (name_line_h - swatch) // 2
        try:
            draw.rounded_rectangle(
                [row_x, sy, row_x + swatch, sy + swatch],
                radius=swatch_radius, fill=gcol_rgb,
            )
        except AttributeError:
            draw.rectangle([row_x, sy, row_x + swatch, sy + swatch],
                           fill=gcol_rgb)
        tag_x = row_x + swatch + swatch_gap
        text_y = row_y + 2 * s
        draw.text((tag_x, text_y), tag, fill=(60, 60, 60), font=font_tag)
        tag_w = font_tag.getbbox(tag + '  ')[2]
        draw.text((tag_x + tag_w, text_y), name,
                  fill=(25, 25, 25), font=font_label)

        dy = row_y + name_line_h + 4 * s
        for dl in desc_lines:
            draw.text((tag_x, dy), dl, fill=(95, 95, 95), font=font_desc)
            dy += desc_line_h

        row_y += row_height(desc_lines)
        if i < len(row_blocks) - 1:
            row_y += row_gap

    # Downscale for final output (smoothing => crisper text)
    if s > 1:
        final_size = (total_w // s, total_h // s)
        img = img.resize(final_size, Image.LANCZOS)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_base.with_suffix('.png')
    pdf_path = out_base.with_suffix('.pdf')
    img.save(png_path, 'PNG', optimize=True)
    img.save(pdf_path, 'PDF', resolution=200.0)
    return png_path, pdf_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('html_dir', type=Path)
    ap.add_argument('naming_txt', type=Path)
    ap.add_argument('out_dir', type=Path)
    ap.add_argument('--no-shading', action='store_true',
                    help='ignore margin alpha and use solid group base colors')
    ap.add_argument('--font-size', type=int, default=13)
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
        tokens, group_colors, group_base_rgb = parse_html(hp)
        out_name = hp.stem.replace('interactive_doc_', 'doc_')
        out_base = args.out_dir / out_name
        png, pdf = render(tokens, group_colors, group_base_rgb,
                          group_names, out_base, no_shading=args.no_shading,
                          font_size=args.font_size)
        html_copy = out_base.with_suffix('.html')
        if hp.resolve() != html_copy.resolve():
            shutil.copy2(hp, html_copy)
        print(f'  {hp.name}: {len(tokens)} tokens -> '
              f'{png.name}, {pdf.name}, {html_copy.name}')


if __name__ == '__main__':
    main()
