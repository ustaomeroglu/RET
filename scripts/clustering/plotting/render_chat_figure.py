#!/usr/bin/env python
"""Render a horizontal chat figure: User bubble | GPT bubble (side by side).

Input text file uses `### Role` headers, as in doc_0_explanation.txt.

Usage:
    python scripts/clustering/render_chat_figure.py \
        <text_file> <output_base> \
        [--width 2400] [--user-frac 0.32]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _break_long_word(word: str, max_w: int, font) -> list[str]:
    out: list[str] = []
    cur = ''
    for ch in word:
        trial = cur + ch
        if font.getbbox(trial)[2] <= max_w:
            cur = trial
        else:
            if cur:
                out.append(cur)
            cur = ch
    if cur:
        out.append(cur)
    return out


def wrap(text: str, max_w: int, font) -> list[str]:
    out: list[str] = []
    cur = ''
    for w in text.split(' '):
        if font.getbbox(w)[2] > max_w:
            if cur:
                out.append(cur)
                cur = ''
            chunks = _break_long_word(w, max_w, font)
            out.extend(chunks[:-1])
            cur = chunks[-1]
            continue
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


def parse_chat(text: str):
    sections = []
    cur_role = None
    cur_lines: list[str] = []
    for line in text.splitlines():
        m = re.match(r'^###\s+(.+?)\s*$', line)
        if m:
            if cur_role is not None:
                sections.append((cur_role, '\n'.join(cur_lines).strip('\n')))
            cur_role = m.group(1).strip()
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_role is not None:
        sections.append((cur_role, '\n'.join(cur_lines).strip('\n')))
    return sections


def _looks_monospace(p: str) -> bool:
    return any(tok in p for tok in ('<|', '|>', '↵'))


def _parse_blocks(body: str, inner_w: int, font_body, font_bold,
                  font_mono, bullet_indent):
    heading_re = re.compile(r'^\d+\.\s+[A-Za-z][A-Za-z0-9\s/\-]{0,40}$')
    bullet_re = re.compile(r'^[–\-]\s+(.+)$')
    paragraphs = re.split(r'\n{2,}', body.strip())
    blocks: list[tuple] = []
    for i, p in enumerate(paragraphs):
        p = p.strip('\n')
        if not p.strip():
            continue
        lines_in = p.split('\n')
        for j, line in enumerate(lines_in):
            line = line.rstrip()
            if not line.strip():
                continue
            if _looks_monospace(line):
                blocks.append(('mono', wrap(line, inner_w, font_mono)))
            elif j == 0 and heading_re.match(line.strip()):
                blocks.append(('head', wrap(line.strip(), inner_w, font_bold)))
            elif bullet_re.match(line.strip()):
                content = bullet_re.match(line.strip()).group(1)
                blocks.append((
                    'bullet',
                    wrap(content, inner_w - bullet_indent, font_body),
                ))
            else:
                blocks.append(('body', wrap(line, inner_w, font_body)))
        if i < len(paragraphs) - 1:
            blocks.append(('gap', []))
    return blocks


def render_chat_figure(text: str, width: int, user_frac: float = 0.32,
                       s: int = 2, font_size: int = 18) -> Image.Image:
    W = width * s
    font_body = load_font('DejaVuSans.ttf', font_size * s)
    font_role = load_font('DejaVuSans-Bold.ttf', (font_size + 4) * s)
    font_bold = load_font('DejaVuSans-Bold.ttf', font_size * s)
    font_mono = load_font('DejaVuSansMono.ttf', (font_size - 4) * s)

    pad_x = 28 * s
    pad_y = 24 * s
    bubble_pad = 18 * s
    col_gap = 28 * s
    role_gap = 6 * s
    bullet_indent = 18 * s

    b_asc, b_dsc = font_body.getmetrics()
    body_h = b_asc + b_dsc + 2 * s
    m_asc, m_dsc = font_mono.getmetrics()
    mono_h = m_asc + m_dsc
    r_asc, r_dsc = font_role.getmetrics()
    role_h = r_asc + r_dsc + 2 * s
    para_gap = 8 * s

    content_w = W - pad_x * 2
    # Diagonal layout: User top-right, GPT bottom-left spanning full width.
    user_w = int(content_w * 0.55)
    gpt_w = content_w

    def _lh(kind):
        return mono_h if kind == 'mono' else body_h

    def bubble_height(blocks) -> int:
        h = bubble_pad * 2
        for kind, lines in blocks:
            if kind == 'gap':
                h += para_gap
            else:
                h += len(lines) * _lh(kind)
        return h

    sections = parse_chat(text)
    if not sections:
        raise SystemExit('No `### Role` chat sections found.')

    cols = []  # each: (role, style, blocks, col_w)
    for role, body in sections:
        style = 'user' if role.lower().startswith('user') else 'assistant'
        w = user_w if style == 'user' else gpt_w
        inner_w = w - bubble_pad * 2
        blocks = _parse_blocks(body, inner_w, font_body, font_bold,
                               font_mono, bullet_indent)
        cols.append((role, style, blocks, w))

    user_col = next((c for c in cols if c[1] == 'user'), None)
    gpt_col = next((c for c in cols if c[1] != 'user'), None)
    if user_col is None or gpt_col is None:
        raise SystemExit('Expected exactly one User and one assistant section.')

    user_bubble_h = bubble_height(user_col[2])
    gpt_bubble_h = bubble_height(gpt_col[2])

    vert_gap = 24 * s
    user_block_h = role_h + role_gap + user_bubble_h
    gpt_block_h = role_h + role_gap + gpt_bubble_h

    total_h = pad_y + user_block_h + vert_gap + gpt_block_h + pad_y

    img = Image.new('RGB', (W, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    def place(col, x, y):
        role, style, blocks, w = col
        if style == 'user':
            bg = (240, 245, 252)
            border = (210, 222, 238)
            role_color = (40, 82, 148)
        else:
            bg = (248, 244, 238)
            border = (234, 224, 208)
            role_color = (152, 88, 30)
        draw.text((x + 2 * s, y), role, fill=role_color, font=font_role)
        y += role_h + role_gap
        bh = bubble_height(blocks)
        try:
            draw.rounded_rectangle(
                [x, y, x + w, y + bh],
                radius=16 * s, fill=bg, outline=border,
                width=max(1, s // 2),
            )
        except AttributeError:
            draw.rectangle([x, y, x + w, y + bh], fill=bg, outline=border)
        inner_x = x + bubble_pad
        inner_y = y + bubble_pad
        for kind, lines in blocks:
            if kind == 'gap':
                inner_y += para_gap
                continue
            if kind == 'mono':
                font = font_mono
                color = (60, 60, 60)
                indent = 0
            elif kind == 'head':
                font = font_bold
                color = (20, 20, 20)
                indent = 0
            elif kind == 'bullet':
                font = font_body
                color = (25, 25, 25)
                indent = bullet_indent
            else:
                font = font_body
                color = (25, 25, 25)
                indent = 0
            lh = _lh(kind)
            for i, line in enumerate(lines):
                if kind == 'bullet' and i == 0:
                    draw.text((inner_x, inner_y), '–',
                              fill=(110, 110, 110), font=font_body)
                draw.text((inner_x + indent, inner_y), line,
                          fill=color, font=font)
                inner_y += lh

    # GPT: bottom-left
    gpt_x = pad_x
    gpt_y = pad_y + user_block_h + vert_gap

    # User: top, right-aligned with GPT's right edge.
    gpt_right = gpt_x + gpt_col[3]
    user_x = gpt_right - user_col[3]
    user_y = pad_y
    place(user_col, user_x, user_y)
    place(gpt_col, gpt_x, gpt_y)

    # Downscale 2x to final
    if s > 1:
        img = img.resize((img.size[0] // s, img.size[1] // s), Image.LANCZOS)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('text_file', type=Path)
    ap.add_argument('output_base', type=Path)
    ap.add_argument('--width', type=int, default=2400)
    ap.add_argument('--user-frac', type=float, default=0.32)
    ap.add_argument('--font-size', type=int, default=18)
    args = ap.parse_args()

    text = args.text_file.read_text(encoding='utf-8')
    img = render_chat_figure(text, args.width, args.user_frac,
                             font_size=args.font_size)

    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.output_base.with_suffix('.png'), 'PNG', optimize=True)
    img.save(args.output_base.with_suffix('.pdf'), 'PDF', resolution=200.0)
    print(f'Saved {args.output_base.with_suffix(".png")} '
          f'({img.size[0]}x{img.size[1]})')


if __name__ == '__main__':
    main()
