#!/usr/bin/env python
"""Compose a paper figure: narrative text (left) + clustering image (right).

If the text file contains `### Role` headers, the left column is rendered
as a stack of chat bubbles (role-labeled).  Otherwise it falls back to a
plain document layout.

Usage:
    python scripts/clustering/compose_doc_with_explanation.py \
        <text_file> <image_file> <output_base> \
        [--left-title "..."] [--right-title "..."]
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
    """Split a word that doesn't fit on one line into character chunks."""
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
    words = text.split(' ')
    out: list[str] = []
    cur = ''
    for w in words:
        if font.getbbox(w)[2] > max_w:
            # Flush current line, then break long word
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
    """Return list of (role, body) tuples split on `### Role` headers.

    Returns empty list if no chat headers are present.
    """
    if not re.search(r'(?m)^###\s+\S', text):
        return []
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


def _looks_monospace(para: str) -> bool:
    return any(tok in para for tok in ('<|', '|>', '↵'))


def render_chat_column(text: str, width: int, s: int = 2) -> Image.Image:
    font_body = load_font('DejaVuSans.ttf', 18 * s)
    font_role = load_font('DejaVuSans-Bold.ttf', 21 * s)
    font_bold = load_font('DejaVuSans-Bold.ttf', 18 * s)
    font_mono = load_font('DejaVuSansMono.ttf', 14 * s)

    pad_x = 32 * s
    pad_y = 26 * s
    bubble_pad = 16 * s
    bubble_gap = 20 * s
    role_gap = 5 * s

    content_w = width - pad_x * 2
    inner_w = content_w - bubble_pad * 2

    b_asc, b_dsc = font_body.getmetrics()
    body_h = b_asc + b_dsc + s  # tight line spacing
    m_asc, m_dsc = font_mono.getmetrics()
    mono_h = m_asc + m_dsc
    r_asc, r_dsc = font_role.getmetrics()
    role_h = r_asc + r_dsc + 2 * s
    para_gap = 6 * s

    bullet_indent = 14 * s
    heading_re = re.compile(r'^\d+\.\s+[A-Za-z][A-Za-z0-9\s/\-]{0,40}$')
    bullet_re = re.compile(r'^[–\-]\s+(.+)$')

    sections = parse_chat(text)
    bubbles = []
    for role, body in sections:
        style = 'user' if role.lower().startswith('user') else 'assistant'
        paragraphs = re.split(r'\n{2,}', body.strip())
        blocks = []
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
        bubbles.append((role, style, blocks))

    def _lh(kind):
        if kind == 'mono':
            return mono_h
        return body_h

    def bubble_height(blocks) -> int:
        h = bubble_pad * 2
        for kind, lines in blocks:
            if kind == 'gap':
                h += para_gap
            else:
                h += len(lines) * _lh(kind)
        return h

    total_h = pad_y
    for (role, style, blocks) in bubbles:
        total_h += role_h + role_gap + bubble_height(blocks) + bubble_gap
    total_h = total_h - bubble_gap + pad_y

    img = Image.new('RGB', (width, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = pad_y
    for (role, style, blocks) in bubbles:
        if style == 'user':
            bg = (240, 245, 252)
            border = (210, 222, 238)
            role_color = (40, 82, 148)
        else:
            bg = (248, 244, 238)
            border = (234, 224, 208)
            role_color = (152, 88, 30)

        draw.text((pad_x + 2 * s, y), role, fill=role_color, font=font_role)
        y += role_h + role_gap

        bh = bubble_height(blocks)
        try:
            draw.rounded_rectangle(
                [pad_x, y, pad_x + content_w, y + bh],
                radius=14 * s, fill=bg, outline=border,
                width=max(1, s // 2),
            )
        except AttributeError:
            draw.rectangle(
                [pad_x, y, pad_x + content_w, y + bh],
                fill=bg, outline=border,
            )

        inner_x = pad_x + bubble_pad
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

        y += bh + bubble_gap

    return img


def render_text_column(text: str, width: int, title: str = '',
                       s: int = 2) -> Image.Image:
    """Fallback: plain article-style rendering when no chat headers."""
    font_body = load_font('DejaVuSans.ttf', 15 * s)
    font_head = load_font('DejaVuSans-Bold.ttf', 18 * s)
    font_title = load_font('DejaVuSans-Bold.ttf', 20 * s)

    pad_x = 32 * s
    pad_y = 28 * s
    content_w = width - pad_x * 2
    bullet_indent = 22 * s

    b_asc, b_dsc = font_body.getmetrics()
    body_h = b_asc + b_dsc + 6 * s
    h_asc, h_dsc = font_head.getmetrics()
    head_h = h_asc + h_dsc + 8 * s
    t_asc, t_dsc = font_title.getmetrics()
    title_h = t_asc + t_dsc + 10 * s

    blocks: list[tuple] = []
    for p in re.split(r'\n{2,}', text.strip()):
        if not p.strip():
            continue
        for line in p.split('\n'):
            if not line.strip():
                continue
            m = re.match(r'^(\d+)\.\s+(.+)$', line.strip())
            if m and len(line.strip()) < 90:
                blocks.append(('head', wrap(line.strip(), content_w, font_head), 0))
                continue
            m = re.match(r'^-\s+(.+)$', line.strip())
            if m:
                blocks.append(('bullet',
                               wrap(m.group(1), content_w - bullet_indent, font_body),
                               bullet_indent))
                continue
            blocks.append(('para', wrap(line.strip(), content_w, font_body), 0))
        blocks.append(('space', [], 0))

    total_h = pad_y * 2
    if title:
        total_h += title_h + 12 * s
    for kind, lines, _ in blocks:
        if kind == 'space':
            total_h += body_h // 2
        elif kind == 'head':
            total_h += 10 * s + len(lines) * head_h
        else:
            total_h += len(lines) * body_h

    img = Image.new('RGB', (width, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = pad_y
    if title:
        draw.text((pad_x, y), title, fill=(15, 15, 15), font=font_title)
        y += title_h
        draw.line([(pad_x, y + 2 * s), (width - pad_x, y + 2 * s)],
                  fill=(210, 210, 210), width=max(1, s // 2))
        y += 12 * s

    for kind, lines, indent in blocks:
        if kind == 'space':
            y += body_h // 2
            continue
        if kind == 'head':
            y += 10 * s
            for line in lines:
                draw.text((pad_x, y), line, fill=(15, 15, 15), font=font_head)
                y += head_h
        elif kind == 'bullet':
            for i, line in enumerate(lines):
                if i == 0:
                    draw.text((pad_x + 4 * s, y), '•',
                              fill=(70, 70, 70), font=font_body)
                draw.text((pad_x + indent, y), line,
                          fill=(35, 35, 35), font=font_body)
                y += body_h
        else:
            for line in lines:
                draw.text((pad_x + indent, y), line,
                          fill=(35, 35, 35), font=font_body)
                y += body_h

    return img


def _right_with_caption(right_raw: Image.Image, title: str,
                        s: int) -> Image.Image:
    if not title:
        return right_raw
    strip_w = right_raw.size[0] * s
    strip_h = 64 * s
    strip = Image.new('RGB', (strip_w, strip_h), (255, 255, 255))
    dt = ImageDraw.Draw(strip)
    tfont = load_font('DejaVuSans-Bold.ttf', 20 * s)
    dt.text((32 * s, 14 * s), title, fill=(15, 15, 15), font=tfont)
    dt.line([(32 * s, 56 * s), (strip_w - 32 * s, 56 * s)],
            fill=(210, 210, 210), width=1)
    strip = strip.resize(
        (right_raw.size[0], strip.size[1] // s), Image.LANCZOS,
    )
    out = Image.new(
        'RGB',
        (right_raw.size[0], right_raw.size[1] + strip.size[1]),
        (255, 255, 255),
    )
    out.paste(strip, (0, 0))
    out.paste(right_raw, (0, strip.size[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('text_file', type=Path)
    ap.add_argument('image_file', type=Path)
    ap.add_argument('output_base', type=Path)
    ap.add_argument('--left-title', default='')
    ap.add_argument('--right-title',
                    default='RET cluster trajectory (unsupervised)')
    args = ap.parse_args()

    s = 2
    text = args.text_file.read_text(encoding='utf-8')
    left_w = 1050 * s

    if parse_chat(text):
        left_big = render_chat_column(text, left_w, s=s)
    else:
        left_big = render_text_column(text, left_w, title=args.left_title, s=s)
    left_img = left_big.resize(
        (left_big.size[0] // s, left_big.size[1] // s), Image.LANCZOS,
    )

    right_raw = Image.open(args.image_file).convert('RGB')
    right_img = _right_with_caption(right_raw, args.right_title, s)

    lh, rh = left_img.size[1], right_img.size[1]
    target_h = max(lh, rh)
    if lh < target_h:
        padded = Image.new('RGB', (left_img.size[0], target_h), (255, 255, 255))
        padded.paste(left_img, (0, 0))
        left_img = padded
    if rh < target_h:
        padded = Image.new('RGB', (right_img.size[0], target_h), (255, 255, 255))
        padded.paste(right_img, (0, 0))
        right_img = padded

    gap = 28
    total_w = left_img.size[0] + gap + right_img.size[0]
    composite = Image.new('RGB', (total_w, target_h), (255, 255, 255))
    composite.paste(left_img, (0, 0))
    ImageDraw.Draw(composite).line(
        [(left_img.size[0] + gap // 2, 20),
         (left_img.size[0] + gap // 2, target_h - 20)],
        fill=(220, 220, 220), width=1,
    )
    composite.paste(right_img, (left_img.size[0] + gap, 0))

    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    composite.save(args.output_base.with_suffix('.png'), 'PNG', optimize=True)
    composite.save(args.output_base.with_suffix('.pdf'), 'PDF', resolution=200.0)
    print(f'Saved {args.output_base.with_suffix(".png")} '
          f'({composite.size[0]}x{composite.size[1]})')


if __name__ == '__main__':
    main()
