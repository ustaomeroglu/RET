#!/usr/bin/env python
"""Inject LLM-generated cluster/group names into interactive_doc_*.html files.

Usage:
    python scripts/clustering/inject_names.py <naming_file.txt> <html_file_or_dir>

Examples:
    # Single file
    python scripts/clustering/inject_names.py \\
        .../cluster_naming/Gpt5-2naming.txt \\
        .../test/interactive_doc_0.html

    # All interactive_doc_*.html in a directory
    python scripts/clustering/inject_names.py \\
        .../cluster_naming/Gpt5-2naming.txt \\
        .../test/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_naming_file(
    path: Path,
) -> tuple[dict[int, tuple[str, str]], dict[int, tuple[str, str]]]:
    """Parse an LLM naming output file.

    Recognizes two line formats:
        G<id>: <name> | <description>
        C<id> [G<id>]: <name> | <description>
        C<id>: <name> | <description>          (older format, no group tag)

    Returns (cluster_names, group_names) where each value is (name, description).
    Non-matching lines are silently skipped.
    """
    cluster_names: dict[int, tuple[str, str]] = {}
    group_names: dict[int, tuple[str, str]] = {}

    group_re = re.compile(r"^\s*G(\d+)\s*:\s*(.+?)\s*\|\s*(.+?)\s*$")
    cluster_re = re.compile(
        r"^\s*C(\d+)\s*(?:\[G\d+\])?\s*:\s*(.+?)\s*\|\s*(.+?)\s*$"
    )

    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        m = group_re.match(line)
        if m:
            gid = int(m.group(1))
            group_names[gid] = (m.group(2).strip(), m.group(3).strip())
            continue

        m = cluster_re.match(line)
        if m:
            cid = int(m.group(1))
            cluster_names[cid] = (m.group(2).strip(), m.group(3).strip())
            continue

    return cluster_names, group_names


def _build_js_dict(names: dict[int, tuple[str, str]]) -> str:
    """Build a JS object literal from a names dict."""
    if not names:
        return "{}"
    entries = []
    for k in sorted(names):
        name_esc = json.dumps(names[k][0])
        desc_esc = json.dumps(names[k][1])
        entries.append(f"{k}: {{name: {name_esc}, desc: {desc_esc}}}")
    return "{" + ", ".join(entries) + "}"


_LEGEND_HOVER_JS = """
// --- Legend hover: show cluster/group names (injected) ---
document.querySelectorAll('.legend-item[data-cluster]').forEach(function(el) {
  el.addEventListener('mouseenter', function(ev) {
    var cid = el.dataset.cluster;
    if (CLUSTER_NAMES[cid]) {
      showTip(ev, 'C' + cid + ': ' + CLUSTER_NAMES[cid].name + '\\n' + CLUSTER_NAMES[cid].desc);
    }
  });
  el.addEventListener('mouseleave', hideTip);
});
document.querySelectorAll('.legend-group[data-group]').forEach(function(el) {
  el.addEventListener('mouseenter', function(ev) {
    var gid = el.dataset.group;
    if (GROUP_NAMES[gid]) {
      showTip(ev, 'G' + gid + ': ' + GROUP_NAMES[gid].name + '\\n' + GROUP_NAMES[gid].desc);
    }
  });
  el.addEventListener('mouseleave', hideTip);
});
// --- Legend sub-tabs (Legend / Names) ---
document.querySelectorAll('.legend-tab-btn').forEach(function(btn) {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.legend-tab-btn').forEach(function(b) { b.classList.remove('active'); });
    document.querySelectorAll('.legend-tab-content').forEach(function(c) { c.classList.remove('active'); });
    btn.classList.add('active');
    var target = document.getElementById('ltab-' + btn.dataset.ltab);
    if (target) target.classList.add('active');
  });
});
"""

_NAMES_PANEL_CSS = """
.legend-tabs { display:flex; gap:0; margin-bottom:4px; }
.legend-tab-btn { padding:3px 10px; font-size:11px; border:1px solid #ccc;
  background:#eee; cursor:pointer; border-bottom:none; border-radius:4px 4px 0 0; }
.legend-tab-btn.active { background:white; font-weight:bold; }
.legend-tab-content { display:none; }
.legend-tab-content.active { display:block; }
.names-panel { max-height:260px; overflow-y:auto; padding:6px;
  border:1px solid #ddd; border-radius:0 4px 4px 4px; font-size:12px; }
.names-group-header { font-weight:bold; margin-top:6px; padding:2px 6px; }
.names-cluster { margin-left:12px; padding:1px 4px; }
.names-swatch { display:inline-block; width:10px; height:10px; border-radius:2px;
  margin-right:5px; vertical-align:middle; }
"""


def _parse_legend_structure(content: str) -> tuple[dict[int, str], dict[int, list[int]], dict[int, str]]:
    """Extract cluster colors, group→clusters membership, and group colors from
    the existing <div class="legend">...</div> block.

    Returns (cluster_colors, group_to_clusters, group_colors).
    """
    cluster_colors: dict[int, str] = {}
    group_to_clusters: dict[int, list[int]] = {}
    group_colors: dict[int, str] = {}

    # Extract the legend div content (handles the single-line layout)
    m = re.search(r'<div class="legend">(.*?)</div>\s*(?:<div class="legend-tab-content"|<div class="container">)', content, re.DOTALL)
    if not m:
        m = re.search(r'<div class="legend">(.*?)</div>', content, re.DOTALL)
    if not m:
        return cluster_colors, group_to_clusters, group_colors
    legend_html = m.group(1)

    # Parse legend-group spans (if any)
    group_pat = re.compile(
        r'<span class="legend-group"\s+data-group="(\d+)"[^>]*?border:\s*2px solid ([^;"]+)[^>]*?>(.*?)</span>\s*(?=<span class="legend-group"|$)',
        re.DOTALL,
    )
    for gm in group_pat.finditer(legend_html):
        gid = int(gm.group(1))
        group_colors[gid] = gm.group(2).strip()
        inner = gm.group(3)
        members: list[int] = []
        for cm in re.finditer(
            r'<span class="legend-item"\s+data-cluster="(\d+)"[^>]*?background:\s*([^;"]+)[^>]*?>',
            inner,
        ):
            cid = int(cm.group(1))
            cluster_colors[cid] = cm.group(2).strip()
            members.append(cid)
        group_to_clusters[gid] = members

    # Also parse any top-level legend-items (no grouping case)
    if not group_to_clusters:
        for cm in re.finditer(
            r'<span class="legend-item"\s+data-cluster="(\d+)"[^>]*?background:\s*([^;"]+)[^>]*?>',
            legend_html,
        ):
            cid = int(cm.group(1))
            cluster_colors[cid] = cm.group(2).strip()

    return cluster_colors, group_to_clusters, group_colors


def _build_names_tab_html(
    cluster_names: dict[int, tuple[str, str]],
    group_names: dict[int, tuple[str, str]],
    cluster_colors: dict[int, str],
    group_to_clusters: dict[int, list[int]],
    group_colors: dict[int, str],
) -> str:
    import html as _html

    parts: list[str] = []
    parts.append('<div class="legend-tab-content" id="ltab-names">')
    parts.append('<div class="names-panel">')

    if group_to_clusters:
        for g in sorted(group_to_clusters):
            gname, gdesc = group_names.get(g, (f"Group {g}", ""))
            gcol = group_colors.get(g, "#888")
            parts.append(
                f'<div class="names-group-header" style="border-left:4px solid {gcol}">'
                f'G{g}: {_html.escape(gname)}'
            )
            if gdesc:
                parts.append(f' &mdash; <span style="font-weight:normal">{_html.escape(gdesc)}</span>')
            parts.append('</div>')
            for ci in group_to_clusters[g]:
                cname, cdesc = cluster_names.get(ci, (f"Cluster {ci}", ""))
                ccol = cluster_colors.get(ci, "#ccc")
                parts.append(
                    f'<div class="names-cluster">'
                    f'<span class="names-swatch" style="background:{ccol}"></span>'
                    f'C{ci}: {_html.escape(cname)}'
                )
                if cdesc:
                    parts.append(f' &mdash; <span style="color:#666">{_html.escape(cdesc)}</span>')
                parts.append('</div>')
    else:
        for ci in sorted(cluster_colors):
            cname, cdesc = cluster_names.get(ci, (f"Cluster {ci}", ""))
            ccol = cluster_colors.get(ci, "#ccc")
            parts.append(
                f'<div class="names-cluster">'
                f'<span class="names-swatch" style="background:{ccol}"></span>'
                f'C{ci}: {_html.escape(cname)}'
            )
            if cdesc:
                parts.append(f' &mdash; <span style="color:#666">{_html.escape(cdesc)}</span>')
            parts.append('</div>')

    parts.append('</div>')  # names-panel
    parts.append('</div>')  # ltab-names
    return "".join(parts)


def _inject_names_tab(content: str, names_tab_html: str) -> str:
    """Wrap the existing legend div in a Legend/Names tab structure.

    Inserts (before <div class="legend">):
        <div class="legend-tabs">...buttons...</div>
        <div class="legend-tab-content active" id="ltab-legend">
    and (after the legend div's closing tag):
        </div>   <!-- close ltab-legend -->
        <div class="legend-tab-content" id="ltab-names">...</div>
    Idempotent: if the tab bar is already present, does nothing.
    """
    if 'class="legend-tab-btn' in content and 'data-ltab="names"' in content:
        return content  # already injected

    # Ensure CSS for tab buttons + names panel is present; if missing, add it.
    if ".legend-tab-btn" not in content:
        content = content.replace("</style>", _NAMES_PANEL_CSS + "\n</style>", 1)
    elif ".names-panel" not in content:
        content = content.replace("</style>", _NAMES_PANEL_CSS + "\n</style>", 1)

    tab_bar = (
        '<div class="legend-tabs">'
        '<button class="legend-tab-btn active" data-ltab="legend">Legend</button>'
        '<button class="legend-tab-btn" data-ltab="names">Names</button>'
        '</div>'
        '<div class="legend-tab-content active" id="ltab-legend">'
    )

    idx = content.find('<div class="legend">')
    if idx < 0:
        return content
    content = content[:idx] + tab_bar + content[idx:]

    # Find the matching </div> for the legend div. Since <div class="legend">
    # is a flat container of spans, its closing </div> is immediately before
    # <div class="container">.
    container_idx = content.find('<div class="container">')
    if container_idx < 0:
        return content
    # Insert closing of ltab-legend and the names tab right before container
    injection = '</div>' + names_tab_html  # close ltab-legend + add names tab
    content = content[:container_idx] + injection + content[container_idx:]
    return content

_TOKEN_TIP_JS = """\
var tipText = sp.title;
    var cid = sp.dataset.cluster;
    var gid = sp.dataset.group;
    var nameParts = [];
    if (gid !== undefined && GROUP_NAMES[gid]) nameParts.push('G' + gid + ': ' + GROUP_NAMES[gid].name);
    if (cid !== undefined && CLUSTER_NAMES[cid]) nameParts.push('C' + cid + ': ' + CLUSTER_NAMES[cid].name);
    if (nameParts.length > 0) tipText += '\\n' + nameParts.join(' | ');
    showTip(ev, tipText);"""


def inject_into_html(
    html_path: Path,
    cluster_names: dict[int, tuple[str, str]],
    group_names: dict[int, tuple[str, str]],
) -> bool:
    """Inject CLUSTER_NAMES/GROUP_NAMES into an HTML file.

    Handles both new HTML (with placeholders) and old HTML (without them).
    Returns True if modifications were made.
    """
    content = html_path.read_text(encoding="utf-8")

    cluster_js = _build_js_dict(cluster_names)
    group_js = _build_js_dict(group_names)
    cluster_repl = f"var CLUSTER_NAMES = {cluster_js};"
    group_repl = f"var GROUP_NAMES = {group_js};"

    changed = False
    has_placeholder = "var CLUSTER_NAMES = " in content

    if has_placeholder:
        # New HTML: replace existing placeholders (or previously injected)
        content, n1 = re.subn(
            r"var CLUSTER_NAMES = \{[^;]*\};",
            lambda _: cluster_repl,
            content,
            count=1,
        )
        content, n2 = re.subn(
            r"var GROUP_NAMES = \{[^;]*\};",
            lambda _: group_repl,
            content,
            count=1,
        )
        changed = (n1 + n2) > 0
    else:
        # Old HTML: inject variables after const STRIDE=... line
        stride_pat = r"(const STRIDE=\d+, N_LABELS=\d+, N_TOKENS=\d+;)"
        m = re.search(stride_pat, content)
        if m:
            inject_vars = f"\n{cluster_repl}\n{group_repl}"
            content = content[:m.end()] + inject_vars + content[m.end():]
            changed = True

        # Patch old tooltip: replace showTip(ev, sp.title) with name-aware version
        if "showTip(ev, sp.title);" in content and "var tipText = sp.title;" not in content:
            content = content.replace(
                "showTip(ev, sp.title);",
                _TOKEN_TIP_JS,
                1,  # only replace the first occurrence (token hover)
            )

        # Inject legend hover JS if not already present
        if "Legend hover: show cluster/group names" not in content:
            # Insert before </script></body>
            content = content.replace(
                "</script></body></html>",
                _LEGEND_HOVER_JS + "\n</script></body></html>",
            )

    # Inject the Names sub-tab if it isn't already present.
    if 'data-ltab="names"' not in content:
        cluster_colors, group_to_clusters, group_colors = _parse_legend_structure(content)
        if cluster_colors or group_to_clusters:
            names_tab_html = _build_names_tab_html(
                cluster_names, group_names, cluster_colors, group_to_clusters, group_colors
            )
            new_content = _inject_names_tab(content, names_tab_html)
            if new_content != content:
                content = new_content
                changed = True

    html_path.write_text(content, encoding="utf-8")
    return changed


def main():
    parser = argparse.ArgumentParser(
        description="Inject cluster/group names into interactive HTML files.",
    )
    parser.add_argument(
        "naming_file",
        type=Path,
        help="Path to the LLM naming output .txt file",
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Path to an HTML file or a directory containing interactive_doc_*.html",
    )
    args = parser.parse_args()

    if not args.naming_file.exists():
        print(f"Error: naming file not found: {args.naming_file}", file=sys.stderr)
        sys.exit(1)

    cluster_names, group_names = parse_naming_file(args.naming_file)
    print(
        f"Parsed {len(cluster_names)} cluster names, "
        f"{len(group_names)} group names from {args.naming_file.name}"
    )

    if not cluster_names and not group_names:
        print("Warning: no names parsed — check file format.", file=sys.stderr)
        sys.exit(1)

    # Collect target HTML files
    if args.target.is_dir():
        html_files = sorted(args.target.glob("interactive_doc_*.html"))
        if not html_files:
            print(f"No interactive_doc_*.html files found in {args.target}")
            sys.exit(1)
    elif args.target.is_file():
        html_files = [args.target]
    else:
        print(f"Error: target not found: {args.target}", file=sys.stderr)
        sys.exit(1)

    for html_path in html_files:
        ok = inject_into_html(html_path, cluster_names, group_names)
        if ok:
            print(f"  Injected names into {html_path.name}")
        else:
            print(
                f"  Skipped {html_path.name} "
                f"(no CLUSTER_NAMES/GROUP_NAMES placeholders found)"
            )


if __name__ == "__main__":
    main()
