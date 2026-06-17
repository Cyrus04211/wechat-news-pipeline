"""Export Markdown reports to simple standalone HTML."""

from __future__ import annotations

import html
import re
from pathlib import Path


def markdown_to_html(md_text: str, title: str = "Research Report") -> str:
    lines = md_text.splitlines()
    body_parts: list[str] = []
    in_list = False
    in_table = False
    table_rows: list[str] = []

    def close_list():
        nonlocal in_list
        if in_list:
            body_parts.append("</ul>")
            in_list = False

    def close_table():
        nonlocal in_table, table_rows
        if in_table and table_rows:
            body_parts.append("<table>")
            for i, row in enumerate(table_rows):
                tag = "th" if i == 0 else "td"
                cells = "".join(f"<{tag}>{html.escape(c.strip())}</{tag}>" for c in row.split("|")[1:-1])
                body_parts.append(f"<tr>{cells}</tr>")
            body_parts.append("</table>")
            table_rows = []
            in_table = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            close_list()
            in_table = True
            table_rows.append(stripped)
            continue
        close_table()

        if stripped.startswith("# "):
            close_list()
            body_parts.append(f"<h1>{html.escape(stripped[2:])}</h1>")
        elif stripped.startswith("## "):
            close_list()
            body_parts.append(f"<h2>{html.escape(stripped[3:])}</h2>")
        elif stripped.startswith("### "):
            close_list()
            body_parts.append(f"<h3>{html.escape(stripped[4:])}</h3>")
        elif stripped.startswith("- "):
            if not in_list:
                body_parts.append("<ul>")
                in_list = True
            content = _inline_format(stripped[2:])
            body_parts.append(f"<li>{content}</li>")
        elif stripped.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            if m:
                alt, src = m.group(1), m.group(2)
                body_parts.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>')
        elif not stripped:
            close_list()
            body_parts.append("<br>")
        else:
            close_list()
            body_parts.append(f"<p>{_inline_format(stripped)}</p>")

    close_list()
    close_table()

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; line-height: 1.6; color: #1a1a1a; }
    h1 { border-bottom: 2px solid #2563eb; padding-bottom: 0.3rem; }
    h2 { color: #1e40af; margin-top: 2rem; }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }
    th, td { border: 1px solid #d1d5db; padding: 0.4rem 0.6rem; text-align: left; }
    th { background: #eff6ff; }
    img { max-width: 100%; height: auto; }
    figure { margin: 1rem 0; }
    figcaption { font-size: 0.85rem; color: #6b7280; }
    ul { padding-left: 1.2rem; }
    a { color: #2563eb; }
    .meta { color: #6b7280; font-size: 0.9rem; }
    """

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
{''.join(body_parts)}
</body>
</html>"""


def _inline_format(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<a href="\2" target="_blank" rel="noopener">\1</a>',
        text,
    )
    return text


def export_html_from_markdown(md_path: str, html_path: str, title: str = "") -> str:
    md_text = Path(md_path).read_text(encoding="utf-8")
    if not title:
        for line in md_text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
    html_text = markdown_to_html(md_text, title=title or "Research Report")
    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    Path(html_path).write_text(html_text, encoding="utf-8")
    return html_path
