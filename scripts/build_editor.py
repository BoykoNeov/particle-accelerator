"""Bundle the editor into one self-contained HTML file.

``editor/index.html`` loads ``accsim-optics.js`` and ``presets.js`` as separate
scripts so the physics core can be unit-tested under Node and the presets read as
JSON. A single file is handier to send around or publish, so this inlines both::

    .venv/Scripts/python.exe scripts/build_editor.py [OUT.html]

Default output: ``W:/temp/claude/accsim-editor/accsim-editor.html`` (a scratch
location, deliberately outside the repository — the bundle is a build product).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EDITOR = ROOT / "editor"
DEFAULT_OUT = Path("W:/temp/claude/accsim-editor/accsim-editor.html")


def bundle() -> str:
    html = (EDITOR / "index.html").read_text(encoding="utf-8")

    def inline(match: re.Match[str]) -> str:
        src = match.group(1)
        code = (EDITOR / src).read_text(encoding="utf-8")
        # A literal "</script>" inside the JS would end the inline block early.
        code = code.replace("</script>", "<\\/script>")
        return f"<script>\n/* inlined from editor/{src} */\n{code}\n</script>"

    out, n = re.subn(r'<script src="([^"]+\.js)"></script>', inline, html)
    if n != 2:
        raise SystemExit(f"expected to inline 2 scripts, found {n}")
    return out


def main(argv: list[str]) -> None:
    out = Path(argv[1]) if len(argv) > 1 else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    text = bundle()
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(text)} bytes)")


if __name__ == "__main__":
    main(sys.argv)
