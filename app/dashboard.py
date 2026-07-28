"""Serves the dashboard page.

The page itself is a static, self-contained HTML file — no CDN, no external
fonts, no network at render time. A dashboard that needs the internet to draw is
one that breaks in exactly the situation you built it to observe.

It reads the same `/metrics`, `/metrics/recent` and `/healthz` endpoints any
other client would. There is no privileged dashboard API, so anything the page
shows can be checked with `curl`.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.responses import HTMLResponse

STATIC_DIR = Path(__file__).parent.parent / "static"
DASHBOARD_PATH = STATIC_DIR / "dashboard.html"

_FALLBACK = """<!doctype html><meta charset="utf-8">
<title>Dashboard unavailable</title>
<body style="font:14px system-ui;padding:40px;max-width:34em">
<h1 style="font-size:18px">Dashboard file missing</h1>
<p>Expected <code>static/dashboard.html</code>. The gateway is unaffected —
<code>/metrics</code> and <code>/metrics/recent</code> still serve JSON.</p>
</body>"""


def render() -> HTMLResponse:
    """Return the page, or an honest placeholder if it is missing.

    A missing static file must not 500: the dashboard is an observability
    surface, and losing it should never look like the gateway itself failing.
    """
    if not DASHBOARD_PATH.exists():
        return HTMLResponse(_FALLBACK, status_code=200)
    return HTMLResponse(
        DASHBOARD_PATH.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},  # always reflect the deployed file
    )
