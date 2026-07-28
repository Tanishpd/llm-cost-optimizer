"""Tests for static/dashboard.html — the 380 lines nothing used to execute.

This file existed for two milestones with exactly one assertion about it
("contains no `http://`"), and it is where the worst bug of the project lived:
`${LABEL[r.tier] ?? r.tier || "—"}` is a **SyntaxError** — mixing `??` with
`||` needs parentheses — which blanked the entire page while 441 tests passed.
It was found by taking a screenshot.

So this module attacks that class directly, in four escalating steps:

1. the JavaScript **parses** (`node --check`),
2. every `el("x")` the JS looks up **exists** in the HTML,
3. every URL the JS fetches is a **real route** on the app,
4. `render()` **actually runs** against a real `/metrics` payload under a DOM
   shim, which is the only one of the four that can catch a logic error.

Step 4 is why this is worth more than a linter: it executes the renderer with
the exact JSON the gateway produces, so a field renamed in `telemetry.py`
breaks the test rather than the page.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.telemetry import _empty_metrics

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = ROOT / "static/dashboard.html"

# `node --check` is the parse gate. CI's ubuntu-latest ships node, and the
# workflow runs it as its own step too, so a machine without node degrades to
# "the other three checks still run" rather than to silence.
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")


def script_source() -> str:
    """The contents of the page's <script> block."""
    html = DASHBOARD.read_text()
    blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(blocks) == 1, f"expected one script block, found {len(blocks)}"
    return blocks[0]


# --------------------------------------------------------------------------
# 1. it parses
# --------------------------------------------------------------------------


@needs_node
def test_the_dashboard_javascript_parses(tmp_path):
    """The regression that motivated this whole file.

    A SyntaxError anywhere in the block means the browser executes *none* of
    it: no render, no polling, a blank page. Nothing else in the test suite or
    in ruff can see it, because it is not Python.
    """
    js = tmp_path / "dashboard.js"
    js.write_text(script_source())
    proc = subprocess.run([NODE, "--check", str(js)], capture_output=True, text=True)
    assert proc.returncode == 0, f"dashboard JS does not parse:\n{proc.stderr}"


@needs_node
def test_the_parse_check_would_actually_fail(tmp_path):
    """Guard the guard.

    A syntax check that cannot fail is worse than none — it reads as coverage.
    This injects the exact defect from the original bug and asserts the tool
    rejects it.
    """
    js = tmp_path / "broken.js"
    js.write_text('const x = a ?? b || c;')      # needs parens; SyntaxError
    proc = subprocess.run([NODE, "--check", str(js)], capture_output=True, text=True)
    assert proc.returncode != 0, "node --check accepted invalid JS"


# --------------------------------------------------------------------------
# 2. the JS and the HTML agree about the DOM
# --------------------------------------------------------------------------


def test_every_element_the_js_looks_up_exists_in_the_html():
    """`el("kpis")` against a div someone renamed is a silent no-op: the page
    renders, that section is just permanently empty."""
    html = DASHBOARD.read_text()
    looked_up = set(re.findall(r'\bel\(\s*"([^"]+)"', html))
    defined = set(re.findall(r'\bid="([^"]+)"', html))

    assert looked_up, "found no el() calls — has the helper been renamed?"
    missing = sorted(looked_up - defined)
    assert not missing, f"JS looks up ids that the HTML does not define: {missing}"


def test_every_element_defined_is_actually_used():
    """The other direction. A leftover id is harmless but usually means a
    section was removed from the JS and forgotten in the markup."""
    html = DASHBOARD.read_text()
    looked_up = set(re.findall(r'\bel\(\s*"([^"]+)"', html))
    defined = set(re.findall(r'\bid="([^"]+)"', html))
    assert not sorted(defined - looked_up), "unused ids in the markup"


# --------------------------------------------------------------------------
# 3. the JS calls routes that exist
# --------------------------------------------------------------------------


def test_every_endpoint_the_dashboard_fetches_is_a_real_route():
    """The dashboard is a client of this API like any other.

    Renaming `/metrics/recent` would leave the suite green and the live log
    permanently empty — the page keeps rendering, so nothing looks broken.
    """
    html = DASHBOARD.read_text()
    called = {
        u.split("?")[0]
        for u in re.findall(r'(?:api|fetch)\(\s*"(/[^"]*)"', html)
    }
    served = {getattr(r, "path", "") for r in app.routes}

    assert called, "found no fetch targets — has the api() helper changed?"
    missing = sorted(called - served)
    assert not missing, f"dashboard fetches routes the app does not serve: {missing}"


# --------------------------------------------------------------------------
# 4. it renders real data without throwing
# --------------------------------------------------------------------------


HARNESS = r"""
// Minimal DOM + browser shim. Enough for render(); no jsdom dependency.
const seen = {};
function stubElement(id) {
  const node = {
    id,
    value: "",
    set innerHTML(v) { seen[id] = String(v); },
    get innerHTML() { return seen[id] || ""; },
    set textContent(v) { seen[id] = String(v); },
    get textContent() { return seen[id] || ""; },
    style: {},
    classList: { add() {}, remove() {}, toggle() {} },
    appendChild() {}, addEventListener() {},
    // Returns a node rather than null: the real page has these children, and
    // a shim that returned null would make the test fail on its own
    // incompleteness instead of on the dashboard.
    querySelector: () => stubElement(id + " >"),
    querySelectorAll: () => [],
  };
  return node;
}
const elements = {};
globalThis.document = {
  getElementById(id) {
    if (!(id in elements)) elements[id] = stubElement(id);
    return elements[id];
  },
  createElement: () => stubElement("created"),
  addEventListener() {},
};
globalThis.window = { prompt: () => null };
globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => ({}) });
globalThis.setInterval = () => 0;
globalThis.setTimeout = () => 0;

__SCRIPT__

// Drive the renderer with the payloads the gateway actually produced.
const payload = __PAYLOAD__;
render(payload.metrics, payload.recent, payload.health);

console.log(JSON.stringify({ populated: Object.keys(seen).sort() }));
"""


def _run_render(payload: dict, tmp_path: Path) -> dict:
    src = HARNESS.replace("__SCRIPT__", script_source()).replace(
        "__PAYLOAD__", json.dumps(payload))
    js = tmp_path / "harness.mjs"
    js.write_text(src)
    proc = subprocess.run([NODE, str(js)], capture_output=True, text=True)
    assert proc.returncode == 0, f"render() threw:\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


@needs_node
def test_render_survives_a_real_metrics_payload(tmp_path):
    """The strong one: execute the renderer against JSON this gateway made.

    A field renamed in `telemetry.py` now breaks a test instead of a panel.
    """
    with TestClient(app) as c:
        metrics = c.get("/metrics").json()
        recent = c.get("/metrics/recent?count=12").json()["requests"]
        health = c.get("/healthz").json()

    out = _run_render({"metrics": metrics, "recent": recent, "health": health}, tmp_path)

    # The panels that carry data. Named explicitly rather than "everything the
    # shim saw": `health` is styled through a child node and `banner` only
    # appears when degraded, so a blanket assertion would be testing the shim's
    # bookkeeping rather than the dashboard.
    for panel in ("kpis", "stack", "legend", "latency", "log", "mode",
                  "saved", "saved-pct", "baseline"):
        assert panel in out["populated"], f"render() left #{panel} empty"


@needs_node
def test_render_survives_a_cold_gateway_with_no_traffic(tmp_path):
    """First load, before anyone has sent a request. Division by zero in a
    rate calculation would blank the page for exactly the person seeing it
    for the first time."""
    out = _run_render(
        {"metrics": _empty_metrics(), "recent": [], "health": None}, tmp_path)
    assert "kpis" in out["populated"]


@needs_node
def test_render_survives_redis_being_down(tmp_path):
    """Degraded mode still has to draw. This is the payload `/metrics` returns
    when Redis is unreachable."""
    out = _run_render(
        {"metrics": _empty_metrics(degraded=["redis_unavailable"]),
         "recent": [], "health": {"status": "degraded", "dependencies": {}}},
        tmp_path)
    assert "kpis" in out["populated"]


@needs_node
def test_render_survives_an_unknown_tier(tmp_path):
    """Adding a tier to `cascade.py` is a data change (D14). The dashboard
    must not blank because it met a tier name it has no colour for — which is
    precisely what the `??`/`||` line was trying to handle when it broke."""
    metrics = _empty_metrics()
    metrics["requests"] = 3
    metrics["tiers"]["served"]["local_medium"] = 3
    out = _run_render(
        {"metrics": metrics,
         "recent": [{"trace_id": "abc", "tier": "local_medium", "path": ["local_medium"],
                     "reason": "a new tier", "latency_ms": {"total": 12.0},
                     "cost_micro_usd": "0", "created_at": "1785258422.7"}],
         "health": None},
        tmp_path)
    assert "log" in out["populated"], "the live log blanked on an unknown tier"
