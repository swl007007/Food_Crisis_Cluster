"""Measure MLflow UI load in a real headless Edge via the DevTools protocol (Windows Python).

Run with a Windows interpreter that has `websockets` (no installs):
  python.exe browser_probe.py --out result.json URL [URL ...]

For each URL: fresh tab, navigate, wait until the runs grid shows rows (or a compare table),
and record wall time to first rows, network bytes of run-search responses, total encoded
bytes, JS heap and Performance metrics. Read-only: it only loads pages.
--screenshot-dir saves a PNG of each page once ready; --fixed-wait N skips the grid check and
waits N seconds (for run, model and registry pages).
"""

import argparse
import asyncio
import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.request

import websockets

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
READY_JS = """(() => {
  const rows = document.querySelectorAll('.ag-row').length;
  const txt = document.body ? document.body.innerText : '';
  const m = txt.match(/(\\d+) matching runs/);
  const cmp = document.querySelectorAll('table tr, [role="row"]').length;
  return JSON.stringify({rows, matching: m ? +m[1] : null, cmp});
})()"""


async def probe(ws_url, url, timeout, ready, shot=None, fixed_wait=None):
    async with websockets.connect(ws_url, max_size=None) as ws:
        msg_id = 0
        events = []

        async def call(method, params=None):
            nonlocal msg_id
            msg_id += 1
            mine = msg_id
            await ws.send(json.dumps({"id": mine, "method": method, "params": params or {}}))
            while True:
                m = json.loads(await ws.recv())
                if m.get("id") == mine:
                    return m.get("result", {})
                events.append(m)

        for d in ("Page.enable", "Network.enable", "Runtime.enable", "Performance.enable"):
            await call(d)
        await call("Network.setCacheDisabled", {"cacheDisabled": True})
        t0 = time.perf_counter()
        await call("Page.navigate", {"url": url})
        state, t_ready = None, None
        if fixed_wait:
            await asyncio.sleep(fixed_wait)
            t_ready = time.perf_counter() - t0
        while not fixed_wait and time.perf_counter() - t0 < timeout:
            r = await call("Runtime.evaluate", {"expression": READY_JS, "returnByValue": True})
            state = json.loads(r.get("result", {}).get("value") or "{}")
            if ready(state):
                t_ready = time.perf_counter() - t0
                break
            await asyncio.sleep(0.1)
        await asyncio.sleep(1.5)  # let trailing requests finish before reading counters
        if shot:
            img = await call("Page.captureScreenshot", {"format": "png"})
            with open(shot, "wb") as f:
                f.write(base64.b64decode(img["data"]))
        perf = {m["name"]: m["value"] for m in (await call("Performance.getMetrics")).get("metrics", [])}
        # drain remaining buffered events
        try:
            while True:
                events.append(json.loads(await asyncio.wait_for(ws.recv(), 0.2)))
        except Exception:
            pass
        urls, done = {}, {}
        for e in events:
            if e.get("method") == "Network.requestWillBeSent":
                urls[e["params"]["requestId"]] = e["params"]["request"]["url"]
            elif e.get("method") == "Network.loadingFinished":
                done[e["params"]["requestId"]] = e["params"]["encodedDataLength"]
        search = [(urls.get(k, ""), v) for k, v in done.items() if "runs/search" in urls.get(k, "")]
        return {"url": url, "seconds_to_ready": t_ready, "state": state,
                "search_requests": len(search), "search_bytes": sum(v for _, v in search),
                "all_bytes": sum(done.values()), "requests": len(done),
                "js_heap_used_mb": round(perf.get("JSHeapUsedSize", 0) / 1e6, 1),
                "script_seconds": round(perf.get("ScriptDuration", 0), 2),
                "task_seconds": round(perf.get("TaskDuration", 0), 2),
                "layout_seconds": round(perf.get("LayoutDuration", 0), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=9714)  # outside Windows excluded port ranges
    ap.add_argument("--timeout", type=float, default=90)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--screenshot-dir", default=None)
    ap.add_argument("--fixed-wait", type=float, default=None)
    a = ap.parse_args()
    profile = tempfile.mkdtemp(prefix="ipcch-mlflow-probe-")
    edge = subprocess.Popen([EDGE, "--headless=new", "--disable-gpu", f"--remote-debugging-port={a.port}",
                             f"--user-data-dir={profile}", "--window-size=1600,1000", "--no-first-run",
                             "--disable-extensions", "about:blank"])
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/json/version", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        results = []
        if a.screenshot_dir:
            os.makedirs(a.screenshot_dir, exist_ok=True)
        for n, url in enumerate(a.urls):
            compare = "compare-runs" in url
            ready = (lambda s: (s.get("cmp") or 0) > 3) if compare else (lambda s: (s.get("rows") or 0) > 0)
            for i in range(a.repeat):
                req = urllib.request.Request(f"http://127.0.0.1:{a.port}/json/new?about:blank", method="PUT")
                tab = json.loads(urllib.request.urlopen(req).read())
                try:
                    shot = (os.path.join(a.screenshot_dir, f"page{n + 1:02d}.png")
                            if a.screenshot_dir and i == a.repeat - 1 else None)
                    r = asyncio.run(probe(tab["webSocketDebuggerUrl"], url, a.timeout, ready, shot, a.fixed_wait))
                    r["screenshot"] = shot
                    r["attempt"] = i + 1
                    results.append(r)
                    print(json.dumps(r), flush=True)
                finally:
                    urllib.request.urlopen(f"http://127.0.0.1:{a.port}/json/close/{tab['id']}")
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
    finally:
        edge.terminate()


if __name__ == "__main__":
    main()
