"""Generation-pinned, byte-capped download of AlphaEarth 2017 COGs (EE interpreter; requester-pays bucket).

Each object in the manifest is pinned by path, generation and size. Per object: one metadata GET (generation-bound)
for crc32c, then a streamed media download with ``userProject`` to ``<name>.part``; size and crc32c are verified
before the rename. Every received byte (retries included) counts toward ``max_download_bytes``; at most
``max_object_attempts`` per object; local scratch is checked against ``max_scratch_bytes`` before each object.
Socket timeouts are explicit. Usage: python aef_fetch.py --manifest M --dest DIR [--only PATH ...]
"""
import argparse, base64, hashlib, json, os, shutil, sys, time
from pathlib import Path

BUCKET = "alphaearth_foundations"


def ledger_bytes(dest: Path) -> int:
    p = dest / "download_ledger.jsonl"
    return sum(json.loads(x).get("bytes", 0) for x in p.read_text().splitlines()) if p.exists() else 0


def log(dest: Path, rec: dict) -> None:
    with open(dest / "download_ledger.jsonl", "a") as h:
        h.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **rec}) + "\n")


def crc32c_file(path: Path) -> str:
    import google_crc32c
    c = google_crc32c.Checksum()
    with open(path, "rb") as h:
        for chunk in iter(lambda: h.read(1 << 24), b""):
            c.update(chunk)
    return base64.b64encode(c.digest()).decode()


def fetch(manifest: dict, dest: Path, storage, only=None) -> list:
    lim = manifest["limits"]
    out = []
    for obj in manifest["objects"]:
        if only and obj["path"] not in only:
            continue
        name = obj["path"].split(f"gs://{BUCKET}/", 1)[1]
        local = dest / Path(name).name
        if local.exists():
            if local.stat().st_size == obj["size"] and crc32c_file(local) == obj.get("crc32c_verified"):
                out.append({"path": obj["path"], "local": str(local), "skipped": True})
                continue
            raise SystemExit(f"{local} exists but does not match its pinned identity; not overwritten")
        scratch = sum(f.stat().st_size for f in dest.glob("*") if f.is_file())
        if scratch + obj["size"] > int(lim["max_scratch_bytes"]):
            raise SystemExit(f"scratch cap: {scratch} + {obj['size']} > {lim['max_scratch_bytes']}")
        meta = storage.objects().get(bucket=BUCKET, object=name, generation=obj["generation"], userProject=manifest["user_project"]).execute()
        log(dest, {"path": obj["path"], "event": "metadata", "bytes": 0, "generation": meta.get("generation"), "size": meta.get("size"), "crc32c": meta.get("crc32c")})
        if int(meta["generation"]) != int(obj["generation"]) or int(meta["size"]) != int(obj["size"]):
            raise SystemExit(f"{obj['path']}: live generation/size differ from the pin")
        for attempt in range(1, int(lim["max_object_attempts"]) + 1):
            if ledger_bytes(dest) + obj["size"] > int(lim["max_download_bytes"]):
                raise SystemExit(f"download byte cap reached before {obj['path']}")
            part = local.with_name(local.name + ".part")
            t0, got = time.monotonic(), 0
            try:
                from googleapiclient.http import MediaIoBaseDownload
                req = storage.objects().get_media(bucket=BUCKET, object=name, generation=obj["generation"], userProject=manifest["user_project"])
                with open(part, "wb") as h:
                    dl = MediaIoBaseDownload(h, req, chunksize=64 << 20)
                    done = False
                    while not done:
                        status, done = dl.next_chunk(num_retries=0)
                        got = h.tell()
                ok = None
            except Exception as error:  # noqa: BLE001
                ok = str(error)[:300]
                got = part.stat().st_size if part.exists() else got
            log(dest, {"path": obj["path"], "event": "download", "attempt": attempt, "bytes": got, "seconds": round(time.monotonic() - t0, 3), "error": ok})
            if ok is None and part.stat().st_size == obj["size"] and crc32c_file(part) == meta["crc32c"]:
                part.replace(local)
                obj["crc32c_verified"] = meta["crc32c"]
                out.append({"path": obj["path"], "local": str(local), "bytes": got, "seconds": round(time.monotonic() - t0, 3), "crc32c": meta["crc32c"]})
                break
            if part.exists():
                part.unlink()
        else:
            raise SystemExit(f"{obj['path']}: failed after {lim['max_object_attempts']} attempts")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--only", nargs="*")
    a = p.parse_args(argv)
    manifest = json.loads(Path(a.manifest).read_text())
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != manifest["manifest_sha256"]:
        raise SystemExit("fetch manifest differs from its frozen sha256")
    import ee, httplib2, google_auth_httplib2  # noqa: E401
    from googleapiclient.discovery import build
    ee.Initialize(project=manifest["user_project"], http_transport=httplib2.Http(timeout=60))
    http = google_auth_httplib2.AuthorizedHttp(ee.data.get_persistent_credentials(), http=httplib2.Http(timeout=float(manifest["limits"]["http_timeout_s"])))
    storage = build("storage", "v1", http=http, cache_discovery=False)
    res = fetch(manifest, Path(a.dest), storage, a.only)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
