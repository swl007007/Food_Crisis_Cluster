# Print matching SOURCE-cell lines (code/markdown only; outputs never read) from notebooks.
import json, re, sys, glob
pat = re.compile(sys.argv[1])
for path in sys.argv[2:]:
    try:
        nb = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        print("ERR", path, e); continue
    for ci, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") not in ("code", "markdown"): continue
        src = c.get("source", ""); src = "".join(src) if isinstance(src, list) else src
        for li, line in enumerate(src.splitlines()):
            if pat.search(line):
                print(f"{path.split('/')[-1]}:cell{ci}:line{li+1}: {line.strip()[:220]}")
