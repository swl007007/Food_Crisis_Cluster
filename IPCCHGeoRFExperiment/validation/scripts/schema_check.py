import sys,re,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'IPCCHGeoRFExperiment'))
import prepare_data as p
s=(Path(__file__).parent/'.trellis/tasks/archive/2026-09/09-19-ipcch-binary-georf-pipeline/research/secondary-predictors.md').read_text(encoding="utf-8")
b=re.findall(r"```text\n(.*?)```",s,re.S)
names=[x.strip() for block in b[:6] for x in block.splitlines() if x.strip()]
assert names==list(p.RAW_FEATURE_COLUMNS),(len(names),names)
print('PASS approved whitelist exact 70 names/order')
