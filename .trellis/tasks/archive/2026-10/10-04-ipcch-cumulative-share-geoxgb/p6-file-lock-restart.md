# User-authorized P6 restart after file lock

User selected: “我暂停 Dropbox 同步，再用冻结代码和新 run ID 重跑”.

The failed scientific run `p6-formal-20261004` remains intact and incomplete.
Its first H1/G1 quartet reached model persistence, then Windows reported
WinError32 at the model-directory rename. Dropbox was running, but the precise
locking process was not independently identified; do not claim that attribution
as proven. No Stage1 map was completed.

Use unchanged implementation6798df2, unchanged scientific configs, pinned
Windows environment and the same registered executor/audit run/base. The user
will pause Dropbox synchronization. First perform a small no-fit file-write /
directory-rename/read check under the existing runs directory; record result.
Then use a NEW scientific run ID for fresh prepare and learn-map, retaining
the agreed map/request-budget checkpoint before predict.

Do not change package code, add retry logic, redirect the output root, mark
Dropbox ignore streams, delete the failed attempt, overwrite its incomplete
markers or reuse its partial model entry as a completed fit. This permission
is the unchanged-recipe restart selected by the user. If the file operation
fails again, stop and report with evidence rather than looping retries.

Record this operational decision plus old/new scientific run IDs in task
evidence. The Trellis audit run is not restarted or rebound.
