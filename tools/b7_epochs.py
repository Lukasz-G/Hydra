"""Print the epoch wall times a Hydra run recorded, plus their median.

Usage:
  python tools/b7_epochs.py runs/<name>/metrics.jsonl
"""
import json
import statistics
import sys

path = sys.argv[1]
xs = []
with open(path, encoding="utf-8") as fh:
    for line in fh:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if "epoch_seconds" in row:
            xs.append(float(row["epoch_seconds"]))

print("HYDRA_EPOCH_SECONDS", [round(x, 1) for x in xs])
if xs:
    print("HYDRA_EPOCH_MEDIAN %.1f" % statistics.median(xs))
    print("HYDRA_EPOCH_MEDIAN_MIN %.2f" % (statistics.median(xs) / 60.0))
