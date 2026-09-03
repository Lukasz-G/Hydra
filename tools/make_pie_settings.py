"""Generate a Pie settings file by overriding the installed package defaults.

Usage:
  PYTHON_WITH_PIE tools/make_pie_settings.py DATA_DIR MODEL_DIR OUT_JSON [NAME]

DATA_DIR holds train/dev/test.tsv in our converted format (token/lemma/pos/
morph, tab-separated, no header, blank lines between sentences). Merging into
pie's own default_settings.json keeps us honest to their published recipe:
only the data paths, the reader shape, and the joint-LM auxiliary (their
paper's OOV lever) are pinned; every model hyperparameter stays their default.
"""
import json
import sys
from pathlib import Path

import pie

data_dir, model_dir, out_json = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
name = sys.argv[4] if len(sys.argv) > 4 else data_dir.name

defaults = json.loads((Path(pie.__file__).parent / "default_settings.json").read_text())
defaults.update({
    "modelname": name,
    "modelpath": str(model_dir),
    "input_path": str(data_dir / "train.tsv"),
    "dev_path": str(data_dir / "dev.tsv"),
    "test_path": str(data_dir / "test.tsv"),
    "sep": "\t",
    "header": False,
    "tasks_order": ["lemma", "pos", "morph"],
    "breakline_ref": None,
    "max_sent_len": 35,
    "tasks": [
        {"name": "lemma", "target": True, "context": "sentence", "level": "char",
         "decoder": "attentional", "settings": {"bos": True, "eos": True,
                                                "lower": False, "target": "lemma"}},
        {"name": "pos"},
        {"name": "morph"},
    ],
    "include_lm": True,   # their joint-LM auxiliary (Manjavacas et al. 2019)
    "device": "cuda",
    "verbose": True,
})
model_dir.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(defaults, indent=1), encoding="utf-8")
print("wrote", out_json)
