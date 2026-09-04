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
from json_minify import json_minify  # pie's own defaults file is JSONC (// comments)

data_dir, model_dir, out_json = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
name = sys.argv[4] if len(sys.argv) > 4 else data_dir.name

defaults_text = (Path(pie.__file__).parent / "default_settings.json").read_text()
defaults = json.loads(json_minify(defaults_text))
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
        # each task needs its OWN "schedule" dict (even if empty): pie's
        # settings.merge_task_defaults() does `task[k] = task.get(k, default)`
        # without copying, so any task missing "schedule" shares the SAME
        # dict object as every other task missing it. TaskScheduler then does
        # `tasks[name] = task.get('schedule', {}); tasks[name]['target'] = ...`
        # against that shared object, so setting pos/morph's target=False
        # after lemma's target=True silently clobbers lemma's flag back to
        # False too (verified by reproducing settings_from_file() + check_settings()
        # + TaskScheduler() on the actual generated settings.json). With
        # target permanently False, pie never saves a checkpoint (torch.save
        # is gated on is_target) and patience/early-stop can never fire
        # either -- training would run to the full epochs=500 cap and
        # produce nothing. Giving each task its own dict avoids the aliasing.
        {"name": "lemma", "target": True, "context": "sentence", "level": "char",
         "decoder": "attentional", "schedule": {},
         "settings": {"bos": True, "eos": True, "lower": False, "target": "lemma"}},
        {"name": "pos", "schedule": {}},
        {"name": "morph", "schedule": {}},
    ],
    "include_lm": True,   # their joint-LM auxiliary (Manjavacas et al. 2019)
    "device": "cuda",
    "verbose": True,
    # NOTE: do NOT set run_test=True here -- despite the name, pie's
    # scripts/train.py gates its final model.save() on `not settings.run_test`
    # (run_test=True is meant for hyperparameter-search runs that don't need
    # a persisted model). The test-set evaluation print ("Evaluating best
    # model on test set") happens unconditionally whenever test_path is set,
    # regardless of run_test, so the default (False) already gives us both
    # the test score AND a saved model.
    # pie's own default (epochs=500, patience=100) is unboundedly expensive;
    # with the schedule-aliasing bug above fixed, pie's checkpointing and
    # early-stop now actually work, so this tighter budget costs nothing in
    # model quality, only wall-clock (mirrors the RNNTagger patience wrapper)
    "patience": 15,
})
model_dir.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(defaults, indent=1), encoding="utf-8")
print("wrote", out_json)
