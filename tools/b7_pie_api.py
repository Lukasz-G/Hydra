"""Print the installed pie's public surface.

Batch 7 has to get per-token predictions out of a trained pie model so
tools/score_baseline.py can score it the way RNNTagger is scored. Guessing
the API would waste a 12-hour training run, so the driver prints what is
actually there while the training runs, and the prediction step is written
against this output.
"""
import inspect
import pkgutil

import pie

print("version", getattr(pie, "__version__", "?"))
print("file", pie.__file__)
print("top", [n for n in dir(pie) if not n.startswith("_")])
print("submodules", sorted(m.name for m in pkgutil.iter_modules(pie.__path__)))

for name in ("tagger", "models", "models.model", "settings", "scripts.tag"):
    try:
        mod = __import__("pie." + name, fromlist=["x"])
    except Exception as exc:                      # noqa: BLE001
        print(name, "import failed:", exc)
        continue
    classes = [n for n, o in inspect.getmembers(mod, inspect.isclass)]
    funcs = [n for n, o in inspect.getmembers(mod, inspect.isfunction)]
    print(name, "classes:", classes)
    print(name, "functions:", funcs)

for target in ("pie.tagger.Tagger", "pie.models.BaseModel", "pie.models.SimpleModel"):
    mod_name, _, attr = target.rpartition(".")
    try:
        obj = getattr(__import__(mod_name, fromlist=["x"]), attr)
    except Exception as exc:                      # noqa: BLE001
        print(target, "unavailable:", exc)
        continue
    print("--", target)
    for meth in ("__init__", "load", "tag", "predict", "tag_file"):
        fn = getattr(obj, meth, None)
        if fn is None:
            continue
        try:
            print("   ", meth, inspect.signature(fn))
        except (TypeError, ValueError):
            print("   ", meth, "(signature unavailable)")
