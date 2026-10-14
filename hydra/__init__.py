"""Hydra: a tagger-lemmatiser for historical German.

    import hydra
    nlp = hydra.load("de-4corpus")
    doc = nlp("Was daz mere vil lip")

`load` and the classes it returns live in `hydra.pipeline`; they are imported
lazily here so that `import hydra` costs nothing until a model is asked for,
and so the training code can import this package without pulling in torch.
"""
__version__ = "2.0.0"

__all__ = ["load", "Hydra", "Doc", "Token", "__version__"]


def __getattr__(name):
    if name in ("load", "Hydra", "Doc", "Token"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
