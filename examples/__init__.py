"""CPU-friendly walkthroughs for the personal GMTM / fusion code.

Import these as ``python -m examples.<module>`` from the repository root.
They add ``model/`` to ``sys.path`` via :mod:`examples.common` and never
touch CMU pickle files unless you pass an explicit path.
"""

__all__ = [
    "common",
    "synthetic_data",
    "eval_metrics",
]