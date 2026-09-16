"""Ensure ``model/`` is on ``sys.path`` before any test imports ``models``."""

from examples.common import MODEL_DIR, REPO_ROOT  # noqa: F401
