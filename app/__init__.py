"""The gateway package.

`__version__` lives here and nowhere else. It was hardcoded in three places —
`pyproject.toml`, `app/main.py`'s FastAPI constructor, and implicitly in
whatever a reader assumed — which is how a release ends up reporting a version
it is not. `scripts/check_docs.py` asserts they agree.
"""

__version__ = "0.5.0"
