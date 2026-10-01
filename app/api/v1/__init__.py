"""API version 1.

Versioning lives at the routing layer: app/api/v1/ holds the routers, and
app/api/deps.py stays outside it because pagination and session handling are
not version-specific.

`schemas/` is likewise unversioned for now. When a breaking change forces a v2,
only the models that actually changed get copied under schemas/v2/ -- copying
all of them per version is how a codebase ends up with four near-identical
copies of the same shape.
"""

from app.api.v1.router import api_router

__all__ = ["api_router"]
