"""Shared across every version of the contract.

Unversioned, at the schemas/ root rather than inside v1/ -- the same call as
api/deps.py sitting outside api/v1/. Page[T] and the two model configs are the
envelope and the house rules, not the shapes themselves. If v2 ever moves to
cursor pagination it defines its own Page; until then, one definition.


ORM   -- responses. Built from SQLAlchemy Row objects, hence from_attributes.
WRITE -- requests. ``extra="forbid"`` is load-bearing rather than tidy:
         FastAPI silently drops a field it does not recognise, so without it a
         client POSTing ``risk_level`` gets a 201 and no risk level, and
         ``GET /deals?stalled_dayz=30`` returns the whole unfiltered pipeline
         with a 200 -- which looks exactly like a working filter. With it,
         both are a 422 naming the field."""

from typing import Generic, List, TypeVar

from pydantic import BaseModel, ConfigDict


ORM = ConfigDict(from_attributes=True)
WRITE = ConfigDict(extra="forbid")


T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    """Envelope for every list endpoint.

    `total` is the count *before* limit/offset, so a table can render page
    numbers without a second request.
    """

    items: List[T]
    total: int
    limit: int
    offset: int
