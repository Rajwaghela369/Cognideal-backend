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

from pydantic import BaseModel, ConfigDict, Field


ORM = ConfigDict(from_attributes=True)
WRITE = ConfigDict(extra="forbid")


T = TypeVar("T")


class ListQuery(BaseModel):
    """Base for the filter model of a **paginated** list endpoint.

    `limit` and `offset` live here, inside the filter model, rather than in a
    separate `Depends(pagination)`. That is not a style preference -- the two
    arrangements are mutually exclusive, and the other one does not work.

    A filter model bound with `Annotated[Filters, Query()]` is validated
    against the *whole* query string, so with `extra="forbid"` every parameter
    the model does not declare is rejected before any other dependency is
    consulted. `?limit=2` was therefore a 422 on every paginated endpoint
    while the response envelope kept reporting `"limit": 50` -- the
    dependency's default, echoed back. It looked exactly like a working
    paginated API serving page one, which is the only page it could serve.

    Dropping `extra="forbid"` would also have fixed it, and would have given
    back the silent typo that rule exists to catch: `?stalled_dayz=30`
    returning the entire unfiltered pipeline with a 200. Keeping the rule and
    moving the two fields inside the model costs nothing and keeps both
    guarantees.

    **Inherit this only where the handler actually paginates.** An endpoint
    that returns a bare list must keep rejecting `?limit=`, because silently
    accepting a window it does not apply is the same lie in the other
    direction.
    """

    model_config = WRITE

    #: Bounded, not unbounded: `le=200` is what stops one request asking for
    #: the entire table. The default matches what every client gets today.
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class Page(BaseModel, Generic[T]):
    """Envelope for every list endpoint.

    `total` is the count *before* limit/offset, so a table can render page
    numbers without a second request.
    """

    items: List[T]
    total: int
    limit: int
    offset: int
