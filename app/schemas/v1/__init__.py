"""Version 1 of the wire contract -- mirrors app/api/v1/routes/.

Versioned because these models *are* the contract, not just a convenience for
building it: `extra="forbid"` on DealCreate is a promise about what v1 accepts,
and a v2 that accepts one more field is a genuinely different model that has to
coexist with this one.

When v2 arrives it does NOT copy this package. A v2 module imports whatever is
unchanged:

    from app.schemas.v1.deal.core import DealCounts, DealListItem   # unchanged
    class DealDetail(BaseModel): ...                                # changed

Copying every model per version is how a codebase ends up with four
near-identical definitions of one shape and no way to tell which differ on
purpose.
"""
