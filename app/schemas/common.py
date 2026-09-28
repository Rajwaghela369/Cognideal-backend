from typing import Generic, List, TypeVar

from pydantic import BaseModel

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
