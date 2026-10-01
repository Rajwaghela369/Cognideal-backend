"""The guard ``app/models/__init__.py`` promises in its docstring.

The failure it prevents is silent: ``alembic/env.py`` does
``from app.models import *``, which respects ``__all__``, so a model left out of
that list is invisible to autogenerate. No error -- the table simply never gets
created, and it is discovered much later by a query that returns nothing.
"""

import app.models as registry
from app.db.base import Base


def _mapped_classes():
    return [
        mapper.class_
        for mapper in Base.registry.mappers
        if getattr(mapper.class_, "__tablename__", None)
    ]


def test_every_table_has_a_mapped_class():
    mapped_tables = {cls.__tablename__ for cls in _mapped_classes()}
    for table in Base.metadata.tables:
        assert table in mapped_tables, "table %r has no mapped class" % table


def test_every_mapped_class_is_exported():
    """The actual failure mode: present in the module, absent from __all__."""
    for cls in _mapped_classes():
        assert cls.__name__ in registry.__all__, (
            "%s is mapped but missing from app.models.__all__, so alembic "
            "autogenerate cannot see its table" % cls.__name__
        )


def test_star_import_reaches_every_table():
    namespace = {}
    exec("from app.models import *", namespace)  # noqa: S102 -- this is the thing under test
    reachable = {
        value.__tablename__
        for value in namespace.values()
        if hasattr(value, "__tablename__")
    }
    assert reachable == set(Base.metadata.tables)


def test_all_entries_exist():
    """A stale name in __all__ breaks the star import outright."""
    for name in registry.__all__:
        assert hasattr(registry, name), "__all__ names %r, which does not exist" % name
