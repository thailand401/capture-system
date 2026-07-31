"""Declarative base and shared mixins for ORM models."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, MetaData, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Consistent constraint/index naming convention so Alembic autogenerate
# produces stable, predictable names across all migrations.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# BigInteger in production (Postgres BIGINT), but plain Integer on SQLite.
# SQLite only auto-populates a primary key via its ROWID when the column's
# type affinity is exactly INTEGER — BIGINT does not get that treatment —
# so this keeps production types correct while letting the in-memory
# SQLite test database autoincrement primary keys correctly.
BigIntPK = BigInteger().with_variant(Integer, "sqlite")


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampMixin:
    """Adds a server-generated ``created_at`` column to a model."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
