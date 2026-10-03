"""Abstract base class for all data source plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import pyarrow as pa


@dataclass
class DateRange:
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"start {self.start} must be <= end {self.end}")


class SourcePlugin(ABC):
    """
    Every data source implements this interface.
    fetch() writes raw records to the archive and returns the count of records written.
    It must be idempotent: re-running for the same date range is safe.
    """

    name: str
    terms_url: str
    rate_limit_per_second: float

    @abstractmethod
    def fetch(self, date_range: DateRange) -> int:
        """Fetch records for the given date range. Returns number of records written."""
        ...

    @property
    @abstractmethod
    def schema(self) -> pa.Schema:
        """PyArrow schema for the raw records this source produces."""
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the source is reachable and returning expected data."""
        ...
