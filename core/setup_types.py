"""Shared interfaces for the startup setup window and installers."""

from dataclasses import dataclass
from typing import Callable, Literal


Progress = Callable[[int, int | None], None]
InstallLog = Callable[[str], None]


class DownloadInterrupted(RuntimeError):
    """A temporary connection failure during an install."""


@dataclass(frozen=True)
class InstallationItem:
    key: str
    label: str
    category: Literal["library", "model"]
    is_ready: Callable[[], bool]
    install: Callable[[Progress], None]
    install_with_log: Callable[[Progress, InstallLog], None] | None = None
