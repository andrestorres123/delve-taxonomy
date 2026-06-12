"""Unified console output with verbosity levels."""

from __future__ import annotations

from contextlib import contextmanager
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, Generator, Optional

if TYPE_CHECKING:
    from rich.console import Console as RichConsole
    from rich.status import Status


class Verbosity(IntEnum):
    """Verbosity levels for console output."""

    SILENT = 0  # SDK default - no output
    QUIET = 1  # Errors only
    NORMAL = 2  # Spinners + checkmarks (CLI default)
    VERBOSE = 3  # Progress bars with ETA
    DEBUG = 4  # Everything + internal state


class Console:
    """Unified output manager respecting verbosity levels.

    Provides consistent output across CLI and SDK with configurable verbosity.
    Uses rich for styled terminal output when verbosity warrants it.

    Args:
        verbosity: The verbosity level. Defaults to SILENT for SDK usage.
    """

    def __init__(self, verbosity: Verbosity = Verbosity.SILENT) -> None:
        """Initialize console with given verbosity level."""
        self.verbosity = verbosity
        self._rich_console: Optional[RichConsole] = None
        self._current_status: Optional[Status] = None

    def _get_rich(self) -> RichConsole:
        """Lazily initialize and return the rich console."""
        if self._rich_console is None:
            from rich.console import Console as RichConsole

            self._rich_console = RichConsole()
        return self._rich_console

    @contextmanager
    def status(self, message: str) -> Generator[Optional[Status], None, None]:
        """Show spinner during operation.

        Args:
            message: Status message to display.

        Yields:
            The Status object (NORMAL+) or None (below NORMAL).
        """
        if self.verbosity >= Verbosity.NORMAL:
            with self._get_rich().status(message) as status:
                self._current_status = status
                try:
                    yield status
                finally:
                    self._current_status = None
        else:
            yield None

    def update_status(self, message: str) -> None:
        """Update current status message.

        Args:
            message: New status message.
        """
        if self._current_status:
            self._current_status.update(message)

    def success(self, message: str) -> None:
        """Show success checkmark.

        Args:
            message: Success message to display.
        """
        if self.verbosity >= Verbosity.NORMAL:
            self._get_rich().print(f"[green]\u2713[/green] {message}")

    def error(self, message: str) -> None:
        """Show error message to stderr.

        Args:
            message: Error message to display.
        """
        if self.verbosity >= Verbosity.QUIET:
            from rich.console import Console as RichConsole

            err_console = RichConsole(stderr=True)
            err_console.print(f"[red]\u2717[/red] {message}")

    def warning(self, message: str) -> None:
        """Show warning message (DEBUG only).

        Args:
            message: Warning message to display.
        """
        if self.verbosity >= Verbosity.DEBUG:
            self._get_rich().print(f"[yellow]![/yellow] {message}")

    def debug(self, message: str) -> None:
        """Show debug information (DEBUG only).

        Args:
            message: Debug message to display.
        """
        if self.verbosity >= Verbosity.DEBUG:
            self._get_rich().print(f"[dim]{message}[/dim]")

    def info(self, message: str) -> None:
        """Show info message (VERBOSE+).

        Args:
            message: Info message to display.
        """
        if self.verbosity >= Verbosity.VERBOSE:
            self._get_rich().print(message)

    @contextmanager
    def progress(
        self, total: int, description: str
    ) -> Generator[Callable[[int], None], None, None]:
        """Show progress bar (VERBOSE+) or spinner (NORMAL).

        Args:
            total: Total number of items to process.
            description: Description of the operation.

        Yields:
            An advance function that increments progress by n (default 1).
        """
        if self.verbosity >= Verbosity.VERBOSE:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self._get_rich(),
            ) as progress:
                task = progress.add_task(description, total=total)
                yield lambda n=1: progress.advance(task, n)
        elif self.verbosity >= Verbosity.NORMAL:
            # Just show spinner, no bar
            with self.status(description):
                yield lambda n=1: None  # No-op advance
        else:
            yield lambda n=1: None  # Silent

    def print(self, message: str = "") -> None:
        """Print a message (NORMAL+).

        Args:
            message: Message to print.
        """
        if self.verbosity >= Verbosity.NORMAL:
            self._get_rich().print(message)

    def rule(self, title: str = "") -> None:
        """Print a horizontal rule (VERBOSE+).

        Args:
            title: Optional title for the rule.
        """
        if self.verbosity >= Verbosity.VERBOSE:
            self._get_rich().rule(title)


class NullConsole(Console):
    """Silent console for SDK usage - all operations are no-ops."""

    def __init__(self) -> None:
        """Initialize a silent console."""
        super().__init__(Verbosity.SILENT)
