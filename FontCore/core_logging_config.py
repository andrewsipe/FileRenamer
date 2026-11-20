"""Unified logging and output management for font processing.

Combines Python logging for diagnostics with console_styles-powered HandlerAPI
for UX events. Tracks metrics and prints a final summary.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from pathlib import Path
from typing import Dict, Optional, Tuple


class Verbosity(IntEnum):
    """Verbosity levels following Ubuntu CLI guidelines."""

    QUIET = 0  # Minimal output, errors only
    BRIEF = 1  # Normal user interface messages (default)
    VERBOSE = 2  # Descriptive, thorough descriptions
    DEBUG = 3  # Internal execution steps, developer-focused
    TRACE = 4  # System-generated information


VERBOSITY_TO_LEVEL = {
    Verbosity.QUIET: logging.ERROR,
    Verbosity.BRIEF: logging.INFO,
    Verbosity.VERBOSE: logging.INFO,
    Verbosity.DEBUG: logging.DEBUG,
    Verbosity.TRACE: logging.DEBUG,
}

# Backward compatibility aliases
Verbosity.LOW = Verbosity.QUIET
Verbosity.NORMAL = Verbosity.BRIEF
Verbosity.HIGH = Verbosity.VERBOSE


class MetricsTracker:
    def __init__(self) -> None:
        self.processed: int = 0
        self.saved: int = 0
        self.skipped: int = 0
        self.errors: int = 0
        self.naming_methods: Dict[str, int] = {}
        # Extraction usage metrics
        self.extraction_method_counts: Dict[str, int] = {}
        self.extraction_method_sites: Dict[str, Dict[str, int]] = {}

    def increment(self, metric: str) -> None:
        if hasattr(self, metric):
            setattr(self, metric, getattr(self, metric) + 1)
            if metric in {"saved", "skipped", "errors"}:
                self.processed += 1

    def track_naming_method(self, method: str) -> None:
        if not method:
            return
        self.naming_methods[method] = self.naming_methods.get(method, 0) + 1

    def track_extraction_usage(
        self, method_label: Optional[str], site_label: Optional[str]
    ) -> None:
        if not method_label or not site_label:
            return
        method = method_label
        sites = self.extraction_method_sites.setdefault(method, {})
        self.extraction_method_counts[method] = (
            self.extraction_method_counts.get(method, 0) + 1
        )
        sites[site_label] = sites.get(site_label, 0) + 1


class HandlerAPI:
    """
    User-facing extraction steps (per font, at default verbosity):

    1. ``discovered`` — format and *how* the bytes were obtained (network, Base64, …).
    2. ``mapping`` (verbose only) — *how the filename was chosen* (URL, CSS family, …).
    3. ``saved`` — final path on disk.

    Arrows are avoided in ``mapping`` so lines are not read as “rename A to B”.
    """

    def __init__(self, verbosity: Verbosity, metrics: MetricsTracker) -> None:
        self.verbosity = verbosity
        self.metrics = metrics

    def discovered(
        self,
        filename: str,
        format_str: str,
        source: Optional[str] = None,
        handler_name: Optional[str] = None,
    ) -> None:
        if self.verbosity < Verbosity.BRIEF:
            return
        import FontCore.core_console_styles as cs

        if source:
            # Use provided casing/label as-is (already normalized upstream)
            source_display = source
            cs.StatusIndicator("discovered").add_file(filename).add_message(
                f"{format_str.upper()} · {source_display}"
            ).emit()
        else:
            cs.StatusIndicator("discovered").add_file(filename).add_message(
                f"{format_str.upper()}"
            ).emit()

    def mapping(
        self,
        key: str,
        value: str,
        handler_name: Optional[str] = None,
        context: Optional[str] = None,
    ) -> None:
        if self.verbosity < Verbosity.VERBOSE:
            return
        import FontCore.core_console_styles as cs

        prefix = f"[{handler_name}] " if handler_name else ""
        msg = f"{prefix}{key}: {value}"
        if context:
            msg += f" ({context})"
        cs.StatusIndicator("mapping").add_message(msg).emit()

    def saved(
        self,
        filepath: Path,
        naming_method: str,
        handler_name: Optional[str] = None,
        extraction_method: Optional[str] = None,
        site_label: Optional[str] = None,
    ) -> None:
        import FontCore.core_console_styles as cs

        cs.StatusIndicator("saved").add_file(str(filepath), filename_only=True).emit()
        self.metrics.increment("saved")
        self.metrics.track_naming_method(naming_method)
        if extraction_method and site_label:
            self.metrics.track_extraction_usage(extraction_method, site_label)

    def info(
        self,
        message: str,
        handler_name: Optional[str] = None,
        verbose_only: bool = True,
    ) -> None:
        min_level = Verbosity.VERBOSE if verbose_only else Verbosity.BRIEF
        if self.verbosity < min_level:
            return
        import FontCore.core_console_styles as cs

        prefix = f"[{handler_name}] " if handler_name else ""
        cs.StatusIndicator("info").add_message(f"{prefix}{message}").emit()

    def warning(
        self,
        message: str,
        handler_name: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> None:
        if self.verbosity < Verbosity.BRIEF:
            return
        import FontCore.core_console_styles as cs

        indicator = cs.StatusIndicator("warning")
        if filename:
            indicator.add_file(filename)
        prefix = f"[{handler_name}] " if handler_name else ""
        indicator.with_explanation(f"{prefix}{message}").emit()

    def error(
        self,
        message: str,
        handler_name: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> None:
        import FontCore.core_console_styles as cs

        indicator = cs.StatusIndicator("error")
        if filename:
            indicator.add_file(filename)
        prefix = f"[{handler_name}] " if handler_name else ""
        indicator.with_explanation(f"{prefix}{message}").emit()
        self.metrics.increment("errors")


def print_summary(metrics: MetricsTracker, console=None) -> None:
    if metrics.processed == 0:
        return
    import FontCore.core_console_styles as cs

    # Access current verbosity to gate bullet sections
    current_verbosity = Verbosity.BRIEF
    try:
        from typing import cast

        global _handler_api
        if _handler_api is not None:
            current_verbosity = cast(HandlerAPI, _handler_api).verbosity
    except Exception:
        pass
    cs.emit(f"\n{'=' * 60}", console=console)
    cs.StatusIndicator("info").add_message("Font extraction finished").emit(console)
    cs.emit("", console=console)
    cs.fmt_processing_summary(
        dry_run=False,
        updated=metrics.saved,
        unchanged=metrics.skipped,
        errors=metrics.errors,
        console=console,
    )
    if metrics.naming_methods:
        # Bullet summary only for BRIEF/VERBOSE verbosity
        if current_verbosity >= Verbosity.BRIEF:
            cs.StatusIndicator("info").add_message(
                "How output filenames were chosen:"
            ).emit(console)

            # Separate site handlers from core categories
            site_handlers = {}
            core_methods = {}
            for method, count in metrics.naming_methods.items():
                if method.startswith("Site handler:"):
                    # Extract handler class label after prefix
                    label = method.split(":", 1)[1].strip()
                    site_handlers[label] = site_handlers.get(label, 0) + count
                else:
                    core_methods[method] = core_methods.get(method, 0) + count

            # Alphabetize core categories (by label)
            for label, count in sorted(
                core_methods.items(), key=lambda x: x[0].lower()
            ):
                cs.emit(
                    f"{cs.indent(1)}• {label}: {cs.fmt_count(count)}", console=console
                )

            # Group site handlers alphabetically
            if site_handlers:
                cs.emit(f"{cs.indent(1)}• Site handlers:", console=console)
                for label, count in sorted(
                    site_handlers.items(), key=lambda x: x[0].lower()
                ):
                    cs.emit(
                        f"{cs.indent(2)}- {label}: {cs.fmt_count(count)}",
                        console=console,
                    )

        # Else: at LOW, omit bullets entirely

    # Extraction methods grouped by site
    if metrics.extraction_method_counts and current_verbosity >= Verbosity.BRIEF:
        cs.StatusIndicator("info").add_message(
            "Where fonts were detected (by site):"
        ).emit(console)

        # Methods alphabetized
        for method in sorted(
            metrics.extraction_method_counts.keys(), key=lambda x: x.lower()
        ):
            cs.emit(f"{cs.indent(1)}• {method}:", console=console)
            site_counts = metrics.extraction_method_sites.get(method, {})
            # Sites sorted by count desc
            for site, count in sorted(
                site_counts.items(), key=lambda x: x[1], reverse=True
            ):
                cs.emit(f"{cs.indent(2)}- {site} ({count})", console=console)
    cs.emit(f"{'=' * 60}\n", console=console)


_logger: Optional[logging.Logger] = None
_handler_api: Optional[HandlerAPI] = None
_metrics: Optional[MetricsTracker] = None
_initialized: bool = False


def setup_logging(
    verbosity: Verbosity = Verbosity.BRIEF, console=None
) -> Tuple[logging.Logger, HandlerAPI, MetricsTracker]:
    global _logger, _handler_api, _metrics, _initialized
    level = VERBOSITY_TO_LEVEL[verbosity]

    if _initialized:
        if _handler_api:
            _handler_api.verbosity = verbosity
        logging.getLogger().setLevel(level)
        if _logger is not None:
            _logger.setLevel(level)
        return (_logger, _handler_api, _metrics)

    _metrics = MetricsTracker()
    _handler_api = HandlerAPI(verbosity, _metrics)

    # Standard StreamHandler prints Rich markup literally (e.g. "[orange1]…[/orange1]").
    # RichHandler with markup=True renders the same tags as StatusIndicator / emit().
    rich_handlers = []
    try:
        from FontCore.core_console_styles import RICH_AVAILABLE, get_console

        if RICH_AVAILABLE:
            rich_console = get_console()
            if rich_console is not None:
                from rich.logging import RichHandler

                rich_handlers.append(
                    RichHandler(
                        console=rich_console,
                        level=level,
                        markup=True,
                        rich_tracebacks=True,
                        show_time=False,
                        show_path=False,
                        show_level=True,
                    )
                )
    except Exception:
        rich_handlers = []

    if rich_handlers:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=rich_handlers,
            force=True,
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(levelname)s: %(message)s",
            force=True,
        )

    _logger = logging.getLogger("FontExtractor")
    _logger.setLevel(level)
    _initialized = True
    return (_logger, _handler_api, _metrics)


_cached_api: Optional[HandlerAPI] = None
_cached_metrics: Optional[MetricsTracker] = None


def get_api() -> HandlerAPI:
    """Lazy accessor for the structured logging API singleton."""
    global _cached_api
    if _cached_api is None:
        _, _cached_api, _ = setup_logging()
    return _cached_api


def get_metrics() -> MetricsTracker:
    """Lazy accessor for the logging metrics singleton."""
    global _cached_metrics
    if _cached_metrics is None:
        _, _, _cached_metrics = setup_logging()
    return _cached_metrics


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
