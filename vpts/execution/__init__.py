"""Forward paper-walk execution layer — survivorship-free evidence, paper only.

See :mod:`vpts.execution.paper_trader`. This package never places a real order or
moves money; it logs the system's dated decisions and resolves them, first-touch,
against the bars that subsequently arrive.
"""
from vpts.execution.paper_trader import (
    PaperLedger,
    PaperOrder,
    build_order,
    resolve_order,
    run_paper_walk,
)

__all__ = [
    "PaperOrder",
    "PaperLedger",
    "build_order",
    "resolve_order",
    "run_paper_walk",
]
