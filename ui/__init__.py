"""Terminal presentation for AEGIS.

Presentation is deliberately separate from orchestration and tool policy so
the CLI can evolve without creating a second execution path.
"""

from .terminal import TerminalUI

__all__ = ["TerminalUI"]
