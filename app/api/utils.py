"""Shared utilities for API routers"""
import logging
from typing import List

from api.schemas import HistoryMessage

logger = logging.getLogger(__name__)


def serialize_history(history: List[HistoryMessage]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in history]


def check_history_injection(history: List[HistoryMessage], injection_guard) -> bool:
    """Return True if any history message fails injection guard."""
    for idx, msg in enumerate(history):
        try:
            if not injection_guard.check(msg.content)["is_safe"]:
                return True
        except Exception:
            logger.error("injection_guard.check() failed on history message %d", idx, exc_info=True)
            return True
    return False
