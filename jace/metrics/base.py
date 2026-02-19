"""Shared types and XML helpers for metric extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from xml.etree.ElementTree import Element


@dataclass
class ExtractedMetric:
    """A single metric extracted from command output."""
    metric: str
    value: float
    unit: str = ""
    tags: dict = field(default_factory=dict)
    is_counter: bool = False


# ── XML helpers (handle Junos namespace-prefixed elements) ───────────

def xml_findall(element: Any, tag: str) -> list:
    """Find all descendant elements by tag name, ignoring XML namespaces."""
    return element.findall(f".//{{*}}{tag}")


def xml_findtext(element: Any, tag: str, default: str = "") -> str:
    """Find text content of first matching descendant, ignoring namespaces."""
    text = element.findtext(f".//{{*}}{tag}")
    return text.strip() if text is not None else default


def xml_float(element: Any, tag: str, default: float = 0.0) -> float:
    """Extract a float value from an XML element by tag name."""
    text = xml_findtext(element, tag)
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default
