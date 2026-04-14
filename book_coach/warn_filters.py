"""LangChain still touches Pydantic v1 paths; on Python 3.14+ that emits a noisy UserWarning."""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)
