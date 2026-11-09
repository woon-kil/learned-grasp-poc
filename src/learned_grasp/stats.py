"""시험 집계."""

from __future__ import annotations

import math


def wilson_interval(successes: int, total: int, z: float = 1.959963985) -> tuple[float, float]:
    """이항 비율의 Wilson 구간. 합격선의 입증이 아니라 불확실성 표시다."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("successes와 total이 유효하지 않다")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def percentile(values: list[float], q: float) -> float:
    """선형 보간 백분위."""
    if not values:
        raise ValueError("값이 없다")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
