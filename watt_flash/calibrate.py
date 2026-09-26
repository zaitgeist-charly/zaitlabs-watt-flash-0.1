"""Per-bucket temperature scaling of the option logits."""
from typing import Dict

import numpy as np

from .layout import QTYPE_NAMES


def bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-20" if k <= 20 else "21+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def temperature_for(temps: Dict[str, float], qtype: int, k: int) -> float:
    return temps.get(bucket(qtype, k), temps.get(QTYPE_NAMES[int(qtype)], 1.0))


def probabilities(z: np.ndarray, qtype: int, temps: Dict[str, float]) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / temperature_for(temps, qtype, len(z))
    p = np.exp(z - z.max())
    return p / p.sum()
