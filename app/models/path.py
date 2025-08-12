import numpy as np
from typing import List
from dataclasses import dataclass
from app.models.edge import Edge


@dataclass
class Path:
    edges: List[Edge]
    total_cost: np.float32