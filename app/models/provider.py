from typing import Optional

import numpy as np
from app.models.intersection import Intersection
from app.models.path import Path
from app.models.tdg import dataclass


@dataclass
class Provider:
    name: str
    location: Intersection
    available_quantity: int
    departure_instant: np.float32 = 0
    best_path: Optional[Path] = None
    transportation_time: Optional[np.float32] = None

    def __str__(self) -> str:
        time_str = f"{self.transportation_time} seconds" if self.transportation_time is not None else "Not calculated"
        return f"{self.name} - {self.location.name} - {self.available_quantity} units available - Estimated time: {time_str}"
