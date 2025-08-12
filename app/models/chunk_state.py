from enum import Enum

import numpy as np
from app.constants import Array6

class ChunkState(Enum):
    """
    Enum representing the state of a chunk in the transportation network.
    """
    EMPTY = 0
    PARTIALLY_FILLED = 1
    FILLED = 2
    OVERLOADED = 3

    @staticmethod
    def nominal_velocities_vector() -> Array6[np.float]:
        """
        Returns a vector of nominal velocities for each chunk state.
        """
        return np.array([1.0, 0.8, 0.5, 0.2, 0.1, 0.0], dtype=float)
