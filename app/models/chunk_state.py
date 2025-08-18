from enum import Enum

import numpy as np
from app.constants import Array6

class ChunkState(Enum):
    """
    Enum representing the 6 traffic states in the transportation network.
    Combines traffic density (Fluide/Dense/Embouteillé) × weather (Sec/Pluie)
    """
    FLUIDE_SEC = 0        # État 0: Fluide + Sec
    FLUIDE_PLUIE = 1      # État 1: Fluide + Pluie  
    DENSE_SEC = 2         # État 2: Dense + Sec
    DENSE_PLUIE = 3       # État 3: Dense + Pluie
    EMBOUTEILLE_SEC = 4   # État 4: Embouteillé + Sec
    EMBOUTEILLE_PLUIE = 5 # État 5: Embouteillé + Pluie

    @staticmethod
    def nominal_velocities_vector() -> Array6[np.float32]:
        """
        Returns a vector of nominal velocities for each traffic state (km/h).
        Based on ANNEXE 2-Matrices.pdf velocity vectors.
        """
        return np.array([50.0, 40.0, 30.0, 25.0, 15.0, 10.0], dtype=np.float32)
