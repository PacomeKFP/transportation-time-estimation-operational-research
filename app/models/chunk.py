import numpy as np
from dataclasses import dataclass


@dataclass
class Chunk:
    length: np.float32