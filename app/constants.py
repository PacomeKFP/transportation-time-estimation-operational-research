
from typing import Callable, Literal, Annotated, TypeVar
from numpy.typing import NDArray
import numpy as np

INFINTY = float("inf")
DELTA_T = 0.1  # seconds
MAX_NUMBER = 999
STATE_DIMENSION = 6
DType = TypeVar("DType", bound=np.generic)
Array6 = Annotated[NDArray[DType], Literal[STATE_DIMENSION, 1]]
Matrix6 = Annotated[Callable[[NDArray[DType]], NDArray[DType]], Literal[STATE_DIMENSION, STATE_DIMENSION]]

MAX_NUMBER = 999