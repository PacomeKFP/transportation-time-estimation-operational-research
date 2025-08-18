from dataclasses import dataclass
from typing import Callable
import numpy as np
from app.models import chunk
from app.models.chunk import Chunk
from app.models.edge import Edge
from app.constants import DELTA_T, Array6, Matrix6
from app.models.chunk_state import ChunkState

@dataclass
class ChunkTransportationTimeEstimator:
    chunk: Chunk
    time_origin: np.float32
    states_distribution: Array6[np.float32]
    P: Matrix6[np.float32]
    time_differential: np.float32 = DELTA_T


    def estimate(self) -> np.float32:
        traveled_distance = 0.0
        time_estimation = 0.0
        while traveled_distance < self.chunk.length:
            velocity = self._compute_distributed_velocity(self.states_distribution, ChunkState.nominal_velocities_vector())
            temporal_distance = velocity * self.time_differential
            if temporal_distance + traveled_distance >= self.chunk.length:
                temporal_distance = self.chunk.length - traveled_distance
            traveling_time = temporal_distance / velocity
            time_estimation += traveling_time
            traveled_distance += temporal_distance

            self.states_distribution = self.states_distribution @ self.P(traveling_time)

        return time_estimation

    def _compute_distributed_velocity(self, distribution: Array6[np.float32], nominal_velocities: Array6[np.float32]) -> np.float32:
        return np.sum(distribution * nominal_velocities)


@dataclass
class EdgeTransportationTimeEstimator:
    states_distribution: Array6[np.float32]
    P: Matrix6[np.float32]
    time_differential: np.float32 = DELTA_T


    
    def estimate(self, edge: Edge, time_origin: np.float32) -> np.float32:
        total_time_estimation = 0.0
        for chunk in edge.chunks:
            estimator = ChunkTransportationTimeEstimator(
                chunk=chunk,
                time_origin=time_origin,
                states_distribution=self.states_distribution,
                time_differential=self.time_differential,
                P=self.P
            )
            total_time_estimation += estimator.estimate()
        return total_time_estimation