from dataclasses import dataclass


@dataclass
class OrdaResult:
    t_star: int
    path: list
    total_cost: float