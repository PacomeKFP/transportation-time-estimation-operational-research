from ast import Tuple
from dataclasses import dataclass, field
from typing import List, Optional


import numpy as np
from app.models.intersection import Intersection
from app.models.provider import Provider
from app.models.tdg import TDG
from app.constants import DELTA_T, MAX_NUMBER
from app.solvers.object_orda import ObjectOrdaSolver



@dataclass
class MainApplication:
    providers: List[Provider]
    location: Intersection
    map: Optional[TDG] = None
    requested_quantity: int = 0
    time_differential: int = DELTA_T
    time_window: Tuple[int, int] = (0, 100)
    preselected_providers: List[Provider] = field(default_factory=list)
    sorted_providers: List[Provider] = field(default_factory=list)

    def execute(self, top: int = 5) -> None:
        self._preselect_provider()
        self._estimate_transportation_time_for_each_provider()
        self._sort_providers()
        self._display_top_providers(top)

    def _preselect_provider(self) -> List[Provider]:
        self.preselected_providers = []
        for provider in self.providers:
            if provider.available_quantity >= self.requested_quantity:
                self.preselected_providers.append(provider)
        if not self.preselected_providers:
            raise ValueError(
                "No providers available with sufficient quantity.")
        return self.preselected_providers

    def _estimate_transportation_time_for_each_provider(self) -> None:
        for provider in self.preselected_providers:
            orda_results = ObjectOrdaSolver.solve(
                tdg=self.map,
                source=provider.location,
                destination=self.location,
                time_window=self.time_window
            )

            provider.departure_instant = orda_results.t_star
            
            provider.best_path = orda_results.path
            # TODO: verifier si le coût total est bien calculé
            provider.best_path.total_cost = orda_results.total_cost
            
            

    def _sort_providers(self) -> List[Provider]:
        self.sorted_providers = sorted(
            self.preselected_providers,
            key=lambda p: p.best_path.total_cost if p.best_path and p.best_path.total_cost is not None else MAX_NUMBER
        )
        return self.sorted_providers

    def _display_top_providers(self, number: int = 5) -> List[Provider]:
        print("The best providers are:")
        for i in range(min(number, len(self.sorted_providers))):
            provider = self.sorted_providers[i]
            print(f"{i+1}. {provider}")

        return self.sorted_providers[:number]