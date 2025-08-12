from dataclasses import dataclass
from typing import Optional, Set, Tuple, List, Callable

from trash.massah_dataclass import Edge
from app.intersection import Intersection


@dataclass
class TDG:
    intersections: Set[Intersection]
    edges: Set[Edge]
    weight_function: Callable[[Edge, float], float]
    
    def get_predecessors(self, intersection: Intersection) -> List[Intersection]:
        """Retourne tous les prédécesseurs d'une intersection"""
        predecessors = []
        for edge in self.edges:
            if edge.extremity == intersection:
                predecessors.append(edge.origin)
        return predecessors
    
    def find_edge(self, origin: Intersection, extremity: Intersection) -> Optional[Edge]:
        """Trouve l'arête entre deux intersections"""
        for edge in self.edges:
            if edge.origin == origin and edge.extremity == extremity:
                return edge
        return None