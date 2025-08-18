from app.models.chunk import Chunk
from dataclasses import dataclass
from typing import List
from app.models.intersection import Intersection

@dataclass
class Edge:
    origin: Intersection
    extremity: Intersection
    chunks: List[Chunk]
    
    def __hash__(self):
        return hash((self.origin.name, self.extremity.name))
    
    def __eq__(self, other):
        if isinstance(other, Edge):
            return (self.origin == other.origin and 
                   self.extremity == other.extremity)
        return False    