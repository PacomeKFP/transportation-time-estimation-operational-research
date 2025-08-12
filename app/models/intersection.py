from typing import Optional
from dataclasses import dataclass


@dataclass
class Intersection:
    name: str
    label: Optional[str] = None
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Intersection):
            return self.name == other.name
        return False
    
    def __repr__(self):
        return f"Intersection({self.name})"