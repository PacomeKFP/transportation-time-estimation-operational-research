from dataclasses import dataclass
from typing import Set, List, Optional, Tuple, Callable
import numpy as np

INFINITY = 1e700

@dataclass
class Chunk:
    length: np.float32
    
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

@dataclass
class Path:
    edges: List[Edge]
    total_cost: np.float32

@dataclass
class Graph:
    intersections: Set[Intersection]
    edges: Set[Edge]
    
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

def massah(graph: Graph, 
           source: Intersection, 
           destination: Intersection, 
           time_window: Tuple[int, int],
           weight_function: Callable[[Edge, float], float]) -> Tuple[int, Optional[Path]]:
    """
    Algorithme Massah adapté aux dataclasses
    
    Args:
        graph: Le graphe avec intersections et arêtes
        source: Intersection source
        destination: Intersection destination
        time_window: (t_min, t_max) fenêtre de temps
        weight_function: fonction qui calcule le poids d'une arête au temps t
    
    Returns:
        (t_star, path): meilleur temps de départ et chemin optimal
    """
    T = time_window
    
    # Initialiser G_l : g[intersection][t] = temps d'arrivée minimal
    g = dict()
    for intersection in graph.intersections:
        g[intersection] = dict()
        for t in range(T[0], T[1]):
            g[intersection][t] = INFINITY

    # Initialiser H_k_l : h[origin][extremity][t] = temps d'arrivée via cette arête
    h = dict()
    for intersection in graph.intersections:
        h[intersection] = dict()
    
    for edge in graph.edges:
        if edge.origin not in h:
            h[edge.origin] = dict()
        h[edge.origin][edge.extremity] = dict()
        for t in range(T[0], T[1]):
            h[edge.origin][edge.extremity][t] = INFINITY

    # Initialiser le nœud source
    for t in range(T[0], T[1]):
        g[source][t] = t  # Temps d'arrivée = temps de départ pour le source

    # Boucle principale
    iteration = 0
    max_iterations = len(graph.intersections) * (T[1] - T[0])
    
    while iteration < max_iterations:
        iteration += 1
        
        # Mettre à jour les h_k_l(t)
        for edge in graph.edges:
            for t in range(T[0], T[1]):
                if g[edge.origin][t] < INFINITY:
                    # Calculer le temps de voyage sur cette arête
                    travel_time = weight_function(edge, g[edge.origin][t])
                    arrival_time = g[edge.origin][t] + travel_time
                    h[edge.origin][edge.extremity][t] = arrival_time

        # Vérifier la convergence
        can_stop = True

        # Mettre à jour les g_l(t)
        for intersection in graph.intersections:
            predecessors = graph.get_predecessors(intersection)
            
            for t in range(T[0], T[1]):
                if predecessors:
                    new_value = min(h[pred][intersection][t] for pred in predecessors)
                    if abs(g[intersection][t] - new_value) > 1e-10:
                        can_stop = False
                    g[intersection][t] = new_value

        if can_stop:
            break

    # Récupérer le meilleur instant de départ
    t_star = T[0]
    for t in range(T[0], T[1]):
        if g[destination][t] < g[destination][t_star]:
            t_star = t

    # Si pas de chemin trouvé
    if g[destination][t_star] >= INFINITY:
        return t_star, None

    # Construire le chemin optimal
    path_edges = []
    current_intersection = destination
    total_cost = 0.0
    current_time = t_star  # Suivre le temps actuel lors de la reconstruction
    
    while current_intersection != source:
        found = False
        
        for edge in graph.edges:
            if edge.extremity == current_intersection:
                expected_arrival = h[edge.origin][edge.extremity][t_star]
                
                if abs(expected_arrival - g[current_intersection][t_star]) < 1e-10:
                    path_edges.insert(0, edge)
                    
                    # Calculer le coût de cette arête en utilisant le temps d'arrivée à l'origine
                    departure_time = g[edge.origin][t_star]
                    if departure_time < INFINITY:
                        edge_cost = weight_function(edge, departure_time)
                    else:
                        # Si le temps de départ est infini, utiliser un poids par défaut
                        edge_cost = float(edge.chunks[0].length) if edge.chunks else 1.0
                    
                    total_cost += edge_cost
                    current_intersection = edge.origin
                    found = True
                    break
        
        if not found:
            # Impossible de reconstruire le chemin
            print(f"DEBUG: Impossible de trouver le prédécesseur de {current_intersection.name}")
            return t_star, None

    # Créer l'objet Path
    optimal_path = Path(edges=path_edges, total_cost=np.float32(total_cost))
    
    return t_star, optimal_path


# Exemple d'utilisation
def exemple_reproduction_exacte():
    """Reproduction exacte de l'exemple original en orienté objet"""
    
    # Créer les intersections correspondant aux nœuds 0, 1, 2, 3, 4, 5
    intersections = {
        Intersection("0"),
        Intersection("1"), 
        Intersection("2"),
        Intersection("3"),
        Intersection("4"),
        Intersection("5")
    }
    
    # Créer un mapping pour faciliter la création des arêtes
    node_map = {name: intersection for intersection in intersections 
                for name in [intersection.name]}
    
    # Créer les arêtes exactement comme dans l'exemple original
    edges_data = [
        (1, 0, [2.0, 1.5]),      
        (0, 1, [2.0, 1.5]),      
        (1, 3, [1.5]),           
        (0, 2, [1.0, 1.0, 1.0]), 
        (1, 4, [3.0, 2.0]),      
        (2, 3, [2.0]),           
        (2, 4, [1.5, 1.0]),      
        (3, 5, [2.5, 1.5]),      
        (4, 5, [1.0, 1.0]),      
        (1, 5, [6.0]),           
    ]
    
    edges = set()
    for origin_id, extremity_id, chunk_values in edges_data:
        origin = node_map[str(origin_id)]
        extremity = node_map[str(extremity_id)]
        chunks = [Chunk(np.float32(value)) for value in chunk_values]
        edges.add(Edge(origin, extremity, chunks))
    
    # Créer le graphe
    graph = Graph(intersections, edges)
    
    # Fonction de poids EXACTEMENT identique à l'exemple original
    def weight_function(edge: Edge, departure_time: float) -> float:
        """Fonction de poids identique: sum(chunks) * cos(π * departure_time)"""
        # Vérifier que departure_time est valide
        if not np.isfinite(departure_time) or departure_time >= INFINITY:
            return float('inf')
        
        # Calculer la somme des chunks (équivalent à edge[2] dans l'original)
        chunk_sum = sum(float(chunk.length) for chunk in edge.chunks)
        
        # Appliquer la même formule
        weight = chunk_sum * np.cos(np.pi * departure_time)
        
        return float(weight)
    
    print("=== REPRODUCTION EXACTE DE L'EXEMPLE ORIGINAL ===")
    print("Graphe:")
    print("  Intersections:", sorted([i.name for i in intersections]))
    print("  Arêtes:")
    for edge in sorted(edges, key=lambda e: (e.origin.name, e.extremity.name)):
        chunk_values = [float(c.length) for c in edge.chunks]
        print(f"    {edge.origin.name} -> {edge.extremity.name}: {chunk_values}")
    print()
    
    # Chercher le plus court chemin EXACTEMENT comme l'original
    source = node_map["0"]      # Nœud 0
    destination = node_map["5"] # Nœud 5  
    time_window = (0, 5)        # Fenêtre (0, 5)
    
    print(f"Recherche du plus court chemin de {source.name} à {destination.name}")
    print(f"Fenêtre de temps: {time_window}")
    print(f"Fonction de poids: sum(chunks) * cos(π * t)")
    print()
    
    # Exécuter l'algorithme
    t_star, optimal_path = massah(graph, source, destination, time_window, weight_function)
    
    print("=== RÉSULTAT ===")
    print(f"Meilleur temps de départ: {t_star}")
    
    if optimal_path:
        print(f"Chemin optimal: {[(e.origin.name, e.extremity.name) for e in optimal_path.edges]}")
        print(f"Coût total: {optimal_path.total_cost:.6f}")
        
        # Affichage détaillé du chemin
        print("\nDétail du chemin:")
        current_time = t_star
        for i, edge in enumerate(optimal_path.edges):
            weight = weight_function(edge, current_time)
            chunk_values = [float(c.length) for c in edge.chunks]
            print(f"  {edge.origin.name} -> {edge.extremity.name}: chunks={chunk_values}, "
                  f"poids={weight:.6f} au temps {current_time}")
            current_time += weight
            
        print(f"\nTemps d'arrivée final: {current_time:.6f}")
    else:
        print("Chemin optimal: []")
        print("Aucun chemin trouvé!")
    
    print()
    
    # Comparaison avec différents temps de départ pour vérifier la fonction cosinus
    print("=== ANALYSE DE LA FONCTION DE POIDS ===")
    test_edge = list(edges)[0]  # Prendre une arête pour tester
    chunk_sum = sum(float(c.length) for c in test_edge.chunks)
    
    print(f"Test sur l'arête {test_edge.origin.name}->{test_edge.extremity.name} (chunks={[float(c.length) for c in test_edge.chunks]})")
    print(f"sum(chunks) = {chunk_sum}")
    print("Évolution du poids selon le temps:")
    
    for t in [0, 0.5, 1.0, 1.5, 2.0]:
        weight = weight_function(test_edge, t)
        cos_value = np.cos(np.pi * t)
        print(f"  t={t}: cos(π*{t})={cos_value:.3f}, poids={weight:.3f}")


if __name__ == "__main__":
    exemple_reproduction_exacte()
