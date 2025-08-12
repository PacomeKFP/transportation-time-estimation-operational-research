from typing import Optional, Tuple

import numpy as np
from app.models.orda_result import OrdaResult
from app.models.tdg import TDG
from app.models.intersection import Intersection
from app.models.edge import Edge
from app.models.path import Path
from app.constants import INFINITY

class ObjectOrdaSolver:
    
    @staticmethod
    def solve(tdg: TDG, 
            source: Intersection, 
            destination: Intersection, 
            time_window: Tuple[int, int]) -> OrdaResult:
        """
        Algorithme Massah adapté aux dataclasses
        
        Args:
            graph: Le graphe avec intersections et arêtes
            source: Intersection source
            destination: Intersection destination
            time_window: (t_min, t_max) fenêtre de temps
            weight_function: fonction qui calcule le poids d'une arête au temps t
        
        Returns:
            (t_star, path, total_cost): meilleur temps de départ, chemin optimal et coût total
        """
        T = time_window
        
        # Initialiser G_l : g[intersection][t] = temps d'arrivée minimal
        g = dict()
        for intersection in tdg.intersections:
            g[intersection] = dict()
            for t in range(T[0], T[1]):
                g[intersection][t] = INFINITY

        # Initialiser H_k_l : h[origin][extremity][t] = temps d'arrivée via cette arête
        h = dict()
        for intersection in tdg.intersections:
            h[intersection] = dict()
        
        for edge in tdg.edges:
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
        max_iterations = len(tdg.intersections) * (T[1] - T[0])
        
        while iteration < max_iterations:
            iteration += 1
            
            # Mettre à jour les h_k_l(t)
            for edge in tdg.edges:
                for t in range(T[0], T[1]):
                    if g[edge.origin][t] < INFINITY:
                        # Calculer le temps de voyage sur cette arête
                        travel_time = tdg.weight_function(edge, g[edge.origin][t])
                        arrival_time = g[edge.origin][t] + travel_time
                        h[edge.origin][edge.extremity][t] = arrival_time

            # Vérifier la convergence
            can_stop = True

            # Mettre à jour les g_l(t)
            for intersection in tdg.intersections:
                predecessors = tdg.get_predecessors(intersection)
                
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
            return OrdaResult(t_star, None, INFINITY)

        # Construire le chemin optimal
        path_edges = []
        current_intersection = destination
        total_cost = 0.0
        current_time = t_star  # Suivre le temps actuel lors de la reconstruction
        
        while current_intersection != source:
            found = False
            
            for edge in tdg.edges:
                if edge.extremity == current_intersection:
                    expected_arrival = h[edge.origin][edge.extremity][t_star]
                    
                    if abs(expected_arrival - g[current_intersection][t_star]) < 1e-10:
                        path_edges.insert(0, edge)
                        
                        # Calculer le coût de cette arête en utilisant le temps d'arrivée à l'origine
                        departure_time = g[edge.origin][t_star]
                        if departure_time < INFINITY:
                            edge_cost = tdg.weight_function(edge, departure_time)
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
                return OrdaResult(t_star, None, INFINITY)

        # Créer l'objet Path
        optimal_path = Path(edges=path_edges, total_cost=np.float32(total_cost))
        
        return OrdaResult(t_star, optimal_path, g[destination][t_star])