"""
Implémentation des vrais algorithmes d'Orda & Rom (1990)
"Shortest-Path and Minimum-Delay Algorithms in Networks with Time-Dependent Edge-Length"

Implémente les 4 algorithmes authentiques :
- UW1 : Unrestricted Waiting pour temps de départ donné
- UW2 : Unrestricted Waiting pour intervalle de temps  
- SW1 : Source Waiting pour temps de départ donné
- SW2 : Source Waiting pour intervalle de temps
"""

from typing import Optional, Tuple, List, Dict, Callable
import numpy as np
from collections import defaultdict
import heapq

from app.models.orda_result import OrdaResult
from app.models.tdg import TDG
from app.models.intersection import Intersection
from app.models.edge import Edge
from app.models.path import Path
from app.constants import INFINITY


class OrdaRom1990Solver:
    """
    Implémentation fidèle des algorithmes d'Orda & Rom (1990)
    """
    
    @staticmethod
    def _compute_optimal_waiting_function(edge: Edge, tdg: TDG, time_domain: Tuple[float, float]) -> Callable[[float], float]:
        """
        Calcule la fonction Dik(t) = min{τ + dik(t + τ) | τ ≥ 0}
        Cette fonction représente le délai total optimal (attente + voyage) pour traverser une arête.
        
        Args:
            edge: L'arête (vi, vj)
            tdg: Le graphe time-dependent
            time_domain: Domaine temporel (t_min, t_max)
            
        Returns:
            Fonction Dik(t) qui donne le délai optimal pour partir à l'instant t
        """
        def Dik(t: float) -> float:
            if t < time_domain[0] or t > time_domain[1]:
                return INFINITY
            
            # Pour chaque temps d'attente possible τ ≥ 0
            min_delay = INFINITY
            
            # Échantillonnage des temps d'attente (approximation discrète)
            max_wait = min(24.0, time_domain[1] - t)  # Max 24h d'attente
            wait_samples = np.linspace(0, max_wait, 100)
            
            for tau in wait_samples:
                departure_time = t + tau
                if departure_time <= time_domain[1]:
                    travel_time = tdg.weight_function(edge, departure_time)
                    total_delay = tau + travel_time
                    min_delay = min(min_delay, total_delay)
            
            return min_delay if min_delay < INFINITY else INFINITY
        
        return Dik

    @staticmethod
    def solve_UW1(tdg: TDG, 
                  source: Intersection, 
                  destination: Intersection, 
                  start_time: float,
                  time_domain: Tuple[float, float] = (0, 100)) -> OrdaResult:
        """
        Algorithme UW1 (Unrestricted Waiting) d'Orda & Rom
        Pour un temps de départ donné, trouve le chemin optimal avec attente libre.
        
        Implémentation fidèle de l'Algorithm UW1 du paper original.
        
        Args:
            tdg: Graphe dépendant du temps
            source: Nœud source 
            destination: Nœud destination
            start_time: Temps de départ fixé
            time_domain: Domaine temporel pour les fonctions
            
        Returns:
            OrdaResult avec chemin optimal, temps et coût
        """
        # Étape 1: Précalculer les fonctions Dik(t) pour toutes les arêtes
        edge_delay_functions = {}
        for edge in tdg.edges:
            edge_delay_functions[edge] = OrdaRom1990Solver._compute_optimal_waiting_function(
                edge, tdg, time_domain
            )
        
        # Étape 2: Algorithme UW1 - version Dijkstra généralisée
        # Initialisation
        X = {}  # X[node] = temps d'arrivée minimal permanent
        Y = {}  # Y[node] = temps d'arrivée minimal temporaire
        f = {}  # f[node] = prédécesseur dans l'arbre optimal
        
        # Initialisation du source
        X[source] = start_time
        f[source] = None
        
        # Initialisation des autres nœuds
        for node in tdg.intersections:
            if node != source:
                Y[node] = INFINITY
                f[node] = None
        
        # Nœud courant
        current = source
        
        # Algorithme principal UW1
        while current is not None:
            # Pour tous les voisins du nœud courant
            for edge in tdg.edges:
                if edge.origin == current:
                    neighbor = edge.extremity
                    
                    # Si le voisin n'a pas encore de label permanent
                    if neighbor not in X:
                        # Calculer le nouveau temps d'arrivée via cette arête
                        Dik_func = edge_delay_functions[edge]
                        new_arrival = X[current] + Dik_func(X[current])
                        
                        # Mettre à jour si c'est mieux
                        if new_arrival < Y.get(neighbor, INFINITY):
                            Y[neighbor] = new_arrival
                            f[neighbor] = current
            
            # Trouver le prochain nœud à traiter (label permanent)
            next_node = None
            min_arrival = INFINITY
            
            for node in tdg.intersections:
                if node not in X and Y.get(node, INFINITY) < min_arrival:
                    min_arrival = Y[node]
                    next_node = node
            
            if next_node is not None:
                X[next_node] = Y[next_node]
                current = next_node
            else:
                current = None
        
        # Vérifier si la destination est atteignable
        if destination not in X:
            return OrdaResult(start_time, None, INFINITY)
        
        # Reconstruction du chemin optimal
        path_edges = []
        current_node = destination
        total_cost = 0.0
        
        while f[current_node] is not None:
            predecessor = f[current_node]
            
            # Trouver l'arête entre predecessor et current_node
            connecting_edge = None
            for edge in tdg.edges:
                if edge.origin == predecessor and edge.extremity == current_node:
                    connecting_edge = edge
                    break
            
            if connecting_edge:
                path_edges.insert(0, connecting_edge)
                # Calculer le coût de cette arête
                Dik_func = edge_delay_functions[connecting_edge]
                edge_cost = Dik_func(X[predecessor])
                total_cost += edge_cost
            
            current_node = predecessor
        
        # Créer le chemin optimal
        optimal_path = Path(edges=path_edges, total_cost=np.float32(total_cost))
        arrival_time = X[destination]
        travel_time = arrival_time - start_time
        
        return OrdaResult(start_time, optimal_path, travel_time)

    @staticmethod
    def solve_UW2(tdg: TDG,
                  source: Intersection,
                  destination: Intersection, 
                  time_interval: Tuple[float, float]) -> OrdaResult:
        """
        Algorithme UW2 (Unrestricted Waiting) d'Orda & Rom
        Pour un intervalle de temps de départ, trouve le temps optimal et le chemin.
        
        Utilise des fonctions au lieu de valeurs scalaires.
        """
        T_min, T_max = time_interval
        
        # Précalculer les fonctions Dik pour toutes les arêtes
        edge_delay_functions = {}
        for edge in tdg.edges:
            edge_delay_functions[edge] = OrdaRom1990Solver._compute_optimal_waiting_function(
                edge, tdg, time_interval
            )
        
        # Fonctions d'arrivée Xi(t) et Yik(t)
        X = {}  # X[node] = fonction temps d'arrivée
        Y = {}  # Y[(node_from, node_to)] = fonction temps d'arrivée via arête
        
        # Initialisation des fonctions
        for node in tdg.intersections:
            if node == source:
                # Pour le source: X_s(t) = t
                X[node] = lambda t: t if T_min <= t <= T_max else INFINITY
            else:
                # Pour les autres: X_i(t) = ∞
                X[node] = lambda t: INFINITY
        
        # Initialiser les fonctions Y pour toutes les arêtes
        for edge in tdg.edges:
            Y[(edge.origin, edge.extremity)] = lambda t: INFINITY
        
        # Algorithme UW2 - itérations jusqu'à convergence
        max_iterations = len(tdg.intersections) * 10
        converged = False
        
        for iteration in range(max_iterations):
            if converged:
                break
            
            old_functions = {node: X[node] for node in X}
            
            # Mise à jour des fonctions Yik(t)
            for edge in tdg.edges:
                origin, extremity = edge.origin, edge.extremity
                Dik_func = edge_delay_functions[edge]
                
                def make_Y_function(orig_func, delay_func):
                    def Y_func(t):
                        if T_min <= t <= T_max:
                            x_val = orig_func(t)
                            if x_val < INFINITY:
                                return x_val + delay_func(x_val)
                        return INFINITY
                    return Y_func
                
                Y[(origin, extremity)] = make_Y_function(X[origin], Dik_func)
            
            # Mise à jour des fonctions Xi(t)
            for node in tdg.intersections:
                if node != source:
                    # Trouver tous les prédécesseurs de ce nœud
                    predecessors = []
                    for edge in tdg.edges:
                        if edge.extremity == node:
                            predecessors.append(edge.origin)
                    
                    if predecessors:
                        def make_X_function(preds, node_dest):
                            def X_func(t):
                                if T_min <= t <= T_max:
                                    min_val = INFINITY
                                    for pred in preds:
                                        y_val = Y[(pred, node_dest)](t)
                                        min_val = min(min_val, y_val)
                                    return min_val
                                return INFINITY
                            return X_func
                        
                        X[node] = make_X_function(predecessors, node)
            
            # Vérifier la convergence (approximation)
            converged = True
            test_points = np.linspace(T_min, T_max, 10)
            for node in tdg.intersections:
                for t in test_points:
                    if abs(X[node](t) - old_functions[node](t)) > 1e-6:
                        converged = False
                        break
                if not converged:
                    break
        
        # Trouver le temps de départ optimal
        test_times = np.linspace(T_min, T_max, 100)
        best_time = T_min
        best_travel_time = INFINITY
        
        for t in test_times:
            arrival_time = X[destination](t)
            if arrival_time < INFINITY:
                travel_time = arrival_time - t
                if travel_time < best_travel_time:
                    best_travel_time = travel_time
                    best_time = t
        
        if best_travel_time >= INFINITY:
            return OrdaResult(T_min, None, INFINITY)
        
        # Utiliser UW1 pour reconstruire le chemin optimal au temps optimal
        return OrdaRom1990Solver.solve_UW1(tdg, source, destination, best_time, time_interval)

    @staticmethod
    def solve_SW1(tdg: TDG,
                  source: Intersection,
                  destination: Intersection,
                  start_time: float,
                  time_domain: Tuple[float, float] = (0, 100)) -> OrdaResult:
        """
        Algorithme SW1 (Source Waiting) d'Orda & Rom
        Attente autorisée uniquement au source, temps de départ donné.
        
        Basé sur le Theorem 3 du paper: pour des fonctions continues,
        SW donne le même résultat que UW.
        """
        # Étape 1: Résoudre avec UW1 pour obtenir le chemin topologique optimal
        uw_result = OrdaRom1990Solver.solve_UW1(tdg, source, destination, start_time, time_domain)
        
        if uw_result.path is None:
            return OrdaResult(start_time, None, INFINITY)
        
        # Étape 2: Calculer les temps de départ pour chaque nœud sans attente intermédiaire
        # (Algorithme SW1 du paper)
        
        path_edges = uw_result.path.edges
        arrival_times = {}
        departure_times = {}
        
        # Calculer les temps d'arrivée le long du chemin UW
        current_time = start_time
        current_node = source
        arrival_times[source] = start_time
        departure_times[source] = start_time
        
        for edge in path_edges:
            travel_time = tdg.weight_function(edge, departure_times[edge.origin])
            arrival_time = departure_times[edge.origin] + travel_time
            arrival_times[edge.extremity] = arrival_time
            departure_times[edge.extremity] = arrival_time  # Pas d'attente intermédiaire
        
        # Étape 3: Calculer le temps d'attente source nécessaire (backward)
        target_arrival = arrival_times[destination]
        
        # Recalculer backward pour trouver le temps de départ source optimal
        current_arrival = target_arrival
        
        for edge in reversed(path_edges):
            # Trouver le temps de départ qui mène à current_arrival
            origin_node = edge.origin
            
            # Résoudre: departure_time + weight(edge, departure_time) = current_arrival
            # Approximation par recherche
            best_departure = start_time
            min_error = INFINITY
            
            for test_departure in np.linspace(time_domain[0], time_domain[1], 1000):
                predicted_arrival = test_departure + tdg.weight_function(edge, test_departure)
                error = abs(predicted_arrival - current_arrival)
                if error < min_error:
                    min_error = error
                    best_departure = test_departure
            
            current_arrival = best_departure
        
        # Temps d'attente source
        source_waiting_time = max(0, current_arrival - start_time)
        optimal_departure_time = start_time + source_waiting_time
        
        # Recalculer le coût total avec le nouveau timing
        total_cost = 0.0
        current_time = optimal_departure_time
        
        for edge in path_edges:
            travel_time = tdg.weight_function(edge, current_time)
            total_cost += travel_time
            current_time += travel_time
        
        # Créer le résultat avec le chemin et le timing SW
        sw_path = Path(edges=path_edges, total_cost=np.float32(total_cost))
        final_arrival_time = optimal_departure_time + total_cost
        travel_time = final_arrival_time - start_time
        
        return OrdaResult(optimal_departure_time, sw_path, travel_time)

    @staticmethod
    def solve_SW2(tdg: TDG,
                  source: Intersection,
                  destination: Intersection,
                  time_interval: Tuple[float, float]) -> OrdaResult:
        """
        Algorithme SW2 (Source Waiting) d'Orda & Rom
        Attente autorisée uniquement au source, intervalle de temps.
        
        Utilise UW2 puis calcule les fonctions d'attente source WAIT(s, w, t).
        """
        # Étape 1: Résoudre avec UW2
        uw2_result = OrdaRom1990Solver.solve_UW2(tdg, source, destination, time_interval)
        
        if uw2_result.path is None:
            return uw2_result
        
        # Étape 2: Pour le SW, le résultat est équivalent pour des fonctions continues
        # (Theorem 3 et Corollary 2 du paper)
        
        # Le temps de départ optimal est déjà calculé par UW2
        # Pour SW, on utilise ce temps avec attente source si nécessaire
        
        return OrdaRom1990Solver.solve_SW1(
            tdg, source, destination, uw2_result.t_star, time_interval
        )

    @staticmethod  
    def solve(tdg: TDG,
              source: Intersection,
              destination: Intersection,
              time_window: Tuple[float, float],
              algorithm: str = "UW2") -> OrdaResult:
        """
        Interface unifiée pour tous les algorithmes d'Orda & Rom (1990)
        
        Args:
            tdg: Graphe time-dependent
            source: Nœud source
            destination: Nœud destination  
            time_window: Intervalle de temps de départ
            algorithm: "UW1", "UW2", "SW1", "SW2"
            
        Returns:
            OrdaResult avec la solution optimale
        """
        start_time = time_window[0]
        
        if algorithm == "UW1":
            return OrdaRom1990Solver.solve_UW1(tdg, source, destination, start_time, time_window)
        elif algorithm == "UW2":
            return OrdaRom1990Solver.solve_UW2(tdg, source, destination, time_window)
        elif algorithm == "SW1":
            return OrdaRom1990Solver.solve_SW1(tdg, source, destination, start_time, time_window)
        elif algorithm == "SW2":
            return OrdaRom1990Solver.solve_SW2(tdg, source, destination, time_window)
        else:
            raise ValueError(f"Algorithme inconnu: {algorithm}. Utilisez UW1, UW2, SW1, ou SW2")