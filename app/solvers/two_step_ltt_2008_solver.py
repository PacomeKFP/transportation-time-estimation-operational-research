"""
Implémentation de l'algorithme TWO-STEP-LTT de Ding et al. (2008)
"Finding Time-Dependent Shortest Paths over Large Graphs"

Implémente l'algorithme TWO-STEP-LTT qui améliore l'approche OR (Orda-Rom)
en découplant path-selection et time-refinement pour une meilleure efficacité.

L'algorithme fonctionne en deux étapes :
1. Time Refinement : Calcul des fonctions d'arrivée gi(t) avec Dijkstra-based
2. Path Selection : Sélection rapide du chemin optimal
"""

from typing import Optional, Tuple, List, Dict, Set, Callable
import numpy as np
from collections import defaultdict
import heapq

from app.models.orda_result import OrdaResult
from app.models.tdg import TDG
from app.models.intersection import Intersection
from app.models.edge import Edge
from app.models.path import Path
from app.constants import INFINITY


class TwoStepLTT2008Solver:
    """
    Implémentation fidèle de l'algorithme TWO-STEP-LTT de Ding et al. (2008)
    """
    
    @staticmethod
    def _is_fifo_graph(tdg: TDG, time_domain: Tuple[float, float]) -> bool:
        """
        Vérifie si le graphe satisfait la propriété FIFO.
        FIFO: wi,j(t1) ≤ t∆ + wi,j(t1 + t∆) pour t∆ ≥ 0
        
        Args:
            tdg: Graphe time-dependent
            time_domain: Domaine temporel
            
        Returns:
            True si le graphe est FIFO
        """
        T_min, T_max = time_domain
        test_points = np.linspace(T_min, T_max - 1, 20)
        deltas = [0.1, 0.5, 1.0, 2.0]
        
        for edge in tdg.edges:
            for t0 in test_points:
                for t_delta in deltas:
                    if t0 + t_delta <= T_max:
                        w_t0 = tdg.weight_function(edge, t0)
                        w_t0_delta = tdg.weight_function(edge, t0 + t_delta)
                        
                        # Propriété FIFO : w(t0) ≤ t_delta + w(t0 + t_delta)
                        if w_t0 > t_delta + w_t0_delta + 1e-6:
                            return False
        return True

    @staticmethod
    def timeRefinement(tdg: TDG, 
                      source: Intersection, 
                      destination: Intersection,
                      time_interval: Tuple[float, float]) -> Dict[Intersection, Callable[[float], float]]:
        """
        Algorithme 3: timeRefinement du paper TWO-STEP-LTT
        
        Calcule les fonctions d'arrivée earliest gi(t) pour tous les nœuds
        en utilisant une approche Dijkstra-based avec raffinement incrémental.
        
        Args:
            tdg: Graphe time-dependent
            source: Nœud source
            destination: Nœud destination  
            time_interval: Intervalle temporel [ts, te]
            
        Returns:
            Dictionnaire {nœud: fonction gi(t)} des temps d'arrivée
        """
        ts, te = time_interval
        
        # Initialisation des fonctions gi(t) et des intervalles τi
        g = {}  # g[node] = fonction d'arrivée
        tau = {}  # tau[node] = borne droite de l'intervalle raffiné
        
        # gs(t) = t pour le source (ligne 1)
        g[source] = lambda t: t if ts <= t <= te else INFINITY
        tau[source] = ts
        
        # gi(t) = ∞ pour les autres nœuds (lignes 2-3)
        for node in tdg.intersections:
            if node != source:
                g[node] = lambda t: INFINITY
                tau[node] = ts
        
        # Priority queue Q contenant (τi, gi(t)) ordonné par gi(τi) (ligne 4)
        Q = []
        for node in tdg.intersections:
            priority = g[node](tau[node])
            heapq.heappush(Q, (priority, tau[node], node, g[node]))
        
        # Boucle principale (lignes 5-19)
        while len(Q) >= 2:
            # Dequeue le nœud avec la plus petite arrivée (ligne 6)
            _, tau_i, node_i, g_i = heapq.heappop(Q)
            
            # Head de Q pour obtenir gk(τk) (ligne 7)
            if Q:
                priority_k, tau_k, node_k, g_k = Q[0]
                g_k_tau_k = g_k(tau_k)
            else:
                g_k_tau_k = INFINITY
            
            # Calcul de Δ = min{wf,i(gk(τk)) | (vf, vi) ∈ E} (ligne 8)
            delta = INFINITY
            for edge in tdg.edges:
                if edge.extremity == node_i:
                    w_val = tdg.weight_function(edge, g_k_tau_k)
                    delta = min(delta, w_val)
            
            if delta == INFINITY:
                delta = 0
            
            # Calcul de τ'i = max{t | gi(t) ≤ gk(τk) + Δ} (ligne 9)
            tau_prime_i = te
            target_val = g_k_tau_k + delta
            
            # Recherche du τ'i optimal
            test_times = np.linspace(tau_i, te, 1000)
            for t in test_times:
                if g_i(t) <= target_val:
                    tau_prime_i = t
                else:
                    break
            
            # Mise à jour des fonctions gj(t) pour les voisins (lignes 10-13)
            updated_nodes = set()
            
            for edge in tdg.edges:
                if edge.origin == node_i:
                    neighbor = edge.extremity
                    
                    # Création de la nouvelle fonction g'j(t) (ligne 11)
                    def make_updated_g(old_g_func, edge_ref, g_i_func):
                        def new_g(t):
                            if tau_i <= t <= tau_prime_i:
                                # g'j(t) = gi(t) + wi,j(gi(t))
                                g_i_val = g_i_func(t)
                                if g_i_val < INFINITY:
                                    w_val = tdg.weight_function(edge_ref, g_i_val)
                                    return g_i_val + w_val
                                return INFINITY
                            else:
                                return old_g_func(t)
                        return new_g
                    
                    # gj(t) = min{gj(t), g'j(t)} (ligne 12)
                    old_g_j = g[neighbor]
                    new_g_j_partial = make_updated_g(old_g_j, edge, g_i)
                    
                    def make_min_g(old_func, new_partial_func):
                        def min_g(t):
                            return min(old_func(t), new_partial_func(t))
                        return min_g
                    
                    g[neighbor] = make_min_g(old_g_j, new_g_j_partial)
                    updated_nodes.add(neighbor)
            
            # Mise à jour de τi (ligne 14)
            tau[node_i] = tau_prime_i
            
            # Vérification de la condition de terminaison (lignes 15-19)
            if tau[node_i] >= te:
                if node_i == destination:
                    # Terminaison avec succès (ligne 17)
                    return g
                else:
                    # Continuer avec les autres nœuds (ligne 19)
                    pass
            else:
                # Remettre le nœud dans la queue pour traitement ultérieur
                priority = g[node_i](tau[node_i])
                heapq.heappush(Q, (priority, tau[node_i], node_i, g[node_i]))
            
            # Mettre à jour Q pour les nœuds modifiés (ligne 13)
            new_Q = []
            for priority, tau_val, node, g_func in Q:
                if node in updated_nodes:
                    new_priority = g[node](tau[node])
                    new_Q.append((new_priority, tau[node], node, g[node]))
                else:
                    new_Q.append((priority, tau_val, node, g_func))
            
            Q = new_Q
            heapq.heapify(Q)
        
        return g

    @staticmethod
    def pathSelection(tdg: TDG,
                     g_functions: Dict[Intersection, Callable[[float], float]],
                     source: Intersection,
                     destination: Intersection, 
                     optimal_start_time: float) -> Optional[Path]:
        """
        Algorithme 2: pathSelection du paper TWO-STEP-LTT
        
        Sélectionne le chemin optimal basé sur les fonctions gi(t) calculées
        et le temps de départ optimal t*.
        
        Args:
            tdg: Graphe time-dependent
            g_functions: Fonctions d'arrivée gi(t) calculées par timeRefinement
            source: Nœud source
            destination: Nœud destination
            optimal_start_time: Temps de départ optimal t*
            
        Returns:
            Chemin optimal ou None si aucun chemin trouvé
        """
        path_edges = []
        current_node = destination
        total_cost = 0.0
        
        # Algorithme de reconstruction backward (lignes 3-7)
        while current_node != source:
            found_predecessor = False
            
            # Pour chaque arête entrante (ligne 4)
            for edge in tdg.edges:
                if edge.extremity == current_node:
                    predecessor = edge.origin
                    
                    # Vérifier la condition gi(t*) + wi,j(gi(t*)) = gj(t*) (ligne 5)
                    g_i_val = g_functions[predecessor](optimal_start_time)
                    g_j_val = g_functions[current_node](optimal_start_time)
                    
                    if g_i_val < INFINITY:
                        w_i_j_val = tdg.weight_function(edge, g_i_val)
                        expected_arrival = g_i_val + w_i_j_val
                        
                        # Tolérance numérique pour la comparaison
                        if abs(expected_arrival - g_j_val) < 1e-6:
                            # Prédécesseur trouvé (ligne 6)
                            current_node = predecessor
                            path_edges.insert(0, edge)
                            total_cost += w_i_j_val
                            found_predecessor = True
                            break
            
            if not found_predecessor:
                # Impossible de reconstruire le chemin
                return None
        
        # Retourner le chemin optimal (ligne 8)
        return Path(edges=path_edges, total_cost=np.float32(total_cost))

    @staticmethod
    def solve(tdg: TDG,
              source: Intersection,
              destination: Intersection,
              time_window: Tuple[float, float]) -> OrdaResult:
        """
        Algorithme 1: TWO-STEP-LTT principal du paper
        
        Résout le problème TDSP en deux étapes :
        1. Time Refinement pour calculer les fonctions gi(t)
        2. Path Selection pour trouver le chemin optimal
        
        Args:
            tdg: Graphe time-dependent
            source: Nœud source
            destination: Nœud destination
            time_window: Fenêtre temporelle [ts, te]
            
        Returns:
            OrdaResult avec la solution optimale
        """
        ts, te = time_window
        
        # Vérification optionnelle FIFO
        is_fifo = TwoStepLTT2008Solver._is_fifo_graph(tdg, time_window)
        if not is_fifo:
            print("Attention: Le graphe n'est pas FIFO. L'algorithme peut ne pas être optimal.")
        
        # Étape 1: Time Refinement (ligne 1)
        g_functions = TwoStepLTT2008Solver.timeRefinement(tdg, source, destination, time_window)
        
        # Vérification de l'atteignabilité (ligne 2)
        g_e = g_functions[destination]
        if all(g_e(t) == INFINITY for t in np.linspace(ts, te, 100)):
            # Pas de chemin trouvé (ligne 6)
            return OrdaResult(ts, None, INFINITY)
        
        # Trouver le temps de départ optimal t* (ligne 3)
        # t* = argmin[t∈T]{ge(t) - t}
        best_start_time = ts
        best_travel_time = INFINITY
        
        test_times = np.linspace(ts, te, 1000)
        for t in test_times:
            arrival_time = g_e(t)
            if arrival_time < INFINITY:
                travel_time = arrival_time - t
                if travel_time < best_travel_time:
                    best_travel_time = travel_time
                    best_start_time = t
        
        if best_travel_time >= INFINITY:
            return OrdaResult(ts, None, INFINITY)
        
        # Étape 2: Path Selection (ligne 4)
        optimal_path = TwoStepLTT2008Solver.pathSelection(
            tdg, g_functions, source, destination, best_start_time
        )
        
        if optimal_path is None:
            return OrdaResult(best_start_time, None, INFINITY)
        
        # Retourner la solution (ligne 5)
        return OrdaResult(best_start_time, optimal_path, best_travel_time)

    @staticmethod
    def solve_with_waiting_analysis(tdg: TDG,
                                   source: Intersection,
                                   destination: Intersection,
                                   time_window: Tuple[float, float]) -> Tuple[OrdaResult, Dict]:
        """
        Version étendue qui retourne aussi l'analyse des temps d'attente.
        
        Returns:
            Tuple (OrdaResult, analyse_dict) où analyse_dict contient :
            - 'g_functions': Fonctions gi(t) calculées
            - 'is_fifo': Booléen indiquant si le graphe est FIFO
            - 'waiting_times': Temps d'attente à chaque nœud
        """
        result = TwoStepLTT2008Solver.solve(tdg, source, destination, time_window)
        
        # Calculer l'analyse supplémentaire
        g_functions = TwoStepLTT2008Solver.timeRefinement(tdg, source, destination, time_window)
        is_fifo = TwoStepLTT2008Solver._is_fifo_graph(tdg, time_window)
        
        # Calcul des temps d'attente (pour graphes FIFO, waiting = 0 selon Theorem 5.1)
        waiting_times = {}
        if result.path:
            current_time = result.t_star
            for edge in result.path.edges:
                origin_node = edge.origin
                # Pour FIFO: temps d'attente optimal = 0
                waiting_times[origin_node] = 0.0
                current_time += tdg.weight_function(edge, current_time)
        
        analysis = {
            'g_functions': g_functions,
            'is_fifo': is_fifo,
            'waiting_times': waiting_times,
            'time_complexity_estimate': f"O(({len(tdg.intersections)} log {len(tdg.intersections)} + {len(tdg.edges)})α(T))",
            'space_complexity_estimate': f"O(({len(tdg.intersections)} + {len(tdg.edges)})α(T))"
        }
        
        return result, analysis