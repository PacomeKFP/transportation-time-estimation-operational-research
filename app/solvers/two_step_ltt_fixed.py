"""
Implémentation corrigée de l'algorithme TWO-STEP-LTT de Ding et al. (2008)
Utilise des dictionnaires simples pour représenter les fonctions gi(t)
"""

from typing import Optional, Tuple, List, Dict
import heapq

from app.models.orda_result import OrdaResult
from app.models.tdg import TDG
from app.models.intersection import Intersection
from app.models.edge import Edge
from app.models.path import Path
from app.constants import INFINITY


class TwoStepLTTFixed:
    """
    Implémentation corrigée avec dictionnaires simples pour gi(t)
    """
    
    @staticmethod
    def timeRefinement(tdg: TDG, 
                      source: Intersection, 
                      destination: Intersection,
                      time_interval: Tuple[int, int]) -> Dict[Intersection, Dict[int, float]]:
        """
        Algorithme timeRefinement corrigé avec dictionnaires
        
        Args:
            tdg: Graphe time-dependent
            source: Nœud source
            destination: Nœud destination  
            time_interval: Intervalle temporel [ts, te] (entiers)
            
        Returns:
            Dictionnaire {nœud: {temps: valeur}} des temps d'arrivée
        """
        ts, te = time_interval
        
        # Initialisation des fonctions gi(t) comme dictionnaires
        g = {}  # g[node] = {t: valeur}
        tau = {}  # tau[node] = borne droite de l'intervalle raffiné
        
        # gs(t) = t pour le source
        g[source] = {t: float(t) for t in range(ts, te + 1)}
        tau[source] = ts
        
        # gi(t) = ∞ pour les autres nœuds
        for node in tdg.intersections:
            if node != source:
                g[node] = {t: INFINITY for t in range(ts, te + 1)}
                tau[node] = ts
        
        # Priority queue Q - avec Intersection maintenant comparable
        Q = []
        for node in tdg.intersections:
            priority = g[node][tau[node]]
            heapq.heappush(Q, (priority, tau[node], node))
        
        # Boucle principale
        iteration = 0
        while len(Q) >= 2:
            iteration += 1
            if iteration > 100:  # Protection contre boucles infinies
                break
                
            # Dequeue le nœud avec la plus petite arrivée
            _, tau_i, node_i = heapq.heappop(Q)
            
            # Head de Q pour obtenir gk(τk)
            if Q:
                priority_k, tau_k, node_k = Q[0]
                g_k_tau_k = g[node_k][tau_k]
            else:
                g_k_tau_k = INFINITY
            
            # Calcul de Δ = min{wf,i(gk(τk)) | (vf, vi) ∈ E}
            delta = INFINITY
            for edge in tdg.edges:
                if edge.extremity == node_i:
                    w_val = tdg.weight_function(edge, g_k_tau_k)
                    delta = min(delta, w_val)
            
            if delta == INFINITY:
                delta = 0
            
            # Calcul de τ'i = max{t | gi(t) ≤ gk(τk) + Δ}
            tau_prime_i = te
            target_val = g_k_tau_k + delta
            
            # Recherche du τ'i optimal sur les temps entiers
            for t in range(tau_i, te + 1):
                if g[node_i][t] <= target_val:
                    tau_prime_i = t
                else:
                    break
            
            # Mise à jour des fonctions gj(t) pour les voisins
            updated_nodes = set()
            
            for edge in tdg.edges:
                if edge.origin == node_i:
                    neighbor = edge.extremity
                    
                    # Mise à jour de gj(t) sur l'intervalle [τi, τ'i]
                    for t in range(tau_i, tau_prime_i + 1):
                        if t <= te:
                            g_i_val = g[node_i][t]
                            if g_i_val < INFINITY:
                                w_val = tdg.weight_function(edge, g_i_val)
                                new_val = g_i_val + w_val
                                g[neighbor][t] = min(g[neighbor][t], new_val)
                            
                    updated_nodes.add(neighbor)
            
            # Mise à jour de τi
            tau[node_i] = tau_prime_i
            
            # Vérification de la condition de terminaison
            if tau[node_i] >= te:
                if node_i == destination:
                    # Terminaison avec succès
                    return g
                else:
                    # Continuer avec les autres nœuds
                    pass
            else:
                # Remettre le nœud dans la queue
                priority = g[node_i][tau[node_i]]
                heapq.heappush(Q, (priority, tau[node_i], node_i))
            
            # Mettre à jour Q pour les nœuds modifiés
            new_Q = []
            for priority, tau_val, node in Q:
                if node in updated_nodes:
                    new_priority = g[node][tau[node]]
                    new_Q.append((new_priority, tau[node], node))
                else:
                    new_Q.append((priority, tau_val, node))
            
            Q = new_Q
            heapq.heapify(Q)
        
        return g

    @staticmethod
    def pathSelection(tdg: TDG,
                     g_functions: Dict[Intersection, Dict[int, float]],
                     source: Intersection,
                     destination: Intersection, 
                     optimal_start_time: int) -> Optional[Path]:
        """
        Algorithme pathSelection corrigé
        """
        path_edges = []
        current_node = destination
        total_cost = 0.0
        
        # Algorithme de reconstruction backward
        while current_node != source:
            found_predecessor = False
            
            # Pour chaque arête entrante
            for edge in tdg.edges:
                if edge.extremity == current_node:
                    predecessor = edge.origin
                    
                    # Vérifier la condition gi(t*) + wi,j(gi(t*)) = gj(t*)
                    g_i_val = g_functions[predecessor][optimal_start_time]
                    g_j_val = g_functions[current_node][optimal_start_time]
                    
                    if g_i_val < INFINITY:
                        w_i_j_val = tdg.weight_function(edge, g_i_val)
                        expected_arrival = g_i_val + w_i_j_val
                        diff = abs(expected_arrival - g_j_val)
                        
                        # Tolérance numérique élargie pour les calculs Markov (propagation d'états)
                        if diff < 5e-3:
                            # Prédécesseur trouvé
                            current_node = predecessor
                            path_edges.insert(0, edge)
                            total_cost += w_i_j_val
                            found_predecessor = True
                            break
            
            if not found_predecessor:
                # Impossible de reconstruire le chemin
                return None
        
        # Retourner le chemin optimal
        return Path(edges=path_edges, total_cost=total_cost)

    @staticmethod
    def solve(tdg: TDG,
              source: Intersection,
              destination: Intersection,
              time_window: Tuple[int, int]) -> OrdaResult:
        """
        Algorithme TWO-STEP-LTT principal corrigé
        """
        ts, te = time_window
        
        # Étape 1: Time Refinement
        g_functions = TwoStepLTTFixed.timeRefinement(tdg, source, destination, time_window)
        
        # Vérification de l'atteignabilité
        g_e = g_functions[destination]
        if all(g_e[t] == INFINITY for t in range(ts, te + 1)):
            # Pas de chemin trouvé
            return OrdaResult(ts, None, INFINITY)
        
        # Trouver le temps de départ optimal t*
        # t* = argmin[t∈T]{ge(t) - t}
        best_start_time = ts
        best_travel_time = INFINITY
        
        for t in range(ts, te + 1):
            arrival_time = g_e[t]
            if arrival_time < INFINITY:
                travel_time = arrival_time - t
                if travel_time < best_travel_time:
                    best_travel_time = travel_time
                    best_start_time = t
        
        if best_travel_time >= INFINITY:
            return OrdaResult(ts, None, INFINITY)
        
        # Étape 2: Path Selection
        optimal_path = TwoStepLTTFixed.pathSelection(
            tdg, g_functions, source, destination, best_start_time
        )
        
        if optimal_path is None:
            return OrdaResult(best_start_time, None, INFINITY)
        
        # Retourner la solution
        return OrdaResult(best_start_time, optimal_path, best_travel_time)