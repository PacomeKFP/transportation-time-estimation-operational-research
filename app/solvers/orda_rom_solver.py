"""
Implémentation des algorithmes ORDA et ROM pour l'optimisation des chemins
dans les graphes dépendants du temps.

ORDA (Optimal Routing Dynamic Algorithm) : Algorithme de routage optimal dynamique
ROM (Route Optimization Method) : Méthode d'optimisation de routes avec contraintes temporelles

Ces algorithmes classiques sont adaptés pour notre contexte de transport de matériaux.
"""

from typing import Optional, Tuple, List, Dict, Set
import numpy as np
from collections import defaultdict, deque
import heapq

from app.models.orda_result import OrdaResult
from app.models.tdg import TDG
from app.models.intersection import Intersection
from app.models.edge import Edge
from app.models.path import Path
from app.constants import INFINITY


class OrdaRomSolver:
    """
    Solveur combinant les algorithmes ORDA et ROM pour l'optimisation 
    des chemins dans les graphes temporels.
    """
    
    @staticmethod
    def solve_orda(tdg: TDG, 
                   source: Intersection, 
                   destination: Intersection, 
                   time_window: Tuple[int, int]) -> OrdaResult:
        """
        Algorithme ORDA (Optimal Routing Dynamic Algorithm)
        
        Principe : Calcul du plus court chemin avec propagation dynamique
        des coûts en tenant compte de la dépendance temporelle.
        
        Args:
            tdg: Graphe dépendant du temps
            source: Intersection source
            destination: Intersection destination  
            time_window: Fenêtre temporelle (t_min, t_max)
            
        Returns:
            OrdaResult avec le meilleur temps de départ, chemin et coût
        """
        T_min, T_max = time_window
        
        # Structure de données ORDA : coût[noeud][temps] = coût minimal
        cost = defaultdict(lambda: defaultdict(lambda: INFINITY))
        predecessor = defaultdict(lambda: defaultdict(lambda: None))
        
        # Initialisation du nœud source
        for t in range(T_min, T_max):
            cost[source][t] = 0.0
        
        # Propagation ORDA : traitement par ordre temporel croissant
        converged = False
        max_iterations = len(tdg.intersections) * (T_max - T_min)
        iteration = 0
        
        while not converged and iteration < max_iterations:
            iteration += 1
            converged = True
            
            # Pour chaque instant temporel
            for t in range(T_min, T_max - 1):
                
                # Pour chaque arête du graphe
                for edge in tdg.edges:
                    origin = edge.origin
                    destination_node = edge.extremity
                    
                    # Si le nœud origine est atteignable à l'instant t
                    if cost[origin][t] < INFINITY:
                        
                        # Calcul du temps de voyage sur cette arête
                        travel_time = tdg.weight_function(edge, t)
                        
                        # Temps d'arrivée au nœud destination
                        arrival_time = t + travel_time
                        
                        # Vérification que l'arrivée est dans la fenêtre temporelle
                        arrival_slot = int(arrival_time)
                        if T_min <= arrival_slot < T_max:
                            
                            # Coût total via cette arête
                            total_cost = cost[origin][t] + travel_time
                            
                            # Mise à jour si meilleur chemin trouvé
                            if total_cost < cost[destination_node][arrival_slot]:
                                cost[destination_node][arrival_slot] = total_cost
                                predecessor[destination_node][arrival_slot] = (origin, t, edge)
                                converged = False
        
        # Recherche du meilleur instant d'arrivée à destination
        best_arrival_time = T_min
        best_cost = INFINITY
        
        for t in range(T_min, T_max):
            if cost[destination][t] < best_cost:
                best_cost = cost[destination][t]
                best_arrival_time = t
        
        # Si aucun chemin trouvé
        if best_cost >= INFINITY:
            return OrdaResult(T_min, None, INFINITY)
        
        # Reconstruction du chemin optimal
        path_edges = []
        current_node = destination
        current_time = best_arrival_time
        total_path_cost = 0.0
        
        while current_node != source:
            pred_info = predecessor[current_node][current_time]
            if pred_info is None:
                # Impossible de reconstruire le chemin
                return OrdaResult(T_min, None, INFINITY)
            
            pred_node, pred_time, edge = pred_info
            path_edges.insert(0, edge)
            
            # Calculer le coût de cette arête
            edge_cost = tdg.weight_function(edge, pred_time)
            total_path_cost += edge_cost
            
            current_node = pred_node
            current_time = pred_time
        
        # Temps de départ optimal
        t_star = current_time
        
        # Créer l'objet Path
        optimal_path = Path(edges=path_edges, total_cost=np.float32(total_path_cost))
        
        return OrdaResult(t_star, optimal_path, best_cost)
    
    @staticmethod
    def solve_rom(tdg: TDG, 
                  source: Intersection, 
                  destination: Intersection, 
                  time_window: Tuple[int, int],
                  delay_constraint: Optional[float] = None) -> OrdaResult:
        """
        Algorithme ROM (Route Optimization Method)
        
        Principe : Optimisation de routes avec contraintes de délai.
        Utilise une approche de programmation dynamique avec contraintes.
        
        Args:
            tdg: Graphe dépendant du temps
            source: Intersection source
            destination: Intersection destination
            time_window: Fenêtre temporelle
            delay_constraint: Contrainte de délai maximum (optionnel)
            
        Returns:
            OrdaResult avec la solution optimale
        """
        T_min, T_max = time_window
        
        # Structure ROM : [nœud][temps] = (coût_min, délai_min)
        rom_table = defaultdict(lambda: defaultdict(lambda: (INFINITY, INFINITY)))
        parent = defaultdict(lambda: defaultdict(lambda: None))
        
        # Initialisation
        for t in range(T_min, T_max):
            rom_table[source][t] = (0.0, 0.0)  # (coût, délai)
        
        # Algorithme ROM principal
        for iteration in range(len(tdg.intersections)):
            updated = False
            
            for t in range(T_min, T_max - 1):
                for edge in tdg.edges:
                    origin = edge.origin
                    dest = edge.extremity
                    
                    current_cost, current_delay = rom_table[origin][t]
                    
                    if current_cost < INFINITY:
                        # Calcul du coût et délai sur cette arête
                        edge_cost = tdg.weight_function(edge, t)
                        edge_delay = edge_cost  # Dans ce contexte, délai = temps de voyage
                        
                        arrival_time = int(t + edge_cost)
                        if T_min <= arrival_time < T_max:
                            
                            new_cost = current_cost + edge_cost
                            new_delay = current_delay + edge_delay
                            
                            # Vérification de la contrainte de délai
                            if delay_constraint is None or new_delay <= delay_constraint:
                                
                                dest_cost, dest_delay = rom_table[dest][arrival_time]
                                
                                # Critère d'amélioration ROM : coût prioritaire, puis délai
                                improvement = False
                                if new_cost < dest_cost:
                                    improvement = True
                                elif new_cost == dest_cost and new_delay < dest_delay:
                                    improvement = True
                                
                                if improvement:
                                    rom_table[dest][arrival_time] = (new_cost, new_delay)
                                    parent[dest][arrival_time] = (origin, t, edge)
                                    updated = True
            
            if not updated:
                break
        
        # Recherche de la meilleure solution à destination
        best_time = T_min
        best_cost, best_delay = INFINITY, INFINITY
        
        for t in range(T_min, T_max):
            cost, delay = rom_table[destination][t]
            if cost < best_cost or (cost == best_cost and delay < best_delay):
                best_cost, best_delay = cost, delay
                best_time = t
        
        if best_cost >= INFINITY:
            return OrdaResult(T_min, None, INFINITY)
        
        # Reconstruction du chemin
        path_edges = []
        current_node = destination
        current_time = best_time
        total_cost = 0.0
        
        while current_node != source:
            parent_info = parent[current_node][current_time]
            if parent_info is None:
                return OrdaResult(T_min, None, INFINITY)
            
            parent_node, parent_time, edge = parent_info
            path_edges.insert(0, edge)
            
            edge_cost = tdg.weight_function(edge, parent_time)
            total_cost += edge_cost
            
            current_node = parent_node
            current_time = parent_time
        
        t_star = current_time
        optimal_path = Path(edges=path_edges, total_cost=np.float32(total_cost))
        
        return OrdaResult(t_star, optimal_path, best_cost)
    
    @staticmethod
    def solve_hybrid(tdg: TDG, 
                     source: Intersection, 
                     destination: Intersection, 
                     time_window: Tuple[int, int],
                     strategy: str = "orda") -> OrdaResult:
        """
        Solveur hybride combinant ORDA et ROM
        
        Args:
            tdg: Graphe dépendant du temps
            source: Intersection source  
            destination: Intersection destination
            time_window: Fenêtre temporelle
            strategy: "orda", "rom", ou "best" (compare les deux)
            
        Returns:
            OrdaResult avec la meilleure solution
        """
        if strategy == "orda":
            return OrdaRomSolver.solve_orda(tdg, source, destination, time_window)
        elif strategy == "rom":
            return OrdaRomSolver.solve_rom(tdg, source, destination, time_window)
        elif strategy == "best":
            # Exécuter les deux algorithmes et retourner le meilleur
            orda_result = OrdaRomSolver.solve_orda(tdg, source, destination, time_window)
            rom_result = OrdaRomSolver.solve_rom(tdg, source, destination, time_window)
            
            # Comparaison : coût total prioritaire
            if orda_result.total_cost <= rom_result.total_cost:
                return orda_result
            else:
                return rom_result
        else:
            raise ValueError(f"Stratégie inconnue : {strategy}")
    
    @staticmethod
    def solve(tdg: TDG, 
              source: Intersection, 
              destination: Intersection, 
              time_window: Tuple[int, int]) -> OrdaResult:
        """
        Interface principale compatible avec ObjectOrdaSolver
        Par défaut utilise l'algorithme ORDA
        
        Args:
            tdg: Graphe dépendant du temps
            source: Intersection source
            destination: Intersection destination  
            time_window: Fenêtre temporelle
            
        Returns:
            OrdaResult avec la solution optimale
        """
        return OrdaRomSolver.solve_orda(tdg, source, destination, time_window)


# Alias pour une utilisation directe
class OrdaSolver(OrdaRomSolver):
    """Alias pour utiliser uniquement l'algorithme ORDA"""
    
    @staticmethod
    def solve(tdg: TDG, source: Intersection, destination: Intersection, 
              time_window: Tuple[int, int]) -> OrdaResult:
        return OrdaRomSolver.solve_orda(tdg, source, destination, time_window)


class RomSolver(OrdaRomSolver):
    """Alias pour utiliser uniquement l'algorithme ROM"""
    
    @staticmethod
    def solve(tdg: TDG, source: Intersection, destination: Intersection, 
              time_window: Tuple[int, int]) -> OrdaResult:
        return OrdaRomSolver.solve_rom(tdg, source, destination, time_window)