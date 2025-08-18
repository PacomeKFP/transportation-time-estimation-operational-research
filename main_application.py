"""
Application principale d'estimation du temps de transport
Implémente le processus complet du logigramme principal
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import time

# Imports du système existant
from app.models.provider import Provider
from app.models.intersection import Intersection
from app.models.edge import Edge
from app.models.chunk import Chunk
from app.models.tdg import TDG
from app.solvers.two_step_ltt_fixed import TwoStepLTTFixed
from app.time_estimators import EdgeTransportationTimeEstimator

# Imports du parser de données
from data_parser import DataParser, RealisticGraphBuilder, SupplierData


@dataclass
class DeliveryRequest:
    """Demande de livraison"""
    quantity_demanded: float  # m³
    delivery_date: str        # Date de livraison souhaitée
    delivery_hour: int        # Heure de livraison (0-23)
    construction_site: str    # Nom du site de construction


@dataclass
class SupplierEvaluation:
    """Évaluation d'un fournisseur"""
    supplier: SupplierData
    provider: Provider
    estimated_time: float
    total_cost: float
    delivery_window: Tuple[int, int]  # Fenêtre de livraison (heure_min, heure_max)


class TransportationTimeEstimationSystem:
    """Système principal d'estimation du temps de transport"""
    
    def __init__(self, suppliers_file: str, routes_file: str):
        """Initialise le système avec les données des fournisseurs et routes"""
        self.suppliers_data = DataParser.parse_suppliers(suppliers_file)
        self.route_graphs = DataParser.parse_route_graphs(routes_file)
        self.graph_builder = RealisticGraphBuilder(self.suppliers_data, self.route_graphs)
        
        print(f"Système initialisé avec {len(self.suppliers_data)} fournisseurs")
        for supplier in self.suppliers_data:
            print(f"  - {supplier.name}: {supplier.delivery_capacity}m³ à {supplier.location}")
    
    def preselect_suppliers(self, delivery_request: DeliveryRequest) -> List[SupplierData]:
        """Étape 1: Présélection des fournisseurs selon les critères (quantité disponible)"""
        print(f"\n=== PRÉSÉLECTION DES FOURNISSEURS ===")
        print(f"Demande: {delivery_request.quantity_demanded}m³")
        
        available_suppliers = []
        
        for supplier in self.suppliers_data:
            if supplier.delivery_capacity >= delivery_request.quantity_demanded:
                available_suppliers.append(supplier)
                print(f"OK {supplier.name}: {supplier.delivery_capacity}m3 disponibles")
            else:
                print(f"NOK {supplier.name}: {supplier.delivery_capacity}m3 < {delivery_request.quantity_demanded}m3 requis")
        
        print(f"\nFournisseurs présélectionnés: {len(available_suppliers)}")
        return available_suppliers
    
    def estimate_transportation_time(self, supplier: SupplierData, 
                                   delivery_request: DeliveryRequest) -> Tuple[float, TDG, Provider]:
        """Étape 2: Estimation du temps de transport pour un fournisseur"""
        print(f"\n--- Estimation pour {supplier.name} ---")
        
        try:
            # Construire le graphe TDG du fournisseur
            tdg, provider = self.graph_builder.build_supplier_graph(supplier.name)
            
            # Déterminer le nœud de destination (chantier)
            construction_site_node = self.get_construction_site_node(tdg)
            
            # Paramètres temporels - FENÊTRE SIMPLIFIÉE POUR COMPATIBILITÉ
            departure_time = max(0, delivery_request.delivery_hour - 2)  # 2h avant livraison souhaitée
            
            # Fenêtre plus simple pour le solveur TwoStepLTT
            time_window = (0, 24)  # Fenêtre complète pour permettre au solveur de trouver une solution
            
            print(f"  Graphe: {len(tdg.intersections)} intersections, {len(tdg.edges)} arêtes")
            print(f"  Fenêtre temporelle: {time_window}")
            
            # Utiliser TwoStepLTT pour trouver le chemin optimal
            start_time = time.time()
            result = TwoStepLTTFixed.solve(
                tdg, 
                provider.location, 
                construction_site_node, 
                time_window
            )
            calculation_time = time.time() - start_time
            
            if result.path is not None:
                provider.transportation_time = np.float32(result.total_cost)
                provider.best_path = result.path
                
                print(f"  OK Temps estime: {result.total_cost:.2f}h")
                print(f"  OK Depart optimal: {result.t_star}h")
                print(f"  OK Calcul en: {calculation_time:.3f}s")
                
                return result.total_cost, tdg, provider
            else:
                print(f"  NOK Aucun chemin trouve")
                return float('inf'), tdg, provider
                
        except Exception as e:
            print(f"  NOK Erreur: {e}")
            return float('inf'), None, None
    
    def get_construction_site_node(self, tdg: TDG) -> Intersection:
        """Détermine le nœud représentant le chantier (généralement le dernier nœud)"""
        # Heuristique: prendre le nœud avec le nom le plus "élevé" (X9, X8, etc.)
        max_node = None
        max_num = -1
        
        for intersection in tdg.intersections:
            if intersection.name.startswith('X'):
                try:
                    num = int(intersection.name[1:])
                    if num > max_num:
                        max_num = num
                        max_node = intersection
                except ValueError:
                    continue
        
        return max_node if max_node else tdg.intersections[-1]
    
    def multicriteria_selection(self, evaluations: List[SupplierEvaluation], 
                               delivery_request: DeliveryRequest) -> List[SupplierEvaluation]:
        """Étape 3: Tri multicritère (quantité, temps, coût)"""
        print(f"\n=== TRI MULTICRITÈRE ===")
        
        # Filtrer les fournisseurs qui peuvent satisfaire la demande
        valid_evaluations = [
            eval for eval in evaluations 
            if eval.supplier.delivery_capacity >= delivery_request.quantity_demanded
            and eval.estimated_time < float('inf')
        ]
        
        if not valid_evaluations:
            print("Aucun fournisseur valide trouvé!")
            return []
        
        # Tri par temps de transport croissant
        valid_evaluations.sort(key=lambda x: x.estimated_time)
        
        print("Classement des fournisseurs (top 3):")
        for i, eval in enumerate(valid_evaluations[:3]):
            print(f"  {i+1}. {eval.supplier.name}: {eval.estimated_time:.2f}h, {eval.total_cost:.0f} FCFA/m³")
        
        return valid_evaluations
    
    def solve_delivery_optimization(self, delivery_request: DeliveryRequest) -> List[SupplierEvaluation]:
        """Processus principal selon le logigramme"""
        print("=" * 80)
        print("SYSTÈME D'ESTIMATION DU TEMPS DE TRANSPORT")
        print("=" * 80)
        
        # Étape 1: Présélection
        available_suppliers = self.preselect_suppliers(delivery_request)
        
        if not available_suppliers:
            print("Aucun fournisseur disponible pour cette demande!")
            return []
        
        # Étape 2: Estimation du temps de transport pour chaque fournisseur
        evaluations = []
        
        for supplier in available_suppliers:
            estimated_time, tdg, provider = self.estimate_transportation_time(supplier, delivery_request)
            
            # Calcul du coût total (coût matériau + pompe)
            total_cost = supplier.cost_per_m3 + supplier.pump_cost
            
            # Fenêtre de livraison basée sur le temps estimé
            if estimated_time < float('inf'):
                delivery_window = (
                    delivery_request.delivery_hour - int(estimated_time) - 1,
                    delivery_request.delivery_hour - int(estimated_time) + 1
                )
            else:
                delivery_window = (0, 23)  # Fenêtre par défaut si pas de solution
            
            evaluation = SupplierEvaluation(
                supplier=supplier,
                provider=provider,
                estimated_time=estimated_time,
                total_cost=total_cost,
                delivery_window=delivery_window
            )
            evaluations.append(evaluation)
        
        # Étape 3: Tri multicritère
        optimal_suppliers = self.multicriteria_selection(evaluations, delivery_request)
        
        # Résultat final
        print(f"\n=== RÉSULTAT FINAL ===")
        if optimal_suppliers:
            best_supplier = optimal_suppliers[0]
            print(f"Fournisseur recommandé: {best_supplier.supplier.name}")
            print(f"Temps de transport estimé: {best_supplier.estimated_time:.2f}h")
            print(f"Coût total: {best_supplier.total_cost:.0f} FCFA/m³")
            print(f"Fenêtre de départ: {best_supplier.delivery_window[0]}h-{best_supplier.delivery_window[1]}h")
        else:
            print("Aucune solution optimale trouvée")
        
        return optimal_suppliers


def main():
    """Exemple d'utilisation du système complet"""
    
    # Initialiser le système
    system = TransportationTimeEstimationSystem('xl/b.txt', 'xl/a.txt')
    
    # Définir une demande de livraison
    delivery_request = DeliveryRequest(
        quantity_demanded=50.0,  # 50 m³ de béton
        delivery_date="2024-01-15",
        delivery_hour=14,  # Livraison souhaitée à 14h
        construction_site="Chantier Central"
    )
    
    # Résoudre l'optimisation
    optimal_suppliers = system.solve_delivery_optimization(delivery_request)
    
    # Affichage détaillé des solutions (top 3)
    if optimal_suppliers:
        print(f"\n=== TOP 3 SOLUTIONS DÉTAILLÉES ===")
        for i, eval in enumerate(optimal_suppliers[:3]):
            print(f"\nSolution {i+1}: {eval.supplier.name}")
            print(f"  Localisation: {eval.supplier.location}")
            print(f"  Capacite: {eval.supplier.delivery_capacity}m3")
            print(f"  Temps estime: {eval.estimated_time:.2f}h")
            print(f"  Cout: {eval.total_cost:.0f} FCFA/m3")
            print(f"  Qualite: {eval.supplier.quality}")
            
            # Afficher le chemin si disponible
            if hasattr(eval.provider, 'best_path') and eval.provider.best_path:
                path_edges = [f"{e.origin.name}->{e.extremity.name}" for e in eval.provider.best_path.edges]
                print(f"  Chemin: {' -> '.join(path_edges)}")


if __name__ == "__main__":
    main()