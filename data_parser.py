"""
Parser pour les données Excel des fournisseurs et tronçons
Intègre les données réelles avec le système time_estimators.py
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

from app.models.provider import Provider
from app.models.intersection import Intersection
from app.models.edge import Edge
from app.models.chunk import Chunk
from app.models.tdg import TDG
from app.time_estimators import EdgeTransportationTimeEstimator
from app.models.traffic_matrices import traffic_matrices


@dataclass
class SupplierData:
    name: str
    production_capacity: float  # m3/h
    delivery_capacity: float    # m3
    cost_per_m3: float         # FCFA
    pump_cost: float           # FCFA/m3
    quality: str
    location: str


@dataclass 
class RoadSection:
    section_id: str  # T1, T2, etc.
    length: float    # km
    characteristics: str = ""


@dataclass
class RouteArc:
    start_node: str  # X1, X2, etc.
    end_node: str    # X2, X3, etc.
    sections: List[RoadSection]


class DataParser:
    
    @staticmethod
    def parse_suppliers(file_path: str) -> List[SupplierData]:
        """Parse les données des fournisseurs depuis b.txt"""
        suppliers = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Ignorer la ligne d'en-tête
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) >= 7:
                supplier = SupplierData(
                    name=parts[0].strip(),
                    production_capacity=float(parts[1]) if parts[1] else 0.0,
                    delivery_capacity=float(parts[2]) if parts[2] else 0.0,
                    cost_per_m3=float(parts[3].replace(',', '')) if parts[3] else 0.0,
                    pump_cost=float(parts[4].replace(',', '')) if parts[4] else 0.0,
                    quality=parts[5].strip(),
                    location=parts[6].strip()
                )
                suppliers.append(supplier)
        
        return suppliers
    
    @staticmethod
    def parse_route_graphs(file_path: str) -> Dict[str, List[RouteArc]]:
        """Parse les graphes de routes depuis a.txt"""
        graphs = {}
        current_supplier = None
        current_arc = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Détection d'un nouveau fournisseur
            if line.startswith('FOURNISSEUR'):
                current_supplier = line.replace('FOURNISSEUR', '').strip()
                graphs[current_supplier] = []
                current_arc = None
                continue
            
            # Ignorer la ligne d'en-tête
            if line.startswith('Arcs'):
                continue
            
            parts = line.split('\t')
            if len(parts) >= 3:
                arc_name = parts[0].strip()
                section_id = parts[1].strip()
                length_str = parts[2].strip()
                characteristics = parts[3].strip() if len(parts) > 3 else ""
                
                # Si arc_name n'est pas vide et contient un tiret, c'est un nouvel arc
                if arc_name and '-' in arc_name:
                    # Nouveau arc
                    start_node, end_node = arc_name.split('-')
                    current_arc = RouteArc(start_node, end_node, [])
                    graphs[current_supplier].append(current_arc)
                
                # Ajouter le tronçon à l'arc actuel (que arc_name soit vide ou non)
                if current_arc and section_id and length_str:
                    try:
                        length = float(length_str.replace(',', '.'))
                        section = RoadSection(section_id, length, characteristics)
                        current_arc.sections.append(section)
                    except ValueError:
                        continue
        
        return graphs


class MarkovStateManager:
    """Gestionnaire de la propagation des états de Markov à travers le graphe"""
    
    def __init__(self, initial_traffic_level: str = "medium"):
        self.edge_distributions = {}  # Cache des distributions par arête
        self.initial_distribution = traffic_matrices.get_initial_distribution(initial_traffic_level)
        self.P_function = traffic_matrices.create_P_function("average", "dry")
        self.enable_propagation = True  # Flag pour désactiver propagation si nécessaire
    
    def get_edge_distribution(self, edge: Edge, departure_time: float) -> np.ndarray:
        """Récupère ou calcule la distribution d'états pour une arête"""
        edge_key = f"{edge.origin.name}-{edge.extremity.name}"
        
        # Si pas encore calculé, utiliser la distribution initiale
        if edge_key not in self.edge_distributions:
            # Déterminer le niveau de trafic selon l'heure
            hour = int(abs(departure_time)) % 24
            if 6 <= hour <= 9 or 16 <= hour <= 19:
                traffic_level = "heavy"
            elif 22 <= hour or hour <= 5:
                traffic_level = "light"
            else:
                traffic_level = "medium"
            
            self.edge_distributions[edge_key] = traffic_matrices.get_initial_distribution(traffic_level)
        
        return self.edge_distributions[edge_key]
    
    def propagate_distribution(self, edge: Edge, departure_time: float, travel_time: float):
        """Propage la distribution après traversée d'une arête"""
        edge_key = f"{edge.origin.name}-{edge.extremity.name}"
        current_distribution = self.get_edge_distribution(edge, departure_time)
        
        # Appliquer la matrice de transition P(t)
        P_t = self.P_function(travel_time)
        new_distribution = current_distribution @ P_t
        
        # Mettre à jour pour les arêtes suivantes
        self.edge_distributions[edge_key] = new_distribution
        
        return new_distribution


class RealisticGraphBuilder:
    """Construit des graphes TDG réalistes à partir des données parsées"""
    
    def __init__(self, suppliers_data: List[SupplierData], route_graphs: Dict[str, List[RouteArc]]):
        self.suppliers_data = suppliers_data
        self.route_graphs = route_graphs
        self.intersections_cache = {}
        self.state_manager = MarkovStateManager()
    
    def get_intersection(self, node_name: str) -> Intersection:
        """Récupère ou crée une intersection"""
        if node_name not in self.intersections_cache:
            self.intersections_cache[node_name] = Intersection(node_name)
        return self.intersections_cache[node_name]
    
    def create_chunks_from_section(self, section: RoadSection) -> List[Chunk]:
        """Crée des chunks à partir d'une section de route"""
        # Diviser la section en 2-4 chunks selon sa longueur
        section_length = section.length  # km
        
        if section_length < 1.0:
            num_chunks = 2
        elif section_length < 3.0:
            num_chunks = 3
        else:
            num_chunks = 4
        
        chunk_length = section_length / num_chunks
        chunks = []
        
        for i in range(num_chunks):
            chunk = Chunk(length=np.float32(chunk_length))
            chunks.append(chunk)
        
        return chunks
    
    def build_supplier_graph(self, supplier_name: str, construction_site_node: str = "X9") -> Tuple[TDG, Provider]:
        """Construit le graphe TDG pour un fournisseur spécifique"""
        
        # Trouver les données du fournisseur
        supplier_data = None
        for s in self.suppliers_data:
            if supplier_name.upper() in s.name.upper():
                supplier_data = s
                break
        
        if not supplier_data:
            raise ValueError(f"Fournisseur {supplier_name} non trouvé")
        
        # Trouver le graphe de routes avec correspondance manuelle
        route_graph = None
        
        # Correspondances manuelles connues
        name_mapping = {
            'CMCC (F1)': 'CMCC',
            'SAINTE HELENE  (F3)': 'HELENE', 
            'SOMAF (F4)': 'SOMAF'
        }
        
        # Essayer d'abord la correspondance directe
        mapped_name = name_mapping.get(supplier_name)
        if mapped_name and mapped_name in self.route_graphs:
            route_graph = self.route_graphs[mapped_name]
        else:
            # Sinon essayer la correspondance flexible
            for key, graph in self.route_graphs.items():
                if (supplier_name.upper() in key.upper() or 
                    key.upper() in supplier_name.upper() or
                    any(word in key.upper() for word in supplier_name.upper().split())):
                    route_graph = graph
                    break
        
        if not route_graph:
            raise ValueError(f"Graphe de routes pour {supplier_name} non trouvé")
        
        # Construire les intersections et arêtes
        intersections = set()
        edges = []
        
        for arc in route_graph:
            start_intersection = self.get_intersection(arc.start_node)
            end_intersection = self.get_intersection(arc.end_node)
            
            intersections.add(start_intersection)
            intersections.add(end_intersection)
            
            # Créer une arête pour chaque section de l'arc
            for section in arc.sections:
                chunks = self.create_chunks_from_section(section)
                edge = Edge(start_intersection, end_intersection, chunks)
                edges.append(edge)
        
        # Créer le Provider
        source_intersection = self.get_intersection("X1")  # Nœud source typique
        provider = Provider(
            name=supplier_data.name,
            location=source_intersection,
            available_quantity=int(supplier_data.delivery_capacity),
            departure_instant=np.float32(0.0)
        )
        
        # Créer la fonction de poids utilisant EdgeTransportationTimeEstimator avec propagation
        def markov_weight_function(edge: Edge, departure_time: float) -> float:
            # Gérer les cas infinis ou invalides - COMPATIBLE AVEC SOLVEUR
            if (departure_time == float('inf') or departure_time == float('-inf') or 
                departure_time != departure_time or departure_time < 0):
                # Pour le solveur TwoStepLTT, retourner une valeur raisonnable au lieu d'inf
                # Ceci permet au solveur de continuer sa recherche
                total_length = sum(chunk.length for chunk in edge.chunks)
                return float(total_length * 0.02)  # Fallback: 20 min/km
            
            # Normaliser le temps dans une plage raisonnable
            normalized_time = abs(departure_time) % 24.0
            
            # Récupérer la distribution d'états pour cette arête (avec propagation)
            current_distribution = self.state_manager.get_edge_distribution(edge, normalized_time)
            
            # Fonction P(t) = e^(Qt) avec matrice moyenne
            P_function = self.state_manager.P_function
            
            try:
                # Créer l'estimateur d'arête avec la distribution propagée
                estimator = EdgeTransportationTimeEstimator(
                    states_distribution=current_distribution,
                    P=P_function
                )
                
                # Calculer le temps de transport via chaînes de Markov
                travel_time = estimator.estimate(edge, np.float32(normalized_time))
                
                # Vérifier que le résultat est valide
                if np.isnan(travel_time) or np.isinf(travel_time) or travel_time <= 0:
                    # Fallback si estimation Markov échoue
                    total_length = sum(chunk.length for chunk in edge.chunks)
                    travel_time = total_length * 0.02
                
                # Propager la distribution pour les arêtes suivantes (si activé)
                if self.state_manager.enable_propagation:
                    self.state_manager.propagate_distribution(edge, normalized_time, float(travel_time))
                
                return max(0.01, float(travel_time))  # Minimum 0.01h pour éviter zéro
                
            except Exception as e:
                # En cas d'erreur, utiliser fallback
                total_length = sum(chunk.length for chunk in edge.chunks)
                return max(0.01, float(total_length * 0.02))
        
        # Créer le TDG
        tdg = TDG(list(intersections), edges, markov_weight_function)
        
        return tdg, provider


def main():
    """Exemple d'utilisation"""
    # Parser les données
    suppliers = DataParser.parse_suppliers('xl/b.txt')
    route_graphs = DataParser.parse_route_graphs('xl/a.txt')
    
    print("=== FOURNISSEURS PARSÉS ===")
    for supplier in suppliers:
        print(f"{supplier.name}: {supplier.delivery_capacity}m³ à {supplier.location}")
    
    print("\n=== GRAPHES DE ROUTES (DÉTAILLÉ) ===")
    for supplier_name, arcs in route_graphs.items():
        print(f"\n{supplier_name}:")
        for arc in arcs:
            sections_info = [f"{s.section_id}({s.length}km)" for s in arc.sections]
            print(f"  {arc.start_node}-{arc.end_node}: {', '.join(sections_info)} (Total: {len(arc.sections)} tronçons)")
    
    # Construire des graphes réalistes
    builder = RealisticGraphBuilder(suppliers, route_graphs)
    
    try:
        tdg, provider = builder.build_supplier_graph("SOMAF")
        print(f"\n=== GRAPHE CONSTRUIT POUR {provider.name} ===")
        print(f"Intersections: {len(tdg.intersections)}")
        print(f"Arêtes: {len(tdg.edges)}")
        total_chunks = sum(len(edge.chunks) for edge in tdg.edges)
        print(f"Chunks total: {total_chunks}")
        
        # Détail des arêtes
        print("\nDétail des arêtes:")
        for i, edge in enumerate(tdg.edges):
            print(f"  Arête {i+1}: {edge.origin.name}->{edge.extremity.name}, {len(edge.chunks)} chunks")
        
    except Exception as e:
        print(f"Erreur: {e}")


if __name__ == "__main__":
    main()