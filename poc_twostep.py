"""
PoC pour tester TwoStepLTT corrigé avec 5 graphes de taille croissante
"""

import time
import sys
import os
import random
import numpy as np

# Ajouter le chemin du projet
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models.intersection import Intersection
from app.models.edge import Edge
from app.models.chunk import Chunk
from app.models.tdg import TDG
from app.solvers.two_step_ltt_fixed import TwoStepLTTFixed


def create_chunks(min_chunks: int = 2, max_chunks: int = 8) -> list:
    """Crée une liste de chunks avec des longueurs aléatoires"""
    num_chunks = random.randint(min_chunks, max_chunks)
    chunks = []
    for i in range(num_chunks):
        # Longueur aléatoire entre 0.5 et 3.0
        length = random.uniform(0.5, 3.0)
        chunks.append(Chunk(length=np.float32(length)))
    return chunks

from app.time_estimators import EdgeTransportationTimeEstimator

def create_weight_function(base_weight: float = 10.0):
    """Crée une fonction de poids simple time-dependent"""
    def weight_func(edge: Edge, departure_time: float) -> float:
        # Fonction simple: poids de base + variation sinusoïdale
        import math
        variation = 5.0 + math.log((10 + departure_time) / 10.0) * sum(c.length for c in edge.chunks)
        return  variation + base_weight
    return EdgeTransportationTimeEstimator.es


def create_graph_1() -> TDG:
    """Graphe 1: 3 nœuds, 3 arêtes (triangle)"""
    print("Création du Graphe 1: Triangle (3 nœuds, 3 arêtes)")
    
    # Nœuds
    a = Intersection("A")
    b = Intersection("B") 
    c = Intersection("C")
    intersections = [a, b, c]
    
    # Arêtes avec chunks (2-8 chunks par arête)
    edges = [
        Edge(a, b, create_chunks()),
        Edge(b, c, create_chunks()),
        Edge(a, c, create_chunks())
    ]
    
    return TDG(intersections, edges, create_weight_function(8.0))


def create_graph_2() -> TDG:
    """Graphe 2: 4 nœuds, 5 arêtes"""
    print("Création du Graphe 2: (4 nœuds, 5 arêtes)")
    
    # Réutilise les nœuds du graphe 1 + 1 nouveau
    a = Intersection("A")
    b = Intersection("B") 
    c = Intersection("C")
    d = Intersection("D")
    intersections = [a, b, c, d]
    
    # Arêtes: graphe 1 + nouvelles arêtes
    edges = [
        Edge(a, b, create_chunks()),
        Edge(b, c, create_chunks()),
        Edge(a, c, create_chunks()),
        Edge(c, d, create_chunks()),
        Edge(a, d, create_chunks())
    ]
    
    return TDG(intersections, edges, create_weight_function(10.0))


def create_graph_3() -> TDG:
    """Graphe 3: 5 nœuds, 8 arêtes"""
    print("Création du Graphe 3: (5 nœuds, 8 arêtes)")
    
    # Réutilise + 1 nouveau nœud
    a = Intersection("A")
    b = Intersection("B") 
    c = Intersection("C")
    d = Intersection("D")
    e = Intersection("E")
    intersections = [a, b, c, d, e]
    
    # Arêtes: graphe 2 + nouvelles arêtes
    edges = [
        Edge(a, b, create_chunks()),
        Edge(b, c, create_chunks()),
        Edge(a, c, create_chunks()),
        Edge(c, d, create_chunks()),
        Edge(a, d, create_chunks()),
        Edge(d, e, create_chunks()),
        Edge(b, e, create_chunks()),
        Edge(a, e, create_chunks())
    ]
    
    return TDG(intersections, edges, create_weight_function(12.0))


def create_graph_4() -> TDG:
    """Graphe 4: 6 nœuds, 12 arêtes"""
    print("Création du Graphe 4: (6 nœuds, 12 arêtes)")
    
    # Réutilise + 1 nouveau nœud
    a = Intersection("A")
    b = Intersection("B") 
    c = Intersection("C")
    d = Intersection("D")
    e = Intersection("E")
    f = Intersection("F")
    from string import ascii_uppercase
    intersections = [Intersection(name) for name in ascii_uppercase]
    intersections += [Intersection(f'{name}+{name}') for name in ascii_uppercase]
    
    # Arêtes: graphe 3 + nouvelles arêtes
    edges = [
        Edge(a, b, create_chunks()),
        Edge(b, c, create_chunks()),
        Edge(a, c, create_chunks()),
        Edge(c, d, create_chunks()),
        Edge(a, d, create_chunks()),
        Edge(d, e, create_chunks()),
        Edge(b, e, create_chunks()),
        Edge(a, e, create_chunks()),
        Edge(e, f, create_chunks()),
        Edge(c, f, create_chunks()),
        Edge(d, f, create_chunks()),
        Edge(b, f, create_chunks())
    ]
    
    return TDG(intersections, edges, create_weight_function(15.0))


def create_graph_5() -> TDG:
    """Graphe 5: 7 nœuds, 16 arêtes"""
    print("Création du Graphe 5: (7 nœuds, 16 arêtes)")
    
    # Réutilise + 1 nouveau nœud
    a = Intersection("A")
    b = Intersection("B") 
    c = Intersection("C")
    d = Intersection("D")
    e = Intersection("E")
    f = Intersection("F")
    g = Intersection("G")
    intersections = [a, b, c, d, e, f, g]
    
    # Arêtes: graphe 4 + nouvelles arêtes
    edges = [
        Edge(a, b, create_chunks()),
        Edge(b, c, create_chunks()),
        Edge(a, c, create_chunks()),
        Edge(c, d, create_chunks()),
        Edge(a, d, create_chunks()),
        Edge(d, e, create_chunks()),
        Edge(b, e, create_chunks()),
        Edge(a, e, create_chunks()),
        Edge(e, f, create_chunks()),
        Edge(c, f, create_chunks()),
        Edge(d, f, create_chunks()),
        Edge(b, f, create_chunks()),
        Edge(f, g, create_chunks()),
        Edge(e, g, create_chunks()),
        Edge(a, g, create_chunks()),
        Edge(c, g, create_chunks())
    ]
    
    return TDG(intersections, edges, create_weight_function(18.0))


def test_graph(graph: TDG, graph_name: str) -> dict:
    """Teste un graphe avec TwoStepLTT et mesure le temps"""
    print(f"\n=== Test {graph_name} ===")
    
    # Compter le nombre total de chunks
    total_chunks = sum(len(edge.chunks) for edge in graph.edges)
    avg_chunks = total_chunks / len(graph.edges) if graph.edges else 0
    
    print(f"Nœuds: {len(graph.intersections)}, Arêtes: {len(graph.edges)}")
    print(f"Chunks total: {total_chunks}, Moyenne par arête: {avg_chunks:.1f}")
    
    # Paramètres du test
    source = graph.intersections[0]  # Premier nœud (A)
    destination = graph.intersections[-1]  # Dernier nœud
    time_window = (0, 24)  # Fenêtre de 24 heures (entiers)
    
    print(f"Source: {source.name}, Destination: {destination.name}")
    print(f"Fenêtre temporelle: {time_window}")
    
    # Mesure du temps
    start_time = time.time()
    
    try:
        result = TwoStepLTTFixed.solve(graph, source, destination, time_window)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        print(f"Succes en {execution_time:.4f} secondes")
        print(f"Temps de départ optimal: {result.t_star}")
        print(f"Temps de voyage: {result.total_cost:.2f}")
        
        if result.path:
            path_str = " -> ".join([f"{edge.origin.name}-{edge.extremity.name}" for edge in result.path.edges])
            print(f"Chemin: {path_str}")
        
        return {
            'success': True,
            'execution_time': execution_time,
            'nodes': len(graph.intersections),
            'edges': len(graph.edges),
            'optimal_start': result.t_star,
            'travel_time': result.total_cost
        }
        
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Erreur: {str(e)}")
        print(f"Temps avant erreur: {execution_time:.4f} secondes")
        
        return {
            'success': False,
            'execution_time': execution_time,
            'nodes': len(graph.intersections),
            'edges': len(graph.edges),
            'error': str(e)
        }


def main():
    """Fonction principale du PoC"""
    print("PoC TwoStepLTT Corrige - Test de Performance")
    print("=" * 60)
    
    # Créer les 5 graphes
    graphs = [
        (create_graph_1(), "Graphe 1"),
        (create_graph_2(), "Graphe 2"),
        (create_graph_3(), "Graphe 3"),
        (create_graph_4(), "Graphe 4"),
        (create_graph_5(), "Graphe 5")
    ]
    
    results = []
    
    # Tester chaque graphe
    for graph, name in graphs:
        result = test_graph(graph, name)
        results.append(result)
    
    # Résumé des performances
    print("\n" + "=" * 60)
    print("RESUME DES PERFORMANCES")
    print("=" * 60)
    print(f"{'Graphe':<12} {'Nœuds':<8} {'Arêtes':<8} {'Temps (s)':<12} {'Statut':<10}")
    print("-" * 60)
    
    for i, result in enumerate(results, 1):
        status = "OK" if result['success'] else "Erreur"
        print(f"Graphe {i:<6} {result['nodes']:<8} {result['edges']:<8} {result['execution_time']:<12.4f} {status}")
    
    # Analyse de la scalabilité
    successful_results = [r for r in results if r['success']]
    if len(successful_results) > 1:
        print(f"\nANALYSE DE SCALABILITE")
        print("-" * 40)
        
        times = [r['execution_time'] for r in successful_results]
        nodes = [r['nodes'] for r in successful_results]
        edges = [r['edges'] for r in successful_results]
        
        print(f"Temps min: {min(times):.4f}s")
        print(f"Temps max: {max(times):.4f}s")
        if min(times) > 0:
            print(f"Ratio max/min: {max(times)/min(times):.2f}x")
        else:
            print("Ratio max/min: N/A (temps trop rapides)")
        
        # Facteur de croissance approximatif
        if len(times) >= 2:
            if times[0] > 0:
                growth_factor = times[-1] / times[0]
                print(f"Facteur de croissance du temps: {growth_factor:.2f}x")
            else:
                print("Facteur de croissance du temps: N/A (temps trop rapides)")
            
            node_factor = nodes[-1] / nodes[0]
            edge_factor = edges[-1] / edges[0]
            
            print(f"Facteur de croissance des nœuds: {node_factor:.2f}x")
            print(f"Facteur de croissance des arêtes: {edge_factor:.2f}x")
    
    print(f"\nTest termine avec {len(successful_results)}/{len(results)} graphes reussis")


if __name__ == "__main__":
    main()