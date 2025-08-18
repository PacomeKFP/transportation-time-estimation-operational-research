"""
Debug du main_application.py pour identifier les problèmes
"""

import numpy as np
from data_parser import DataParser, RealisticGraphBuilder

def debug_graph_construction():
    """Debug la construction des graphes"""
    print("=== DEBUG CONSTRUCTION GRAPHES ===")
    
    # Parser les données
    suppliers = DataParser.parse_suppliers('xl/b.txt')
    route_graphs = DataParser.parse_route_graphs('xl/a.txt')
    
    print(f"Fournisseurs trouvés: {len(suppliers)}")
    print(f"Graphes de routes: {len(route_graphs)}")
    
    builder = RealisticGraphBuilder(suppliers, route_graphs)
    
    # Tester chaque fournisseur
    for supplier in suppliers:
        print(f"\n--- Test {supplier.name} ---")
        try:
            tdg, provider = builder.build_supplier_graph(supplier.name)
            print(f"  OK Graphe cree: {len(tdg.intersections)} intersections, {len(tdg.edges)} aretes")
            
            # Vérifier la structure du graphe
            print(f"  Intersections: {[i.name for i in tdg.intersections]}")
            print(f"  Arêtes:")
            for i, edge in enumerate(tdg.edges):
                print(f"    {i+1}: {edge.origin.name} -> {edge.extremity.name} ({len(edge.chunks)} chunks)")
            
            # Tester la fonction de poids
            if tdg.edges:
                test_edge = tdg.edges[0]
                try:
                    weight = tdg.weight_function(test_edge, 10.0)
                    print(f"  OK Fonction poids OK: {weight}")
                except Exception as e:
                    print(f"  NOK Erreur fonction poids: {e}")
                    
        except Exception as e:
            print(f"  NOK Erreur construction: {e}")

def debug_solver():
    """Debug le solveur TwoStepLTT"""
    print("\n=== DEBUG SOLVEUR ===")
    
    from app.solvers.two_step_ltt_fixed import TwoStepLTTFixed
    from app.models.intersection import Intersection
    from app.models.edge import Edge
    from app.models.chunk import Chunk
    from app.models.tdg import TDG
    
    # Créer un graphe simple pour test
    print("Création d'un graphe simple...")
    
    a = Intersection("A")
    b = Intersection("B")
    c = Intersection("C")
    
    # Arêtes simples avec chunks
    chunks1 = [Chunk(length=np.float32(1.0)), Chunk(length=np.float32(1.0))]
    chunks2 = [Chunk(length=np.float32(1.5))]
    
    edge1 = Edge(a, b, chunks1)
    edge2 = Edge(b, c, chunks2)
    
    # Fonction de poids simple et déterministe
    def simple_weight(edge, departure_time):
        total_length = sum(chunk.length for chunk in edge.chunks)
        base_speed = 50.0  # km/h
        return float(total_length / base_speed)
    
    tdg = TDG([a, b, c], [edge1, edge2], simple_weight)
    
    print(f"Graphe test: {len(tdg.intersections)} intersections, {len(tdg.edges)} arêtes")
    
    # Tester le solveur
    try:
        time_window = (0, 24)
        print(f"Test solveur avec fenêtre {time_window}")
        
        result = TwoStepLTTFixed.solve(tdg, a, c, time_window)
        
        if result.path is not None:
            print(f"  OK Solution trouvee!")
            print(f"    Temps optimal: {result.t_star}")
            print(f"    Cout total: {result.total_cost}")
            print(f"    Chemin: {len(result.path.edges)} aretes")
        else:
            print(f"  NOK Aucune solution trouvee")
            print(f"    Temps: {result.t_star}")
            print(f"    Cout: {result.total_cost}")
            
    except Exception as e:
        print(f"  NOK Erreur solveur: {e}")
        import traceback
        traceback.print_exc()

def debug_real_supplier():
    """Debug avec un vrai fournisseur"""
    print("\n=== DEBUG FOURNISSEUR RÉEL ===")
    
    suppliers = DataParser.parse_suppliers('xl/b.txt')
    route_graphs = DataParser.parse_route_graphs('xl/a.txt')
    builder = RealisticGraphBuilder(suppliers, route_graphs)
    
    # Tester SOMAF qui a des données
    try:
        tdg, provider = builder.build_supplier_graph("SOMAF (F4)")
        print(f"SOMAF - Graphe: {len(tdg.intersections)} intersections, {len(tdg.edges)} arêtes")
        
        # Identifier source et destination
        source = provider.location
        destination = None
        
        # Trouver la destination (nœud avec numéro le plus élevé)
        max_num = -1
        for intersection in tdg.intersections:
            if intersection.name.startswith('X'):
                try:
                    num = int(intersection.name[1:])
                    if num > max_num:
                        max_num = num
                        destination = intersection
                except ValueError:
                    continue
        
        if not destination:
            destination = tdg.intersections[-1]
            
        print(f"Source: {source.name}, Destination: {destination.name}")
        
        # Test avec fonction de poids simplifiée
        def debug_weight(edge, departure_time):
            total_length = sum(chunk.length for chunk in edge.chunks)
            return float(total_length * 0.02)  # ~20 minutes par km
        
        # Remplacer la fonction de poids
        tdg.weight_function = debug_weight
        
        # Tester le solveur
        from app.solvers.two_step_ltt_fixed import TwoStepLTTFixed
        
        time_window = (0, 5)  # Fenêtre réduite pour debug
        print(f"Test avec fenêtre réduite: {time_window}")
        
        result = TwoStepLTTFixed.solve(tdg, source, destination, time_window)
        
        if result.path is not None:
            print(f"  OK Solution trouvee!")
            print(f"    Temps optimal: {result.t_star}")
            print(f"    Cout total: {result.total_cost}")
        else:
            print(f"  NOK Aucune solution - cout: {result.total_cost}")
            
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_graph_construction()
    debug_solver()
    debug_real_supplier()