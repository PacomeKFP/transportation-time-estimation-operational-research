"""
Test du solveur avec une fonction de poids simplifiée
"""

import numpy as np
from data_parser import DataParser, RealisticGraphBuilder
from app.solvers.two_step_ltt_fixed import TwoStepLTTFixed

def test_simple_solver():
    """Test avec fonction de poids simple pour isoler le problème"""
    print("=== TEST SOLVEUR SIMPLIFIÉ ===")
    
    suppliers = DataParser.parse_suppliers('xl/b.txt')
    route_graphs = DataParser.parse_route_graphs('xl/a.txt')
    builder = RealisticGraphBuilder(suppliers, route_graphs)
    
    try:
        tdg, provider = builder.build_supplier_graph("SOMAF (F4)")
        
        print(f"Graphe SOMAF: {len(tdg.intersections)} intersections, {len(tdg.edges)} arêtes")
        
        # Remplacer par une fonction de poids très simple
        def simple_weight(edge, departure_time):
            """Fonction de poids constante pour test"""
            if departure_time == float('inf') or departure_time == float('-inf') or departure_time != departure_time:
                return float('inf')
            
            total_length = sum(chunk.length for chunk in edge.chunks)
            return float(total_length * 0.02)  # 20 min/km = 0.02h/km
        
        # Remplacer la fonction de poids
        tdg.weight_function = simple_weight
        
        # Test de la fonction
        test_edge = tdg.edges[0]
        for time in [0, 6, 12, 18]:
            weight = tdg.weight_function(test_edge, float(time))
            print(f"Poids simple à {time}h: {weight:.4f}h")
        
        # Test du solveur
        source = provider.location
        destination = None
        
        # Trouver X9
        for intersection in tdg.intersections:
            if intersection.name == "X9":
                destination = intersection
                break
        
        if destination:
            print(f"\nTest solveur: {source.name} -> {destination.name}")
            
            # Test avec fenêtre simple
            time_window = (0, 5)
            print(f"Fenêtre temporelle: {time_window}")
            
            try:
                result = TwoStepLTTFixed.solve(tdg, source, destination, time_window)
                
                if result.path is not None:
                    print(f"OK Solution trouvee!")
                    print(f"  Temps optimal: {result.t_star:.2f}h")
                    print(f"  Cout total: {result.total_cost:.4f}h")
                    path_edges = [f"{e.origin.name}->{e.extremity.name}" for e in result.path.edges]
                    print(f"  Chemin: {' -> '.join(path_edges)}")
                else:
                    print(f"NOK Aucune solution trouvee")
                    print(f"  Cout: {result.total_cost}")
                    
            except Exception as e:
                print(f"NOK Erreur solveur: {e}")
                import traceback
                traceback.print_exc()
        
        # Test avec fonction de poids Markov originale
        print(f"\n=== RETOUR FONCTION MARKOV ===")
        
        # Récréer le graphe avec la fonction Markov
        tdg2, provider2 = builder.build_supplier_graph("SOMAF (F4)")
        
        print("Test fonction Markov originale:")
        test_edge2 = tdg2.edges[0]
        for time in [0, 6]:
            try:
                weight = tdg2.weight_function(test_edge2, float(time))
                print(f"Poids Markov à {time}h: {weight:.6f}h")
                
                # Vérifier s'il y a des valeurs problématiques
                if np.isnan(weight) or np.isinf(weight):
                    print(f"  WARNING: Valeur problematique detectee!")
                    
            except Exception as e:
                print(f"Erreur poids Markov à {time}h: {e}")
        
        # Test solveur avec Markov
        try:
            result2 = TwoStepLTTFixed.solve(tdg2, provider2.location, destination, (0, 2))
            if result2.path is not None:
                print(f"OK Markov: {result2.total_cost:.6f}h")
            else:
                print(f"NOK Markov: {result2.total_cost}")
        except Exception as e:
            print(f"NOK Erreur Markov: {e}")
    
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_solver()