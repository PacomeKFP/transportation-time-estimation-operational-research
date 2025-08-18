"""
Test de l'intégration des chaînes de Markov
"""

import numpy as np
from data_parser import DataParser, RealisticGraphBuilder
from app.solvers.two_step_ltt_fixed import TwoStepLTTFixed
from app.models.traffic_matrices import traffic_matrices

def test_markov_integration():
    """Test de l'intégration complète"""
    print("=== TEST INTÉGRATION MARKOV ===")
    
    # Test des matrices Q et P
    print("\n1. Test des matrices de transition:")
    Q_avg = traffic_matrices.get_Q_matrix("average")
    print(f"Matrice Q moyenne: {Q_avg.shape}")
    print(f"Somme des lignes (doit être ~0): {np.sum(Q_avg, axis=1)}")
    
    # Test de P(t)
    P_func = traffic_matrices.create_P_function("average")
    P_1 = P_func(1.0)  # P(1 heure)
    print(f"P(1h) shape: {P_1.shape}")
    print(f"P(1h) somme des lignes (doit être ~1): {np.sum(P_1, axis=1)}")
    
    # Test des distributions
    print("\n2. Test des distributions d'états:")
    for level in ["light", "medium", "heavy"]:
        dist = traffic_matrices.get_initial_distribution(level)
        print(f"Distribution {level}: {dist}, somme: {np.sum(dist)}")
    
    # Test de la construction de graphe
    print("\n3. Test construction graphe avec Markov:")
    suppliers = DataParser.parse_suppliers('xl/b.txt')
    route_graphs = DataParser.parse_route_graphs('xl/a.txt')
    builder = RealisticGraphBuilder(suppliers, route_graphs)
    
    try:
        tdg, provider = builder.build_supplier_graph("SOMAF (F4)")
        print(f"Graphe SOMAF construit: {len(tdg.intersections)} intersections, {len(tdg.edges)} arêtes")
        
        # Test de la fonction de poids Markov
        print("\n4. Test fonction de poids Markov:")
        test_edge = tdg.edges[0]
        
        for time in [0, 6, 12, 18]:
            try:
                weight = tdg.weight_function(test_edge, float(time))
                print(f"  Poids à {time}h: {weight:.4f}h")
            except Exception as e:
                print(f"  Erreur à {time}h: {e}")
        
        # Test du solveur avec fenêtre très large
        print("\n5. Test solveur avec Markov:")
        source = provider.location
        destination = None
        
        # Trouver destination (nœud le plus élevé)
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
        
        if destination:
            print(f"Source: {source.name}, Destination: {destination.name}")
            
            # Test avec plusieurs fenêtres
            for window in [(0, 24), (0, 12), (6, 18)]:
                print(f"\n  Test fenêtre {window}:")
                try:
                    result = TwoStepLTTFixed.solve(tdg, source, destination, window)
                    
                    if result.path is not None:
                        print(f"    OK - Temps: {result.t_star:.2f}h, Coût: {result.total_cost:.4f}h")
                        path_edges = [f"{e.origin.name}->{e.extremity.name}" for e in result.path.edges]
                        print(f"    Chemin: {' -> '.join(path_edges)}")
                    else:
                        print(f"    NOK - Aucune solution, coût: {result.total_cost}")
                        
                except Exception as e:
                    print(f"    ERREUR: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("Destination non trouvée!")
    
    except Exception as e:
        print(f"Erreur construction graphe: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_markov_integration()