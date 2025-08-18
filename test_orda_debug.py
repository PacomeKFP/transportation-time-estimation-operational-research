"""
Test simple avec le solveur ORDA pour debug
"""

from data_parser import DataParser, RealisticGraphBuilder
from app.solvers.orda_rom_1990_solver import OrdaRom1990Solver

def test_orda_simple():
    """Test ORDA pour comprendre le problème"""
    print("=== TEST SOLVEUR ORDA ===")
    
    suppliers = DataParser.parse_suppliers('xl/b.txt')
    route_graphs = DataParser.parse_route_graphs('xl/a.txt')
    builder = RealisticGraphBuilder(suppliers, route_graphs)
    
    try:
        # Test SOMAF
        tdg, provider = builder.build_supplier_graph("SOMAF (F4)")
        
        print(f"SOMAF:")
        print(f"  Source: {provider.location.name}")
        print(f"  Graphe: {len(tdg.intersections)} intersections, {len(tdg.edges)} arêtes")
        
        # Destination
        destination = None
        for intersection in tdg.intersections:
            if intersection.name == "X9":
                destination = intersection
                break
        
        print(f"  Destination: {destination.name}")
        
        # Test fonction de poids
        print(f"\\nTest fonction poids:")
        for edge in tdg.edges[:2]:  # 2 premières arêtes
            weight = tdg.weight_function(edge, 0.0)
            print(f"  {edge.origin.name}->{edge.extremity.name}: {weight:.6f}h")
            
            if weight == float('inf'):
                print(f"    PROBLEME: Poids infini!")
                return
        
        # Test ORDA
        print(f"\\nTest ORDA solver:")
        try:
            result = OrdaRom1990Solver.solve(tdg, provider.location, destination)
            
            if hasattr(result, 'path') and result.path is not None:
                print(f"  OK: Coût {result.total_cost:.6f}h")
                path_edges = [f"{e.origin.name}->{e.extremity.name}" for e in result.path.edges]
                print(f"  Chemin: {' -> '.join(path_edges)}")
            else:
                print(f"  NOK: Pas de chemin - {result}")
                
        except Exception as e:
            print(f"  ERREUR ORDA: {e}")
            import traceback
            traceback.print_exc()
        
        # Test fonction poids simple pour comparaison
        print(f"\\n=== COMPARAISON FONCTION SIMPLE ===")
        
        def simple_weight(edge, departure_time):
            if departure_time == float('inf') or departure_time != departure_time:
                return float('inf')
            total_length = sum(chunk.length for chunk in edge.chunks)
            return float(total_length * 0.02)
        
        # Backup et test
        original_weight = tdg.weight_function
        tdg.weight_function = simple_weight
        
        try:
            result_simple = OrdaRom1990Solver.solve(tdg, provider.location, destination)
            
            if hasattr(result_simple, 'path') and result_simple.path is not None:
                print(f"  Simple OK: Coût {result_simple.total_cost:.6f}h")
            else:
                print(f"  Simple NOK: {result_simple}")
                
        except Exception as e:
            print(f"  ERREUR Simple: {e}")
        
        # Remettre fonction originale
        tdg.weight_function = original_weight
        
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()

def explain_multicriteria():
    """Explication du tri multicritère"""
    print(f"\\n=== TRI MULTICRITÈRE ===")
    print(f"Le tri multicritère dans main_application.py:")
    print(f"1. Filtre: garde seulement fournisseurs avec capacity >= demande ET temps < inf")
    print(f"2. Tri: classe par temps de transport croissant (plus rapide = meilleur)")
    print(f"3. Critères: (quantité, temps, coût) mais seul le TEMPS est utilisé pour le tri")
    print(f"4. Si AUCUN fournisseur n'a temps < inf → 'Aucun fournisseur valide'")

if __name__ == "__main__":
    test_orda_simple()
    explain_multicriteria()