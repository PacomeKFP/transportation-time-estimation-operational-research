"""
Debug des appels à la fonction de poids par le solveur
"""

import numpy as np
from data_parser import DataParser, RealisticGraphBuilder
from app.solvers.two_step_ltt_fixed import TwoStepLTTFixed

class DebugWeightWrapper:
    """Wrapper pour débugger les appels à la fonction de poids"""
    
    def __init__(self, original_function):
        self.original_function = original_function
        self.call_count = 0
        self.errors = []
        self.infinite_results = []
    
    def __call__(self, edge, departure_time):
        self.call_count += 1
        
        try:
            result = self.original_function(edge, departure_time)
            
            if np.isinf(result) or np.isnan(result):
                self.infinite_results.append({
                    'call': self.call_count,
                    'edge': f"{edge.origin.name}->{edge.extremity.name}",
                    'time': departure_time,
                    'result': result
                })
                print(f"  CALL {self.call_count}: {edge.origin.name}->{edge.extremity.name} @ {departure_time} -> {result}")
            
            return result
            
        except Exception as e:
            self.errors.append({
                'call': self.call_count,
                'edge': f"{edge.origin.name}->{edge.extremity.name}",
                'time': departure_time,
                'error': str(e)
            })
            print(f"  ERROR {self.call_count}: {edge.origin.name}->{edge.extremity.name} @ {departure_time} -> {e}")
            return float('inf')
    
    def report(self):
        print(f"\n=== RAPPORT DEBUG ===")
        print(f"Appels totaux: {self.call_count}")
        print(f"Erreurs: {len(self.errors)}")
        print(f"Valeurs infinies: {len(self.infinite_results)}")
        
        if self.errors:
            print(f"\nErreurs détaillées:")
            for error in self.errors[:5]:  # Top 5
                print(f"  {error}")
        
        if self.infinite_results:
            print(f"\nValeurs infinies:")
            for inf_res in self.infinite_results[:5]:  # Top 5
                print(f"  {inf_res}")

def debug_solver_calls():
    """Debug détaillé du solveur"""
    print("=== DEBUG APPELS SOLVEUR ===")
    
    suppliers = DataParser.parse_suppliers('xl/b.txt')
    route_graphs = DataParser.parse_route_graphs('xl/a.txt')
    builder = RealisticGraphBuilder(suppliers, route_graphs)
    
    try:
        tdg, provider = builder.build_supplier_graph("SOMAF (F4)")
        
        # Wrapper la fonction de poids
        debug_wrapper = DebugWeightWrapper(tdg.weight_function)
        tdg.weight_function = debug_wrapper
        
        # Trouver destination
        destination = None
        for intersection in tdg.intersections:
            if intersection.name == "X9":
                destination = intersection
                break
        
        print(f"Test solveur: {provider.location.name} -> {destination.name}")
        
        # Test avec fenêtre très petite d'abord
        print(f"\n1. Test fenêtre (0, 1):")
        try:
            result1 = TwoStepLTTFixed.solve(tdg, provider.location, destination, (0, 1))
            print(f"   Résultat: path={result1.path is not None}, cost={result1.total_cost}")
        except Exception as e:
            print(f"   ERREUR: {e}")
        
        debug_wrapper.report()
        
        # Reset et test avec fenêtre plus large
        print(f"\n2. Test fenêtre (0, 5):")
        debug_wrapper.call_count = 0
        debug_wrapper.errors = []
        debug_wrapper.infinite_results = []
        
        try:
            result2 = TwoStepLTTFixed.solve(tdg, provider.location, destination, (0, 5))
            print(f"   Résultat: path={result2.path is not None}, cost={result2.total_cost}")
        except Exception as e:
            print(f"   ERREUR: {e}")
            import traceback
            traceback.print_exc()
        
        debug_wrapper.report()
        
        # Test manuel de la fonction de poids avec valeurs extrêmes
        print(f"\n3. Test valeurs extrêmes:")
        test_edge = tdg.edges[0]
        test_values = [float('-inf'), float('inf'), float('nan'), -1.0, 100.0]
        
        for val in test_values:
            try:
                weight = debug_wrapper.original_function(test_edge, val)
                print(f"   f({val}) = {weight}")
            except Exception as e:
                print(f"   f({val}) = ERROR: {e}")
        
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_solver_calls()