"""
Debug du ChunkTransportationTimeEstimator
"""

import numpy as np
from app.models.edge import Edge
from app.models.chunk import Chunk
from app.models.intersection import Intersection
from app.time_estimators import EdgeTransportationTimeEstimator, ChunkTransportationTimeEstimator
from app.models.traffic_matrices import traffic_matrices

def debug_chunk_estimator():
    """Debug détaillé du système d'estimation"""
    print("=== DEBUG CHUNK ESTIMATOR ===")
    
    # Créer un chunk simple
    chunk = Chunk(length=np.float32(1.0))  # 1 km
    print(f"Chunk: {chunk.length}km")
    
    # Distribution et matrice P
    distribution = traffic_matrices.get_initial_distribution("medium")
    P_function = traffic_matrices.create_P_function("average")
    
    print(f"Distribution initiale: {distribution}")
    print(f"Somme distribution: {np.sum(distribution)}")
    
    # Test P(t) pour différentes valeurs
    print(f"\nTest P(t):")
    for t in [0.01, 0.1, 1.0]:
        P_t = P_function(t)
        print(f"  P({t}): shape={P_t.shape}, somme_lignes={np.sum(P_t, axis=1)}")
    
    # Test ChunkTransportationTimeEstimator
    print(f"\nTest ChunkTransportationTimeEstimator:")
    try:
        estimator = ChunkTransportationTimeEstimator(
            chunk=chunk,
            time_origin=np.float32(0.0),
            states_distribution=distribution,
            P=P_function,
            time_differential=np.float32(0.1)
        )
        
        print(f"Estimateur créé, démarrage estimation...")
        
        # Ajouter debug dans estimate()
        traveled_distance = 0.0
        time_estimation = 0.0
        iterations = 0
        max_iterations = 1000  # Limite de sécurité
        
        while traveled_distance < chunk.length and iterations < max_iterations:
            # Vitesses nominales depuis chunk_state.py
            nominal_velocities = np.array([50.0, 40.0, 30.0, 25.0, 15.0, 10.0], dtype=np.float32)
            velocity = np.sum(estimator.states_distribution * nominal_velocities)
            
            print(f"  Iter {iterations}: distance={traveled_distance:.4f}, vitesse={velocity:.2f} km/h")
            
            if velocity <= 0:
                print(f"    ERREUR: Vitesse nulle ou négative!")
                break
                
            temporal_distance = velocity * estimator.time_differential
            if temporal_distance + traveled_distance >= chunk.length:
                temporal_distance = chunk.length - traveled_distance
            
            traveling_time = temporal_distance / velocity
            time_estimation += traveling_time
            traveled_distance += temporal_distance
            
            # Mise à jour distribution
            P_t = estimator.P(traveling_time)
            estimator.states_distribution = estimator.states_distribution @ P_t
            
            iterations += 1
            
            # Debug premières itérations
            if iterations <= 5:
                print(f"    temporal_distance={temporal_distance:.4f}, traveling_time={traveling_time:.6f}")
                print(f"    nouvelle_distribution={estimator.states_distribution}")
        
        if iterations >= max_iterations:
            print(f"  ERREUR: Trop d'itérations ({iterations})")
            return float('inf')
        
        print(f"  Résultat: {time_estimation:.6f}h en {iterations} itérations")
        
    except Exception as e:
        print(f"  ERREUR: {e}")
        import traceback
        traceback.print_exc()
    
    # Test EdgeTransportationTimeEstimator
    print(f"\nTest EdgeTransportationTimeEstimator:")
    try:
        # Créer une arête simple
        start = Intersection("A")
        end = Intersection("B")
        chunks = [Chunk(length=np.float32(0.5)), Chunk(length=np.float32(0.5))]
        edge = Edge(start, end, chunks)
        
        edge_estimator = EdgeTransportationTimeEstimator(
            states_distribution=distribution,
            P=P_function
        )
        
        result = edge_estimator.estimate(edge, np.float32(0.0))
        print(f"  Edge estimation: {result:.6f}h")
        
        if np.isinf(result) or np.isnan(result):
            print(f"  PROBLEME: Valeur infinie ou NaN!")
        
    except Exception as e:
        print(f"  ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_chunk_estimator()