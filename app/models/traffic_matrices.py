"""
Matrices de transition pour le système de chaînes de Markov
Basé sur ANNEXE 2-Matrices.pdf

6 états de trafic: (Fluide/Dense/Embouteillé) × (Sec/Pluie)
États: 0=Fluide-Sec, 1=Fluide-Pluie, 2=Dense-Sec, 3=Dense-Pluie, 4=Embouteillé-Sec, 5=Embouteillé-Pluie
"""

import numpy as np
from typing import Callable
from app.constants import Array6, Matrix6

def matrix_exp(A: np.ndarray, max_terms: int = 20) -> np.ndarray:
    """
    Calcule l'exponentielle de matrice e^A en utilisant la série de Taylor
    e^A = I + A + A²/2! + A³/3! + ...
    """
    n = A.shape[0]
    result = np.eye(n, dtype=A.dtype)  # I (matrice identité)
    term = np.eye(n, dtype=A.dtype)   # Term courant
    
    for k in range(1, max_terms + 1):
        term = term @ A / k  # A^k / k!
        result += term
        
        # Arrêter si la convergence est atteinte
        if np.max(np.abs(term)) < 1e-10:
            break
    
    return result

class TrafficMatrices:
    """Gestionnaire des matrices Q et P pour les chaînes de Markov"""
    
    def __init__(self):
        # Matrices Q extraites du PDF ANNEXE 2-Matrices.pdf
        # Q11: Zone urbaine - saison sèche
        self.Q11 = np.array([
            [-0.30,  0.15,  0.10,  0.05,  0.00,  0.00],
            [ 0.20, -0.40,  0.00,  0.15,  0.05,  0.00],
            [ 0.10,  0.00, -0.35,  0.05,  0.15,  0.05],
            [ 0.05,  0.20,  0.10, -0.45,  0.00,  0.10],
            [ 0.00,  0.05,  0.20,  0.00, -0.40,  0.15],
            [ 0.00,  0.00,  0.05,  0.25,  0.20, -0.50]
        ], dtype=np.float32)
        
        # Q12: Zone urbaine - saison des pluies
        self.Q12 = np.array([
            [-0.40,  0.25,  0.05,  0.10,  0.00,  0.00],
            [ 0.15, -0.50,  0.00,  0.25,  0.10,  0.00],
            [ 0.05,  0.00, -0.45,  0.10,  0.20,  0.10],
            [ 0.10,  0.30,  0.05, -0.55,  0.00,  0.10],
            [ 0.00,  0.10,  0.25,  0.00, -0.50,  0.15],
            [ 0.00,  0.00,  0.10,  0.30,  0.15, -0.55]
        ], dtype=np.float32)
        
        # Q21: Zone périurbaine - saison sèche
        self.Q21 = np.array([
            [-0.25,  0.10,  0.10,  0.05,  0.00,  0.00],
            [ 0.15, -0.35,  0.00,  0.15,  0.05,  0.00],
            [ 0.15,  0.00, -0.30,  0.05,  0.10,  0.00],
            [ 0.05,  0.15,  0.10, -0.40,  0.00,  0.10],
            [ 0.00,  0.05,  0.15,  0.00, -0.35,  0.15],
            [ 0.00,  0.00,  0.00,  0.20,  0.20, -0.40]
        ], dtype=np.float32)
        
        # Q22: Zone périurbaine - saison des pluies
        self.Q22 = np.array([
            [-0.35,  0.20,  0.05,  0.10,  0.00,  0.00],
            [ 0.10, -0.45,  0.00,  0.25,  0.10,  0.00],
            [ 0.10,  0.00, -0.40,  0.10,  0.15,  0.05],
            [ 0.10,  0.25,  0.05, -0.50,  0.00,  0.10],
            [ 0.00,  0.10,  0.20,  0.00, -0.45,  0.15],
            [ 0.00,  0.00,  0.05,  0.25,  0.15, -0.45]
        ], dtype=np.float32)
        
        # Vecteurs de vitesses pour chaque état (km/h)
        # Basé sur le PDF: [Fluide-Sec, Fluide-Pluie, Dense-Sec, Dense-Pluie, Embouteillé-Sec, Embouteillé-Pluie]
        self.velocity_vector = np.array([50.0, 40.0, 30.0, 25.0, 15.0, 10.0], dtype=np.float32)
        
        # Matrice Q moyenne
        self.Q_avg = (self.Q11 + self.Q12 + self.Q21 + self.Q22) / 4.0
        
    def get_Q_matrix(self, zone_type: str = "average", season: str = "dry") -> np.ndarray:
        """
        Retourne la matrice Q appropriée
        
        Args:
            zone_type: "urban", "periurban", "average"
            season: "dry", "wet"
        """
        if zone_type == "average":
            return self.Q_avg
        elif zone_type == "urban" and season == "dry":
            return self.Q11
        elif zone_type == "urban" and season == "wet":
            return self.Q12
        elif zone_type == "periurban" and season == "dry":
            return self.Q21
        elif zone_type == "periurban" and season == "wet":
            return self.Q22
        else:
            return self.Q_avg
    
    def create_P_function(self, zone_type: str = "average", season: str = "dry") -> Callable[[float], np.ndarray]:
        """
        Crée une fonction P(t) = e^(Qt) pour une matrice Q donnée
        
        Args:
            zone_type: Type de zone
            season: Saison
            
        Returns:
            Fonction qui calcule P(t) = e^(Qt)
        """
        Q = self.get_Q_matrix(zone_type, season)
        
        def P_function(t: float) -> np.ndarray:
            """Calcule P(t) = e^(Qt)"""
            return matrix_exp(Q * t).astype(np.float32)
        
        return P_function
    
    def get_velocity_vector(self) -> Array6[np.float32]:
        """Retourne le vecteur des vitesses nominales"""
        return self.velocity_vector
    
    def get_initial_distribution(self, traffic_level: str = "medium") -> Array6[np.float32]:
        """
        Retourne une distribution initiale d'états
        
        Args:
            traffic_level: "light", "medium", "heavy"
        """
        if traffic_level == "light":
            # Plus de probabilité d'états fluides
            return np.array([0.4, 0.2, 0.2, 0.1, 0.08, 0.02], dtype=np.float32)
        elif traffic_level == "heavy":
            # Plus de probabilité d'états embouteillés
            return np.array([0.1, 0.05, 0.2, 0.15, 0.3, 0.2], dtype=np.float32)
        else:  # medium
            # Distribution équilibrée
            return np.array([0.25, 0.15, 0.25, 0.15, 0.15, 0.05], dtype=np.float32)

# Instance globale pour utilisation dans le système
traffic_matrices = TrafficMatrices()