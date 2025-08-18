# Rapport d'Analyse des Vrais Algorithmes ORDA-ROM

## Introduction

Après analyse des documents sources originaux, il est clair que l'implémentation actuelle dans `orda_rom_solver.py` ne correspond pas aux vrais algorithmes d'Orda et Rom. Ce rapport présente une analyse détaillée des algorithmes authentiques basée sur les deux papers académiques fournis.

## 1. Les Vrais Algorithmes ORDA-ROM (1990)

### 1.1 Contexte et Modèles de Waiting

L'article original d'Orda & Rom (1990) définit **trois modèles d'attente** pour les graphes time-dependent :

#### **Unrestricted Waiting (UW)**
- Attente illimitée autorisée à chaque nœud
- Deux algorithmes : **UW1** et **UW2**

#### **Forbidden Waiting (FW)**  
- Aucune attente autorisée nulle part
- Peut mener à des chemins infinis avec délai fini
- Complexité NP-hard

#### **Source Waiting (SW)**
- Attente autorisée uniquement au nœud source
- Deux algorithmes : **SW1** et **SW2**

### 1.2 Algorithmes UW1 et UW2

#### **UW1 - Pour un temps de départ donné**

```
Algorithm UW1:
1. Initialization: Xs ← ts; fs ← NIL; ∀k≠s Yk ← ∞, Xk ← NULL, fk ← NIL; j ← s;
2. For all neighbors k of j for which Xk = NULL, do:
   a. Yk ← min{Yk, Xj + Djk(Xj)}
   b. If Yk changed in Step 2(a), then set fk ← j.
3. If all nodes have nonnull X-value, then stop.
   Otherwise, let l be a node for which Xl = NULL and such that Yl ≤ Yk ∀k for which Xk = NULL.
   Set Xl ← Yl, j ← l, and proceed with Step 2.
```

**Fonction clé** : `Dik(t) = min{τ + dik(t + τ) | τ ≥ 0}`

#### **UW2 - Pour tous les temps de départ**

Utilise des **fonctions** au lieu de valeurs scalaires :
- `Xi(t)` : temps d'arrivée minimal au nœud i pour temps de départ t
- `Yik(t)` : temps d'arrivée à k via i pour temps de départ t

### 1.3 Algorithmes SW1 et SW2

#### **SW1 - Temps de départ donné**
1. Calcule d'abord le chemin optimal avec UW1
2. Reconstitue les temps de départ en "backward" depuis la destination
3. Calcule le temps d'attente source nécessaire

#### **SW2 - Tous les temps de départ**
Utilise UW2 puis calcule les fonctions `WAIT(s, w, t)` d'attente source.

## 2. L'Algorithme TWO-STEP-LTT (2008)

### 2.1 Innovation de Ding et al.

L'article de 2008 propose **TWO-STEP-LTT** qui améliore OR (Orda-Rom) :

#### **Étape 1 - Time Refinement**
- Découple la sélection de chemin du raffinement temporel
- Algorithme basé sur Dijkstra
- Complexité : `O((n log n + m)α(T))`

#### **Étape 2 - Path Selection**  
- Sélection rapide du chemin optimal
- Complexité : `O(mα(T))`

### 2.2 Avantages par rapport à OR

1. **Découplement** : Sépare path-selection et time-refinement
2. **Efficacité** : Meilleure complexité que OR : `O(nmα(T))`
3. **Graphes FIFO** : Optimisé pour les réseaux routiers
4. **Scalabilité** : Peut traiter de gros graphes

### 2.3 Algorithme TWO-STEP-LTT Détaillé

```
Algorithm TWO-STEP-LTT(GT, vs, ve, T):
1. {gi(t)} ← timeRefinement(GT, vs, ve, T);
2. if ¬(ge(t) = ∞ for the entire [ts, te]) then
3.   t* ← argmin[t∈T]{ge(t) - t};
4.   p* ← pathSelection(GT, {gi(t)}, vs, ve, t*);
5.   return (t*, p*);
6. else return ∅;
```

## 3. Différences Fondamentales

### 3.1 ORDA-ROM vs TWO-STEP-LTT

| Aspect | ORDA-ROM (1990) | TWO-STEP-LTT (2008) |
|--------|------------------|----------------------|
| **Approche** | Bellman-Ford généralisé | Dijkstra-based |
| **Complexité** | O(nmα(T)) | O((n log n + m)α(T)) |
| **Fonctions** | hk,l(t) + gi(t) | Seulement gi(t) |
| **Graphes** | Généraux | FIFO optimisé |
| **Stratégie** | Path-selection + time-refinement couplés | Découplage en 2 étapes |

### 3.2 Modèles d'Attente

- **UW** : Plus court chemin avec attente libre
- **SW** : Équivalent à UW pour fonctions continues  
- **FW** : Problème NP-hard, chemins potentiellement infinis

## 4. Erreurs de l'Implémentation Actuelle

### 4.1 Problèmes Identifiés

1. **Algorithmes inventés** : Les méthodes `solve_orda()` et `solve_rom()` ne correspondent à aucun algorithme des papers
2. **Confusion conceptuelle** : "ORDA" et "ROM" ne sont pas deux algorithmes séparés
3. **Structure incorrecte** : L'implémentation ressemble plus à TWO-STEP-LTT qu'à OR
4. **Fonction Dik manquante** : Pas d'implémentation de l'attente optimale
5. **Modèles d'attente ignorés** : Pas de distinction UW/SW/FW

### 4.2 Code Problématique

```python
# INCORRECT - N'existe pas dans les papers
def solve_orda(tdg, source, destination, time_window):
    # Cette méthode est une invention
    
def solve_rom(tdg, source, destination, time_window):  
    # Cette méthode est aussi une invention
```

## 5. Implémentations Correctes Recommandées

### 5.1 Pour Orda-Rom UW1 (Temps de départ fixé)

```python
class OrdaRomSolver:
    @staticmethod
    def solve_UW1(tdg: TDG, source: Intersection, destination: Intersection, 
                  start_time: float) -> OrdaResult:
        """Algorithme UW1 d'Orda-Rom pour temps de départ fixé"""
        # 1. Calculer Dik(t) = min{τ + dik(t + τ) | τ ≥ 0} pour chaque edge
        # 2. Appliquer Dijkstra généralisé avec Dik au lieu de dik
        # 3. Retourner chemin optimal et coût
```

### 5.2 Pour TWO-STEP-LTT (Amélioration moderne)

```python
class TwoStepLTTSolver:
    @staticmethod
    def solve(tdg: TDG, source: Intersection, destination: Intersection,
              time_window: Tuple[float, float]) -> OrdaResult:
        """Algorithme TWO-STEP-LTT de Ding et al. (2008)"""
        # 1. Time Refinement - Calculer gi(t) pour tous les nœuds
        # 2. Path Selection - Sélectionner chemin optimal
```

## 6. Recommandations

### 6.1 Actions Immédiates

1. **Supprimer** l'implémentation actuelle incorrecte
2. **Implémenter UW1** pour les cas simples (temps de départ fixé)
3. **Implémenter TWO-STEP-LTT** pour l'efficacité sur gros graphes
4. **Ajouter support** pour les modèles d'attente (UW/SW/FW)

### 6.2 Architecture Suggérée

```python
# Structure recommandée
class TimeDependendentSolver:
    def solve_UW1(self, start_time)      # Orda-Rom pour temps fixé
    def solve_UW2(self, time_interval)   # Orda-Rom pour intervalle
    def solve_SW1(self, start_time)      # Source waiting
    def solve_TWO_STEP_LTT(self, time_interval)  # Approche moderne efficace
```

## 7. Conclusion

L'implémentation actuelle ne correspond pas aux algorithmes historiques d'Orda-Rom. Une refonte complète est nécessaire pour :

1. **Respecter les algorithmes originaux** UW1, UW2, SW1, SW2
2. **Implémenter correctement** la fonction d'attente optimale Dik(t)
3. **Ajouter TWO-STEP-LTT** pour l'efficacité moderne
4. **Supporter les modèles d'attente** appropriés

Cette analyse révèle un écart significatif entre l'implémentation actuelle et les algorithmes académiques de référence.