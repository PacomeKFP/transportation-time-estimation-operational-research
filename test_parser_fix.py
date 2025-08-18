"""Test du parser avec une logique corrigée"""

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class RoadSection:
    section_id: str
    length: float
    characteristics: str = ""

@dataclass
class RouteArc:
    start_node: str
    end_node: str
    sections: List[RoadSection]

def parse_route_graphs_fixed(file_path: str) -> Dict[str, List[RouteArc]]:
    """Parse corrigé"""
    graphs = {}
    current_supplier = None
    current_arc = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Nouveau fournisseur
        if line_stripped.startswith('FOURNISSEUR'):
            current_supplier = line_stripped.replace('FOURNISSEUR', '').strip()
            graphs[current_supplier] = []
            current_arc = None
            continue
        
        # Ignorer en-tête
        if line_stripped.startswith('Arcs'):
            continue
        
        parts = line.split('\t')
        if len(parts) >= 3:
            arc_name = parts[0].strip()
            section_id = parts[1].strip()
            length_str = parts[2].strip()
            characteristics = parts[3].strip() if len(parts) > 3 else ""
            
            # Nouveau arc si arc_name non vide
            if arc_name and '-' in arc_name:
                start_node, end_node = arc_name.split('-')
                current_arc = RouteArc(start_node, end_node, [])
                graphs[current_supplier].append(current_arc)
            
            # Ajouter tronçon si possible
            if current_arc and section_id and length_str:
                try:
                    length = float(length_str.replace(',', '.'))
                    section = RoadSection(section_id, length, characteristics)
                    current_arc.sections.append(section)
                    print(f"Ajouté: {section_id}({length}km) à {current_arc.start_node}-{current_arc.end_node}")
                except ValueError:
                    continue
    
    return graphs

# Test
graphs = parse_route_graphs_fixed('xl/a.txt')

print("=== RÉSULTATS ===")
for supplier_name, arcs in graphs.items():
    print(f"\n{supplier_name}:")
    for arc in arcs:
        sections_info = [f"{s.section_id}({s.length}km)" for s in arc.sections]
        print(f"  {arc.start_node}-{arc.end_node}: {', '.join(sections_info)} (Total: {len(arc.sections)} tronçons)")