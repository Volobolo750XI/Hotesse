"""Avion niveau 1 (1 | 1, le tout premier) : identique a avion_complet.glb,
mais les 2 cabines de toilettes sont VIDES (cloisons, facade et portes gardees ;
pas de cuvette, lavabo, robinet, miroir : ca s'achete par dalles).

Usage : python3 generate_avion_niveau1_wc_vides.py [sortie.glb]
"""
import os
import sys

import generate_avion_exterieur as E
import generate_cabin as G
from generate_cabin import arch


def toilettes_vides(parts):
    z0, z1 = G.LAV
    G.build_open_zone(parts, z0, z1)
    G.build_ceiling(parts, z0, z1, lights=False)
    G.floor(parts, z0, z1)
    G.bulkhead(parts, z0, thickness=0.0001)
    G.bulkhead(parts, z1, [arch(cx, 0.72, G.IN_DOOR_H) for cx in G.LAV_CX])
    for x in (-2 * G.LAV_HALF, 0.0, 2 * G.LAV_HALF):
        parts.add("panel", G.B(x - G.PART_T / 2, x + G.PART_T / 2, 0, G.CEILING.y[-1] - 0.07, z0, z1 - G.PART_T))
    for cx in G.LAV_CX:
        a, b = cx - G.LAV_HALF + 0.02, cx + G.LAV_HALF - 0.02
        G.floor(parts, z0, z1 - G.PART_T, a + 0.03, b - 0.03, key="vinyl", y=0.002)
        parts.add("light", G.B(cx - 0.20, cx + 0.20, G.CEILING.y[-1] - 0.12, G.CEILING.y[-1] - 0.11, z0 + 0.6, z0 + 1.2))
        G.flat_door(f"Porte_WC_{'D' if cx > 0 else 'G'}", "x", cx, cx - 0.36, cx + 0.36,
                    z1 - G.PART_T / 2, hinge_at_start=cx > 0)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(G.HERE, "avion_niveau1_wc_vides.glb")
    G.build_lavatories = toilettes_vides
    sc = E.build()
    for n in [n for n in sc.graph.nodes_geometry if n.startswith("WC_")]:    # pas de cuvettes
        sc.delete_geometry(sc.graph[n][1])
    sc.export(out)
    print(out, sum(len(g.faces) for g in sc.geometry.values()), "triangles")
