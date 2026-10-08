"""Variante plus petite de l'avion : configuration 2-2 (une seule allee).

Reutilise generate_cabin.py en changeant la section du fuselage et
l'amenagement : pas de coffres centraux, et a l'avant les pieces se suivent
(entree -> cuisine -> chambre -> cockpit) au lieu d'un couloir central.

Usage : python3 generate_avion_2x2.py [sortie.glb]
"""
import os
import sys

import generate_cabin as G
from generate_cabin import Profile, arch, bulkhead, build_ceiling, build_open_zone, flat_door, floor, B

# section de petit avion de ligne : ~2,8 m de large interieur
G.WALL = Profile([(1.25, 0.00), (1.33, 0.35), (1.39, 0.80), (1.40, 1.20),
                  (1.36, 1.48), (1.27, 1.64)])
G.BIN_BOTTOM = Profile([(1.27, 1.64), (1.06, 1.655), (0.86, 1.68)])
G.BIN_FACE = Profile([(0.86, 1.68), (0.79, 1.76), (0.77, 1.88), (0.80, 1.99), (0.88, 2.06)])
G.CEILING = Profile([(0.88, 2.06), (0.80, 2.14), (0.62, 2.19), (0.30, 2.215), (0.0, 2.22)])
G.WALL_FULL = Profile([(1.25, 0.00), (1.33, 0.35), (1.39, 0.80), (1.40, 1.20),
                       (1.37, 1.50), (1.28, 1.78), (1.10, 1.98), (0.88, 2.06)])
G.BIN_PROFILES = [G.WALL, G.BIN_BOTTOM, G.BIN_FACE, G.CEILING]
G.OPEN_PROFILES = [G.WALL_FULL, G.CEILING]
G.FLOOR_X = G.WALL.x[0]
G.OUTLINE_BINS = G.outline(G.BIN_PROFILES)
G.OUTLINE_OPEN = G.outline(G.OPEN_PROFILES)

G.CENTER_BINS = False
G.AISLE_LINES = (-0.42, 0.42)       # une allee centrale
G.CEIL_LIGHT_X = (0.55,)
G.PAX_DOOR_W = 0.95
G.LAV_HALF = 0.62
G.LAV_CX = (-0.62, 0.62)
G.BED_X = 0.46                      # lit contre la paroi droite
G.COCKPIT_SCALE = (0.70, 0.92)

# longueurs des zones
G.N_WINDOWS = 16
G.LENGTH = 2 * G.MARGIN + (G.N_WINDOWS - 1) * G.PITCH
DOOR_ZONE, LAV_LEN, GALLEY_LEN, ROOM_LEN, COCKPIT_LEN = 2.2, 1.6, 3.0, 3.4, 2.6
G.REAR_DOOR = (-DOOR_ZONE, 0.0)
G.LAV = (G.REAR_DOOR[0] - LAV_LEN, G.REAR_DOOR[0])
G.CABIN = (0.0, G.LENGTH)
G.FRONT_DOOR = (G.LENGTH, G.LENGTH + DOOR_ZONE)
G.CREW = (G.FRONT_DOOR[1], G.FRONT_DOOR[1] + GALLEY_LEN + ROOM_LEN)
G.COCKPIT = (G.CREW[1], G.CREW[1] + COCKPIT_LEN)

DOOR_X, DOOR_W, DOOR_H = -0.75, 0.70, 1.90   # portes alignees sur le cote gauche


def build_crew(parts):
    """Entree -> cuisine (vide) -> chambre (avec le lit) -> cockpit."""
    z0, z1 = G.CREW
    zg = z0 + GALLEY_LEN
    build_open_zone(parts, z0, z1)
    build_ceiling(parts, z0, z1)
    floor(parts, z0, z1)
    floor(parts, z0 + G.PART_T, zg - G.PART_T, key="vinyl", y=0.002)   # sol de la cuisine
    G.IN_DOOR_H = DOOR_H
    a, b = DOOR_X - DOOR_W / 2, DOOR_X + DOOR_W / 2
    for z, name, hinge_start in ((z0 + G.PART_T, "Porte_Cuisine", True),
                                 (zg, "Porte_Chambre", True),
                                 (z1, "Porte_Cockpit", False)):
        bulkhead(parts, z, [arch(DOOR_X, DOOR_W, DOOR_H)])
        flat_door(name, "x", DOOR_X, a, b, z - G.PART_T / 2, hinge_at_start=hinge_start)
    G.IN_DOOR_H = 2.00
    # liseuses murales de part et d'autre du lit
    for x in (G.BED_X - 1.0, G.BED_X + 0.85):
        parts.add("light", B(x - 0.05, x + 0.05, 1.25, 1.33, z1 - G.PART_T - 0.02, z1 - G.PART_T))


G.build_crew = build_crew

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(G.HERE, "avion_2x2.glb")
    sc = G.build()
    sc.export(out)
    total = G.COCKPIT[1] - G.LAV[0]
    print(f"{out}: longueur totale {total:.2f} m, {sum(len(g.faces) for g in sc.geometry.values())} triangles")
