"""Avion niveau 2 : le meme avion complet, mais cabine elargie pour 2 | 2 sieges.

Le niveau 1 (avion_complet.glb) a une cabine de 2,8 m de large : 1 siege de chaque cote.
Ici la cabine fait 3,6 m (comme un vrai A320) : 2 sieges (60 cm chacun) de chaque cote + l'allee.
Les hauteurs, la longueur, la position et l'exterieur A320 ne changent pas
(le fuselage exterieur est assez large : 2,02 m de rayon).

Meme structure que avion_complet.glb : pieces separees, portes a part (pivot sur la charniere).

Usage : python3 generate_avion_niveau2.py [sortie.glb]
"""
import os
import sys

import numpy as np

import generate_avion_2x2  # noqa: F401  (configure le petit avion)
import generate_cabin as G
from generate_cabin import Profile

F = 1.30            # elargissement de la cabine (2,8 m -> 3,6 m)


def large(pts):
    return Profile([(x * F, y) for x, y in pts])


G.WALL = large([(1.25, 0.00), (1.33, 0.35), (1.39, 0.80), (1.40, 1.20), (1.36, 1.48), (1.27, 1.64)])
G.BIN_BOTTOM = large([(1.27, 1.64), (1.06, 1.655), (0.86, 1.68)])
G.BIN_FACE = large([(0.86, 1.68), (0.79, 1.76), (0.77, 1.88), (0.80, 1.99), (0.88, 2.06)])
G.CEILING = large([(0.88, 2.06), (0.80, 2.14), (0.62, 2.19), (0.30, 2.215), (0.0, 2.22)])
G.WALL_FULL = large([(1.25, 0.00), (1.33, 0.35), (1.39, 0.80), (1.40, 1.20),
                     (1.37, 1.50), (1.28, 1.78), (1.10, 1.98), (0.88, 2.06)])
G.BIN_PROFILES = [G.WALL, G.BIN_BOTTOM, G.BIN_FACE, G.CEILING]
G.OPEN_PROFILES = [G.WALL_FULL, G.CEILING]
G.FLOOR_X = G.WALL.x[0]
G.OUTLINE_BINS = G.outline(G.BIN_PROFILES)
G.OUTLINE_OPEN = G.outline(G.OPEN_PROFILES)

G.AISLE_LINES = (-0.32, 0.32)       # allee centrale (60 cm)
G.CEIL_LIGHT_X = (0.72,)
G.LAV_HALF = 0.80
G.LAV_CX = (-0.80, 0.80)
G.BED_X = 0.60

import generate_avion_exterieur as E  # noqa: E402  (apres le reglage de la cabine)

# hublots et portes : l'embrasure va du mur interieur jusqu'a la peau exterieure
MUR_MAX = float(G.WALL.x.max())
G.WIN_DEPTH = E.R_EXT - MUR_MAX + 0.05
G.PAX_REVEAL = E.R_EXT - float(G.WALL_FULL.x[np.argmin(np.abs(G.WALL_FULL.y - 1.0))]) + 0.05


def nose_scale(t):
    """Le cockpit (plus large) suit toujours le nez exterieur."""
    zi = G.COCKPIT[0] + t * (G.COCKPIT[1] - G.COCKPIT[0])
    cy, hy, hz = E.ext_section(zi)
    sx = min(1.0, (hz - 0.30) / MUR_MAX)
    sy = min(1.0, (cy + hy - 0.15) / G.CEILING.y[-1])
    return sx, sy


G.nose_scale = nose_scale
E.nose_scale = nose_scale

# portes passagers : battant 8 cm plus pres du mur (sinon il depasserait de la peau)
import inspect  # noqa: E402
_src = inspect.getsource(G.pax_door)
for a, b in (("np.array([hinge_z]), 0.15)", "np.array([hinge_z]), 0.07)"),
             ("leaf, 0.12, 0.03", "leaf, 0.04, 0.03"), ("leaf, 0.20, 0.03", "leaf, 0.12, 0.03"),
             ("leaf.exterior.coords, 0.12, 0.20", "leaf.exterior.coords, 0.04, 0.12"),
             ("port, 0.117, 0.02", "port, 0.037, 0.02"), ("0.556), 0.118, 0.01", "0.556), 0.038, 0.01"),
             ("np.array([door_z - 0.33]), 0.12)", "np.array([door_z - 0.33]), 0.04)")):
    assert a in _src, a
    _src = _src.replace(a, b)
exec(_src, G.__dict__)

MARGE = 0.03


def epouser_la_peau(sc):
    """Aux deux bouts (nez et queue qui se retrecissent), ramene ce qui depasse a l'interieur de la peau."""
    for node in sc.graph.nodes_geometry:
        T, g = sc.graph[node]
        if g.startswith("A320") or g in ("Fuselage", "Details") or not np.allclose(T, np.eye(4)):
            continue
        m = sc.geometry[g]
        v = m.vertices.copy()
        for i, (x, y, z) in enumerate(v):
            cy, hy, hz = E.ext_section(-z)
            r = np.hypot(x / max(hz - MARGE, 1e-3), (y - cy) / max(hy - MARGE, 1e-3))
            if r > 1.0:
                v[i] = [x / r, cy + (y - cy) / r, z]
        m.vertices = v


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(G.HERE, "avion_niveau2.glb")
    sc = E.build()
    epouser_la_peau(sc)
    sc.export(out)
    print(f"{out}: cabine {2 * MUR_MAX:.2f} m de large, {sum(len(g.faces) for g in sc.geometry.values())} triangles")
