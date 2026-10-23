"""Avion niveau 2 : la zone des sieges est elargie pour 2 | 2 sieges.

Seule la CABINE (la zone des sieges) passe de 2,8 m a 3,65 m de large.
Tout le reste est IDENTIQUE au niveau 1 (avion_complet.glb) : entrees avec les portes passagers,
toilettes, cuisine, chambre, cockpit, et donc les 9 portes au meme endroit.
=> dans UEFN, les memes portes / boutons / devices servent pour tous les avions.
Une cloison de raccord ferme l'ecart entre la cabine large et les zones d'entree.

Exterieur : le meme A320, mais ailes un peu reculees et reacteurs sous les ailes.

Usage : python3 generate_avion_niveau2.py [sortie.glb]
"""
import os
import sys

import numpy as np
from shapely.geometry import box as sbox

import generate_avion_2x2  # noqa: F401  (configure le petit avion)
import generate_cabin as G
from generate_cabin import Profile, extrude_xy

import generate_avion_exterieur as E  # noqa: E402

F = 1.30            # elargissement de la cabine (2,8 m -> 3,65 m)

REGLAGES = ("WALL", "BIN_BOTTOM", "BIN_FACE", "CEILING", "WALL_FULL", "BIN_PROFILES", "OPEN_PROFILES",
            "FLOOR_X", "OUTLINE_BINS", "OUTLINE_OPEN", "AISLE_LINES", "CEIL_LIGHT_X", "WIN_DEPTH", "PAX_REVEAL")
ETROIT = {k: getattr(G, k) for k in REGLAGES}


def large(pts):
    return Profile([(x * F, y) for x, y in pts])


def cabine_large():
    W = large([(1.25, 0.00), (1.33, 0.35), (1.39, 0.80), (1.40, 1.20), (1.36, 1.48), (1.27, 1.64)])
    BB = large([(1.27, 1.64), (1.06, 1.655), (0.86, 1.68)])
    BF = large([(0.86, 1.68), (0.79, 1.76), (0.77, 1.88), (0.80, 1.99), (0.88, 2.06)])
    C = large([(0.88, 2.06), (0.80, 2.14), (0.62, 2.19), (0.30, 2.215), (0.0, 2.22)])
    WF = large([(1.25, 0.00), (1.33, 0.35), (1.39, 0.80), (1.40, 1.20),
                (1.37, 1.50), (1.28, 1.78), (1.10, 1.98), (0.88, 2.06)])
    bins, ouvert = [W, BB, BF, C], [WF, C]
    return {"WALL": W, "BIN_BOTTOM": BB, "BIN_FACE": BF, "CEILING": C, "WALL_FULL": WF,
            "BIN_PROFILES": bins, "OPEN_PROFILES": ouvert, "FLOOR_X": W.x[0],
            "OUTLINE_BINS": G.outline(bins), "OUTLINE_OPEN": G.outline(ouvert),
            "AISLE_LINES": (-0.32, 0.32), "CEIL_LIGHT_X": (0.72,),
            # embrasure des hublots : du mur jusqu'a la peau exterieure
            "WIN_DEPTH": E.R_EXT - float(W.x.max()) + 0.05, "PAX_REVEAL": ETROIT["PAX_REVEAL"]}


LARGE = cabine_large()


def regler(d):
    for k, v in d.items():
        setattr(G, k, v)


def installer():
    """La cabine (zone des sieges) en large, tout le reste en etroit (comme le niveau 1)."""
    cabine, plafond, sol = G.build_cabin, G.build_ceiling, G.floor

    def build_cabin(parts):
        regler(LARGE)
        cabine(parts)
        # cloisons de raccord : ecart entre la cabine large et les zones etroites, aux deux bouts
        raccord = LARGE["OUTLINE_BINS"].difference(ETROIT["OUTLINE_OPEN"]).intersection(sbox(-9, 0, 9, 9))
        for z in G.CABIN:
            parts.add("panel", extrude_xy(raccord, z - 0.02, z + 0.02))

    def selon_zone(f):
        def g(parts, z0, z1, *a, **k):
            regler(LARGE if (z0, z1) == tuple(G.CABIN) else ETROIT)
            return f(parts, z0, z1, *a, **k)
        return g

    def etroit(f):
        def g(*a, **k):
            regler(ETROIT)
            return f(*a, **k)
        return g

    G.build_cabin = build_cabin
    G.build_ceiling = selon_zone(plafond)
    G.floor = selon_zone(sol)
    for nom in ("build_open_zone", "build_lavatories", "build_crew", "build_cockpit"):
        setattr(G, nom, etroit(getattr(G, nom)))


RECUL_AILE = 0.6        # m vers l'arriere : ailes, carenage du ventre, train principal
RECUL_MOTEUR = 3.4      # m vers l'arriere : reacteurs (leur centre sous le milieu de l'aile)


def reculer_ailes(sc):
    for node in sc.graph.nodes_geometry:
        T, g = sc.graph[node]
        if not g.startswith("A320"):
            continue
        m = sc.geometry[g]
        v = m.vertices.copy()
        nom = g.split("_")[1]
        if nom == "Cylinder002":
            # aile + carenage du ventre (pas l'empennage, a l'arriere)
            k = (v[:, 2] < 5.0) & ((np.abs(v[:, 0]) > 1.9) | (v[:, 1] < -0.3))
            v[k, 2] += RECUL_AILE
        elif nom in ("Cylinder007", "Cylinder013"):
            v[:, 2] += RECUL_MOTEUR
        elif nom in ("Cylinder009", "Cylinder017", "Cylinder018", "Cylinder019", "Cylinder020", "Cylinder021"):
            v[:, 2] += RECUL_AILE
        else:
            continue
        m.vertices = v


def toilettes_vides(parts):
    """2 cabines de toilettes VIDES : cloisons, facade et portes, mais rien dedans
    (pas de lavabo, miroir, cuvette : ca s'achete par dalles)."""
    z0, z1 = G.LAV
    G.build_open_zone(parts, z0, z1)
    G.build_ceiling(parts, z0, z1, lights=False)
    G.floor(parts, z0, z1)
    G.bulkhead(parts, z0, thickness=0.0001)
    G.bulkhead(parts, z1, [G.arch(cx, 0.72, G.IN_DOOR_H) for cx in G.LAV_CX])
    for x in (-2 * G.LAV_HALF, 0.0, 2 * G.LAV_HALF):
        parts.add("panel", G.B(x - G.PART_T / 2, x + G.PART_T / 2, 0, G.CEILING.y[-1] - 0.07, z0, z1 - G.PART_T))
    for cx in G.LAV_CX:
        a, b = cx - G.LAV_HALF + 0.02, cx + G.LAV_HALF - 0.02
        G.floor(parts, z0, z1 - G.PART_T, a + 0.03, b - 0.03, key="vinyl", y=0.002)
        parts.add("light", G.B(cx - 0.20, cx + 0.20, G.CEILING.y[-1] - 0.12, G.CEILING.y[-1] - 0.11, z0 + 0.6, z0 + 1.2))
        G.flat_door(f"Porte_WC_{'D' if cx > 0 else 'G'}", "x", cx, cx - 0.36, cx + 0.36,
                    z1 - G.PART_T / 2, hinge_at_start=cx > 0)


def enlever_cuvettes(sc):
    for n in [n for n in sc.graph.nodes_geometry if n.startswith("WC_")]:
        sc.delete_geometry(sc.graph[n][1])


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(G.HERE, "avion_niveau2.glb")
    G.build_lavatories = toilettes_vides
    installer()
    sc = E.build()
    reculer_ailes(sc)
    enlever_cuvettes(sc)
    sc.export(out)
    print(f"{out}: cabine {2 * float(LARGE['WALL'].x.max()):.2f} m de large, "
          f"{sum(len(g.faces) for g in sc.geometry.values())} triangles")
