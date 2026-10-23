"""Avion niveau 2 avec une GRANDE zone Business : l'avion 2 | 2, allonge de ~8,5 m.

La zone Business (8,5 m) est inseree juste derriere la porte avant, devant la classe eco.
L'avant de l'avion (cockpit, chambre, cuisine, porte avant) ne bouge PAS ;
c'est l'arriere (eco, porte arriere, toilettes, queue) qui recule de 8,5 m.
Exterieur : le meme A320, avec un troncon de fuselage en plus derriere les ailes.
Les sieges, le rideau et la deco Business sont des objets a part (dalles).

Usage : python3 generate_avion_niveau2_business.py [sortie.glb]
"""
import os
import sys

import numpy as np
import trimesh

import generate_avion_2x2  # noqa: F401  (configure le petit avion)
import generate_cabin as G

# ---------------------------------------------------------------- allongement
N_EN_PLUS = 16                          # hublots en plus (16 x 0,53 m = 8,48 m)
EXTRA = N_EN_PLUS * G.PITCH
FIN_ECO = G.CABIN[1]                    # avant de la cabine (ne bouge pas)
G.N_WINDOWS += N_EN_PLUS
G.LENGTH += EXTRA
G.CABIN = (G.CABIN[0] - EXTRA, G.CABIN[1])
G.REAR_DOOR = (G.REAR_DOOR[0] - EXTRA, G.REAR_DOOR[1] - EXTRA)
G.LAV = (G.LAV[0] - EXTRA, G.LAV[1] - EXTRA)
DEBUT_BUSINESS = FIN_ECO - EXTRA         # la Business va de DEBUT_BUSINESS a FIN_ECO

import generate_avion_exterieur as E  # noqa: E402

# l'arriere de l'A320 (derriere les ailes) recule de EXTRA : troncon droit insere en X_COUPE
X_COUPE = 1.0                           # derriere le bout des ailes, avant l empennage (repere A320)
_zi_base = E.zi_of


def zi_of(x):
    return E.OFF - x * E.K - (EXTRA if x > X_COUPE else 0.0)


def x_of(zi):
    z_coupe = E.OFF - X_COUPE * E.K
    if zi >= z_coupe:
        return (E.OFF - zi) / E.K
    if zi <= z_coupe - EXTRA:
        return (E.OFF - zi - EXTRA) / E.K
    return X_COUPE                      # dans le troncon ajoute : section constante


E.zi_of = zi_of
E.x_of = x_of
E.STRAIGHT = (G.REAR_DOOR[0] + 0.5, zi_of(E.MID_X[0]))
E.NOSE_TIP = zi_of(E.NOSE[0][0])
E.TAIL_TIP = zi_of(E.TAIL[-1][0])

import generate_avion_niveau2 as N  # noqa: E402  (cabine large 2 | 2, ailes reculees)
from generate_cabin import arch, extrude_xy, mat  # noqa: E402
from shapely.geometry import box as sbox  # noqa: E402


# ---------------------------------------------------------------- hublots
def profondeur_hublot():
    """Profondeur exacte de l'embrasure : du mur (au centre du hublot) jusqu'a la peau exterieure."""
    W = N.LARGE["WALL"]
    def dist(d):
        p, _ = W.map(np.array([G.WIN_CENTER_S]), np.array([0.0]), d)
        return np.hypot(p[0, 0], p[0, 1] - E.CY_EXT) - E.R_EXT
    a, b = 0.0, 1.0
    for _ in range(60):
        m = (a + b) / 2
        if dist(m) > 0:
            b = m
        else:
            a = m
    return a


N.LARGE["WIN_DEPTH"] = profondeur_hublot() + 0.004    # arrive pile sur la peau (4 mm dans l'epaisseur)

# ---------------------------------------------------------------- separation eco / Business
G.M["curtain"] = mat("Rideau", (0.16, 0.20, 0.33), 0.9)
G.NAMES["curtain"] = "Rideau"
PASSAGE_L, PASSAGE_H = 0.80, 2.00        # passage dans l'allee


def rideau(cote, z):
    """Un pan de rideau plisse, tire sur le cote du passage (cote = -1 gauche, +1 droite)."""
    x_bord = cote * PASSAGE_L / 2
    largeur = 0.20                       # rideau ouvert : ramasse sur 20 cm
    nx, ny = 40, 30
    V, F = [], []
    for j in range(ny):
        y = 0.04 + (PASSAGE_H - 0.10) * j / (ny - 1)
        for i in range(nx):
            u = i / (nx - 1)
            x = x_bord - cote * largeur * u
            dz = 0.035 * np.sin(u * 9 * np.pi) * (0.8 + 0.2 * j / ny)    # plis
            V.append((x, y, z + dz))
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, c, d = j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i
            F += [[a, b, c], [a, c, d], [a, c, b], [a, d, c]]          # deux faces (visible des 2 cotes)
    return trimesh.Trimesh(np.array(V), np.array(F), process=False)


def separation(parts):
    z = DEBUT_BUSINESS                    # entre l'eco (derriere) et la Business (devant)
    contour = N.LARGE["OUTLINE_BINS"]
    passage = arch(0.0, PASSAGE_L, PASSAGE_H)
    cloison = contour.difference(passage).intersection(sbox(-9, 0, 9, 9))
    parts.add("panel", extrude_xy(cloison, z - 0.03, z + 0.03))
    # cadre clair autour du passage
    cadre = passage.buffer(0.04).difference(passage).intersection(sbox(-9, 0, 9, 9))
    parts.add("reveal", extrude_xy(cadre, z - 0.045, z + 0.045))
    # tringle + 2 pans de rideau ouverts
    tr = trimesh.creation.cylinder(radius=0.012, height=PASSAGE_L + 0.10, sections=16)
    tr.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    tr.apply_translation([0, PASSAGE_H - 0.03, z + 0.06])
    parts.add("steel", tr)
    for cote in (-1, 1):
        parts.add("curtain", rideau(cote, z + 0.06))


def ajouter_separation():
    cabine = G.build_cabin

    def build_cabin(parts):
        cabine(parts)
        separation(parts)
    G.build_cabin = build_cabin

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(G.HERE, "avion_niveau2_business.glb")
    G.build_lavatories = N.toilettes_vides
    N.installer()
    ajouter_separation()
    sc = E.build()
    N.reculer_ailes(sc)
    N.enlever_cuvettes(sc)
    sc.export(out)
    print(f"{out}: longueur {E.NOSE_TIP - E.TAIL_TIP:.1f} m (+{EXTRA:.2f} m), "
          f"Business de z={-FIN_ECO:.2f} a z={-DEBUT_BUSINESS:.2f} (glTF), "
          f"{sum(len(g.faces) for g in sc.geometry.values())} triangles")
