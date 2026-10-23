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

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(G.HERE, "avion_niveau2_business.glb")
    N.installer()
    sc = E.build()
    N.reculer_ailes(sc)
    sc.export(out)
    print(f"{out}: longueur {E.NOSE_TIP - E.TAIL_TIP:.1f} m (+{EXTRA:.2f} m), "
          f"Business de z={-FIN_ECO:.2f} a z={-DEBUT_BUSINESS:.2f} (glTF), "
          f"{sum(len(g.faces) for g in sc.geometry.values())} triangles")
