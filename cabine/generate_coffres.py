"""Coffres à bagages de l'avion complet (intérieur 2-2), avec un vrai rangement.

Dans avion_complet.glb, les coffres ("Coffres") sont des volumes pleins. Ici ils sont creux :
  - plancher (où l'on pose les bagages), fond qui suit le fuselage, séparations entre
    compartiments et flancs aux deux bouts ;
  - une porte par compartiment, en noeud séparé dont l'origine est sur la charnière
    (en haut) : on l'ouvre en la faisant tourner vers le haut autour de l'axe X du noeud
    (l'axe de l'avion).

Les coordonnées sont exactement celles de avion_complet.glb : il suffit de poser le modèle
au même endroit que l'avion (même position, même rotation).
Les noeuds s'appellent Coffre_D1..D6 / Coffre_G1..G6 (D = droite, G = gauche, 1 = arrière).

Usage : python3 generate_coffres.py [sortie.glb]
"""
import os
import sys

import numpy as np
import trimesh
from shapely.geometry import Polygon, box
from trimesh.transformations import translation_matrix

import generate_avion_2x2  # noqa: F401  (configure l'intérieur 2-2)
import generate_cabin as G
from generate_cabin import Profile, rrect, surface

# fond du coffre : suit le fuselage, du bas du coffre jusqu'au plafond
BIN_BACK = Profile([(1.27, 1.64), (1.28, 1.78), (1.10, 1.98), (0.88, 2.06)])
EP = 0.012          # épaisseur des panneaux
INTERIEUR = G.mat("CoffreInterieur", (0.66, 0.68, 0.71), 0.8)  # gris de la garniture intérieure
JEU = 0.004         # jeu entre deux portes


def section():
    """Coupe (x, y) de l'intérieur d'un coffre, côté droit."""
    pts = list(G.BIN_BOTTOM.points()) + list(G.BIN_FACE.points())[1:] + list(BIN_BACK.points())[::-1][1:]
    return Polygon(pts).buffer(0)


def plaque(poly, z, ep=EP):
    """Panneau plat (séparation / flanc) dans le plan (x, y), centré en z."""
    m = G.extrude(poly, ep)
    m.apply_translation([0, 0, z - ep / 2])
    return m


def build(out):
    z0, z1 = G.CABIN
    debuts = list(np.arange(z0, z1 - 0.1, G.BIN_LEN))
    bornes = debuts + [z1]
    coque, dedans = [], []
    # plancher : dessous blanc (vu de la cabine), dessus gris (où l'on pose les bagages)
    coque.append(surface(G.BIN_BOTTOM, box(z0, 0, z1, G.BIN_BOTTOM.length)))
    dedans.append(surface(G.BIN_BOTTOM, box(z0, 0.01, z1, G.BIN_BOTTOM.length - 0.005), EP))
    # fond et dessus, le long du fuselage
    dedans.append(surface(BIN_BACK, box(z0, 0, z1, BIN_BACK.length), -0.002))
    # séparations entre compartiments + flancs aux deux bouts
    sec = section()
    for z in bornes[1:-1]:
        dedans.append(plaque(sec, z))
    for z in (bornes[0], bornes[-1]):
        coque.append(plaque(sec, z))
    # petites lèvres sous le bord des portes (butée) : bande sur le bord avant du plancher
    s_bord = G.BIN_BOTTOM.length
    coque.append(surface(G.BIN_BOTTOM, box(z0, s_bord - 0.03, z1, s_bord), -0.006))

    scene = trimesh.Scene()
    for cote, signe in (("D", 1), ("G", -1)):
        for nom_g, liste, matiere in (("Coffres", coque, G.M["bin"]), ("CoffresInterieur", dedans, INTERIEUR)):
            meshes = [m for m in liste if m is not None]
            coq = trimesh.util.concatenate([trimesh.Trimesh(m.vertices, m.faces, process=False) for m in meshes])
            if signe < 0:
                coq = G.mirror(coq)
            coq.apply_transform(G.FLIP)
            coq.visual = trimesh.visual.TextureVisuals(material=matiere)
            scene.add_geometry(coq, node_name=f"{nom_g}_{cote}", geom_name=f"{nom_g}_{cote}")

        # portes : charnière en haut du profil de la face (point (0.88, 2.06))
        hx, hy = G.BIN_FACE.x[-1], G.BIN_FACE.y[-1]
        for i, (a, b) in enumerate(zip(bornes[:-1], bornes[1:]), start=1):
            nom = f"Coffre_{cote}{i}"
            porte = surface(G.BIN_FACE, box(a + JEU, 0.004, b - JEU, G.BIN_FACE.length - 0.004), strip=0.01)
            dos = surface(G.BIN_FACE, box(a + JEU, 0.004, b - JEU, G.BIN_FACE.length - 0.004), 0.018, strip=0.01)
            loquet = surface(G.BIN_FACE, rrect((a + b) / 2, G.BIN_FACE.length - 0.075, 0.16, 0.035, 0.015), -0.0015, 0.01)
            pieces = {"bin": [porte], "interieur": [dos], "dark": [loquet]}
            pivot = np.array([hx * signe, hy, (a + b) / 2])
            pg = G.to_gltf(pivot)
            scene.graph.update(frame_from=scene.graph.base_frame, frame_to=nom, matrix=translation_matrix(pg))
            for k, ms in pieces.items():
                m = trimesh.util.concatenate([trimesh.Trimesh(x.vertices, x.faces, process=False) for x in ms if x is not None])
                if signe < 0:
                    m = G.mirror(m)
                m.apply_transform(G.FLIP)
                m.apply_translation(-pg)
                matiere = INTERIEUR if k == "interieur" else G.M[k]
                nk = "Interieur" if k == "interieur" else G.NAMES[k]
                m.visual = trimesh.visual.TextureVisuals(material=matiere)
                scene.add_geometry(m, node_name=f"{nom}_{nk}", geom_name=f"{nom}_{nk}", parent_node_name=nom)
    scene.export(out)
    print(out, sum(len(g.faces) for g in scene.geometry.values()), "triangles,", len(debuts), "compartiments par côté")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(G.HERE, "coffres.glb"))
