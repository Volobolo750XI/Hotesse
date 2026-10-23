"""Siege Business : un bloc de 2 places cote a cote (pour la cabine 2 | 2), style "suite".

Coque creme avec lisere dore, parois hautes, assise et dossier en cuir creme, appui-tete blanc,
oreiller, accoudoir central, une console en bois avec petit ecran de chaque cote, repose-pieds.
Repere : metres, Y en haut, le siege regarde vers -Z (l'avant de l'avion), origine au sol au centre.
Pieces separees (une par materiau) comme les autres modeles.

Usage : python3 generate_siege_business.py [sortie.glb]
"""
import os
import sys

import numpy as np
import trimesh
from shapely.geometry import box, Point
from trimesh.transformations import rotation_matrix, translation_matrix

HERE = os.path.dirname(os.path.abspath(__file__))


def mat(nom, rgb, rough=0.6, metal=0.0):
    return trimesh.visual.material.PBRMaterial(name=nom, baseColorFactor=[*rgb, 255],
                                               roughnessFactor=rough, metallicFactor=metal, doubleSided=True)


M = {
    "coque": mat("CoqueCreme", (232, 224, 205), 0.45),
    "cuir": mat("CuirCreme", (245, 236, 214), 0.55),
    "sombre": mat("CoqueSombre", (68, 64, 68), 0.5),
    "bois": mat("Bois", (150, 98, 58), 0.5),
    "or": mat("Dore", (205, 168, 95), 0.3, 0.8),
    "blanc": mat("Blanc", (246, 246, 244), 0.8),
    "ecran": mat("Ecran", (18, 22, 30), 0.2),
}
PIECES = {k: [] for k in M}


def ajouter(k, m):
    PIECES[k].append(m)


def bloc(x0, x1, y0, y1, z0, z1, r=0.03):
    """Boite aux coins arrondis (vue de dessus), du sol y0 a y1."""
    r = min(r, (x1 - x0) / 2.01, (z1 - z0) / 2.01)
    poly = box(x0 + r, z0 + r, x1 - r, z1 - r).buffer(r, resolution=6)
    m = trimesh.creation.extrude_polygon(poly, y1 - y0)
    # extrude_polygon : polygone en (x, y), hauteur en z -> on remet Y en haut
    m.apply_transform(rotation_matrix(-np.pi / 2, [1, 0, 0]))
    m.apply_translation([0, y0, 0])
    # apres rotation, l'ancien y (notre z) est passe en -z : on corrige
    v = m.vertices.copy()
    v[:, 2] = -v[:, 2]
    m = trimesh.Trimesh(v, m.faces[:, ::-1], process=True)
    return m


def dalle(l, h, e, r=0.04):
    """Plaque arrondie (largeur l en x, hauteur h en y, epaisseur e en z), centree en bas au milieu."""
    r = min(r, l / 2.01, h / 2.01)
    poly = box(-l / 2 + r, r, l / 2 - r, h - r).buffer(r, resolution=6)
    m = trimesh.creation.extrude_polygon(poly, e)
    m.apply_translation([0, 0, -e / 2])
    return m


def poser(m, rot_x_deg=0.0, pos=(0, 0, 0)):
    m = m.copy()
    m.apply_transform(rotation_matrix(np.radians(rot_x_deg), [1, 0, 0]))
    m.apply_translation(pos)
    return m


def rbox(l, h, e, r=0.05, bombe=0.0, n=4):
    """Boite aux bords tous arrondis (rayon r), centree ; bombe = gonflement des grandes faces."""
    r = min(r, l / 2.01, h / 2.01, e / 2.01)
    m = trimesh.creation.box((1, 1, 1))
    for _ in range(n):
        m = m.subdivide()
    v = m.vertices * np.array([l, h, e])
    demi = np.array([l / 2 - r, h / 2 - r, e / 2 - r])
    inner = np.clip(v, -demi, demi)
    d = v - inner
    nd = np.linalg.norm(d, axis=1, keepdims=True)
    v = inner + r * d / np.maximum(nd, 1e-9)
    if bombe:
        u = np.clip(1 - (v[:, 0] / (l / 2)) ** 2, 0, 1) * np.clip(1 - (v[:, 1] / (h / 2)) ** 2, 0, 1)
        v[:, 2] += np.sign(v[:, 2]) * bombe * u
    return trimesh.Trimesh(v, m.faces, process=True)


def coussin(l, h, e, r=0.05):
    return rbox(l, h, e, min(0.06, e / 2.2), bombe=e * 0.25)


# ---------------------------------------------------------------- dimensions
PLACE = 0.53          # largeur d'une place
ACCOUDOIR = 0.08      # accoudoir central
CONSOLE = 0.20        # console cote hublot (une seule, sinon le bloc ne rentre pas en 2 | 2)
COTE = 0.02           # petite paroi cote allee
LARGEUR = 2 * PLACE + ACCOUDOIR + CONSOLE + COTE  # 1,36 m : entre l'allee et le hublot
PROF = 1.55           # profondeur du bloc (de l'arriere du dossier au repose-pieds)


def place(cx):
    """Une place : assise, dossier galbe, appui-tete, repose-pieds individuel, oreiller."""
    zb = PROF / 2            # arriere
    # base sombre sous l'assise
    ajouter("cuir", poser(rbox(PLACE - 0.10, 0.34, 0.56, 0.04), 0, (cx, 0.20, zb - 0.55)))
    # assise bombee
    ajouter("cuir", poser(rbox(PLACE - 0.05, 0.62, 0.15, 0.06, bombe=0.03), 90, (cx, 0.45, zb - 0.55)))
    # dossier galbe, incline vers l'arriere
    ajouter("cuir", poser(rbox(PLACE - 0.07, 0.86, 0.17, 0.07, bombe=0.035), -12, (cx, 0.93, zb - 0.22)))
    # appui-tete blanc
    ajouter("blanc", poser(rbox(0.30, 0.20, 0.09, 0.04, bombe=0.02), -12, (cx, 1.20, zb - 0.36)))
    # oreiller contre le dossier
    ajouter("blanc", poser(rbox(0.36, 0.26, 0.12, 0.06, bombe=0.04), -18, (cx, 0.67, zb - 0.40)))
    # repose-pieds individuel (pouf creme) devant la place
    ajouter("cuir", poser(rbox(PLACE - 0.16, 0.36, 0.34, 0.06, bombe=0.02), 0, (cx, 0.18, -PROF / 2 + 0.22)))


def construire(cote_hublot=1):
    """cote_hublot = +1 : console a droite (+x) ; -1 : version miroir (bloc de gauche)."""
    for k in PIECES:
        PIECES[k].clear()
    half = LARGEUR / 2
    zb, zf = PROF / 2, -PROF / 2
    # x croissant : allee (-half) -> hublot (+half)
    x_allee = -half + COTE
    xs = [x_allee + PLACE / 2, x_allee + PLACE + ACCOUDOIR + PLACE / 2]
    for cx in xs:
        place(cx)
    ajouter("coque", bloc(-half, half, 0.0, 0.10, zf, zb, 0.06))                       # socle
    ajouter("coque", poser(rbox(LARGEUR, 1.18, 0.10, 0.04), 0, (0, 0.59, zb + 0.03)))   # dos du bloc (creme)
    ajouter("or", poser(rbox(LARGEUR + 0.01, 0.02, 0.11, 0.01), 0, (0, 1.18, zb + 0.03)))   # lisere
    # console cote hublot : caisson creme + plateau bois + lisere dore + ecran
    a, b = half - CONSOLE, half
    ajouter("sombre", bloc(a, b, 0.0, 0.70, zf + 0.20, zb, 0.04))
    ajouter("bois", bloc(a + 0.01, b - 0.01, 0.70, 0.73, zf + 0.21, zb - 0.02, 0.03))
    ajouter("or", bloc(a, b, 0.73, 0.745, zf + 0.20, zb, 0.03))
    ajouter("ecran", poser(dalle(0.17, 0.13, 0.015, 0.01), -20, ((a + b) / 2, 0.75, zf + 0.42)))
    # parois hautes (coquille) : cote allee et cote hublot, lisere dore
    for xp in (-half, half - 0.03):
        ajouter("sombre", bloc(xp, xp + 0.03, 0.0 if xp < 0 else 0.70, 1.10, zf + 0.55, zb, 0.012))
        ajouter("or", bloc(xp - 0.003, xp + 0.033, 1.10, 1.115, zf + 0.55, zb, 0.012))
    # accoudoir central + separation d'intimite
    xa = x_allee + PLACE
    ajouter("sombre", bloc(xa, xa + ACCOUDOIR, 0.0, 0.62, zf + 0.55, zb, 0.03))
    ajouter("or", bloc(xa, xa + ACCOUDOIR, 0.62, 0.635, zf + 0.55, zb, 0.03))
    ajouter("sombre", bloc(xa + ACCOUDOIR / 2 - 0.015, xa + ACCOUDOIR / 2 + 0.015, 0.635, 1.05, zf + 0.75, zb, 0.01))
    if cote_hublot < 0:
        for k, lst in PIECES.items():
            for m in lst:
                m.apply_transform(np.diag([-1, 1, 1, 1]))
                m.invert()
    sc = trimesh.Scene()
    for k, lst in PIECES.items():
        if lst:
            m = trimesh.util.concatenate(lst)
            m.visual = trimesh.visual.TextureVisuals(material=M[k])
            sc.add_geometry(m, node_name=f"SiegeBusiness_{M[k].name}", geom_name=f"SiegeBusiness_{M[k].name}")
    return sc


if __name__ == "__main__":
    for nom, cote in (("siege_business_droite.glb", 1), ("siege_business_gauche.glb", -1)):
        out = os.path.join(HERE, nom)
        sc = construire(cote)
        sc.export(out)
        b = sc.bounds
        print(f"{out}: {b[1][0]-b[0][0]:.2f} x {b[1][2]-b[0][2]:.2f} m, {b[1][1]-b[0][1]:.2f} m de haut, "
              f"{sum(len(g.faces) for g in sc.geometry.values())} triangles")
