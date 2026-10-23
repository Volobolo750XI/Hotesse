"""Siege Business realiste : une place dans sa coque enveloppante (style A350 / Air France).

- coque en U (vue de dessus) : haute derriere (1,25 m), elle descend vers l'avant ; bord superieur arrondi,
  exterieur blanc laque, interieur en tissu bleu nuit ;
- siege rembourre : assise en 2 coussins, dossier galbe avec couture, appui-tete, coussin raye ;
- repose-pieds (pouf) devant, tablette laterale en bois, lampe de lecture, ceinture ;
- textures generees : tissu, cuir, bois, laque (couleur + rugosite), UV par projection.
Repere : metres, Y en haut, le passager regarde vers -Z. Origine au sol, au centre de la place.
Une place = un fichier ; en 2 | 2 on en pose 2 cote a cote de chaque cote de l'allee.

Usage : python3 generate_siege_business_v2.py [sortie.glb]
"""
import io
import os
import sys

import numpy as np
import trimesh
from PIL import Image, ImageFilter
from trimesh.transformations import rotation_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)

# ------------------------------------------------------------------ textures
def bruit(n, echelle, oct=4):
    out = np.zeros((n, n))
    for o in range(oct):
        f = 2 ** o
        g = rng.random((max(2, int(n / echelle * f)), max(2, int(n / echelle * f))))
        im = Image.fromarray((g * 255).astype(np.uint8)).resize((n, n), Image.BICUBIC)
        out += np.asarray(im, float) / 255 / f
    return out / sum(1 / 2 ** o for o in range(oct))


def tex_tissu(rgb, n=512):
    b = bruit(n, 64, 3)
    x = np.arange(n)
    trame = (np.sin(x[:, None] * 2 * np.pi / 6) * np.sin(x[None, :] * 2 * np.pi / 6)) * 0.5 + 0.5
    v = 0.86 + 0.10 * b + 0.06 * trame
    return np.clip(np.array(rgb)[None, None, :] * v[..., None], 0, 255).astype(np.uint8)


def tex_matelasse(rgb, n=512):
    """Tissu avec des panneaux cousus : bandes horizontales bombees separees par des coutures."""
    b = bruit(n, 64, 3)
    y = np.arange(n)[:, None] / n
    bande = 0.10                                       # une bande = 0,1 UV (5 cm avec echelle 0.5)
    t = (y % bande) / bande
    bombe = 0.90 + 0.10 * np.sin(np.pi * t)            # chaque bande est bombee
    couture = np.exp(-((t - 0.0) / 0.03) ** 2) + np.exp(-((t - 1.0) / 0.03) ** 2)
    pts = ((np.arange(n)[None, :] // 6) % 2) * 0.5 + 0.5   # points de couture
    v = bombe * (0.92 + 0.08 * b) * (1 - 0.35 * couture * pts)
    v = np.repeat(v, n, axis=1) if v.shape[1] == 1 else v
    return np.clip(np.array(rgb)[None, None, :] * v[..., None], 0, 255).astype(np.uint8)


def tex_cuir(rgb, n=512):
    grain = bruit(n, 8, 3)
    v = 0.92 + 0.10 * (grain - 0.5) * 2 * 0.5
    return np.clip(np.array(rgb)[None, None, :] * v[..., None], 0, 255).astype(np.uint8)


def tex_bois(n=512):
    y = np.linspace(0, 1, n)[:, None]
    b = bruit(n, 128, 4)
    veines = np.sin((y * 40 + b * 6) * np.pi) * 0.5 + 0.5
    base = np.array([138, 92, 56]); clair = np.array([176, 124, 80])
    c = base + (clair - base) * (veines[..., None] ** 2)
    c = c * (0.92 + 0.12 * bruit(n, 32, 2)[..., None])
    return np.clip(c, 0, 255).astype(np.uint8)


def tex_rayures(n=256):
    x = np.arange(n)[None, :].repeat(n, 0)
    r = ((x // 16) % 2 == 0)
    c = np.where(r[..., None], np.array([34, 42, 72]), np.array([226, 220, 206]))
    c = c * (0.94 + 0.08 * bruit(n, 16, 2)[..., None])
    return np.clip(c, 0, 255).astype(np.uint8)


def tex_uni(rgb, n=64, var=0.03):
    return np.clip(np.array(rgb)[None, None, :] * (1 + var * (bruit(n, 8, 2)[..., None] - 0.5)), 0, 255).astype(np.uint8)


def mat(nom, img, rough=0.6, metal=0.0, emissive=None):
    im = Image.fromarray(img)
    return trimesh.visual.material.PBRMaterial(name=nom, baseColorTexture=im, roughnessFactor=rough,
                                               metallicFactor=metal, doubleSided=True,
                                               emissiveFactor=emissive)


M = {
    "laque": mat("CoqueLaqueBlanche", tex_uni((236, 236, 232)), 0.25),
    "exterieur": mat("CoqueBleuNuit", tex_cuir((34, 42, 70)), 0.4),
    "tissu": mat("TissuBleuNuit", tex_tissu((40, 50, 82)), 0.9),
    "cuir": mat("CuirBleuNuit", tex_cuir((44, 54, 88)), 0.55),
    "assise": mat("TissuAssise", tex_matelasse((54, 64, 100)), 0.85),
    "bois": mat("BoisClair", tex_bois(), 0.45),
    "rayures": mat("CoussinRaye", tex_rayures(), 0.85),
    "metal": mat("MetalBrosse", tex_uni((150, 150, 155)), 0.35, 0.9),
    "noir": mat("PlastiqueNoir", tex_uni((25, 27, 32)), 0.5),
    "lumiere": mat("LumiereLampe", tex_uni((255, 236, 200)), 0.3, 0.0, [1.0, 0.9, 0.7]),
    "sangle": mat("Sangle", tex_tissu((30, 34, 46)), 0.8),
    "ecran_on": mat("EcranVeille", tex_uni((14, 20, 34), var=0.4), 0.08, 0.0, [0.02, 0.04, 0.08]),
}
PIECES = {k: [] for k in M}


def uv_boite(m, echelle=1.0):
    """UV par projection selon la normale dominante (comme un triplanaire)."""
    m = m.copy()
    lisse = trimesh.Trimesh(m.vertices, m.faces, process=True)       # sommets soudes -> normales lisses
    vn = lisse.vertex_normals[lisse.faces].reshape(-1, 3)
    m = trimesh.Trimesh(lisse.vertices[lisse.faces].reshape(-1, 3), np.arange(len(lisse.faces) * 3).reshape(-1, 3),
                        process=False)
    m.metadata["normales"] = vn
    n = np.repeat(m.face_normals, 3, axis=0)
    v = m.vertices
    ax = np.argmax(np.abs(n), axis=1)
    uv = np.where(ax[:, None] == 0, v[:, [2, 1]], np.where(ax[:, None] == 1, v[:, [0, 2]], v[:, [0, 1]]))
    m.metadata["uv"] = uv / echelle
    return m


def ajouter(k, m, echelle=1.0):
    PIECES[k].append(uv_boite(m, echelle))


# ------------------------------------------------------------------ formes
def rbox(l, h, e, r=0.05, bombe=0.0, n=4):
    r = min(r, l / 2.01, h / 2.01, e / 2.01)
    m = trimesh.creation.box((1, 1, 1))
    for _ in range(n):
        m = m.subdivide()
    v = m.vertices * np.array([l, h, e])
    demi = np.array([l / 2 - r, h / 2 - r, e / 2 - r])
    inner = np.clip(v, -demi, demi)
    d = v - inner
    v = inner + r * d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    if bombe:
        u = np.clip(1 - (v[:, 0] / (l / 2)) ** 2, 0, 1) * np.clip(1 - (v[:, 1] / (h / 2)) ** 2, 0, 1)
        v[:, 2] += np.sign(v[:, 2]) * bombe * u
    return trimesh.Trimesh(v, m.faces, process=True)


def pose(m, rx=0.0, ry=0.0, pos=(0, 0, 0)):
    m = m.copy()
    m.apply_transform(rotation_matrix(np.radians(rx), [1, 0, 0]))
    m.apply_transform(rotation_matrix(np.radians(ry), [0, 1, 0]))
    m.apply_translation(pos)
    return m


def cylindre(r, h, pos, axe=(0, 1, 0), sec=24):
    m = trimesh.creation.cylinder(radius=r, height=h, sections=sec)
    a = np.array(axe, float)
    if not np.allclose(a, [0, 0, 1]):
        m.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], a))
    m.apply_translation(pos)
    return m


# ------------------------------------------------------------------ coque en U
L = 0.70          # largeur exterieure de la coque
P = 1.75          # profondeur (de l'arriere de la coque au bout du repose-pieds)
EP = 0.045        # epaisseur de la coque
H_ARR, H_AV = 1.24, 0.80   # hauteur derriere / devant


def chemin_u(n=90):
    """Ligne mediane de la coque, vue de dessus : cote gauche (avant -> arriere), arc, cote droit."""
    R = L / 2 - EP / 2
    zc = P / 2 - R                     # centre de l'arc arriere
    z_av = -P / 2 + 0.70               # la coque s'arrete au niveau des genoux
    n1 = n // 3
    pts = []
    for t in np.linspace(0, 1, n1, endpoint=False):
        pts.append((-R, z_av + (zc - z_av) * t))
    for a in np.linspace(np.pi, 0, n1, endpoint=False):
        pts.append((R * np.cos(a), zc + R * np.sin(a)))
    for t in np.linspace(0, 1, n1):
        pts.append((R, zc + (z_av - zc) * t))
    return np.array(pts), z_av, zc


def hauteur(z, z_av, zc):
    t = np.clip((z - z_av) / (zc - z_av), 0, 1)
    return H_AV + (H_ARR - H_AV) * (0.5 - 0.5 * np.cos(np.pi * t))   # douce, en S


def coque():
    pts, z_av, zc = chemin_u()
    n = len(pts)
    # normales horizontales (vers l'exterieur de la coque)
    tan = np.gradient(pts, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    nor = np.column_stack([tan[:, 1], -tan[:, 0]])          # tourne de -90 : exterieur
    # verifie le sens : au milieu de l'arc la normale doit pointer vers +z (l'arriere)
    if nor[n // 2, 1] < 0:
        nor = -nor
    h = hauteur(pts[:, 1], z_av, zc)
    k = 10                                                   # profil du bord arrondi
    def surface(decal, cle, bas=0.03, marge_haut=0.0):
        V, F = [], []
        ys = np.linspace(0.0, 1.0, 16)
        for i in range(n):
            haut = h[i] - EP / 2 - marge_haut
            b0 = bas if bas >= 0 else h[i] - EP / 2 + bas
            for y in ys:
                p = pts[i] + nor[i] * decal
                V.append((p[0], b0 + y * (haut - b0), p[1]))
        nv = len(ys)
        for i in range(n - 1):
            for j in range(nv - 1):
                a, b, c, d = i * nv + j, (i + 1) * nv + j, (i + 1) * nv + j + 1, i * nv + j + 1
                F += [[a, b, c], [a, c, d]] if decal > 0 else [[a, c, b], [a, d, c]]
        return trimesh.Trimesh(np.array(V), np.array(F), process=False)
    BANDE = 0.07                                             # bandeau blanc en haut de la coque
    ajouter("exterieur", surface(EP / 2, "ext", 0.03, BANDE), 0.5)
    ajouter("laque", surface(EP / 2, "bandeau", -BANDE))
    ajouter("tissu", surface(-EP / 2, "int"), 0.5)
    # bord superieur arrondi (demi-tube) : laque
    V, F = [], []
    an = np.linspace(0, np.pi, k)
    for i in range(n):
        y0 = h[i] - EP / 2
        for a in an:
            p = pts[i] + nor[i] * (EP / 2) * np.cos(a)
            V.append((p[0], y0 + (EP / 2) * np.sin(a), p[1]))
    for i in range(n - 1):
        for j in range(k - 1):
            a_, b_, c_, d_ = i * k + j, (i + 1) * k + j, (i + 1) * k + j + 1, i * k + j + 1
            F += [[a_, c_, b_], [a_, d_, c_]]
    ajouter("laque", trimesh.Trimesh(np.array(V), np.array(F), process=False))
    # chants verticaux a l'avant des deux cotes
    for i in (0, n - 1):
        p0, p1 = pts[i] - nor[i] * EP / 2, pts[i] + nor[i] * EP / 2
        ajouter("laque", trimesh.Trimesh([(p0[0], 0.05, p0[1]), (p1[0], 0.05, p1[1]),
                                           (p1[0], h[i] - EP / 2, p1[1]), (p0[0], h[i] - EP / 2, p0[1])],
                                          [[0, 1, 2], [0, 2, 3]], process=False))
    # plinthe noire au pied de la coque
    return z_av, zc


# ------------------------------------------------------------------ siege
def siege(zc):
    l = L - 2 * EP - 0.10                 # largeur du coussin (entre les parois)
    z_dos = zc + 0.05
    # pied / structure
    ajouter("tissu", pose(rbox(l - 0.04, 0.32, 0.58, 0.03), 0, 0, (0, 0.17, z_dos - 0.47)), 0.5)
    # accoudoir rembourre cote gauche (contre la coque)
    ajouter("cuir", pose(rbox(0.07, 0.06, 0.48, 0.025), 0, 0, (-l / 2 - 0.01, 0.60, z_dos - 0.40)))
    ajouter("laque", pose(rbox(0.06, 0.55, 0.46, 0.02), 0, 0, (-l / 2 - 0.01, 0.29, z_dos - 0.40)))
    # assise : 2 coussins (avant / arriere) avec une couture
    ajouter("assise", pose(rbox(l, 0.13, 0.30, 0.05, bombe=0.02), 90, 0, (0, 0.40, z_dos - 0.30)))
    PIECES["assise"][-1] = uv_boite(pose(rbox(l, 0.30, 0.13, 0.05, bombe=0.02), 90, 0, (0, 0.40, z_dos - 0.32)), 0.5)
    ajouter("assise", pose(rbox(l, 0.26, 0.13, 0.05, bombe=0.02), 90, 0, (0, 0.395, z_dos - 0.60)), 0.5)
    # dossier : coussin bas + coussin haut (couture entre les deux), incline de 14 deg
    for (hh, yc, dz) in ((0.42, 0.64, 0.0), (0.30, 0.99, 0.09)):
        ajouter("assise", pose(rbox(l, hh, 0.14, 0.06, bombe=0.03), -14, 0, (0, yc, z_dos - 0.13 + dz)), 0.5)
    # appui-tete cuir
    ajouter("cuir", pose(rbox(l * 0.62, 0.17, 0.10, 0.045, bombe=0.02), -14, 0, (0, 1.17, z_dos - 0.06)))
    # coussin raye pose contre le dossier
    ajouter("rayures", pose(rbox(0.36, 0.28, 0.12, 0.05, bombe=0.04), -22, 0, (0.03, 0.64, z_dos - 0.36)), 0.3)
    # ceinture : 2 sangles sur l'assise + boucle metal
    for sx in (-1, 1):
        ajouter("sangle", pose(rbox(0.045, 0.006, 0.20, 0.002), 0, 0, (sx * 0.10, 0.475, z_dos - 0.45)))
    ajouter("metal", pose(rbox(0.06, 0.012, 0.05, 0.005), 0, 0, (0, 0.48, z_dos - 0.56)))
    return l


def repose_pieds(z_av):
    # pouf : socle noir + coussin
    ajouter("exterieur", pose(rbox(0.40, 0.30, 0.36, 0.06), 0, 0, (0, 0.15, -P / 2 + 0.22)), 0.5)
    ajouter("cuir", pose(rbox(0.42, 0.07, 0.38, 0.035, bombe=0.0), 0, 0, (0, 0.33, -P / 2 + 0.22)))


def console_et_lampe(zc, z_av):
    # tablette laterale en bois (cote droit, sur la coque) avec bord noir
    x = L / 2 - EP - 0.10
    ajouter("bois", pose(rbox(0.16, 0.025, 0.55, 0.012), 0, 0, (x + 0.02, 0.68, zc - 0.55)), 0.4)
    ajouter("laque", pose(rbox(0.13, 0.66, 0.52, 0.03), 0, 0, (x + 0.035, 0.34, zc - 0.55)))
    # commande du siege (petit pave) sur la tablette
    ajouter("noir", pose(rbox(0.05, 0.012, 0.09, 0.004), 0, 0, (x + 0.02, 0.698, zc - 0.40)))
    # lampe de lecture fixee sur la coque, cote gauche, au-dessus de l'epaule
    xl = -L / 2 + EP + 0.035
    ajouter("noir", cylindre(0.028, 0.11, (xl, 1.08, zc - 0.10), axe=(0.35, -0.35, -1)))
    ajouter("lumiere", cylindre(0.022, 0.004, (xl + 0.02, 1.06, zc - 0.15), axe=(0.35, -0.35, -1)))
    # crochet pour casque
    ajouter("metal", pose(rbox(0.012, 0.05, 0.03, 0.004), 0, 0, (-L / 2 + EP + 0.012, 0.85, zc - 0.25)))


def ecran_dos(zc):
    R = L / 2
    z = zc + R - 0.006                       # encastre dans le dos de la coque (arc arriere)
    ajouter("noir", pose(rbox(0.30, 0.20, 0.02, 0.012), 0, 0, (0, 0.92, z)))
    ajouter("ecran_on", pose(rbox(0.27, 0.17, 0.004, 0.006), 0, 0, (0, 0.92, z + 0.009)))


def construire():
    for k in PIECES:
        PIECES[k].clear()
    z_av, zc = coque()
    siege(zc)
    repose_pieds(z_av)
    console_et_lampe(zc, z_av)
    ecran_dos(zc)
    sc = trimesh.Scene()
    for k, lst in PIECES.items():
        if not lst:
            continue
        m = trimesh.util.concatenate(lst)
        uv = np.vstack([p.metadata["uv"] for p in lst])
        vn = np.vstack([p.metadata["normales"] for p in lst])
        m = trimesh.Trimesh(m.vertices, m.faces, vertex_normals=vn, process=False)
        m.visual = trimesh.visual.TextureVisuals(uv=uv, material=M[k])
        sc.add_geometry(m, node_name=f"SiegeBusiness_{M[k].name}", geom_name=f"SiegeBusiness_{M[k].name}")
    return sc


if __name__ == "__main__":
    # droite : la tablette en bois est a droite (+x) ; gauche : version miroir
    for nom, miroir in (("siege_business_droite.glb", False), ("siege_business_gauche.glb", True)):
        sc = construire()
        if miroir:
            for g in sc.geometry.values():
                g.apply_transform(np.diag([-1, 1, 1, 1]))
                g.invert()
        out = os.path.join(HERE, nom)
        sc.export(out, include_normals=True)
        b = sc.bounds
        print(f"{out}: {b[1][0]-b[0][0]:.2f} x {b[1][2]-b[0][2]:.2f} m, {b[1][1]-b[0][1]:.2f} m de haut, "
              f"{sum(len(g.faces) for g in sc.geometry.values())} triangles")
