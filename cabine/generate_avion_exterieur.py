"""Avion 2-2 complet : l'interieur de generate_avion_2x2.py + l'exterieur d'un A320.

- Le fuselage exterieur est reconstruit a partir des sections mesurees sur
  exterieur_a320.glb (nez, partie droite allongee, cone de queue), avec de vrais
  trous alignes sur les hublots, les portes passagers et le pare-brise.
- Les ailes, moteurs, empennages et trains d'atterrissage sont repris du modele
  A320, a la meme echelle, places autour du fuselage.
- Le nez du cockpit interieur suit la forme du nez exterieur.

Usage : python3 generate_avion_exterieur.py [sortie.glb]
"""
import os
import sys

import numpy as np
import trimesh
from shapely.geometry import box
from shapely.ops import unary_union

import generate_avion_2x2  # noqa: F401  (configure l'interieur 2-2)
import generate_cabin as G
from generate_cabin import Profile, mat, rrect, surface, tube

A320 = os.path.join(G.HERE, "exterieur_a320.glb")

# ------------------------------------------------- sections mesurees sur l'A320
# x (axe de l'avion, nez vers -x), y bas, y haut, demi-largeur
NOSE = [(-2.12, 0.408, 0.441, 0.0), (-2.02, 0.321, 0.530, 0.120), (-1.92, 0.285, 0.593, 0.165),
        (-1.82, 0.267, 0.677, 0.205), (-1.72, 0.260, 0.710, 0.224), (-1.62, 0.255, 0.728, 0.235),
        (-1.52, 0.250, 0.742, 0.245), (-1.42, 0.246, 0.751, 0.253), (-1.32, 0.242, 0.759, 0.258),
        (-1.22, 0.240, 0.764, 0.262), (-1.12, 0.237, 0.769, 0.266)]
TAIL = [(0.70, 0.237, 0.769, 0.266), (0.78, 0.243, 0.768, 0.262), (0.98, 0.257, 0.765, 0.253),
        (1.18, 0.269, 0.763, 0.246), (1.38, 0.299, 0.760, 0.230), (1.58, 0.339, 0.754, 0.207),
        (1.78, 0.381, 0.745, 0.180), (1.98, 0.426, 0.725, 0.145), (2.18, 0.477, 0.705, 0.115),
        (2.38, 0.537, 0.690, 0.082), (2.58, 0.608, 0.670, 0.035), (2.62, 0.635, 0.645, 0.0)]
MID_X = (-1.12, 0.70)            # partie a section constante
A_CY, A_R = 0.503, 0.266         # centre et rayon de la section constante

R_EXT = 1.85                     # rayon exterieur du fuselage (m)
K = R_EXT / A_R                  # echelle A320 -> metres
CY_EXT = 1.0                     # hauteur du centre du fuselage (plancher cabine = 0)
NOSE_LEN = 3.4                   # longueur du nez (le cockpit de 2,6 m est dedans)
TAIL_LEN = (TAIL[-1][0] - MID_X[1]) * K

G.WIN_DEPTH = 0.50               # les embrasures des hublots rejoignent la peau
G.PAX_REVEAL = 0.55              # idem pour les portes passagers
G.M["exterior"] = mat("Fuselage", (0.95, 0.95, 0.96), 0.4)
G.NAMES["exterior"] = "Fuselage"


def section(table, x):
    """(cy, hy, hz) en metres, pour une position x de l'A320."""
    t = np.array(table)
    xs = t[:, 0] if t[0, 0] < t[-1, 0] else t[::-1, 0]
    rows = t if t[0, 0] < t[-1, 0] else t[::-1]
    lo = np.interp(x, xs, rows[:, 1])
    hi = np.interp(x, xs, rows[:, 2])
    hz = np.interp(x, xs, rows[:, 3])
    return ((lo + hi) / 2 - A_CY) * K + CY_EXT, (hi - lo) / 2 * K, hz * K


def ext_section(zi):
    """Section exterieure a la position interne zi (positive vers l'avant)."""
    if zi > G.CREW[1]:
        u = min((zi - G.CREW[1]) / NOSE_LEN, 1.0)
        return section(NOSE, MID_X[0] - u * (MID_X[0] - NOSE[0][0]))
    if zi < G.LAV[0]:
        return section(TAIL, MID_X[1] + (G.LAV[0] - zi) / K)
    return CY_EXT, R_EXT, R_EXT


def on_ellipse(dirs, cy, hy, hz):
    """Projette des directions (dx, dy) sur l'ellipse de section."""
    d = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    r = 1.0 / np.sqrt((d[:, 0] / max(hz, 1e-6)) ** 2 + (d[:, 1] / max(hy, 1e-6)) ** 2)
    return np.column_stack([d[:, 0] * r, cy + d[:, 1] * r])


# ------------------------------------------------ nez du cockpit = nez de l'A320
def nose_scale(t):
    zi = G.COCKPIT[0] + t * (G.COCKPIT[1] - G.COCKPIT[0])
    cy, hy, hz = ext_section(zi)
    sx = min(1.0, (hz - 0.45) / 1.40)
    sy = min(1.0, (cy + hy - 0.15) / G.CEILING.y[-1])
    return sx, sy


G.nose_scale = nose_scale
_build_cockpit = G.build_cockpit


def build_cockpit(parts):
    """Cockpit interieur d'origine + peau exterieure du nez, pare-brise ouvert."""
    _build_cockpit(parts)
    z0, z1 = G.COCKPIT
    ring = G.OUTLINE_OPEN.exterior
    if not ring.is_ccw:
        ring = type(ring)(list(ring.coords)[::-1])
    N, KR = 260, 16                                   # memes anneaux que l'interieur
    base = np.array([ring.interpolate(d).coords[0] for d in np.linspace(0, ring.length, N, endpoint=False)])
    ts = np.linspace(0, 1, KR)
    inner, outer = [], []
    for t in ts:
        sx, sy = nose_scale(t)
        zi = z0 + t * (z1 - z0)
        cy, hy, hz = ext_section(zi)
        p = base * [sx, sy]
        inner.append(np.column_stack([p, np.full(N, zi)]))
        outer.append(np.column_stack([on_ellipse(p - [0, cy], cy, hy, hz), np.full(N, zi)]))
    # radome : on prolonge les memes directions jusqu'a la pointe
    dirs = base * nose_scale(1.0) - [0, ext_section(z1)[0]]
    tip = G.CREW[1] + NOSE_LEN
    for zi in np.linspace(z1, tip, 7)[1:-1]:
        cy, hy, hz = ext_section(zi)
        outer.append(np.column_stack([on_ellipse(dirs, cy, hy, hz), np.full(N, zi)]))
    outer = np.vstack(outer)
    inner = np.vstack(inner)

    def is_window(k, i):
        j = (i + 1) % N
        return k in (8, 9, 11, 12) and all(1.30 < base[q][1] < 2.05 and abs(base[q][0]) > 0.20 for q in (i, j))

    faces, reveal = [], []
    n_rings = len(outer) // N
    for k in range(n_rings - 1):
        for i in range(N):
            j = (i + 1) % N
            a, b, c, d = k * N + i, k * N + j, (k + 1) * N + j, (k + 1) * N + i
            if k < KR - 1 and is_window(k, i):
                # embrasure du pare-brise : relie l'interieur a l'exterieur sur les bords du trou
                for (p, q), nb in (((a, b), (k - 1, i)), ((d, c), (k + 1, i)),
                                   ((a, d), (k, (i - 1) % N)), ((b, c), (k, j))):
                    if not is_window(*nb):
                        reveal.append((p, q))
                continue
            faces += [[a, b, c], [a, c, d]]
    cy, _, _ = ext_section(tip)
    verts = np.vstack([outer, [[0.0, cy, tip]]])
    last = (n_rings - 1) * N
    faces += [[last + i, last + (i + 1) % N, len(verts) - 1] for i in range(N)]
    parts.add("exterior", trimesh.Trimesh(verts, np.array(faces), process=False))
    rv, rf = [], []
    for p, q in reveal:
        o = len(rv)
        rv += [inner[p], inner[q], outer[q], outer[p]]
        rf += [[o, o + 1, o + 2], [o, o + 2, o + 3]]
    if rf:
        parts.add("reveal", trimesh.Trimesh(np.array(rv), np.array(rf), process=False))


G.build_cockpit = build_cockpit


# --------------------------------------------- fuselage : partie droite + queue
def ellipse_profile(cy, hy, hz, n=48):
    th = np.linspace(-np.pi / 2, np.pi / 2, n)
    return Profile(np.column_stack([hz * np.cos(th), cy + hy * np.sin(th)]))


def build_fuselage(parts):
    z0, z1 = G.LAV[0], G.CREW[1]
    skin = ellipse_profile(CY_EXT, R_EXT, R_EXT)
    holes = []
    # hublots : au bout de l'embrasure interieure
    for zw in [G.CABIN[0] + G.MARGIN + i * G.PITCH for i in range(G.N_WINDOWS)]:
        p, _ = G.WALL.map(np.array([G.WIN_CENTER_S]), np.array([zw]), G.WIN_DEPTH)
        holes.append(rrect(zw, skin.s_at_y(p[0, 1]), G.WIN_W + 0.02, G.WIN_H + 0.02, G.WIN_R + 0.01))
    # portes passagers
    for zz in (G.REAR_DOOR, G.FRONT_DOOR):
        zd = (zz[0] + zz[1]) / 2
        s0, s1 = skin.s_at_y(0.06), skin.s_at_y(G.PAX_DOOR_TOP)
        holes.append(rrect(zd, (s0 + s1) / 2, G.PAX_DOOR_W + 0.02, s1 - s0, 0.18))
    poly = box(z0, 0, z1, skin.length).difference(unary_union(holes))
    parts.add("exterior", surface(skin, poly, 0.0, 0.05), sym=True)
    # liseré sombre autour des hublots, vu de l'exterieur
    rims = unary_union([h.buffer(0.03).difference(h) for h in holes[:G.N_WINDOWS]])
    parts.add("dark", surface(skin, rims, 0.004, 0.02), sym=True)

    # cone de queue (ferme)
    n = 96
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    dirs = np.column_stack([np.cos(th), np.sin(th)])
    zs = np.linspace(z0, z0 - TAIL_LEN, 40)
    rings = []
    for zi in zs:
        cy, hy, hz = ext_section(zi)
        rings.append(np.column_stack([on_ellipse(dirs, cy, hy, hz), np.full(n, zi)]))
    verts = np.vstack(rings + [[[0.0, ext_section(zs[-1])[0], zs[-1]]]])
    faces = []
    for k in range(len(zs) - 1):
        for i in range(n):
            j = (i + 1) % n
            faces += [[k * n + i, k * n + j, (k + 1) * n + j], [k * n + i, (k + 1) * n + j, (k + 1) * n + i]]
    last = (len(zs) - 1) * n
    faces += [[last + i, last + (i + 1) % n, len(verts) - 1] for i in range(n)]
    parts.add("exterior", trimesh.Trimesh(verts, np.array(faces), process=False))


# --------------------------------------------- ailes, moteurs, empennages, trains
def a320_parts():
    """Pieces de l'A320 hors fuselage, placees autour du nouveau fuselage (repere interne)."""
    src = trimesh.load(A320, force="scene")
    mid_c = (G.LAV[0] + G.CREW[1]) / 2
    x_c = (MID_X[0] + MID_X[1]) / 2

    def zi_of(x):
        if x < MID_X[0]:
            return G.CREW[1] + (MID_X[0] - x) / (MID_X[0] - NOSE[0][0]) * NOSE_LEN
        if x > MID_X[1]:
            return G.LAV[0] - (x - MID_X[1]) * K
        return mid_c - (x - x_c) * K

    out = []
    for node in src.graph.nodes_geometry:
        T, gname = src.graph[node]
        if gname.startswith("Cube") or gname.startswith("Cylinder.003") or gname.startswith("Cylinder.011"):
            continue  # hublots, portes et vitres peints : remplaces par de vraies ouvertures
        m = src.geometry[gname].copy()
        m.apply_transform(T)
        if gname.startswith("Cylinder.002"):
            # carlingue + ailes + empennages : on retire la peau du fuselage, on garde le reste
            c = m.triangles_center
            keep = np.ones(len(c), bool)
            for f, (x, y, z) in enumerate(c):
                tbl = NOSE if x < MID_X[0] else TAIL if x > MID_X[1] else None
                if tbl:
                    cy, hy, hz = section(tbl, x)
                    cy, hy, hz = (cy - CY_EXT) / K + A_CY, hy / K, hz / K
                else:
                    cy, hy, hz = A_CY, A_R, A_R
                if hz > 1e-3 and (z / hz) ** 2 + ((y - cy) / max(hy, 1e-3)) ** 2 < 1.12 ** 2:
                    keep[f] = False
            m.update_faces(keep)
            m.remove_unreferenced_vertices()
            # ailes / empennages : chaque sommet est place selon sa zone (nez, milieu, queue)
            v = m.vertices
            new = np.column_stack([v[:, 2] * K, (v[:, 1] - A_CY) * K + CY_EXT,
                                   [zi_of(x) for x in v[:, 0]]])
            q = trimesh.Trimesh(new, m.faces, process=False)
            q.visual = m.visual
            out.append((gname, q))
            continue
        else:
            comps = [m]
        for p in comps:
            x_mid = p.centroid[0]
            v = p.vertices
            # A320 : x vers l'arriere, z lateral -> interne : z vers l'avant, x lateral
            new = np.column_stack([v[:, 2] * K, (v[:, 1] - A_CY) * K + CY_EXT,
                                   zi_of(x_mid) - (v[:, 0] - x_mid) * K])
            q = trimesh.Trimesh(new, p.faces, process=False)
            q.visual = p.visual.copy() if hasattr(p.visual, "material") else p.visual
            out.append((gname, q))
    return out


_build = G.build


def build():
    # la peau du fuselage est ajoutee aux pieces avant l'assemblage
    _parts_init = G.Parts.__init__

    def init(self):
        _parts_init(self)

    scene = None
    orig_build_cabin = G.build_cabin

    def build_cabin(parts):
        orig_build_cabin(parts)
        build_fuselage(parts)

    G.build_cabin = build_cabin
    try:
        scene = _build()
    finally:
        G.build_cabin = orig_build_cabin
    for i, (gname, m) in enumerate(a320_parts()):
        m.apply_transform(G.FLIP)
        name = f"A320_{gname.split('_')[0].replace('.', '')}_{i}"
        scene.add_geometry(m, node_name=name, geom_name=name)
    return scene


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(G.HERE, "avion_complet.glb")
    sc = build()
    sc.export(out)
    print(f"{out}: longueur {G.CREW[1] + NOSE_LEN - (G.LAV[0] - TAIL_LEN):.1f} m, "
          f"{sum(len(g.faces) for g in sc.geometry.values())} triangles")
