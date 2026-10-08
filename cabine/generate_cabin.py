"""Genere l'interieur d'un avion gros-porteur (configuration 3-3-3) au format GLB.

De l'arriere vers l'avant :
  - toilettes (2 cabinets WC) contre la cloison du fond
  - zone porte arriere (portes passagers gauche/droite)
  - cabine passagers vide : 2 allees, coffres lateraux + coffres centraux
  - zone porte avant (portes passagers gauche/droite)
  - zone equipage : couloir central, cuisine vide a gauche,
    chambre avec un lit a droite
  - cockpit (sieges pilotes, tableau de bord, pare-brise)

Hublots et pare-brise sont des ouvertures reelles (sans vitre). Chaque entree
a sa porte : chaque porte est un noeud "Porte_..." dont l'origine est sur la
charniere (rotation autour de Y pour l'ouvrir).
Repere glTF : Y vers le haut, l'avant de l'avion vers -Z, origine au debut
de la cabine passagers. Unites en metres.

Usage : python3 generate_cabin.py [sortie.glb]
Les modeles lit.glb et toilettes.glb doivent etre a cote du script.
"""
import os
import sys

import numpy as np
import trimesh
from PIL import Image
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from trimesh.transformations import rotation_matrix, scale_matrix, translation_matrix
from trimesh.visual.material import PBRMaterial

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- parametres
N_WINDOWS = 24
PITCH = 0.53            # entraxe des hublots
MARGIN = 0.75           # distance bout de cabine -> premier hublot
LENGTH = 2 * MARGIN + (N_WINDOWS - 1) * PITCH

# zones le long de l'avion (coordonnee interne : positive vers l'avant)
DOOR_ZONE = 2.6
LAV_LEN = 1.9
CREW_LEN = 5.6
COCKPIT_LEN = 3.0
REAR_DOOR = (-DOOR_ZONE, 0.0)
LAV = (REAR_DOOR[0] - LAV_LEN, REAR_DOOR[0])
CABIN = (0.0, LENGTH)
FRONT_DOOR = (LENGTH, LENGTH + DOOR_ZONE)
CREW = (FRONT_DOOR[1], FRONT_DOOR[1] + CREW_LEN)
COCKPIT = (CREW[1], CREW[1] + COCKPIT_LEN)

WIN_W, WIN_H, WIN_R = 0.27, 0.39, 0.12      # ouverture du hublot
BEZ_W, BEZ_H, BEZ_R = 0.36, 0.50, 0.16      # encadrement en retrait
BEZ_DEPTH, WIN_DEPTH = 0.018, 0.16
WIN_CENTER_S = 1.02     # position du hublot le long du profil de paroi

PAX_DOOR_W = 1.07       # porte passagers type A
PAX_DOOR_TOP = 1.98     # hauteur du haut de porte
IN_DOOR_W, IN_DOOR_H = 0.90, 2.00   # portes interieures

BIN_LEN = 3 * PITCH     # longueur d'une porte de coffre
PART_T = 0.06           # epaisseur des cloisons
CORRIDOR = 0.55         # demi-largeur du couloir equipage (axe des cloisons)
AISLES = (0.80, 1.32)   # bords d'allee (x) : bloc central |x|<0.80, blocs lateraux |x|>1.32


def mat(name, rgb, rough=0.6, metal=0.0, emissive=None, tex=None):
    return PBRMaterial(
        name=name,
        baseColorFactor=[*rgb, 1.0] if tex is None else [1.0, 1.0, 1.0, 1.0],
        baseColorTexture=tex,
        metallicFactor=metal,
        roughnessFactor=rough,
        emissiveFactor=emissive,
        doubleSided=True,
    )


def carpet_texture(size=512, seed=3):
    rng = np.random.default_rng(seed)
    base = np.array([22, 36, 78], float)
    n = rng.normal(0, 1, (size, size))
    # bruit basse frequence + fibres fines
    low = np.array(Image.fromarray(((rng.random((32, 32))) * 255).astype(np.uint8))
                   .resize((size, size), Image.BICUBIC), float) / 255 - 0.5
    lum = 1 + 0.10 * n + 0.10 * low
    img = np.clip(base[None, None, :] * lum[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(img, "RGB")


M = {
    "panel": mat("Panneau", (0.90, 0.905, 0.91), 0.55),
    "bin": mat("Coffre", (0.94, 0.945, 0.95), 0.45),
    "ceiling": mat("Plafond", (0.93, 0.935, 0.94), 0.6),
    "reveal": mat("Embrasure", (0.84, 0.85, 0.86), 0.5),
    "door": mat("Porte", (0.86, 0.87, 0.89), 0.5),
    "seam": mat("Joint", (0.62, 0.64, 0.66), 0.7),
    "dark": mat("JointSombre", (0.35, 0.37, 0.40), 0.7),
    "line": mat("BandeJaune", (0.92, 0.84, 0.38), 0.5, emissive=[0.10, 0.09, 0.03]),
    "light": mat("Lumiere", (1, 1, 1), 0.3, emissive=[1.0, 0.98, 0.95]),
    "psu": mat("PSU", (0.80, 0.81, 0.83), 0.5),
    "carpet": mat("Moquette", (1, 1, 1), 1.0, tex=carpet_texture()),
    "steel": mat("Inox", (0.78, 0.80, 0.82), 0.3, metal=0.8),
    "mirror": mat("Miroir", (0.92, 0.94, 0.96), 0.05, metal=1.0),
    "vinyl": mat("SolVinyle", (0.50, 0.52, 0.55), 0.8),
    "seat": mat("SiegePilote", (0.17, 0.19, 0.23), 0.8),
    "console": mat("Console", (0.22, 0.24, 0.27), 0.6),
    "screen": mat("Ecran", (0.05, 0.10, 0.18), 0.2, emissive=[0.10, 0.30, 0.55]),
    "exit": mat("Exit", (0.2, 0.9, 0.4), 0.4, emissive=[0.10, 0.85, 0.30]),
}
NAMES = {"panel": "Parois", "bin": "Coffres", "ceiling": "Plafond", "reveal": "Embrasures",
         "door": "Portes", "seam": "Joints", "dark": "Details", "line": "BandesSol",
         "light": "Lumieres", "psu": "PSU", "carpet": "Moquette", "steel": "Inox",
         "mirror": "Miroirs", "vinyl": "SolVinyle", "seat": "SiegesPilotes",
         "console": "Cockpit", "screen": "Ecrans", "exit": "Exit"}


# ------------------------------------------------------------------- profils
class Profile:
    """Courbe 2D (x, y) d'une coupe transversale, cote droit (x > 0).

    s = abscisse curviligne. La normale "out" pointe hors de la cabine.
    """

    def __init__(self, ctrl, n=600):
        p = np.asarray(ctrl, float)
        pts = [p[0]]
        ext = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
        for i in range(1, len(ext) - 2):  # Catmull-Rom
            p0, p1, p2, p3 = ext[i - 1:i + 3]
            for t in np.linspace(0, 1, 40)[1:]:
                pts.append(0.5 * (2 * p1 + (-p0 + p2) * t
                                  + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
        pts = np.array(pts)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        self.length = s[-1]
        self.s = np.linspace(0, self.length, n)
        self.x = np.interp(self.s, s, pts[:, 0])
        self.y = np.interp(self.s, s, pts[:, 1])
        tx, ty = np.gradient(self.x), np.gradient(self.y)
        nrm = np.hypot(tx, ty)
        self.nx, self.ny = ty / nrm, -tx / nrm

    def map(self, s, z, d=0.0):
        """(s, z, d) -> point 3D ; d > 0 = vers l'exterieur."""
        s = np.clip(s, 0, self.length)
        nx, ny = np.interp(s, self.s, self.nx), np.interp(s, self.s, self.ny)
        x = np.interp(s, self.s, self.x) + d * nx
        y = np.interp(s, self.s, self.y) + d * ny
        return np.column_stack([x, y, z]), np.column_stack([-nx, -ny, np.zeros_like(nx)])

    def s_at_y(self, y):
        """Abscisse au niveau y (sur la partie montante du profil)."""
        i = int(np.argmax(self.y))
        return float(np.interp(y, self.y[:i + 1], self.s[:i + 1]))

    def s_at_x(self, x):
        xs, ss = self.x, self.s
        if xs[0] > xs[-1]:
            xs, ss = xs[::-1], ss[::-1]
        return float(np.interp(x, xs, ss))

    def points(self, step=0.02):
        s = np.linspace(0, self.length, max(2, int(self.length / step)))
        return np.column_stack([np.interp(s, self.s, self.x), np.interp(s, self.s, self.y)])


# section de gros-porteur : ~5,8 m de large interieur
WALL = Profile([(2.73, 0.00), (2.83, 0.35), (2.90, 0.80), (2.91, 1.20),
                (2.87, 1.50), (2.77, 1.68)])
BIN_BOTTOM = Profile([(2.77, 1.68), (2.48, 1.695), (2.19, 1.72)])
BIN_FACE = Profile([(2.19, 1.72), (2.105, 1.80), (2.08, 1.94), (2.12, 2.07), (2.22, 2.16)])
CEILING = Profile([(2.22, 2.16), (2.12, 2.27), (1.85, 2.39), (1.40, 2.46), (0.90, 2.49), (0.0, 2.51)])
# coffres centraux, suspendus au plafond au-dessus du bloc de 3 sieges du milieu
CENTER_BIN = Profile([(0.94, 2.49), (0.985, 2.33), (0.975, 2.08), (0.93, 1.90),
                      (0.83, 1.80), (0.40, 1.785), (0.0, 1.785)])
# paroi pleine hauteur pour les zones sans coffres (portes, equipage, toilettes)
WALL_FULL = Profile([(2.73, 0.00), (2.83, 0.35), (2.90, 0.80), (2.91, 1.20),
                     (2.88, 1.52), (2.78, 1.84), (2.57, 2.06), (2.22, 2.16)])
BIN_PROFILES = [WALL, BIN_BOTTOM, BIN_FACE, CEILING]
OPEN_PROFILES = [WALL_FULL, CEILING]
FLOOR_X = WALL.x[0]


# ------------------------------------------------------------------- outils
def triangulate(poly):
    v, f = trimesh.creation.triangulate_polygon(poly, engine="earcut")
    return np.asarray(v), np.asarray(f)


def surface(profile, poly, d=0.0, strip=0.025):
    """Plaque le polygone 2D (z, s) sur la surface du profil, decale de d."""
    vs, fs, ns, off = [], [], [], 0
    zmin, smin, zmax, smax = poly.bounds
    edges = np.arange(smin, smax + strip, strip)
    for s0, s1 in zip(edges[:-1], edges[1:]):
        piece = poly.intersection(box(zmin - 1, s0, zmax + 1, min(s1, smax)))
        for g in polys(piece):
            v2, f = triangulate(g)
            p, n = profile.map(v2[:, 1], v2[:, 0], d)
            # orientation : normale geometrique vers la cabine
            a, b, c = p[f[0]]
            if np.dot(np.cross(b - a, c - a), n[f[0][0]]) < 0:
                f = f[:, ::-1]
            vs.append(p), ns.append(n), fs.append(f + off)
            off += len(p)
    if not vs:
        return None
    return trimesh.Trimesh(np.vstack(vs), np.vstack(fs), vertex_normals=np.vstack(ns), process=False)


def tube(profile, ring, d0, d1):
    """Embrasure : relie le contour 2D (z, s) a la profondeur d0 et d1."""
    ring = np.asarray(ring)[:-1]
    a, _ = profile.map(ring[:, 1], ring[:, 0], d0)
    b, _ = profile.map(ring[:, 1], ring[:, 0], d1)
    n = len(ring)
    i = np.arange(n)
    j = (i + 1) % n
    f = np.vstack([np.column_stack([i, j, n + j]), np.column_stack([i, n + j, n + i])])
    return trimesh.Trimesh(np.vstack([a, b]), f, process=False)


def rrect(cx, cy, w, h, r, res=24):
    return box(cx - w / 2 + r, cy - h / 2 + r, cx + w / 2 - r, cy + h / 2 - r).buffer(r, resolution=res)


def arch(cx, w, h, r=0.03, y0=0.0):
    """Ouverture de porte : rectangle a coins hauts arrondis, posee au sol."""
    rect = box(cx - w / 2, y0, cx + w / 2, h - r)
    top = rrect(cx, h - r - 0.001, w, 2 * r, r * 0.999)
    return unary_union([rect, top]).simplify(1e-4)


def B(x0, x1, y0, y1, z0, z1):
    b = trimesh.creation.box([x1 - x0, y1 - y0, z1 - z0])
    b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    return b


def cyl(r, p0, p1, sections=16):
    return trimesh.creation.cylinder(radius=r, segment=[p0, p1], sections=sections)


def polys(geom):
    return [g for g in getattr(geom, "geoms", [geom]) if g.geom_type == "Polygon" and g.area > 1e-6]


def extrude(poly, h):
    return trimesh.util.concatenate([trimesh.creation.extrude_polygon(g, h, engine="earcut") for g in polys(poly)])


def extrude_xy(poly, z0, z1):
    """Polygone dans le plan (x, y) extrude le long de z."""
    m = extrude(poly, z1 - z0)
    m.apply_translation([0, 0, z0])
    return m


def extrude_zy(poly, x0, x1):
    """Polygone dans le plan (z, y) extrude le long de x."""
    m = extrude(poly, x1 - x0)
    v = m.vertices.copy()
    m.vertices = np.column_stack([v[:, 2] + x0, v[:, 1], v[:, 0]])
    return m


def flat_xy(poly, z):
    meshes = []
    for g in polys(poly):
        v, f = triangulate(g)
        meshes.append(trimesh.Trimesh(np.column_stack([v, np.full(len(v), z)]), f))
    return trimesh.util.concatenate(meshes)


def mirror(mesh):
    m = mesh.copy()
    m.apply_transform(np.diag([-1, 1, 1, 1]))
    return m


def outline(profiles):
    right = np.vstack([p.points() for p in profiles])
    left = right[::-1].copy()
    left[:, 0] *= -1
    return Polygon(np.vstack([right, left])).buffer(0)


OUTLINE_BINS = outline(BIN_PROFILES)
OUTLINE_OPEN = outline(OPEN_PROFILES)
CENTER_BIN_SECTION = outline([CENTER_BIN])


class Parts(dict):
    """Meshes ranges par materiau ; sym=True ajoute aussi le cote gauche."""

    def __init__(self):
        super().__init__({k: [] for k in M})

    def add(self, key, mesh, sym=False):
        if mesh is None:
            return
        self[key].append(mesh)
        if sym:
            self[key].append(mirror(mesh))


class Door:
    """Battant de porte : noeud separe, origine sur l'axe de la charniere."""

    def __init__(self, name, pivot):
        self.name, self.pivot, self.parts = name, np.asarray(pivot, float), Parts()

    def mirrored(self, name):
        d = Door(name, self.pivot * [-1, 1, 1])
        for k, v in self.parts.items():
            d.parts[k] = [mirror(m) for m in v]
        return d


DOORS = []


# ------------------------------------------------------------- parois laterales
def openings(parts, profile, shapes, bez_depth, depth):
    """Ouvertures (hublots ou portes) : encadrement en retrait + embrasure."""
    for bz, h in shapes:
        parts.add("panel", tube(profile, bz.exterior.coords, 0, bez_depth), sym=True)
        parts.add("panel", surface(profile, bz.difference(h), bez_depth, 0.02), sym=True)
        parts.add("reveal", tube(profile, h.exterior.coords, bez_depth, depth), sym=True)


def build_cabin(parts):
    z0, z1 = CABIN
    zs = [z0 + MARGIN + i * PITCH for i in range(N_WINDOWS)]
    bezels = [rrect(z, WIN_CENTER_S, BEZ_W, BEZ_H, BEZ_R) for z in zs]
    holes = [rrect(z, WIN_CENTER_S, WIN_W, WIN_H, WIN_R) for z in zs]

    # paroi laterale percee
    wall_poly = box(z0, 0, z1, WALL.length).difference(unary_union(bezels))
    parts.add("panel", surface(WALL, wall_poly), sym=True)
    openings(parts, WALL, zip(bezels, holes), BEZ_DEPTH, WIN_DEPTH)

    # joints de panneaux (verticaux entre hublots + horizontal en bas)
    seams = [box(z - 0.003, 0.02, z + 0.003, WALL.length - 0.01)
             for z in np.arange(z0 + MARGIN - PITCH / 2, z1, PITCH)]
    seams.append(box(z0, 0.42, z1, 0.426))
    seams.append(box(z0, 0.0, z1, 0.012))
    seam_poly = unary_union(seams).difference(unary_union(bezels).buffer(0.01))
    parts.add("seam", surface(WALL, seam_poly, -0.0015), sym=True)

    # coffres lateraux
    parts.add("bin", surface(BIN_BOTTOM, box(z0, 0, z1, BIN_BOTTOM.length)), sym=True)
    parts.add("bin", surface(BIN_FACE, box(z0, 0, z1, BIN_FACE.length), strip=0.01), sym=True)
    bin_seams, latches = [], []
    for z in np.arange(z0 + BIN_LEN, z1 - 0.1, BIN_LEN):
        bin_seams.append(box(z - 0.003, 0.005, z + 0.003, BIN_FACE.length - 0.005))
    for z in np.arange(z0, z1 - 0.1, BIN_LEN):
        latches.append(rrect(z + BIN_LEN / 2, BIN_FACE.length - 0.075, 0.16, 0.035, 0.015))
    bin_seams.append(box(z0, 0.003, z1, 0.008))  # bas de porte
    parts.add("seam", surface(BIN_FACE, unary_union(bin_seams), -0.0015, 0.005), sym=True)
    parts.add("dark", surface(BIN_FACE, unary_union(latches), -0.0015, 0.01), sym=True)
    caps = OUTLINE_OPEN.difference(OUTLINE_BINS).intersection(box(0, 1.5, 4, 3))

    # coffres centraux (au-dessus du bloc de 3 sieges du milieu)
    s_face = CENTER_BIN.s_at_x(0.83)
    parts.add("bin", surface(CENTER_BIN, box(z0, 0, z1, CENTER_BIN.length), strip=0.01), sym=True)
    c_seams = [box(z - 0.003, 0.03, z + 0.003, s_face) for z in np.arange(z0 + BIN_LEN, z1 - 0.1, BIN_LEN)]
    c_seams.append(box(z0, s_face - 0.005, z1, s_face))
    parts.add("seam", surface(CENTER_BIN, unary_union(c_seams), -0.0015, 0.005), sym=True)
    c_latch = [rrect(z + BIN_LEN / 2, s_face - 0.12, 0.16, 0.035, 0.015) for z in np.arange(z0, z1 - 0.1, BIN_LEN)]
    parts.add("dark", surface(CENTER_BIN, unary_union(c_latch), -0.0015, 0.01), sym=True)

    # flancs des coffres aux deux bouts de la cabine
    for z in (z0, z1):
        for g in polys(caps):
            parts.add("bin", flat_xy(g, z), sym=True)
        parts.add("bin", flat_xy(CENTER_BIN_SECTION, z))

    # PSU (services passagers) sous les coffres : panneau + liseuses
    s_mid = CENTER_BIN.s_at_x(0.55)
    for z in zs[::2]:
        zc = z + PITCH / 2
        for prof, s in ((BIN_BOTTOM, 0.22), (CENTER_BIN, s_mid)):
            parts.add("psu", surface(prof, rrect(zc, s, 0.42, 0.20, 0.03), -0.004, 0.02), sym=True)
            for dz in (-0.09, 0.09):
                p, _ = prof.map(np.array([s]), np.array([zc + dz]), -0.004)
                dome = trimesh.creation.icosphere(subdivisions=2, radius=0.022)
                dome.apply_scale([1, 0.45, 1])
                dome.apply_translation(p[0])
                parts.add("light", dome, sym=True)

    # eclairage indirect le long des coffres lateraux et centraux
    parts.add("light", surface(CEILING, box(z0, 0.004, z1, 0.05), -0.002, 0.01), sym=True)
    s_c = CEILING.s_at_x(1.02)
    parts.add("light", surface(CEILING, box(z0, s_c - 0.04, z1, s_c), -0.002, 0.01), sym=True)
    # bandes jaunes le long des deux allees
    for x in (-AISLES[1], -AISLES[0], AISLES[0], AISLES[1]):
        parts.add("line", B(x - 0.02, x + 0.02, 0.0, 0.003, z0 + 0.01, z1 - 0.01))


def pax_door(parts, door_z, name):
    """Porte passagers dans la paroi laterale (de chaque cote), avec battant."""
    s_top = WALL_FULL.s_at_y(PAX_DOOR_TOP)
    hole = rrect(door_z, (0.06 + s_top) / 2, PAX_DOOR_W, s_top - 0.06, 0.18)
    bezel = rrect(door_z, s_top / 2 + 0.01, PAX_DOOR_W + 0.18, s_top + 0.14, 0.26)
    openings(parts, WALL_FULL, [(bezel, hole)], 0.03, 0.26)
    # panneau EXIT au-dessus, seuil inox
    p, _ = WALL_FULL.map(np.array([s_top + 0.16]), np.array([door_z]), -0.04)
    x, y = p[0, 0], p[0, 1]
    parts.add("exit", B(x - 0.03, x + 0.03, y - 0.05, y + 0.05, door_z - 0.17, door_z + 0.17), sym=True)
    parts.add("steel", B(FLOOR_X - 0.35, FLOOR_X + 0.02, 0.0, 0.006, door_z - PAX_DOOR_W / 2, door_z + PAX_DOOR_W / 2),
              sym=True)

    # battant bombe qui suit le fuselage ; charniere cote avant
    leaf = hole.buffer(-0.006)
    hinge_z = door_z + PAX_DOOR_W / 2
    pivot, _ = WALL_FULL.map(np.array([s_top / 2]), np.array([hinge_z]), 0.15)
    d = Door(f"Porte_{name}_D", pivot[0])
    d.parts.add("door", surface(WALL_FULL, leaf, 0.12, 0.03))
    d.parts.add("door", surface(WALL_FULL, leaf, 0.20, 0.03))
    d.parts.add("door", tube(WALL_FULL, leaf.exterior.coords, 0.12, 0.20))
    port = rrect(door_z, s_top * 0.66, 0.20, 0.26, 0.09)   # petit hublot de porte (plein)
    d.parts.add("dark", surface(WALL_FULL, port, 0.117, 0.02))
    d.parts.add("seam", surface(WALL_FULL, box(door_z - 0.4, 0.55, door_z + 0.4, 0.556), 0.118, 0.01))
    hp, hn = WALL_FULL.map(np.array([s_top * 0.5]), np.array([door_z - 0.33]), 0.12)
    d.parts.add("steel", cyl(0.018, hp[0], hp[0] + hn[0] * 0.05))
    d.parts.add("steel", cyl(0.016, hp[0] + hn[0] * 0.05, hp[0] + hn[0] * 0.05 + [0, 0, 0.16]))
    DOORS.append(d)
    DOORS.append(d.mirrored(f"Porte_{name}_G"))
    return bezel


def build_open_zone(parts, z0, z1, door_name=None):
    """Zone sans coffres ; door_name -> porte passagers au milieu, de chaque cote."""
    poly = box(z0, 0, z1, WALL_FULL.length)
    if door_name:
        poly = poly.difference(pax_door(parts, (z0 + z1) / 2, door_name))
    parts.add("panel", surface(WALL_FULL, poly), sym=True)
    seams = unary_union([box(z - 0.003, 0.02, z + 0.003, WALL_FULL.length - 0.01) for z in (z0 + 0.01, z1 - 0.01)])
    parts.add("seam", surface(WALL_FULL, seams, -0.0015), sym=True)


def build_ceiling(parts, z0, z1, lights=True):
    parts.add("ceiling", surface(CEILING, box(z0, 0, z1, CEILING.length), strip=0.02), sym=True)
    seams = [box(z0, 0.62, z1, 0.626)]
    seams += [box(z - 0.003, 0.06, z + 0.003, CEILING.length) for z in np.arange(z0 + BIN_LEN, z1, BIN_LEN)]
    parts.add("seam", surface(CEILING, unary_union(seams), -0.0015, 0.02), sym=True)
    if lights:
        spots = []
        for s in (CEILING.s_at_x(1.6), CEILING.s_at_x(0.5)):
            spots += [box(z - 0.25, s, z + 0.25, s + 0.10) for z in np.arange(z0 + 0.6, z1 - 0.3, 1.1)]
        parts.add("light", surface(CEILING, unary_union(spots), -0.002, 0.02), sym=True)


def floor(parts, z0, z1, x0=-FLOOR_X, x1=FLOOR_X, key="carpet", y=0.0):
    v = np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]])
    m = trimesh.Trimesh(v, [[0, 2, 1], [0, 3, 2]], process=False)
    if key == "carpet":
        m.visual = trimesh.visual.TextureVisuals(uv=np.column_stack([v[:, 0], v[:, 2]]) / 0.8, material=M["carpet"])
    parts.add(key, m)


def bulkhead(parts, z, holes=(), thickness=PART_T):
    """Cloison transversale pleine section (face avant en z), percee des ouvertures."""
    poly = OUTLINE_OPEN.difference(unary_union(list(holes))) if holes else OUTLINE_OPEN
    parts.add("panel", extrude_xy(poly, z - thickness, z))
    for h in holes:  # cadre clair autour des ouvertures
        ring = h.buffer(0.05).difference(h).intersection(box(-9, 0, 9, 9))
        parts.add("reveal", extrude_xy(ring, z - thickness - 0.012, z + 0.012))


def long_partition(parts, x, z0, z1, door_zc, height):
    """Cloison longitudinale (plan x = cte) avec une porte."""
    hole = arch(door_zc, IN_DOOR_W, IN_DOOR_H)
    parts.add("panel", extrude_zy(box(z0, 0, z1, height).difference(hole), x - PART_T / 2, x + PART_T / 2))
    ring = hole.buffer(0.045).difference(hole).intersection(box(z0, 0, z1, 9))
    parts.add("reveal", extrude_zy(ring, x - PART_T / 2 - 0.012, x + PART_T / 2 + 0.012))


def flat_door(name, axis, c, a0, a1, plane, hinge_at_start=True):
    """Battant plat. axis='x' : porte dans une cloison transversale (plan z=plane),
    largeur le long de x ; axis='z' : cloison longitudinale (plan x=plane)."""
    w0, w1, h = a0 + 0.005, a1 - 0.005, IN_DOOR_H - 0.01
    t = 0.035
    hinge = w0 if hinge_at_start else w1
    handle = w1 - 0.08 if hinge_at_start else w0 + 0.08
    if axis == "x":
        d = Door(name, [hinge, 0, plane])
        d.parts.add("door", B(w0, w1, 0.01, h, plane - t / 2, plane + t / 2))
        for sgn in (-1, 1):
            zz = plane + sgn * (t / 2 + 0.04)
            d.parts.add("steel", cyl(0.012, [handle, 1.0, plane + sgn * t / 2], [handle, 1.0, zz]))
            d.parts.add("steel", cyl(0.012, [handle, 1.0, zz], [handle - np.sign(handle - hinge) * 0.13, 1.0, zz]))
    else:
        d = Door(name, [plane, 0, hinge])
        d.parts.add("door", B(plane - t / 2, plane + t / 2, 0.01, h, w0, w1))
        for sgn in (-1, 1):
            xx = plane + sgn * (t / 2 + 0.04)
            d.parts.add("steel", cyl(0.012, [plane + sgn * t / 2, 1.0, handle], [xx, 1.0, handle]))
            d.parts.add("steel", cyl(0.012, [xx, 1.0, handle], [xx, 1.0, handle - np.sign(handle - hinge) * 0.13]))
    DOORS.append(d)


# ------------------------------------------------------------------ toilettes
def build_lavatories(parts):
    z0, z1 = LAV
    build_open_zone(parts, z0, z1)
    build_ceiling(parts, z0, z1, lights=False)
    floor(parts, z0, z1)
    bulkhead(parts, z0, thickness=0.0001)                 # cloison du fond
    xs = (0.0, 0.74, 1.48)                                # murs : centre, entre-deux, cote
    lav_cx = (-0.74, 0.74)
    holes = [arch(cx, 0.72, IN_DOOR_H) for cx in lav_cx]
    bulkhead(parts, z1, holes)                            # facade avec 2 portes
    for x in (-1.48, 0.0, 1.48):
        parts.add("panel", B(x - PART_T / 2, x + PART_T / 2, 0, 2.44, z0, z1 - PART_T))
    for cx in lav_cx:
        a, b = cx - 0.72, cx + 0.72
        floor(parts, z0, z1 - PART_T, a + 0.03, b - 0.03, key="vinyl", y=0.002)
        # lavabo sur le cote exterieur, miroir au-dessus, plafonnier
        side = 1 if cx > 0 else -1
        xa, xb = sorted((cx + side * 0.70, cx + side * 0.25))
        zc = z0 + 1.05
        parts.add("panel", B(xa, xb, 0.0, 0.84, zc - 0.28, zc + 0.28))
        parts.add("steel", B(xa, xb, 0.84, 0.87, zc - 0.30, zc + 0.30))
        bowl = trimesh.creation.cylinder(radius=0.15, height=0.01, sections=32)
        bowl.apply_transform(rotation_matrix(np.pi / 2, [1, 0, 0]))
        bowl.apply_translation([(xa + xb) / 2, 0.872, zc])
        parts.add("dark", bowl)
        xw = cx + side * 0.69
        parts.add("steel", cyl(0.012, [xw, 0.87, zc], [xw, 1.02, zc]))
        parts.add("steel", cyl(0.010, [xw, 1.02, zc], [xw - side * 0.14, 1.02, zc]))
        parts.add("mirror", B(cx + side * 0.705, cx + side * 0.71, 1.10, 1.75, zc - 0.28, zc + 0.28) if side > 0
                  else B(cx - 0.71, cx - 0.705, 1.10, 1.75, zc - 0.28, zc + 0.28))
        parts.add("light", B(cx - 0.20, cx + 0.20, 2.40, 2.41, z0 + 0.6, z0 + 1.2))
        flat_door(f"Porte_WC_{'D' if cx > 0 else 'G'}", "x", cx, cx - 0.36, cx + 0.36,
                  z1 - PART_T / 2, hinge_at_start=cx > 0)


# --------------------------------------------------------------- zone equipage
def build_crew(parts):
    z0, z1 = CREW
    build_open_zone(parts, z0, z1)
    build_ceiling(parts, z0, z1)
    floor(parts, z0, z1)
    floor(parts, z0, z1, -FLOOR_X, -CORRIDOR, key="vinyl", y=0.002)   # sol de la cuisine
    zd = z0 + PART_T + 1.0                                # portes des pieces
    for x in (-CORRIDOR, CORRIDOR):
        long_partition(parts, x, z0 + PART_T, z1 - PART_T, zd, 2.48)
    flat_door("Porte_Cuisine", "z", -CORRIDOR, zd - IN_DOOR_W / 2, zd + IN_DOOR_W / 2, -CORRIDOR)
    flat_door("Porte_Chambre", "z", CORRIDOR, zd - IN_DOOR_W / 2, zd + IN_DOOR_W / 2, CORRIDOR)
    # entree du couloir depuis la zone porte avant, et porte du cockpit
    bulkhead(parts, z0 + PART_T, [arch(0.0, IN_DOOR_W, IN_DOOR_H)])
    flat_door("Porte_Couloir", "x", 0.0, -IN_DOOR_W / 2, IN_DOOR_W / 2, z0 + PART_T / 2)
    bulkhead(parts, z1, [arch(0.0, IN_DOOR_W, IN_DOOR_H)])
    flat_door("Porte_Cockpit", "x", 0.0, -IN_DOOR_W / 2, IN_DOOR_W / 2, z1 - PART_T / 2, hinge_at_start=False)
    # chambre : liseuses murales de part et d'autre du lit (le lit est importe)
    for x in (0.62, 2.68):
        parts.add("light", B(x - 0.06, x + 0.06, 1.25, 1.33, z1 - PART_T - 0.02, z1 - PART_T))


# --------------------------------------------------------------------- cockpit
def nose_scale(t):
    return 1 - 0.75 * t ** 1.5, 1 - 0.45 * t ** 1.8


def build_cockpit(parts):
    z0, z1 = COCKPIT
    ring = OUTLINE_OPEN.exterior
    if not ring.is_ccw:
        ring = type(ring)(list(ring.coords)[::-1])
    N, K = 260, 16
    base = np.array([ring.interpolate(d).coords[0] for d in np.linspace(0, ring.length, N, endpoint=False)])
    ts = np.linspace(0, 1, K)
    verts = []
    for t in ts:
        sx, sy = nose_scale(t)
        verts.append(np.column_stack([base[:, 0] * sx, base[:, 1] * sy, np.full(N, z0 + t * (z1 - z0))]))
    verts = np.vstack(verts)
    shell = []
    for k in range(K - 1):
        for i in range(N):
            j = (i + 1) % N
            window = (k in (8, 9, 11, 12) and all(1.30 < base[q][1] < 2.05 and abs(base[q][0]) > 0.20 for q in (i, j)))
            if window:
                continue
            a, b, c, d = k * N + i, k * N + j, (k + 1) * N + j, (k + 1) * N + i
            shell += [[a, b, c], [a, c, d]]
    parts.add("panel", trimesh.Trimesh(verts, np.array(shell), process=False))
    sx, sy = nose_scale(1.0)
    parts.add("panel", flat_xy(Polygon(base * [sx, sy]).buffer(0), z1))   # bout du nez
    # moquette du cockpit (trapezes successifs)
    for t0, t1 in zip(ts[:-1], ts[1:]):
        w0, w1 = (FLOOR_X - 0.02) * nose_scale(t0)[0], (FLOOR_X - 0.02) * nose_scale(t1)[0]
        za, zb = z0 + t0 * (z1 - z0), z0 + t1 * (z1 - z0)
        v = np.array([[-w0, 0.003, za], [w0, 0.003, za], [w1, 0.003, zb], [-w1, 0.003, zb]])
        m = trimesh.Trimesh(v, [[0, 2, 1], [0, 3, 2]], process=False)
        m.visual = trimesh.visual.TextureVisuals(uv=np.column_stack([v[:, 0], v[:, 2]]) / 0.8, material=M["carpet"])
        parts.add("carpet", m)

    # sieges pilotes (face a l'avant = +z)
    zs = z0 + 1.25
    for x in (-0.55, 0.55):
        parts.add("steel", cyl(0.05, [x, 0.0, zs], [x, 0.30, zs]))
        parts.add("seat", B(x - 0.25, x + 0.25, 0.30, 0.42, zs - 0.25, zs + 0.25))
        back = B(x - 0.24, x + 0.24, 0.0, 0.72, -0.05, 0.05)
        back.apply_transform(rotation_matrix(-0.22, [1, 0, 0]))
        back.apply_translation([0, 0.45, zs - 0.27])
        parts.add("seat", back)
        head = B(x - 0.15, x + 0.15, 0.0, 0.20, -0.05, 0.05)
        head.apply_transform(rotation_matrix(-0.22, [1, 0, 0]))
        head.apply_translation([0, 1.18, zs - 0.43])
        parts.add("seat", head)
        for dx in (-0.27, 0.27):  # accoudoirs
            parts.add("seat", B(x + dx - 0.03, x + dx + 0.03, 0.62, 0.68, zs - 0.20, zs + 0.15))
        sx_ = x + (-0.65 if x < 0 else 0.65)  # console laterale + mini-manche
        parts.add("console", B(sx_ - 0.12, sx_ + 0.12, 0.0, 0.70, zs - 0.10, zs + 0.45))
        parts.add("dark", cyl(0.02, [sx_, 0.70, zs + 0.28], [sx_, 0.86, zs + 0.31]))
    # piedestal central avec manettes
    parts.add("console", B(-0.18, 0.18, 0.0, 0.72, zs - 0.15, zs + 0.75))
    for dx in (-0.05, 0.05):
        parts.add("steel", cyl(0.012, [dx, 0.72, zs + 0.35], [dx, 0.86, zs + 0.30]))
        parts.add("dark", B(dx - 0.03, dx + 0.03, 0.85, 0.89, zs + 0.27, zs + 0.33))
    parts.add("screen", B(-0.13, 0.13, 0.723, 0.726, zs + 0.45, zs + 0.65))
    # tableau de bord + ecrans
    zp = z0 + 2.0
    parts.add("console", B(-1.15, 1.15, 0.0, 1.10, zp, zp + 0.35))
    parts.add("console", B(-1.17, 1.17, 1.10, 1.17, zp - 0.12, zp + 0.25))  # casquette
    for cx in (-0.85, -0.50, 0.50, 0.85):
        parts.add("screen", B(cx - 0.15, cx + 0.15, 0.72, 0.99, zp - 0.006, zp))
    parts.add("screen", B(-0.14, 0.14, 0.88, 1.06, zp - 0.006, zp))
    parts.add("screen", B(-0.14, 0.14, 0.66, 0.84, zp - 0.006, zp))
    # panneau plafond
    parts.add("console", B(-0.40, 0.40, 1.95, 2.02, z0 + 0.8, z0 + 1.7))
    parts.add("screen", B(-0.32, 0.32, 1.948, 1.95, z0 + 0.9, z0 + 1.6))


# ---------------------------------------------------------------- assemblage
FLIP = np.diag([1, 1, -1, 1])  # avant de l'avion vers -Z


def to_gltf(p):
    return np.asarray(p, float) * [1, 1, -1]


def concat(meshes, key):
    if key == "carpet":
        m = trimesh.util.concatenate(meshes)  # conserve les UV
    else:
        m = trimesh.util.concatenate([trimesh.Trimesh(x.vertices, x.faces, process=False) for x in meshes])
    m.apply_transform(FLIP)
    m.visual = trimesh.visual.TextureVisuals(uv=getattr(m.visual, "uv", None), material=M[key])
    return m


def import_model(scene, path, name, matrix):
    """Ajoute un .glb externe sous un noeud parent place par `matrix` (repere glTF)."""
    src = trimesh.load(path, force="scene")
    scene.graph.update(frame_from=scene.graph.base_frame, frame_to=name, matrix=matrix)
    for node in src.graph.nodes_geometry:
        T, gname = src.graph[node]
        scene.add_geometry(src.geometry[gname].copy(), node_name=f"{name}_{node}",
                           geom_name=f"{name}_{gname}", parent_node_name=name, transform=T)


def build():
    DOORS.clear()
    parts = Parts()
    build_cabin(parts)
    build_ceiling(parts, *CABIN, lights=False)
    floor(parts, *CABIN)
    for (z0, z1), name in ((REAR_DOOR, "Arriere"), (FRONT_DOOR, "Avant")):
        build_open_zone(parts, z0, z1, door_name=name)
        build_ceiling(parts, z0, z1)
        floor(parts, z0, z1)
    build_lavatories(parts)
    build_crew(parts)
    build_cockpit(parts)

    scene = trimesh.Scene()
    for k, meshes in parts.items():
        if meshes:
            scene.add_geometry(concat(meshes, k), node_name=NAMES[k], geom_name=NAMES[k])

    # portes : un noeud par battant, origine sur la charniere
    for d in DOORS:
        pivot = to_gltf(d.pivot)
        scene.graph.update(frame_from=scene.graph.base_frame, frame_to=d.name, matrix=translation_matrix(pivot))
        for k, meshes in d.parts.items():
            if meshes:
                m = concat(meshes, k)
                m.apply_translation(-pivot)
                scene.add_geometry(m, node_name=f"{d.name}_{NAMES[k]}", geom_name=f"{d.name}_{NAMES[k]}",
                                   parent_node_name=d.name)

    # lit : tete contre la cloison du cockpit, dans la chambre (cote droit)
    bed = os.path.join(HERE, "lit.glb")
    if os.path.exists(bed):
        z_wall = to_gltf([0, 0, CREW[1] - PART_T])[2]
        T = translation_matrix([1.68, 0.0, z_wall + 1.06]) @ rotation_matrix(np.pi, [0, 1, 0])
        import_model(scene, bed, "Lit", T)
    # toilettes : reservoir contre la cloison du fond, face a l'avant
    wc = os.path.join(HERE, "toilettes.glb")
    if os.path.exists(wc):
        z_back = to_gltf([0, 0, LAV[0]])[2]
        for cx, side in ((-0.74, "G"), (0.74, "D")):
            x = cx + 0.15 if cx < 0 else cx - 0.15   # cote interieur, lavabo cote fuselage
            T = (translation_matrix([x, 0.0, z_back - 0.60])
                 @ rotation_matrix(np.pi, [0, 1, 0]) @ scale_matrix(0.09))
            import_model(scene, wc, f"WC_{side}", T)
    return scene


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "cabine_avion.glb")
    sc = build()
    sc.export(out)
    total = COCKPIT[1] - LAV[0]
    print(f"{out}: longueur totale {total:.2f} m, {sum(len(g.faces) for g in sc.geometry.values())} triangles")
