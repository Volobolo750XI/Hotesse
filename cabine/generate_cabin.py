"""Genere l'interieur d'un avion au format GLB, style epure.

De l'arriere vers l'avant :
  - zone porte arriere (portes passagers gauche/droite)
  - cabine passagers vide (hublots, coffres a bagages)
  - zone porte avant (portes passagers gauche/droite)
  - zone equipage : couloir central, cuisine (galley) a gauche,
    repos equipage (couchettes) a droite
  - cockpit (sieges pilotes, tableau de bord, pare-brise)

Hublots, portes et pare-brise sont des ouvertures reelles (sans vitre).
Repere glTF : Y vers le haut, l'avant de l'avion est vers -Z, l'origine
est au debut de la cabine passagers. Unites en metres.

Usage : python3 generate_cabin.py [sortie.glb]
"""
import sys

import numpy as np
import trimesh
from PIL import Image
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from trimesh.visual.material import PBRMaterial

# ---------------------------------------------------------------- parametres
N_WINDOWS = 24
PITCH = 0.53            # entraxe des hublots
MARGIN = 0.75           # distance bout de cabine -> premier hublot
LENGTH = 2 * MARGIN + (N_WINDOWS - 1) * PITCH

# zones le long de l'avion (coordonnee interne : positive vers l'avant)
DOOR_ZONE = 2.2
CREW_LEN = 3.2
COCKPIT_LEN = 2.6
REAR = (-DOOR_ZONE, 0.0)
CABIN = (0.0, LENGTH)
FRONT = (LENGTH, LENGTH + DOOR_ZONE)
CREW = (FRONT[1], FRONT[1] + CREW_LEN)
COCKPIT = (CREW[1], CREW[1] + COCKPIT_LEN)

WIN_W, WIN_H, WIN_R = 0.27, 0.39, 0.12      # ouverture du hublot
BEZ_W, BEZ_H, BEZ_R = 0.36, 0.50, 0.16      # encadrement en retrait
BEZ_DEPTH, WIN_DEPTH = 0.018, 0.16
WIN_CENTER_S = 1.02     # position du hublot le long du profil de paroi

BIN_LEN = 3 * PITCH     # longueur d'une porte de coffre
PART_T = 0.06           # epaisseur des cloisons
CORRIDOR = 0.50         # demi-largeur du couloir equipage (axe des cloisons)


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
    "seam": mat("Joint", (0.62, 0.64, 0.66), 0.7),
    "dark": mat("JointSombre", (0.35, 0.37, 0.40), 0.7),
    "line": mat("BandeJaune", (0.92, 0.84, 0.38), 0.5, emissive=[0.10, 0.09, 0.03]),
    "light": mat("Lumiere", (1, 1, 1), 0.3, emissive=[1.0, 0.98, 0.95]),
    "psu": mat("PSU", (0.80, 0.81, 0.83), 0.5),
    "carpet": mat("Moquette", (1, 1, 1), 1.0, tex=carpet_texture()),
    "steel": mat("Inox", (0.78, 0.80, 0.82), 0.3, metal=0.8),
    "vinyl": mat("SolGalley", (0.50, 0.52, 0.55), 0.8),
    "cart": mat("Trolley", (0.72, 0.74, 0.77), 0.4, metal=0.3),
    "oven": mat("Four", (0.10, 0.11, 0.13), 0.2),
    "mattress": mat("Matelas", (0.42, 0.52, 0.66), 0.9),
    "pillow": mat("Oreiller", (0.96, 0.96, 0.97), 0.9),
    "seat": mat("SiegePilote", (0.17, 0.19, 0.23), 0.8),
    "console": mat("Console", (0.22, 0.24, 0.27), 0.6),
    "screen": mat("Ecran", (0.05, 0.10, 0.18), 0.2, emissive=[0.10, 0.30, 0.55]),
    "exit": mat("Exit", (0.2, 0.9, 0.4), 0.4, emissive=[0.10, 0.85, 0.30]),
}


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
        return float(np.interp(y, self.y, self.s))

    def points(self, step=0.02):
        s = np.linspace(0, self.length, max(2, int(self.length / step)))
        return np.column_stack([np.interp(s, self.s, self.x), np.interp(s, self.s, self.y)])


WALL = Profile([(1.70, 0.00), (1.80, 0.35), (1.87, 0.80), (1.88, 1.20),
                (1.84, 1.50), (1.74, 1.68)])
BIN_BOTTOM = Profile([(1.74, 1.68), (1.45, 1.695), (1.16, 1.72)])
BIN_FACE = Profile([(1.16, 1.72), (1.075, 1.80), (1.05, 1.94), (1.09, 2.07), (1.19, 2.16)])
CEILING = Profile([(1.19, 2.16), (1.13, 2.24), (0.95, 2.31), (0.60, 2.355), (0.0, 2.37)])
# paroi pleine hauteur pour les zones sans coffres (portes, equipage)
WALL_FULL = Profile([(1.70, 0.00), (1.80, 0.35), (1.87, 0.80), (1.88, 1.20),
                     (1.85, 1.52), (1.75, 1.84), (1.54, 2.06), (1.19, 2.16)])
BIN_PROFILES = [WALL, BIN_BOTTOM, BIN_FACE, CEILING]
OPEN_PROFILES = [WALL_FULL, CEILING]


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
        geoms = getattr(piece, "geoms", [piece])
        for g in geoms:
            if g.geom_type != "Polygon" or g.area < 1e-9:
                continue
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


def arch(cx, w, h, r, y0=0.0):
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

    def merge(self, other):
        for k, v in other.items():
            self[k] += v


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

    # coffres a bagages
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
    # flancs des coffres aux deux bouts de la cabine
    caps = OUTLINE_OPEN.difference(OUTLINE_BINS).intersection(box(0, 1.5, 3, 3))
    for z in (z0, z1):
        for g in getattr(caps, "geoms", [caps]):
            if g.area > 1e-3:
                v, f = triangulate(g)
                parts.add("bin", trimesh.Trimesh(np.column_stack([v, np.full(len(v), z)]), f), sym=True)

    # PSU (services passagers) sous les coffres : panneau + liseuses
    for z in zs[::2]:
        panel = rrect(z + PITCH / 2, 0.24, 0.42, 0.20, 0.03)
        parts.add("psu", surface(BIN_BOTTOM, panel, -0.004, 0.02), sym=True)
        for dz in (-0.09, 0.09):
            p, _ = BIN_BOTTOM.map(np.array([0.22]), np.array([z + PITCH / 2 + dz]), -0.004)
            dome = trimesh.creation.icosphere(subdivisions=2, radius=0.022)
            dome.apply_scale([1, 0.45, 1])
            dome.apply_translation(p[0])
            parts.add("light", dome, sym=True)

    # eclairage indirect le long des coffres
    parts.add("light", surface(CEILING, box(z0, 0.004, z1, 0.05), -0.002, 0.01), sym=True)
    # bandes jaunes au sol
    for x in (-0.57, 0.57):
        parts.add("line", B(x - 0.02, x + 0.02, 0.0, 0.003, z0 + 0.01, z1 - 0.01))


def build_open_zone(parts, z0, z1, door_z=None):
    """Zone sans coffres ; door_z = position d'une porte passagers de chaque cote."""
    poly = box(z0, 0, z1, WALL_FULL.length)
    if door_z is not None:
        s_top = WALL_FULL.s_at_y(1.93)
        hole = rrect(door_z, (0.06 + s_top) / 2, 0.84, s_top - 0.06, 0.16)
        bezel = rrect(door_z, s_top / 2 + 0.01, 1.00, s_top + 0.14, 0.22)
        poly = poly.difference(bezel)
        openings(parts, WALL_FULL, [(bezel, hole)], 0.03, 0.24)
        # seuil au sol et panneau EXIT au-dessus de la porte
        p, _ = WALL_FULL.map(np.array([s_top + 0.17]), np.array([door_z]), -0.04)
        x, y = p[0, 0], p[0, 1]
        parts.add("exit", B(x - 0.03, x + 0.03, y - 0.05, y + 0.05, door_z - 0.17, door_z + 0.17), sym=True)
        parts.add("steel", B(1.40, 1.72, 0.0, 0.006, door_z - 0.45, door_z + 0.45), sym=True)
    parts.add("panel", surface(WALL_FULL, poly), sym=True)
    # joints verticaux aux limites de zone
    seams = unary_union([box(z - 0.003, 0.02, z + 0.003, WALL_FULL.length - 0.01) for z in (z0 + 0.01, z1 - 0.01)])
    parts.add("seam", surface(WALL_FULL, seams, -0.0015), sym=True)


def build_ceiling(parts, z0, z1, lights=True):
    parts.add("ceiling", surface(CEILING, box(z0, 0, z1, CEILING.length), strip=0.02), sym=True)
    seams = [box(z0, 0.62, z1, 0.626)]
    seams += [box(z - 0.003, 0.06, z + 0.003, CEILING.length) for z in np.arange(z0 + BIN_LEN, z1, BIN_LEN)]
    parts.add("seam", surface(CEILING, unary_union(seams), -0.0015, 0.02), sym=True)
    if lights:
        s = CEILING.length - 0.33
        spots = unary_union([box(z - 0.25, s, z + 0.25, s + 0.10) for z in np.arange(z0 + 0.6, z1 - 0.3, 1.1)])
        parts.add("light", surface(CEILING, spots, -0.002, 0.02), sym=True)


def floor(parts, z0, z1, x=WALL.x[0], key="carpet", y=0.0):
    v = np.array([[-x, y, z0], [x, y, z0], [x, y, z1], [-x, y, z1]])
    f = np.array([[0, 2, 1], [0, 3, 2]])
    m = trimesh.Trimesh(v, f, process=False)
    if key == "carpet":
        m.visual = trimesh.visual.TextureVisuals(uv=np.column_stack([v[:, 0], v[:, 2]]) / 0.8, material=M["carpet"])
    parts.add(key, m)


def bulkhead(parts, z, holes=(), thickness=0.0):
    """Cloison transversale pleine section, percee des ouvertures donnees."""
    poly = OUTLINE_OPEN.difference(unary_union(list(holes))) if holes else OUTLINE_OPEN
    if thickness:
        parts.add("panel", extrude_xy(poly, z - thickness, z))
        for h in holes:  # cadre clair autour des ouvertures
            ring = h.buffer(0.05).difference(h)
            parts.add("reveal", extrude_xy(ring, z - thickness - 0.012, z + 0.012))
    else:
        v, f = triangulate(poly)
        parts.add("panel", trimesh.Trimesh(np.column_stack([v, np.full(len(v), z)]), f))


# --------------------------------------------------------------- zone equipage
def galley_unit(parts, z_wall, facing, sink=False):
    """Bloc cuisine (galley) colle a une cloison ; facing = +1 si face vers +z."""
    xa, xb = -1.66, -(CORRIDOR + PART_T / 2) - 0.01
    def zr(d0, d1):
        a, b = z_wall + facing * d0, z_wall + facing * d1
        return min(a, b), max(a, b)
    parts.add("panel", B(xa, xb, 0.0, 0.98, *zr(0, 0.72)))
    front = 0.72
    # emplacements trolleys
    w = 0.32
    for i, cx in enumerate((xa + 0.05 + w / 2, xa + 0.07 + 1.5 * w)):
        parts.add("cart", B(cx - w / 2 + 0.01, cx + w / 2 - 0.01, 0.06, 0.94, *zr(front - 0.02, front + 0.012)))
        parts.add("dark", B(cx - 0.10, cx + 0.10, 0.84, 0.865, *zr(front + 0.012, front + 0.03)))
    # placard bas restant
    cx0 = xa + 0.10 + 2 * w
    parts.add("seam", B(cx0, cx0 + 0.004, 0.05, 0.95, *zr(front, front + 0.004)))
    parts.add("dark", B(xb - 0.08, xb - 0.05, 0.55, 0.80, *zr(front, front + 0.02)))
    # plan de travail inox
    parts.add("steel", B(xa, xb, 0.98, 1.02, *zr(0, front + 0.02)))
    parts.add("steel", B(xa, xb, 1.02, 1.24, *zr(0, 0.03)))  # credence
    if sink:
        cx = (xa + xb) / 2 + 0.15
        parts.add("oven", B(cx - 0.18, cx + 0.18, 1.015, 1.022, *zr(0.20, 0.52)))
        parts.add("steel", cyl(0.012, [cx, 1.02, z_wall + facing * 0.10], [cx, 1.30, z_wall + facing * 0.10]))
        parts.add("steel", cyl(0.010, [cx, 1.30, z_wall + facing * 0.10], [cx, 1.30, z_wall + facing * 0.26]))
    # meubles hauts : fours + cafetiere + placards
    parts.add("panel", B(xa + 0.08, xb, 1.24, 1.95, *zr(0, 0.45)))
    for i, cx in enumerate(np.linspace(xa + 0.30, xb - 0.25, 3)):
        if i < 2:  # fours
            parts.add("steel", B(cx - 0.17, cx + 0.17, 1.42, 1.80, *zr(0.45, 0.46)))
            parts.add("oven", B(cx - 0.13, cx + 0.13, 1.47, 1.72, *zr(0.46, 0.468)))
            parts.add("dark", B(cx - 0.12, cx + 0.12, 1.76, 1.775, *zr(0.46, 0.48)))
        else:  # cafetiere
            parts.add("steel", B(cx - 0.12, cx + 0.12, 1.30, 1.70, *zr(0.45, 0.47)))
            parts.add("oven", cyl(0.045, [cx, 1.36, z_wall + facing * 0.40], [cx, 1.50, z_wall + facing * 0.40]))
            parts.add("screen", B(cx - 0.06, cx + 0.06, 1.58, 1.64, *zr(0.47, 0.475)))
    parts.add("seam", B(xa + 0.08, xb, 1.85, 1.854, *zr(0.45, 0.455)))


def crew_rest(parts, z0, z1):
    """Repos equipage : deux couchettes superposees le long de la paroi droite."""
    xa, xb = 1.02, 1.76
    zc = (z0 + z1) / 2
    za, zb = zc - 1.05, zc + 1.05
    for y_base, y_top in ((0.0, 0.40), (1.12, 1.19)):
        parts.add("panel", B(xa, xb, y_base, y_top, za, zb))
        parts.add("mattress", B(xa + 0.03, xb - 0.03, y_top, y_top + 0.14, za + 0.03, zb - 0.03))
        pil = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        pil.apply_scale([0.26, 0.06, 0.16])
        pil.apply_translation([(xa + xb) / 2, y_top + 0.19, zb - 0.25])
        parts.add("pillow", pil)
        parts.add("light", B(xb - 0.02, xb, y_top + 0.45, y_top + 0.50, zb - 0.45, zb - 0.30))
    parts.add("seam", B(xa - 0.001, xa, 0.38, 0.385, za, zb))
    # panneaux de bout
    for z in (za - 0.04, zb):
        parts.add("panel", B(xa, xb, 0.0, 1.95, z, z + 0.04))
    # echelle
    for dz in (0.0, 0.32):
        parts.add("steel", cyl(0.014, [xa - 0.03, 0.0, za + 0.15 + dz], [xa - 0.03, 1.40, za + 0.15 + dz]))
    for y in (0.55, 0.80, 1.05):
        parts.add("steel", cyl(0.011, [xa - 0.03, y, za + 0.15], [xa - 0.03, y, za + 0.47]))
    # petit casier
    parts.add("panel", B(0.60, 0.98, 0.0, 1.10, z0 + 0.06, z0 + 0.50))
    parts.add("seam", B(0.60, 0.98, 0.55, 0.554, z0 + 0.50, z0 + 0.504))


def build_crew(parts):
    z0, z1 = CREW
    zm = (z0 + z1) / 2
    build_open_zone(parts, z0, z1)
    build_ceiling(parts, z0, z1)
    floor(parts, z0, z1)
    # sol vinyle dans la cuisine
    v = np.array([[-1.70, 0.002, z0], [-CORRIDOR, 0.002, z0], [-CORRIDOR, 0.002, z1], [-1.70, 0.002, z1]])
    parts.add("vinyl", trimesh.Trimesh(v, [[0, 2, 1], [0, 3, 2]]))
    # cloisons longitudinales avec porte au milieu
    door = arch(zm, 0.78, 1.98, 0.12)
    wall = box(z0, 0, z1, 2.34).difference(door)
    for x in (-CORRIDOR, CORRIDOR):
        parts.add("panel", extrude_zy(wall, x - PART_T / 2, x + PART_T / 2))
        parts.add("reveal", extrude_zy(door.buffer(0.045).difference(door).intersection(box(z0, 0, z1, 3)),
                                       x - PART_T / 2 - 0.012, x + PART_T / 2 + 0.012))
    # cloison arriere (vers la porte avant) : passage du couloir
    bulkhead(parts, z0 + PART_T, [arch(0.0, 0.84, 2.02, 0.14)], PART_T)
    # cloison cockpit
    bulkhead(parts, z1, [arch(0.0, 0.80, 1.96, 0.12)], PART_T)
    galley_unit(parts, z1 - PART_T, -1)
    galley_unit(parts, z0 + PART_T, +1, sink=True)
    crew_rest(parts, z0 + PART_T, z1 - PART_T)


# --------------------------------------------------------------------- cockpit
def nose_scale(t):
    return 1 - 0.62 * t ** 1.5, 1 - 0.42 * t ** 1.8


def build_cockpit(parts):
    z0, z1 = COCKPIT
    ring = OUTLINE_OPEN.exterior
    if not ring.is_ccw:
        ring = type(ring)(list(ring.coords)[::-1])
    N, K = 220, 16
    base = np.array([ring.interpolate(d).coords[0] for d in np.linspace(0, ring.length, N, endpoint=False)])
    ts = np.linspace(0, 1, K)
    verts = []
    for t in ts:
        sx, sy = nose_scale(t)
        verts.append(np.column_stack([base[:, 0] * sx, base[:, 1] * sy, np.full(N, z0 + t * (z1 - z0))]))
    verts = np.vstack(verts)
    shell, glass_edge = [], []
    for k in range(K - 1):
        for i in range(N):
            j = (i + 1) % N
            x0, y0 = base[i]
            window = (k in (8, 9, 11, 12) and 1.30 < y0 < 2.02 and abs(x0) > 0.14
                      and 1.30 < base[j][1] < 2.02 and abs(base[j][0]) > 0.14)
            if window:
                continue
            a, b, c, d = k * N + i, k * N + j, (k + 1) * N + j, (k + 1) * N + i
            shell += [[a, b, c], [a, c, d]]
    parts.add("panel", trimesh.Trimesh(verts, np.array(shell), process=False))
    # bout du nez
    sx, sy = nose_scale(1.0)
    cap = Polygon(base * [sx, sy]).buffer(0)
    v, f = triangulate(cap)
    parts.add("panel", trimesh.Trimesh(np.column_stack([v, np.full(len(v), z1)]), f))
    # moquette du cockpit (trapezes successifs)
    for t0, t1 in zip(ts[:-1], ts[1:]):
        w0, w1 = 1.68 * nose_scale(t0)[0], 1.68 * nose_scale(t1)[0]
        za, zb = z0 + t0 * (z1 - z0), z0 + t1 * (z1 - z0)
        v = np.array([[-w0, 0.003, za], [w0, 0.003, za], [w1, 0.003, zb], [-w1, 0.003, zb]])
        m = trimesh.Trimesh(v, [[0, 2, 1], [0, 3, 2]], process=False)
        m.visual = trimesh.visual.TextureVisuals(uv=np.column_stack([v[:, 0], v[:, 2]]) / 0.8, material=M["carpet"])
        parts.add("carpet", m)

    # sieges pilotes (face a l'avant = +z)
    zs = z0 + 1.0
    for x in (-0.52, 0.52):
        parts.add("steel", cyl(0.05, [x, 0.0, zs], [x, 0.30, zs]))
        parts.add("seat", B(x - 0.25, x + 0.25, 0.30, 0.42, zs - 0.25, zs + 0.25))
        back = B(x - 0.24, x + 0.24, 0.0, 0.72, -0.05, 0.05)
        back.apply_transform(trimesh.transformations.rotation_matrix(-0.22, [1, 0, 0]))
        back.apply_translation([0, 0.45, zs - 0.27])
        parts.add("seat", back)
        head = B(x - 0.15, x + 0.15, 0.0, 0.20, -0.05, 0.05)
        head.apply_transform(trimesh.transformations.rotation_matrix(-0.22, [1, 0, 0]))
        head.apply_translation([0, 1.18, zs - 0.43])
        parts.add("seat", head)
        for dx in (-0.27, 0.27):  # accoudoirs
            parts.add("seat", B(x + dx - 0.03, x + dx + 0.03, 0.62, 0.68, zs - 0.20, zs + 0.15))
        # mini-manche lateral
        sx_ = x + (-0.62 if x < 0 else 0.62)
        parts.add("console", B(sx_ - 0.10, sx_ + 0.10, 0.0, 0.70, zs - 0.10, zs + 0.40))
        parts.add("dark", cyl(0.02, [sx_, 0.70, zs + 0.25], [sx_, 0.86, zs + 0.28]))
    # piedestal central avec manettes
    parts.add("console", B(-0.17, 0.17, 0.0, 0.72, zs - 0.15, zs + 0.70))
    for dx in (-0.05, 0.05):
        parts.add("steel", cyl(0.012, [dx, 0.72, zs + 0.35], [dx, 0.86, zs + 0.30]))
        parts.add("dark", B(dx - 0.03, dx + 0.03, 0.85, 0.89, zs + 0.27, zs + 0.33))
    parts.add("screen", B(-0.12, 0.12, 0.723, 0.726, zs + 0.45, zs + 0.62))
    # tableau de bord + ecrans
    zp = z0 + 1.72
    parts.add("console", B(-1.00, 1.00, 0.0, 1.10, zp, zp + 0.35))
    parts.add("console", B(-1.02, 1.02, 1.10, 1.17, zp - 0.12, zp + 0.25))  # casquette
    parts.add("dark", B(-1.00, 1.00, 1.17, 1.18, zp - 0.08, zp + 0.20))
    for cx in (-0.78, -0.45, 0.45, 0.78, 0.0):
        h = 0.18 if cx == 0.0 else 0.26
        y0 = 0.88 - h if cx == 0.0 else 0.72
        parts.add("screen", B(cx - 0.14, cx + 0.14, y0, y0 + h, zp - 0.006, zp))
    parts.add("screen", B(-0.12, 0.12, 0.92, 1.06, zp - 0.006, zp))
    # panneau plafond
    parts.add("console", B(-0.35, 0.35, 1.82, 1.88, z0 + 0.6, z0 + 1.4))
    parts.add("screen", B(-0.28, 0.28, 1.818, 1.82, z0 + 0.7, z0 + 1.3))


# ---------------------------------------------------------------- assemblage
def build():
    parts = Parts()
    build_cabin(parts)
    build_ceiling(parts, *CABIN, lights=False)
    floor(parts, *CABIN)
    for (z0, z1) in (REAR, FRONT):
        build_open_zone(parts, z0, z1, door_z=(z0 + z1) / 2)
        build_ceiling(parts, z0, z1)
        floor(parts, z0, z1)
    bulkhead(parts, REAR[0])           # cloison arriere pleine
    build_crew(parts)
    build_cockpit(parts)

    flip = np.diag([1, 1, -1, 1])  # avant de l'avion vers -Z
    scene = trimesh.Scene()
    names = {"panel": "Parois", "bin": "Coffres", "ceiling": "Plafond", "reveal": "Embrasures",
             "seam": "Joints", "dark": "Details", "line": "BandesSol", "light": "Lumieres", "psu": "PSU",
             "carpet": "Moquette", "steel": "Inox", "vinyl": "SolGalley", "cart": "Trolleys",
             "oven": "Fours", "mattress": "Matelas", "pillow": "Oreillers", "seat": "SiegesPilotes",
             "console": "Cockpit", "screen": "Ecrans", "exit": "Exit"}
    for k, name in names.items():
        meshes = parts[k]
        if not meshes:
            continue
        if k == "carpet":
            m = trimesh.util.concatenate(meshes)  # conserve les UV
        else:
            m = trimesh.util.concatenate([trimesh.Trimesh(x.vertices, x.faces, process=False) for x in meshes])
        m.apply_transform(flip)
        m.visual = trimesh.visual.TextureVisuals(uv=getattr(m.visual, "uv", None), material=M[k])
        scene.add_geometry(m, node_name=name, geom_name=name)
    return scene


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "cabine_avion.glb"
    sc = build()
    sc.export(out)
    total = COCKPIT[1] - REAR[0]
    print(f"{out}: longueur totale {total:.2f} m, {sum(len(g.faces) for g in sc.geometry.values())} triangles")
