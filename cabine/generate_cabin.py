"""Genere l'interieur vide d'une cabine d'avion au format GLB.

Hublots et porte sont des ouvertures reelles (pas de vitre) : on voit a travers.
Repere glTF : Y vers le haut, la cabine s'etend de z=0 (cloison arriere)
a z=-LENGTH (cloison avant avec la porte). Unites en metres.

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
MARGIN = 0.75           # distance cloison -> premier hublot
LENGTH = 2 * MARGIN + (N_WINDOWS - 1) * PITCH
Z0, Z1 = 0.0, LENGTH    # coordonnee "longueur" positive ; convertie en -z a la fin

WIN_W, WIN_H, WIN_R = 0.27, 0.39, 0.12      # ouverture du hublot
BEZ_W, BEZ_H, BEZ_R = 0.36, 0.50, 0.16      # encadrement en retrait
BEZ_DEPTH, WIN_DEPTH = 0.018, 0.16
WIN_CENTER_S = 1.02     # position du hublot le long du profil de paroi

BIN_LEN = 3 * PITCH     # longueur d'une porte de coffre
DOOR_W, DOOR_H, DOOR_R, DOOR_DEPTH = 0.86, 1.86, 0.14, 0.30


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

    def points(self, step=0.02):
        s = np.linspace(0, self.length, max(2, int(self.length / step)))
        return np.column_stack([np.interp(s, self.s, self.x), np.interp(s, self.s, self.y)])


WALL = Profile([(1.70, 0.00), (1.80, 0.35), (1.87, 0.80), (1.88, 1.20),
                (1.84, 1.50), (1.74, 1.68)])
BIN_BOTTOM = Profile([(1.74, 1.68), (1.45, 1.695), (1.16, 1.72)])
BIN_FACE = Profile([(1.16, 1.72), (1.075, 1.80), (1.05, 1.94), (1.09, 2.07), (1.19, 2.16)])
CEILING = Profile([(1.19, 2.16), (1.13, 2.24), (0.95, 2.31), (0.60, 2.355), (0.0, 2.37)])
PROFILES = [WALL, BIN_BOTTOM, BIN_FACE, CEILING]


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


def door_shape(cx, w, h, r):
    rect = box(cx - w / 2, 0, cx + w / 2, h - r)
    top = rrect(cx, h - r - 0.001, w, 2 * r, r * 0.999)
    return unary_union([rect, top]).simplify(1e-4)


def mirror(mesh):
    m = mesh.copy()
    m.apply_transform(np.diag([-1, 1, 1, 1]))
    return m


def window_zs():
    return [Z0 + MARGIN + i * PITCH for i in range(N_WINDOWS)]


# ------------------------------------------------------------------- pieces
def build_side():
    parts = {k: [] for k in M}
    zs = window_zs()
    bezels = [rrect(z, WIN_CENTER_S, BEZ_W, BEZ_H, BEZ_R) for z in zs]
    holes = [rrect(z, WIN_CENTER_S, WIN_W, WIN_H, WIN_R) for z in zs]

    # paroi laterale percee
    wall_poly = box(Z0, 0, Z1, WALL.length).difference(unary_union(bezels))
    parts["panel"].append(surface(WALL, wall_poly))
    for bz, h in zip(bezels, holes):
        parts["panel"].append(tube(WALL, bz.exterior.coords, 0, BEZ_DEPTH))
        parts["panel"].append(surface(WALL, bz.difference(h), BEZ_DEPTH, 0.02))
        parts["reveal"].append(tube(WALL, h.exterior.coords, BEZ_DEPTH, WIN_DEPTH))

    # joints de panneaux (verticaux entre hublots + horizontal en bas)
    seams = [box(z - 0.003, 0.02, z + 0.003, WALL.length - 0.01)
             for z in np.arange(Z0 + MARGIN - PITCH / 2, Z1, PITCH)]
    seams.append(box(Z0, 0.42, Z1, 0.426))
    seams.append(box(Z0, 0.0, Z1, 0.012))
    seam_poly = unary_union(seams).difference(unary_union(bezels).buffer(0.01))
    parts["seam"].append(surface(WALL, seam_poly, -0.0015))

    # coffres a bagages
    parts["bin"].append(surface(BIN_BOTTOM, box(Z0, 0, Z1, BIN_BOTTOM.length)))
    parts["bin"].append(surface(BIN_FACE, box(Z0, 0, Z1, BIN_FACE.length), strip=0.01))
    bin_seams, latches = [], []
    for z in np.arange(Z0 + BIN_LEN, Z1 - 0.1, BIN_LEN):
        bin_seams.append(box(z - 0.003, 0.005, z + 0.003, BIN_FACE.length - 0.005))
    for z in np.arange(Z0, Z1 - 0.1, BIN_LEN):
        latches.append(rrect(z + BIN_LEN / 2, BIN_FACE.length - 0.075, 0.16, 0.035, 0.015))
    bin_seams.append(box(Z0, 0.003, Z1, 0.008))  # bas de porte
    parts["seam"].append(surface(BIN_FACE, unary_union(bin_seams), -0.0015, 0.005))
    parts["dark"].append(surface(BIN_FACE, unary_union(latches), -0.0015, 0.01))

    # PSU (services passagers) sous les coffres : panneau + liseuses
    for z in zs[::2]:
        panel = rrect(z + PITCH / 2, 0.24, 0.42, 0.20, 0.03)
        parts["psu"].append(surface(BIN_BOTTOM, panel, -0.004, 0.02))
        for dz in (-0.09, 0.09):
            p, _ = BIN_BOTTOM.map(np.array([0.22]), np.array([z + PITCH / 2 + dz]), -0.004)
            dome = trimesh.creation.icosphere(subdivisions=2, radius=0.022)
            dome.apply_scale([1, 0.45, 1])
            dome.apply_translation(p[0])
            parts["light"].append(dome)

    # plafond + eclairage indirect + joints
    parts["ceiling"].append(surface(CEILING, box(Z0, 0, Z1, CEILING.length), strip=0.02))
    parts["light"].append(surface(CEILING, box(Z0, 0.004, Z1, 0.05), -0.002, 0.01))
    c_seams = [box(Z0, 0.62, Z1, 0.626)]
    c_seams += [box(z - 0.003, 0.06, z + 0.003, CEILING.length) for z in np.arange(Z0 + BIN_LEN, Z1, BIN_LEN)]
    parts["seam"].append(surface(CEILING, unary_union(c_seams), -0.0015, 0.02))
    return parts


def section_outline():
    right = np.vstack([p.points() for p in PROFILES])
    left = right[::-1].copy()
    left[:, 0] *= -1
    return Polygon(np.vstack([right, left])).buffer(0)


def end_wall(z, with_door):
    poly = section_outline()
    parts = {k: [] for k in M}
    door = door_shape(0.0, DOOR_W, DOOR_H, DOOR_R)
    sign = -1 if z > Z0 else 1  # vers l'interieur de la cabine
    if with_door:
        frame = door_shape(0.0, DOOR_W + 0.14, DOOR_H + 0.07, DOOR_R + 0.07)
        poly = poly.difference(frame)
        # cadre legerement en saillie
        v, f = triangulate(frame.difference(door))
        parts["panel"].append(trimesh.Trimesh(np.column_stack([v, np.full(len(v), z + sign * 0.02)]), f))
        ring = np.asarray(frame.exterior.coords)[:-1]
        parts["panel"].append(_extrude_ring(ring, z, z + sign * 0.02, open_bottom=True))
        ring = np.asarray(door.exterior.coords)[:-1]
        parts["reveal"].append(_extrude_ring(ring, z + sign * 0.02, z - sign * DOOR_DEPTH, open_bottom=True))
        seam = box(-DOOR_W / 2 - 0.07, DOOR_H + 0.25, DOOR_W / 2 + 0.07, DOOR_H + 0.256)
        v, f = triangulate(seam)
        parts["seam"].append(trimesh.Trimesh(np.column_stack([v, np.full(len(v), z + sign * 0.002)]), f))
    v, f = triangulate(poly)
    parts["panel"].append(trimesh.Trimesh(np.column_stack([v, np.full(len(v), z)]), f))
    return parts


def _extrude_ring(ring, z0, z1, open_bottom=False):
    if open_bottom:  # contour ouvert au sol : on retire l'arete au niveau y=0
        n = len(ring)
        keep = [i for i in range(n) if not (ring[i][1] < 1e-6 and ring[(i + 1) % n][1] < 1e-6)]
    else:
        keep = range(len(ring))
    n = len(ring)
    a = np.column_stack([ring, np.full(n, z0)])
    b = np.column_stack([ring, np.full(n, z1)])
    f = []
    for i in keep:
        j = (i + 1) % n
        f += [[i, j, n + j], [i, n + j, n + i]]
    return trimesh.Trimesh(np.vstack([a, b]), np.array(f), process=False)


def floor():
    xw = WALL.x[0]
    v = np.array([[-xw, 0, Z0], [xw, 0, Z0], [xw, 0, Z1], [-xw, 0, Z1]])
    f = np.array([[0, 2, 1], [0, 3, 2]])
    uv = np.column_stack([v[:, 0], v[:, 2]]) / 0.8  # texture repetee tous les 80 cm
    carpet = trimesh.Trimesh(v, f, process=False)
    carpet.visual = trimesh.visual.TextureVisuals(uv=uv, material=M["carpet"])
    lines = []
    for x in (-0.57, 0.57):
        b = trimesh.creation.box([0.04, 0.003, LENGTH - 0.02])
        b.apply_translation([x, 0.0015, (Z0 + Z1) / 2])
        lines.append(b)
    return carpet, lines


# ---------------------------------------------------------------- assemblage
def build():
    groups = {k: [] for k in M}
    side = build_side()
    for k, meshes in side.items():
        meshes = [m for m in meshes if m is not None]
        groups[k] += meshes + [mirror(m) for m in meshes]
    for parts in (end_wall(Z1, True), end_wall(Z0, False)):
        for k, meshes in parts.items():
            groups[k] += meshes
    carpet, lines = floor()
    groups["line"] += lines

    flip = np.diag([1, 1, -1, 1])  # longueur vers -Z (avant de l'avion)
    scene = trimesh.Scene()
    carpet.apply_transform(flip)
    scene.add_geometry(carpet, node_name="Moquette", geom_name="Moquette")
    names = {"panel": "Parois", "bin": "Coffres", "ceiling": "Plafond", "reveal": "Embrasures",
             "seam": "Joints", "dark": "Poignees", "line": "BandesSol", "light": "Lumieres", "psu": "PSU"}
    for k, name in names.items():
        if not groups[k]:
            continue
        m = trimesh.util.concatenate(groups[k])
        m.apply_transform(flip)
        m.visual = trimesh.visual.TextureVisuals(material=M[k])
        scene.add_geometry(m, node_name=name, geom_name=name)
    return scene


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "cabine_avion.glb"
    sc = build()
    sc.export(out)
    print(f"{out}: longueur {LENGTH:.2f} m, {sum(len(g.faces) for g in sc.geometry.values())} triangles")
