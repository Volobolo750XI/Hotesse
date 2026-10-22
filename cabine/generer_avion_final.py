"""Avion final pour UEFN : un seul objet (+ les portes a part).

Part de avion_complet_sans_coffres.glb (meme taille, meme interieur) et :
- retire le bout du nez d'origine reste en double sur le nouveau nez ;
- exterieur comme le modele A320 : hublots noirs en rangee continue,
  pare-brise noir (peints a l'exterieur, transparents vus de l'interieur) ;
- retire les anciens contours gris des hublots peints ;
- materiaux a deux faces (on ne voit plus le ciel a travers l'envers des murs) ;
- regroupe tout ce qui n'est pas une porte en UN seul maillage ("Avion"),
  chaque porte reste un objet a part avec son pivot sur la charniere.

Usage : python3 generer_avion_final.py [sortie.glb]
"""
import io
import os
import sys

import numpy as np
import pygltflib
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate_avion_exterieur as E  # noqa: E402
import generate_cabin as G  # noqa: E402

BASE = os.path.join(HERE, "avion_complet_sans_coffres.glb")
COFFRES = os.path.join(HERE, "coffres.glb")
K, A_CY, CY, OFF = E.K, E.A_CY, E.CY_EXT, E.OFF
A320_TO_GLTF = np.array([[0, 0, K, 0], [0, K, 0, CY - A_CY * K], [K, 0, 0, -OFF], [0, 0, 0, 1]])
NOIR = trimesh.visual.material.PBRMaterial(name="HublotNoir", baseColorFactor=[10, 10, 12, 255],
                                           metallicFactor=0.0, roughnessFactor=0.25, doubleSided=False)


def skin(z):
    """(cy, hy, hz) de la peau exterieure a la position glTF z."""
    return E.ext_section(-z)


def radial(p):
    """Direction vers l'exterieur depuis l'axe du fuselage (p : points glTF)."""
    out = np.zeros_like(p)
    for i, (x, y, z) in enumerate(p):
        cy, hy, hz = skin(z)
        v = np.array([x / max(hz, 1e-3) ** 2, (y - cy) / max(hy, 1e-3) ** 2, 0.0])
        out[i] = v / (np.linalg.norm(v) + 1e-9)
    return out


def hublot(z, yc, w, h, r, cote, n=10):
    """Hublot noir arrondi pose sur la peau (face vers l'exterieur seulement)."""
    us = np.linspace(-w / 2, w / 2, n)
    vs = np.linspace(-h / 2, h / 2, n + 4)
    U, V = np.meshgrid(us, vs)
    # arrondi des coins
    ax, ay = np.maximum(np.abs(U) - (w / 2 - r), 0), np.maximum(np.abs(V) - (h / 2 - r), 0)
    d = np.hypot(ax, ay)
    f = np.where(d > r, r / np.maximum(d, 1e-9), 1.0)
    U = np.sign(U) * (np.minimum(np.abs(U), w / 2 - r) + ax * f)
    V = np.sign(V) * (np.minimum(np.abs(V), h / 2 - r) + ay * f)
    zs, ys = z + U.ravel(), yc + V.ravel()
    xs = []
    for zz, yy in zip(zs, ys):
        cy, hy, hz = skin(zz)
        t = np.clip((yy - cy) / hy, -1, 1)
        xs.append(cote * (hz * np.sqrt(1 - t * t) + 0.006))
    verts = np.column_stack([xs, ys, zs])
    faces = []
    cols = n
    for j in range(len(vs) - 1):
        for i in range(cols - 1):
            a, b, c, d2 = j * cols + i, j * cols + i + 1, (j + 1) * cols + i + 1, (j + 1) * cols + i
            faces += [[a, b, c], [a, c, d2]]
    m = trimesh.Trimesh(verts, np.array(faces), process=False)
    if (m.face_normals[:, 0] * cote).mean() < 0:
        m.invert()
    return m


def hublots_exterieurs():
    """Rangee de hublots noirs comme sur l'A320 : sur les vrais hublots + peints ailleurs."""
    vrais = [-(G.CABIN[0] + G.MARGIN + i * G.PITCH) for i in range(G.N_WINDOWS)]
    p, _ = G.WALL.map(np.array([G.WIN_CENTER_S]), np.array([G.CABIN[0] + G.MARGIN]), G.WIN_DEPTH)
    yc = float(p[0, 1])
    portes = [-(a + b) / 2 for a, b in (G.REAR_DOOR, G.FRONT_DOOR)]
    # etendue des rangees sur l'A320 d'origine
    src = trimesh.load(E.A320, force="scene")
    zz = []
    for node in src.graph.nodes_geometry:
        T, g = src.graph[node]
        if g.startswith("Cube.039") or g.startswith("Cube.001"):
            m = src.geometry[g].copy()
            m.apply_transform(A320_TO_GLTF @ T)
            zz += [m.bounds[0][2], m.bounds[1][2]]
    z_min, z_max = min(zz), max(zz)
    pos = []
    for z in np.arange(vrais[0] + 60 * G.PITCH, vrais[-1] - 60 * G.PITCH, -G.PITCH):
        if not (z_min + 0.15 < z < z_max - 0.15):
            continue
        if min(abs(z - d) for d in portes) < 0.85 or z > portes[0]:
            continue
        pos.append(z)
    meshes = []
    for z in pos:
        for cote in (-1, 1):
            meshes.append(hublot(z, yc, G.WIN_W + 0.07, G.WIN_H + 0.07, G.WIN_R + 0.03, cote))
    return trimesh.util.concatenate(meshes), len(pos)


def parebrise():
    """Vitres noires du cockpit reprises de l'A320 (faces exterieures)."""
    src = trimesh.load(E.A320, force="scene")
    out = []
    for node in src.graph.nodes_geometry:
        T, g = src.graph[node]
        if not (g.startswith("Cylinder.003") or g.startswith("Cylinder.011")):
            continue
        m = src.geometry[g].copy()
        m.apply_transform(A320_TO_GLTF @ T)
        c = m.triangles_center
        keep = (m.face_normals * radial(c)).sum(1) > 0.3
        m.update_faces(keep)
        m.remove_unreferenced_vertices()
        m.vertices += radial(m.vertices) * 0.012
        out.append(m)
    return trimesh.util.concatenate(out)


def boucher_parebrise(fus):
    """Bouche les 4 trous du pare-brise interieur (peau vue de dehors seulement)."""
    m = fus.copy()
    m.merge_vertices(digits_vertex=5)
    e = m.edges_sorted
    u, c = np.unique(e, axis=0, return_counts=True)
    import networkx as nx
    g = nx.Graph()
    g.add_edges_from(u[c == 1])
    verts, faces = [], []
    for comp in nx.connected_components(g):
        p = m.vertices[list(comp)]
        if not (p[:, 2].mean() < -19.0 and p[:, 1].mean() > 1.2 and len(comp) < 200):
            continue
        cyc = nx.cycle_basis(g.subgraph(comp))
        boucle = max(cyc, key=len)
        o = len(verts)
        pts = m.vertices[boucle]
        verts += list(pts) + [pts.mean(0)]
        n = len(boucle)
        faces += [[o + i, o + (i + 1) % n, o + n] for i in range(n)]
    t = trimesh.Trimesh(np.array(verts), np.array(faces), process=False)
    # normales vers l'exterieur
    out = (t.face_normals * radial(t.triangles_center)).sum(1) < 0
    f = t.faces.copy()
    f[out] = f[out][:, ::-1]
    t = trimesh.Trimesh(t.vertices, f, process=False)
    for _ in range(3):
        t = t.subdivide()
    # chaque point est replace sur la peau (meme courbure que le fuselage)
    v = t.vertices.copy()
    for i, (x, y, z) in enumerate(v):
        cy, hy, hz = skin(z)
        k = 1.0 / np.sqrt((x / hz) ** 2 + ((y - cy) / hy) ** 2)
        v[i] = [x * k, cy + (y - cy) * k, z]
    t = trimesh.Trimesh(v, t.faces, process=False)
    mat = fus.visual.material.copy()
    mat.name = "FuselageNez"
    t.visual = trimesh.visual.TextureVisuals(material=mat)
    t.metadata["une_face"] = True
    return t


# couches posees sur une autre surface : decalees vers l'interieur pour ne plus clignoter
DECALAGE = {"Joint": 0.003, "JointSombre": 0.003, "Lumiere": 0.003, "Ecran": 0.004, "Inox": 0.003,
            "Exit": 0.003, "Miroir": 0.003, "PSU": 0.002}


def decoller(m, d):
    """Deplace chaque morceau (d'un bloc) de d vers l'interieur de la cabine."""
    v = m.vertices.copy()
    lab = trimesh.graph.connected_component_labels(m.face_adjacency, node_count=len(m.faces))
    c, n, ar = m.triangles_center, m.face_normals, m.area_faces
    for k in np.unique(lab):
        f = lab == k
        vers = np.column_stack([-c[f, 0], 1.1 - c[f, 1], np.zeros(f.sum())])
        vers /= np.linalg.norm(vers, axis=1, keepdims=True) + 1e-9
        dot = (n[f] * vers).sum(1)
        if np.abs(dot).max() < 0.3:     # face tournee vers l'avant / l'arriere
            vers = np.column_stack([np.zeros(f.sum()), np.zeros(f.sum()), np.sign(-8.0 - c[f, 2])])
            dot = (n[f] * vers).sum(1)
        w = ar[f] * (dot > 0.3)
        if w.sum() == 0:
            continue
        dirn = (n[f] * w[:, None]).sum(0)
        dirn /= np.linalg.norm(dirn) + 1e-9
        idx = np.unique(m.faces[f])
        v[idx] += dirn * d
    m.vertices = v
    return m


def nettoyer(name, m):
    """Corrige une piece de l'avion (repere glTF)."""
    mat = getattr(getattr(m.visual, "material", None), "name", "")
    if mat in DECALAGE and not name.startswith("Porte_"):
        decoller(m, DECALAGE[mat])
    if mat == "SolVinyle":
        m.vertices = m.vertices + [0, 0.008, 0]
    if mat == "BandeJaune":
        m.vertices = m.vertices + [0, 0.004, 0]
    if name.startswith("A320_Cylinder002"):
        # bout du nez d'origine reste en double sur le nouveau nez
        m.update_faces(m.triangles_center[:, 2] > -17.0)
    elif name == "Details":
        # anciens contours gris des hublots, sur la peau exterieure
        m.update_faces(np.abs(m.triangles_center[:, 0]) < 1.6)
    m.remove_unreferenced_vertices()
    return m


def charger():
    corps, portes = [], {}
    for fichier in (BASE, COFFRES):
        s = trimesh.load(fichier)
        racine = s.graph.base_frame
        for node in s.graph.nodes_geometry:
            T, g = s.graph[node]
            m = s.geometry[g].copy()
            parent = s.graph.transforms.parents.get(node, racine)
            if parent != racine and str(parent).startswith(("Porte_", "Coffre_")):
                # porte : geometrie locale, pivot = noeud parent (charniere)
                P, _ = s.graph[parent]
                m.apply_transform(T)
                if str(parent).startswith("Coffre_") and getattr(m.visual.material, "name", "") in DECALAGE:
                    decoller(m, DECALAGE[m.visual.material.name])
                m.apply_transform(np.linalg.inv(P))
                if getattr(m.visual.material, "name", "") == "Joint":
                    m.vertices = m.vertices + m.face_normals.mean(0) * 0.002
                portes.setdefault(parent, [P, []])[1].append((g, m))
            else:
                m.apply_transform(T)
                corps.append((g, nettoyer(g, m)))
    return corps, portes


def ecrire(corps, portes, out):
    """Ecrit le glb : un noeud 'Avion' (un maillage, plusieurs materiaux) + un noeud par porte."""
    sc = trimesh.Scene()
    groupes = {"Avion": (np.eye(4), corps)}
    for k, (P, lst) in portes.items():
        groupes[k] = (P, lst)
    noms = {}
    for gname, (P, lst) in groupes.items():
        for i, (n, m) in enumerate(lst):
            mat = getattr(m.visual, "material", None)
            if mat is not None and mat is not NOIR and not m.metadata.get("une_face"):
                mat = mat.copy() if hasattr(mat, "copy") else mat
                try:
                    mat.doubleSided = True
                except Exception:
                    pass
                m.visual.material = mat
            key = f"{gname}__{i}"
            sc.add_geometry(m, node_name=key, geom_name=key)
            noms[key] = gname
    data = sc.export(file_type="glb")
    gl = pygltflib.GLTF2.load_from_bytes(data)
    # regroupe les primitives par groupe
    prims = {g: [] for g in groupes}
    for node in gl.nodes:
        if node.mesh is not None and node.name in noms:
            prims[noms[node.name]] += gl.meshes[node.mesh].primitives
    gl.meshes = []
    gl.nodes = []
    racine = []
    for gname, (P, _) in groupes.items():
        gl.meshes.append(pygltflib.Mesh(name=gname, primitives=prims[gname]))
        nd = pygltflib.Node(name=gname, mesh=len(gl.meshes) - 1)
        if not np.allclose(P, np.eye(4)):
            nd.matrix = [float(v) for v in P.T.ravel()]
        gl.nodes.append(nd)
        racine.append(len(gl.nodes) - 1)
    gl.scenes = [pygltflib.Scene(nodes=racine)]
    gl.scene = 0
    gl.save_binary(out)


def baguettes(mat):
    """Baguettes de joint sur les raccords mur de cabine / zones de portes (ferment les fentes)."""
    ss = np.linspace(0, G.WALL.length, 48)
    out = []
    for zi in (G.CABIN[0], G.CABIN[1]):
        cols = []
        for dz in (-0.02, 0.02):
            q, _ = G.WALL.map(ss, np.full(len(ss), zi + dz), 0.004)
            cols.append(q)
        n = len(ss)
        v = np.vstack(cols)
        f = [[i, i + 1, n + i + 1] for i in range(n - 1)] + [[i, n + i + 1, n + i] for i in range(n - 1)]
        for cote in (1, -1):
            w = v.copy()
            w[:, 0] *= cote
            w[:, 2] *= -1          # repere interne -> glTF
            m = trimesh.Trimesh(w, np.array(f), process=False)
            out.append(m)
    m = trimesh.util.concatenate(out)
    m.visual = trimesh.visual.TextureVisuals(material=mat)
    return m


def build(out):
    corps, portes = charger()
    hub, n = hublots_exterieurs()
    hub.visual = trimesh.visual.TextureVisuals(material=NOIR)
    pb = parebrise()
    pb.visual = trimesh.visual.TextureVisuals(material=NOIR)
    fus = dict(corps)["Fuselage"]
    corps.append(("Baguettes", baguettes(dict(corps)["Joints"].visual.material)))
    corps += [("HublotsNoirs", hub), ("PareBrise", pb), ("NezBouche", boucher_parebrise(fus))]
    ecrire(corps, portes, out)
    tri = sum(len(m.faces) for _, m in corps) + sum(len(m.faces) for _, l in portes.values() for _, m in l)
    print(f"{out}: 1 objet Avion + {len(portes)} portes, {n} hublots par cote, {tri} triangles")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "avion_final.glb"))
