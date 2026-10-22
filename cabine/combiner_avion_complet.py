"""avion_complet.glb combine en UN seul maillage, sans les portes ni les coffres a bagages.

Rien d'autre ne change : memes pieces, memes materiaux, meme taille, meme position.
Chaque piece devient une partie (un materiau) du maillage unique "Avion".

Usage : python3 combiner_avion_complet.py [sortie.glb]
"""
import io
import os
import sys

import numpy as np
import pygltflib
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "avion_complet.glb")


def combiner(out):
    s = trimesh.load(SOURCE)
    sc = trimesh.Scene()
    n = 0
    for node in s.graph.nodes_geometry:
        T, g = s.graph[node]
        chemin = s.graph.transforms.parents
        # on remonte les parents pour savoir si la piece appartient a une porte
        p, porte = node, False
        while p in chemin:
            p = chemin[p]
            if str(p).startswith("Porte_"):
                porte = True
        if porte or g.startswith("Porte_") or g == "Coffres":
            continue
        m = s.geometry[g].copy()
        m.apply_transform(T)
        sc.add_geometry(m, node_name=f"p{n}", geom_name=f"p{n}")
        n += 1
    gl = pygltflib.GLTF2.load_from_bytes(sc.export(file_type="glb"))
    prims = [p for nd in gl.nodes if nd.mesh is not None for p in gl.meshes[nd.mesh].primitives]
    gl.meshes = [pygltflib.Mesh(name="Avion", primitives=prims)]
    gl.nodes = [pygltflib.Node(name="Avion", mesh=0)]
    gl.scenes = [pygltflib.Scene(nodes=[0])]
    gl.scene = 0
    gl.save_binary(out)
    print(f"{out}: {n} pieces combinees en 1 maillage")


if __name__ == "__main__":
    combiner(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "avion_complet_combine.glb"))
