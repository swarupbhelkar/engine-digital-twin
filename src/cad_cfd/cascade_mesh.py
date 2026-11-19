"""Week 5, step 4a: 2D cascade meshes for Fluent (healthy and eroded blade).

Why make the mesh in Python instead of Ansys Meshing: both blades get EXACTLY
the same mesh settings, so a change in loss comes from the blade, not the mesh.
The fouled case uses the healthy mesh with a rough wall (set in Fluent).

Mesh (Gmsh):
  - Blade: thin quad layers (boundary layer), first cell FIRST_CELL thick,
    growing by GROWTH, so y+ is about 1-2 (needed for the k-omega SST model).
  - Rest of the passage: triangles, small near the blade and in the wake,
    larger towards inlet and outlet.
  - Upper and lower boundaries are an exact copy of each other shifted by
    one pitch, so Fluent can join them as a periodic pair.

Output: cfd/mesh/<case>.msh in Fluent's own text mesh format, with named zones:
  fluid (cells), interior, blade, inlet, outlet, periodic-upper, periodic-lower.
In Fluent, the two periodic zones are linked with one command (see the journal).

Run from the repo root:
    python -m src.cad_cfd.cascade_mesh
"""
import math
from pathlib import Path

import gmsh
import matplotlib.pyplot as plt
import numpy as np

from src.cad_cfd.blade_geometry import domain, profile
from src.cad_cfd.hpc_stage import design_stage

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cfd" / "mesh"

FIRST_CELL = 1.0e-6     # m, first cell height on the blade (y+ about 2)
GROWTH = 1.2            # each layer 20 % thicker than the one below
LAYERS_THICK = 0.6e-3   # m, total boundary-layer mesh thickness
SIZE_BLADE = 0.25e-3    # m, cell size along the blade
SIZE_EDGE = 0.08e-3     # m, at leading and trailing edge
SIZE_WAKE = 0.6e-3      # m, in the wake
SIZE_FAR = 1.5e-3       # m, near inlet and outlet

# Fluent zone ids and boundary types (Fluent file format codes)
ZONES = {"interior": (3, 2), "blade": (4, 3), "inlet": (5, 4), "outlet": (6, 5),
         "periodic-upper": (7, 3), "periodic-lower": (8, 3)}
CELL_ZONE = 2


def build(case: str):
    """Mesh one case with Gmsh. Returns nodes (N x 2), cells (list of node
    index tuples) and boundary edges per zone name."""
    s = design_stage()
    blade = profile(s, case)[:-1]           # drop repeated closing point
    dom = domain(s)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(case)
    geo = gmsh.model.geo

    # Blade: one closed spline starting and ending at the sharp trailing edge
    bp = [geo.addPoint(x, y, 0) for x, y in blade]
    i_le = int(np.argmin(blade[:, 0]))       # leading edge point
    c_suction = geo.addSpline(bp[: i_le + 1])
    c_pressure = geo.addSpline(bp[i_le:] + [bp[0]])
    blade_loop = geo.addCurveLoop([c_suction, c_pressure])

    # Periodic lines: straight inlet part, camber-shaped middle, straight outlet part
    def side(line):
        pts = [geo.addPoint(x, y, 0) for x, y in line]
        return pts, [geo.addLine(pts[0], pts[1]), geo.addSpline(pts[1:-1]),
                     geo.addLine(pts[-2], pts[-1])]

    lo_pts, lo = side(dom["periodic_lower"])
    up_pts, up = side(dom["periodic_upper"])
    inlet = geo.addLine(lo_pts[0], up_pts[0])
    outlet = geo.addLine(lo_pts[-1], up_pts[-1])
    outer = geo.addCurveLoop([lo[0], lo[1], lo[2], outlet, -up[2], -up[1], -up[0], -inlet])
    surf = geo.addPlaneSurface([outer, blade_loop])
    geo.synchronize()

    # Upper boundary = lower boundary shifted by one pitch (exact copy of the mesh)
    shift = [1, 0, 0, 0, 0, 1, 0, s.pitch, 0, 0, 1, 0, 0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(1, up, lo, shift)

    # Cell sizes: small at the blade and in the wake, larger far away
    f = gmsh.model.mesh.field
    d_blade = f.add("Distance")
    f.setNumbers(d_blade, "CurvesList", [c_suction, c_pressure])
    f.setNumber(d_blade, "Sampling", 400)
    t_blade = f.add("Threshold")
    f.setNumber(t_blade, "InField", d_blade)
    f.setNumber(t_blade, "SizeMin", SIZE_BLADE)
    f.setNumber(t_blade, "SizeMax", SIZE_FAR)
    f.setNumber(t_blade, "DistMin", LAYERS_THICK)
    f.setNumber(t_blade, "DistMax", 0.02)

    d_edges = f.add("Distance")
    f.setNumbers(d_edges, "PointsList", [bp[0], bp[i_le]])
    t_edges = f.add("Threshold")
    f.setNumber(t_edges, "InField", d_edges)
    f.setNumber(t_edges, "SizeMin", SIZE_EDGE)
    f.setNumber(t_edges, "SizeMax", SIZE_FAR)
    f.setNumber(t_edges, "DistMin", 0.5e-3)
    f.setNumber(t_edges, "DistMax", 6e-3)

    # Wake: a line from the trailing edge along the exit flow angle
    te = blade[0]
    b2 = math.radians(s.beta2)
    wake_end = te + 1.4 * dom["axial_chord"] * np.array([1.0, math.tan(b2)])
    w0, w1 = geo.addPoint(*te, 0), geo.addPoint(*wake_end, 0)
    wake = geo.addLine(w0, w1)
    geo.synchronize()
    d_wake = f.add("Distance")
    f.setNumbers(d_wake, "CurvesList", [wake])
    f.setNumber(d_wake, "Sampling", 200)
    t_wake = f.add("Threshold")
    f.setNumber(t_wake, "InField", d_wake)
    f.setNumber(t_wake, "SizeMin", SIZE_WAKE)
    f.setNumber(t_wake, "SizeMax", SIZE_FAR)
    f.setNumber(t_wake, "DistMin", 1.5e-3)
    f.setNumber(t_wake, "DistMax", 8e-3)

    size = f.add("Min")
    f.setNumbers(size, "FieldsList", [t_blade, t_edges, t_wake])
    f.setAsBackgroundMesh(size)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # Boundary layer: quads on the blade, fanned round the sharp trailing edge
    bl = f.add("BoundaryLayer")
    f.setNumbers(bl, "CurvesList", [c_suction, c_pressure])
    f.setNumber(bl, "Size", FIRST_CELL)
    f.setNumber(bl, "Ratio", GROWTH)
    f.setNumber(bl, "Thickness", LAYERS_THICK)
    f.setNumber(bl, "Quads", 1)
    f.setNumbers(bl, "FanPointsList", [bp[0]])
    f.setNumber(bl, "SizeFar", SIZE_BLADE)
    f.setAsBoundaryLayer(bl)

    # The wake line is only a sizing guide: keep it out of the mesh
    gmsh.model.geo.remove([(1, wake), (0, w0), (0, w1)])
    geo.synchronize()

    names = {"blade": [c_suction, c_pressure], "inlet": [inlet], "outlet": [outlet],
             "periodic-upper": up, "periodic-lower": lo}
    for name, curves in names.items():
        gmsh.model.addPhysicalGroup(1, curves, name=name)
    gmsh.model.addPhysicalGroup(2, [surf], name="fluid")
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)

    # Pull the mesh out of Gmsh
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    index = {int(t): i for i, t in enumerate(tags)}
    nodes = xyz.reshape(-1, 3)[:, :2]
    cells = []
    for etype, _, enodes in zip(*gmsh.model.mesh.getElements(2)):
        n = {2: 3, 3: 4}[int(etype)]
        cells += [tuple(index[int(t)] for t in c) for c in enodes.reshape(-1, n)]
    bounds = {}
    for name, curves in names.items():
        edges = []
        for c in curves:
            for etype, _, enodes in zip(*gmsh.model.mesh.getElements(1, c)):
                edges += [tuple(index[int(t)] for t in e) for e in enodes.reshape(-1, 2)]
        bounds[name] = edges
    gmsh.finalize()
    return nodes, cells, bounds


def faces_from_cells(nodes, cells, bounds):
    """Every edge once, with the cell on each side.
    Fluent rule (2D, checked in Fluent 2026 R1): c0 is the cell on the LEFT of
    the edge going n0 -> n1 (right-hand rule: z x (n1 - n0) points into c0);
    boundary faces have only c0 (c1 = 0)."""
    owner = {}
    for ci, cell in enumerate(cells, start=1):
        for a, b in zip(cell, cell[1:] + cell[:1]):
            owner.setdefault(frozenset((a, b)), []).append(ci)
    centre = np.array([nodes[list(c)].mean(axis=0) for c in cells])
    zone_of = {}
    for name, edges in bounds.items():
        for a, b in edges:
            zone_of[frozenset((a, b))] = name
    faces = {name: [] for name in ZONES}
    for key, cs in owner.items():
        a, b = tuple(key)
        name = zone_of.get(key, "interior")
        if len(cs) == 1 and name == "interior":
            raise ValueError("boundary edge not in any boundary zone")
        if len(cs) == 2 and name != "interior":
            raise ValueError(f"interior edge marked as {name}")
        # Make the first cell lie to the left of a -> b
        d = nodes[b] - nodes[a]
        left = np.array([-d[1], d[0]])
        if np.dot(centre[cs[0] - 1] - nodes[a], left) < 0:
            a, b = b, a
        faces[name].append((a + 1, b + 1, cs[0], cs[1] if len(cs) == 2 else 0))
    return faces


def write_fluent(path: Path, nodes, cells, faces):
    """Fluent text mesh: nodes (section 10), cells (12), faces (13), zone names (45).
    Counts and indices are in hexadecimal, as the format requires."""
    n_nodes, n_cells = len(nodes), len(cells)
    n_faces = sum(len(v) for v in faces.values())
    out = [f'(0 "cascade mesh from src/cad_cfd/cascade_mesh.py")', "(2 2)",
           f"(10 (0 1 {n_nodes:x} 0 2))",
           f"(12 (0 1 {n_cells:x} 0))",
           f"(13 (0 1 {n_faces:x} 0))",
           f"(10 (1 1 {n_nodes:x} 1 2)("]
    out += [f"{x:.10e} {y:.10e}" for x, y in nodes]
    out.append("))")
    types = " ".join("1" if len(c) == 3 else "3" for c in cells)
    out += [f"(12 ({CELL_ZONE:x} 1 {n_cells:x} 1 0)(", types, "))"]
    first = 1
    for name, (zid, btype) in ZONES.items():
        fz = faces[name]
        last = first + len(fz) - 1
        out.append(f"(13 ({zid:x} {first:x} {last:x} {btype:x} 2)(")
        out += [f"{a:x} {b:x} {c0:x} {c1:x}" for a, b, c0, c1 in fz]
        out.append("))")
        first = last + 1
    out.append(f"(45 ({CELL_ZONE} fluid fluid)())")
    kinds = {2: "interior", 3: "wall", 4: "pressure-inlet", 5: "pressure-outlet"}
    for name, (zid, btype) in ZONES.items():
        out.append(f"(45 ({zid} {kinds[btype]} {name})())")
    path.write_text("\n".join(out) + "\n")


def check_periodic(nodes, faces, pitch):
    """Every node on the upper side must sit exactly one pitch above a lower node."""
    up = sorted({n for f in faces["periodic-upper"] for n in f[:2]})
    lo = sorted({n for f in faces["periodic-lower"] for n in f[:2]})
    lower = nodes[np.array(lo) - 1]
    worst = 0.0
    for n in up:
        d = np.min(np.hypot(*(lower + [0, pitch] - nodes[n - 1]).T))
        worst = max(worst, d)
    return len(up), len(lo), worst


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = design_stage()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, case in enumerate(("healthy", "eroded")):
        nodes, cells, bounds = build(case)
        faces = faces_from_cells(nodes, cells, bounds)
        write_fluent(OUT / f"{case}.msh", nodes, cells, faces)
        n_up, n_lo, worst = check_periodic(nodes, faces, s.pitch)
        quads = sum(len(c) == 4 for c in cells)
        print(f"{case:8s}: {len(cells):6d} cells ({quads} quads), {len(nodes)} nodes, "
              f"{sum(map(len, faces.values()))} faces; periodic nodes {n_up}/{n_lo}, "
              f"worst mismatch {worst:.1e} m")
        for ax, (x0, x1, y0, y1, title) in zip(axes[row], [
                (-0.035, 0.08, -0.06, 0.07, "whole passage"),
                (-0.0015, 0.0025, -0.0015, 0.0025, "leading edge"),
                (0.0285, 0.0315, 0.0215, 0.0245, "trailing edge")]):
            for c in cells:
                xy = nodes[list(c) + [c[0]]]
                if xy[:, 0].max() < x0 or xy[:, 0].min() > x1:
                    continue
                if xy[:, 1].max() < y0 or xy[:, 1].min() > y1:
                    continue
                ax.plot(*xy.T, color="k", lw=0.15)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal")
            ax.set_title(f"{case}: {title}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "16_cascade_mesh.png", dpi=140)
    print("Wrote cfd/mesh/healthy.msh, cfd/mesh/eroded.msh, reports/figures/16_cascade_mesh.png")


if __name__ == "__main__":
    main()
