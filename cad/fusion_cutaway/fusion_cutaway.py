"""Week 5, step 3: build the engine cutaway in Fusion 360 from the step 2 CSVs.

Run inside Fusion 360: Utilities > Add-Ins > Scripts and Add-Ins > Scripts,
"+" (or "Create" > link existing), pick this folder, then Run.

What it builds (one Fusion component per part, so each can get its own colour):
  Spinner, Fan, LPC, Duct, HPC, Burner, HPT, LPT : the flow passage of each part,
      hub radius to tip radius (the space the blades fill), from engine_layout.csv
  Core hub   : solid shaft/disc body under the passage
  Casing     : 20 mm shell outside the core passage
  Nacelle    : 50 mm shell outside the fan
Every part is revolved 270 deg about the engine axis, leaving a 90 deg cutaway.

Fusion's internal unit is cm, so mm values are divided by 10.

The geometry (polygons) is plain Python, so it can be checked outside Fusion:
    python cad/fusion_cutaway/fusion_cutaway.py
prints each polygon and its point count.
"""
import csv
import os

# Folder holding engine_layout.csv and components.csv (the repo's cad folder).
# Default: the folder above this script. Change it if you moved the script.
CAD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUT_ANGLE = "270 deg"
CASING_MM = 20.0
NACELLE_MM = 50.0
FAN_GAP_MM = 10.0       # fan tip to nacelle clearance


def read_csv(name):
    with open(os.path.join(CAD_DIR, name), newline="") as f:
        return [{k: (float(v) if k not in ("station", "name", "component") else v)
                 for k, v in row.items()} for row in csv.DictReader(f)]


def polygons():
    """Closed (x, r) outlines in mm, one per part. Points go round the outline."""
    lay = read_csv("engine_layout.csv")
    comps = {c["component"]: c for c in read_csv("components.csv")}
    fan = comps["Fan"]
    x = [p["x_mm"] for p in lay]
    hub = [p["r_hub_mm"] for p in lay]
    tip = [p["r_tip_mm"] for p in lay]

    def passage(x0, x1):
        """Hub-to-tip strip of the core passage between x0 and x1."""
        xs = [x0] + [xi for xi in x if x0 < xi < x1] + [x1]
        h = [interp(xi, x, hub) for xi in xs]
        t = [interp(xi, x, tip) for xi in xs]
        return list(zip(xs, h)) + list(zip(xs[::-1], t[::-1]))

    out = {}
    for name in ("LPC", "Duct", "HPC", "Combustor", "HPT", "LPT"):
        c = comps[name]
        out["Burner" if name == "Combustor" else name] = passage(c["x_start_mm"], c["x_end_mm"])

    x0, x1 = fan["x_start_mm"], fan["x_end_mm"]
    r_fh, r_ft = fan["r_hub_mm"], fan["r_tip_mm"]
    out["Fan"] = [(x0, r_fh), (x1, r_fh), (x1, r_ft), (x0, r_ft)]
    out["Spinner"] = [(x0 - 0.8 * r_fh, 0.0), (x0, 0.0), (x0, r_fh)]
    # Hub: under the fan, then under the core passage, down to the axis
    out["Core hub"] = ([(x0, 0.0), (x0, r_fh), (x1, r_fh)] + list(zip(x, hub))
                       + [(x[-1], 0.0)])
    out["Casing"] = (list(zip(x, tip))
                     + [(xi, ti + CASING_MM) for xi, ti in zip(x[::-1], tip[::-1])])
    r_n = r_ft + FAN_GAP_MM
    x_n = x0 + 2.6 * (x1 - x0)              # nacelle runs past the LPC
    out["Nacelle"] = [(x0 - 150, r_n + 20), (x_n, r_n), (x_n, r_n + NACELLE_MM),
                      (x0 - 150, r_n + NACELLE_MM + 20)]
    return out


def interp(xi, xs, ys):
    for (xa, ya), (xb, yb) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if xa <= xi <= xb:
            return ya if xb == xa else ya + (yb - ya) * (xi - xa) / (xb - xa)
    return ys[0] if xi < xs[0] else ys[-1]


def clean(poly):
    """Drop repeated consecutive points (zero-length lines break a sketch)."""
    out = []
    for p in poly:
        if not out or abs(p[0] - out[-1][0]) > 1e-6 or abs(p[1] - out[-1][1]) > 1e-6:
            out.append(p)
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) < 1e-6 and abs(out[0][1] - out[-1][1]) < 1e-6:
        out.pop()
    return out


def run(context):
    import adsk.core
    import adsk.fusion
    import traceback

    app = adsk.core.Application.get()
    ui = app.userInterface
    try:
        design = adsk.fusion.Design.cast(app.activeProduct)
        if design is None:
            ui.messageBox("Open a new Fusion design first, then run the script.")
            return
        root = design.rootComponent
        for name, poly in polygons().items():
            occ = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
            comp = occ.component
            comp.name = name
            sk = comp.sketches.add(comp.xYConstructionPlane)
            lines = sk.sketchCurves.sketchLines
            pts = [adsk.core.Point3D.create(px / 10.0, pr / 10.0, 0) for px, pr in clean(poly)]
            first = prev = lines.addByTwoPoints(pts[0], pts[1])
            for p in pts[2:]:
                prev = lines.addByTwoPoints(prev.endSketchPoint, p)
            lines.addByTwoPoints(prev.endSketchPoint, first.startSketchPoint)

            revs = comp.features.revolveFeatures
            inp = revs.createInput(sk.profiles.item(0), comp.xConstructionAxis,
                                   adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
            inp.setAngleExtent(False, adsk.core.ValueInput.createByString(CUT_ANGLE))
            revs.add(inp)
        ui.messageBox("Cutaway built: one component per part. Next: colours (step 3b).")
    except Exception:
        ui.messageBox("Failed:\n{}".format(traceback.format_exc()))


if __name__ == "__main__":
    for k, v in polygons().items():
        print(f"{k:<9} {len(clean(v)):3d} points  x {min(p[0] for p in v):7.1f} .. "
              f"{max(p[0] for p in v):7.1f}  r {min(p[1] for p in v):7.1f} .. "
              f"{max(p[1] for p in v):7.1f} mm")
