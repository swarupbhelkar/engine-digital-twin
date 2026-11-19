"""Week 5, step 3b: colour the cutaway by HPC health and save the three images.

Run inside Fusion AFTER fusion_cutaway has built the model, with the view set
the way you want it (the script keeps your camera and fits the model in it).

What it does:
  1. Gives every part a metal appearance from the Fusion appearance library.
  2. For each moment in cad/health_colours.csv (new, mid-wear, last cycle):
     sets the HPC paint to that colour and saves the viewport as
     reports/figures/15_cutaway_<name>.png (1600 x 900).
  3. Leaves the HPC on the "new" colour at the end.

These are viewport captures (what you see on screen), not ray-traced renders.
For a photo-real version use the Render workspace by hand afterwards.
"""
import csv
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COLOURS = os.path.join(REPO, "cad", "health_colours.csv")
FIGURES = os.path.join(REPO, "reports", "figures")
SHOTS = [("new", "new"), ("mid-wear", "midwear"), ("now (last cycle)", "now")]
WIDTH, HEIGHT = 1600, 900

# Part -> appearance name in the "Fusion Appearance Library"
LOOKS = {"Fan": "Titanium - Satin", "Spinner": "Titanium - Satin",
         "Nacelle": "Aluminum - Satin", "Casing": "Aluminum - Satin",
         "LPC": "Steel - Satin", "Duct": "Steel - Satin", "LPT": "Steel - Satin",
         "Burner": "Nickel - Satin", "HPT": "Nickel - Satin",
         "Core hub": "Paint - Enamel Glossy (Dark Grey)",
         "HPC": "Paint - Enamel Glossy (Green)"}
HIDE = ("Casing", "Nacelle")   # hidden so the whole HPC surface shows its colour


def read_colours():
    with open(COLOURS, newline="") as f:
        return {r["label"]: [int(v) for v in r["rgb"].split()] for r in csv.DictReader(f)}


def design_look(design, name):
    """Appearance already copied into this design, or None."""
    for i in range(design.appearances.count):
        if design.appearances.item(i).name == name:
            return design.appearances.item(i)
    return None


def run(context):
    import traceback

    import adsk.core
    import adsk.fusion

    app = adsk.core.Application.get()
    ui = app.userInterface
    try:
        design = adsk.fusion.Design.cast(app.activeProduct)
        occs = {o.component.name: o for o in design.rootComponent.occurrences}
        missing = [n for n in LOOKS if n not in occs]
        if missing:
            ui.messageBox("Run fusion_cutaway first. Missing parts: " + ", ".join(missing))
            return
        lib = app.materialLibraries.itemByName("Fusion Appearance Library")

        # 1) Appearances; the HPC gets its own copy so its colour can change
        for part, lib_name in LOOKS.items():
            name = "HPC health" if part == "HPC" else lib_name
            look = design_look(design, name) or design.appearances.addByCopy(
                lib.appearances.itemByName(lib_name), name)
            occs[part].appearance = look
        for part in HIDE:
            occs[part].isLightBulbOn = False
        colour = design_look(design, "HPC health").appearanceProperties.itemById("opaque_albedo")

        # 2) One image per moment of the demo engine
        colours = read_colours()
        view = app.activeViewport
        view.fit()
        saved = []
        for label, tag in SHOTS:
            r, g, b = colours[label]
            colour.value = adsk.core.Color.create(r, g, b, 255)
            adsk.doEvents()
            view.refresh()
            path = os.path.join(FIGURES, f"15_cutaway_{tag}.png").replace("\\", "/")
            if view.saveAsImageFile(path, WIDTH, HEIGHT):
                saved.append(os.path.basename(path))
        r, g, b = colours["new"]
        colour.value = adsk.core.Color.create(r, g, b, 255)
        ui.messageBox("Saved to reports/figures:\n  " + "\n  ".join(saved))
    except Exception:
        ui.messageBox("Failed:\n{}".format(traceback.format_exc()))
