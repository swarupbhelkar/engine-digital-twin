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

# Part -> words to look for in the library appearance name (first match wins)
LOOKS = {
    "Fan": ["Titanium - Polished", "Titanium"],
    "Spinner": ["Titanium - Polished", "Titanium"],
    "Nacelle": ["Aluminum - Satin", "Aluminum"],
    "Casing": ["Aluminum - Satin", "Aluminum"],
    "LPC": ["Steel - Satin", "Steel"],
    "Duct": ["Steel - Satin", "Steel"],
    "LPT": ["Steel - Satin", "Steel"],
    "Burner": ["Nickel", "Steel - Satin", "Steel"],
    "HPT": ["Nickel", "Steel - Satin", "Steel"],
    "Core hub": ["Paint - Enamel Glossy (Grey)", "Paint - Enamel Glossy", "Plastic"],
    "HPC": ["Paint - Enamel Glossy (Green)", "Paint - Enamel Glossy", "Plastic"],
}


def read_colours():
    with open(COLOURS, newline="") as f:
        return {r["label"]: [int(v) for v in r["rgb"].split()] for r in csv.DictReader(f)}


def run(context):
    import traceback

    import adsk.core
    import adsk.fusion

    app = adsk.core.Application.get()
    ui = app.userInterface
    try:
        design = adsk.fusion.Design.cast(app.activeProduct)
        root = design.rootComponent
        occs = {o.component.name: o for o in root.occurrences}
        missing = [n for n in LOOKS if n not in occs]
        if missing:
            ui.messageBox("Run fusion_cutaway first. Missing parts: " + ", ".join(missing))
            return

        libs = [lib for lib in app.materialLibraries if "appearance" in lib.name.lower()]
        if not libs:
            ui.messageBox("No appearance library found.")
            return

        def find(words):
            for w in words:
                for lib in libs:
                    a = lib.appearances.itemByName(w)
                    if a:
                        return a
                for lib in libs:            # looser: name contains the words
                    for i in range(lib.appearances.count):
                        a = lib.appearances.item(i)
                        if w.lower() in a.name.lower():
                            return a
            return None

        report = []
        hpc_look = None
        for part, words in LOOKS.items():
            lib_look = find(words)
            if lib_look is None:
                report.append(f"{part}: no appearance found")
                continue
            name = "HPC health" if part == "HPC" else lib_look.name
            look = design.appearances.itemByName(name) or design.appearances.addByCopy(lib_look, name)
            occs[part].appearance = look
            report.append(f"{part}: {lib_look.name}")
            if part == "HPC":
                hpc_look = look

        colour_prop = None
        if hpc_look is not None:
            for i in range(hpc_look.appearanceProperties.count):
                p = hpc_look.appearanceProperties.item(i)
                if isinstance(p, adsk.core.ColorProperty) and not p.isReadOnly:
                    colour_prop = p
                    break
        if colour_prop is None:
            ui.messageBox("Appearances set, but the HPC colour can't be edited:\n" + "\n".join(report))
            return

        colours = read_colours()
        view = app.activeViewport
        saved = []
        for label, tag in SHOTS:
            r, g, b = colours[label]
            colour_prop.value = adsk.core.Color.create(r, g, b, 255)
            adsk.doEvents()
            view.refresh()
            path = os.path.join(FIGURES, f"15_cutaway_{tag}.png").replace("\\", "/")
            if view.saveAsImageFile(path, WIDTH, HEIGHT):
                saved.append(os.path.basename(path))
        r, g, b = colours["new"]
        colour_prop.value = adsk.core.Color.create(r, g, b, 255)

        ui.messageBox("Saved to reports/figures:\n  " + "\n  ".join(saved)
                      + "\n\nAppearances:\n" + "\n".join(report))
    except Exception:
        ui.messageBox("Failed:\n{}".format(traceback.format_exc()))
