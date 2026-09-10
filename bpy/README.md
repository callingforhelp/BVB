# Historical Blender Python exports

This directory contains 526 scripts exported by the earlier Blender-to-Python
workflow. They are retained as historical artifacts and are not maintained in
lockstep with refinements in `blend/`.

The current benchmark accepts an animated native `.blend`, renders its camera,
and evaluates Dual VQA and Latent Similarity directly. No stage reads these
exports or requires a `.blend` → `.py` → `.blend` round trip.

Use the native files in [`blend/`](../blend/) as the reference scene artifacts.
The optional exporter remains under [`addons/`](../addons/README.md) for users
who need source-code exports; it is not an evaluation prerequisite.
