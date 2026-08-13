"""The bundled first-run guided tutorial: a pre-seeded three-speaker sample run.

``voxint tutorial seed`` (and the test suite) build one genuine COMPLETED run from
the committed assets in :mod:`voxint.tutorial.resources`, exhibiting the three
adjudication states the tutorial teaches. The seeding logic lives in
:mod:`voxint.tutorial.seed`; nothing here imports the DB at module import time so
the package stays cheap to import.
"""
