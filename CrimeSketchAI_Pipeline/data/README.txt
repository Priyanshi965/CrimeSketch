CrimeSketch AI — Dataset Folder
═══════════════════════════════

OPTION A — Real paired dataset (CUHK / AR / PRIP-VSGC):
──────────────────────────────────────────────────────────
Place images in:
    data/sketches/    ← one sketch per subject (e.g. subject_001.jpg)
    data/photos/      ← one photo per subject  (e.g. subject_001.jpg)

Filenames must match. Any image extension is accepted: jpg, jpeg, png, bmp, webp.
The pipeline will automatically find matching subject IDs.

OPTION B — Synthetic mode (auto-generate sketches):
──────────────────────────────────────────────────────────
Place any face photos in:
    data/photos/      ← e.g. person1.jpg, person2.png, ...

Run with --synthetic flag:
    python run_pipeline.py --synthetic --n_subjects 500

The pipeline will apply a pencil-sketch simulation (CLAHE + divide-blend)
to each photo to create paired sketch-photo samples automatically.

RECOMMENDED FREE DATASETS:
──────────────────────────────────────────────────────────
• LFW (Labeled Faces in the Wild):
    http://vis-www.cs.umass.edu/lfw/
    Download: lfw-deepfunneled.tgz — use any faces as photos/

• CUHK Face Sketch Database:
    http://mmlab.ie.cuhk.edu.hk/datasets/facesketch.html

• AR Face Database:
    http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html

• PRIP-VSGC (Viewed Sketch + Composite):
    http://www.prip.tuwien.ac.at/

For the paper, the CUHK + AR + PRIP-VSGC + synthetic split was used
as described in Section IV-A.
