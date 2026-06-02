# Rat Thermal Imaging Dataset + Anatomical Segmentation

A radiometric thermal imaging dataset of laboratory rats with pixel-level anatomical
segmentation masks (Background, Head, Body, Tail), together with a U-Net + ResNet-34
segmentation pipeline used for technical validation.

- **Subjects:** 25 rats (`Rat1`–`Rat25`)
- **Frames:** 1,655 quality-controlled frames (~3.4 GB)
- **Per frame:** 
  - a raw `480 × 640` radiometric temperature matrix (CSV, °C), 
  - a single-channel 4-class `uint8` mask (PNG), and 
  - a Jet-colormap thermal rendering (PNG/JPEG)

## Repository layout

Below are the primary directories in this repository. Click the links to view the dedicated `README.md` for each section:

*   **[`code/`](code/README.md)**: Contains the U-Net + ResNet-34 segmentation pipeline, training/evaluation orchestrators, and diagnostic visualization scripts.
*   **[`data_sample/`](data_sample/README.md)**: Contains a two-rat subset (`Rat11` and `Rat18`) for offline smoke-testing.
*   **[`models/`](models/README.md)**: Contains the trained fold checkpoints (e.g., `unet_resnet34_fold2.pt`) and configuration notes.
*   **[`segmentation_unet_runs/`](segmentation_unet_runs/README.md)**: Stores evaluation outputs, including metrics, confusion matrices, and worst-performing frame visualizations.
*   **[`data_full/`](data_full/README.md)**: Reference documentation and loading conventions for the full 25-rat corpus hosted on Figshare.


## Data


Each `data_full/RatN/` directory contains parallel `CSV/`, `Mask/`, and `Thermal Imaging/`
folders paired by a common frame index `N`. Loading conventions are documented in
[`data_full/README.md`](data_full/README.md). A two-rat subset is bundled under
`data_sample/` so the pipeline can be exercised without downloading the full archive.

## Quickstart

```bash
# one-epoch end-to-end smoke test into a temp subdir (no real outputs touched)
python code/segmentation_unet.py --mode smoke

# full inference: evaluate all five folds on their held-out subjects, then aggregate
python code/segmentation_unet.py --mode eval-all

# (re)train the five folds
python code/segmentation_unet.py --mode train-all

# single fold / single image
python code/segmentation_unet.py --mode eval  --fold 1
python code/segmentation_unet.py --mode infer --fold 1 --csv path/to/Thermal_1_CSV.csv

# render the worst-N test images for error analysis - help analyze failure modes
python code/visualize_worst.py 20

# run with no arguments for an interactive menu
python code/segmentation_unet.py
```

