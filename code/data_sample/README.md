# `code/data_sample/`

Lightweight helpers that operate on the two bundled animals under
`<repo>/data_sample/`. Both scripts import the U-Net model and CSV
preprocessor from `<repo>/code/segmentation_unet/`, so there is
no duplicate model code.

## `smoke_test.py` -- end-to-end inference check

Loads `<repo>/models/unet_resnet34_fold2.pt`, runs inference on the
bundled frames, writes the colorized prediction and a blended overlay
back into `data_sample/<rat>/Predicted Mask (color)/` and
`data_sample/<rat>/Predicted Overlay/`, regenerates the ground-truth
`Ground Truth Mask (color)/` and `Ground Truth Overlay/` visualizations,
and reports per-class IoU against the ground-truth masks.

```
python code/data_sample/smoke_test.py            # 1 frame per bundled rat (default)
python code/data_sample/smoke_test.py --all      # every bundled frame
```

Headline 4-panel figure (input | GT overlay | predicted overlay |
prediction mask) is saved to `data_sample/smoke_test_output.png`.

Exit codes:
- `0` -- success;
- `2` -- checkpoint missing at `models/unet_resnet34_fold2.pt`
  (download from Figshare, see `models/README.md`);
- `1` / `3` -- no bundled rats / no frames processed.

## `make_overlay.py` -- regenerate sample visualizations

Reads `data_sample/<rat>/Thermal Imaging/` and `data_sample/<rat>/Mask/`,
writes:
- `data_sample/<rat>/Ground Truth Overlay/` -- thermal image blended with the ground-truth mask;
- `data_sample/<rat>/Ground Truth Mask (color)/` -- colorized mask on a black background.

Uses the same 4-class palette as the prediction outputs
(Head=red, Body=green, Tail=blue, Background transparent / black,
ALPHA=0.45 on the overlay).

```
python code/data_sample/make_overlay.py                       # all bundled rats
python code/data_sample/make_overlay.py path/to/RatXX [...]   # specific dirs
```
