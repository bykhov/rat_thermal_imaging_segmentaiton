# Segmentation Dataset: Rat Thermal Imaging 

## Preface
This dataset contains radiometric infrared thermal images of laboratory rats with
pixel-level anatomical segmentation masks. Each rat was imaged at multiple timepoints
across imaging sessions, yielding a collection that spans a wide range of surface-temperature
regimes.

The full dataset (~3.4 GB, 25 rats, 1,655 frames) includes segmented anatomical regions (head, body, tail) across 25 rat subjects.

The full corpus is hosted on [Figshare](https://doi.org/10.6084/m9.figshare.32309781).

---

Each rat folder contains matched triplets of files;
`Thermal_N.png` in `Thermal Imaging/` corresponds to `Thermal_N.png` in `Mask/`
and `Thermal_N_CSV.csv` in `CSV/`.

```
data_full/
  Rat1/
    CSV/                  raw temperature matrices (one per frame)
    Mask/                 semantic segmentation masks (one per frame)
    Thermal Imaging/      thermal images rendered with Jet colormap (one per frame)
  Rat2/
    ...
  Rat25/
    ...
```

---

## File Formats

The figures below are drawn from the bundled `data_sample/` (`Rat11`, frame 1).

### Thermal images (`Thermal Imaging/Thermal_N.png`)
- Format: PNG or JPEG, uint8 RGB, lossy, ~74 KB/frame
- Resolution: 480 x 640 px
- Visualization of the temperature matrix using the Jet colormap (blue = cold, red = hot)

![Jet-colormap thermal rendering](../data_sample/Rat11/Thermal%20Imaging/Thermal_1.png)

### Temperature matrices (`CSV/Thermal_N_CSV.csv`)
- `CSV/Thermal_<n>_CSV.csv` -- raw radiometric **temperature matrix**
- 480 rows x 640 columns of floating-point temperature values in degrees Celsius, ~2.1 MB/frame.
- Format: CSV with a 2-line header (file path on line 1, empty line 2)
- First data row is prefixed with `"Frame 1,"`, subsequent rows with `","` (label column to skip)
- Reading example:
  ```python
  df = pd.read_csv(path, skiprows=2, header=None)
  temp_matrix = df.iloc[:, 1:].values.astype(np.float32)  # shape (480, 640)
  ```

### Segmentation masks (`Mask/Thermal_N.png`)
- Format: PNG, read with `cv2.IMREAD_UNCHANGED` to preserve raw pixel values
- Pixel value encoding with 4-class labels:

  | Value | Class      |
  |-------|------------|
  | 0     | Background |
  | 1     | Head       |
  | 2     | Body       |
  | 3     | Tail       |

- Reading example:
  ```python
  stream = np.fromfile(str(path), np.uint8)
  mask = cv2.imdecode(stream, cv2.IMREAD_UNCHANGED)  # shape (480, 640), dtype uint8
  ```

![Colorized 4-class mask](../data_sample/Rat11/Ground%20Truth%20Mask%20(color)/Thermal_1.png)

*Colorized rendering of the single-channel label mask (Head = red, Body = green, Tail = blue);
the raw `Mask/` PNG stores discrete class values and is not directly human-viewable.*

If only need temperatures + masks, than ignore `Thermal Imaging/`

