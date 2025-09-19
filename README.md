# brain-vis

Small utilities for mouse-brain visualization:

- **RGBA**: apply transparency masks to RGBA images.
- **Overlay**: colorize a 3D volume and overlay contours.
- **Sunburst app**: interactive Dash app that links a tree (e.g., atlas ontology) to brain slice images.

> This package consolidates three scripts into one installable package.

## Install (editable)

```bash
pip install -e /path/to/brain_vis
```

## Quick start

### RGBA transparency
```python
import numpy as np
from brain_vis.rgba import set_transparency

rgba = np.zeros((100,100,4), dtype=np.uint8); rgba[...,3]=255
mask = np.zeros((100,100), dtype=bool); mask[20:80, 20:80] = True
rgba2 = set_transparency(rgba, mask)
```

### Contour overlay
```python
import numpy as np
from brain_vis.overlay import overlap_contour
import matplotlib.pyplot as plt

vol = np.random.randn(10, 128, 128).astype(np.float32)  # data_by_img
contour = (np.random.rand(10, 128, 128) > 0.95).astype(np.float32)
base_adj, overlayed = overlap_contour(vol, contour, cmin=-2, cmax=2)

plt.imshow(overlayed[5])
plt.axis("off")
```

### Dash sunburst + image viewer
```python
import numpy as np, pandas as pd
from brain_vis.sunburst_app import run_app

# Fake data
Z, H, W = 20, 128, 128
data_by_img = np.random.randn(Z, H, W).astype(np.float32)
annotated_atlas_img = np.random.randint(0, 5, size=(Z,H,W), dtype=np.int32)

df = pd.DataFrame({
    "id":[0,1,2,3,4],
    "parent_id":[-1,0,0,1,1],
    "acronym":["root","A","B","A1","A2"],
    "parent_acronym":["", "root","root","A","A"],
    "name":["root","Region A","Region B","Region A1","Region A2"],
    "rejected":[False, True, False, True, False],
})

# Define a small helper to get subregion ids for a selected id:
def get_subregions_func(df_region, selected_id, return_original=True):
    # Include the selected_id and all descendants
    parents = {row.id: row.parent_id for row in df_region.itertuples()}
    children = {}
    for cid, pid in parents.items():
        children.setdefault(pid, []).append(cid)
    todo, out = [selected_id], set()
    while todo:
        x = todo.pop()
        if x in out: continue
        out.add(x)
        todo.extend(children.get(x, []))
    return df_region[df_region["id"].isin(out)]

app = run_app(
    data_by_region=df,
    data_by_img=data_by_img,
    annotated_atlas_img=annotated_atlas_img,
    outputpath="./out",
    get_subregions_func=get_subregions_func,
    debug=True,
)
# In a script, call: app.run_server(debug=True)
```

## Notes
- For `sunburst_app`, you must pass a `get_subregions_func(data_by_region, id, return_original=True)->DataFrame` so we don't depend on external modules.
- Images are shown with RGBA overlays; TIFF export uses `tifffile`.
