# brain_vis/regions.py
from typing import Optional, List
import numpy as np
import pandas as pd

def get_subregions(
    dataframe: pd.DataFrame,
    region_id: int,
    collected: Optional[List[dict]] = None,
    return_original: bool = False,
) -> pd.DataFrame:
    """
    Recursively collect all descendants of `region_id` from a flat atlas dataframe.

    Expected columns in `dataframe`: "id", "parent_id".
    Returns a DataFrame containing rows for subregions (and, if return_original=True, also the original region).
    """
    if collected is None:
        collected = []

    # direct children of region_id
    subregions = dataframe[dataframe["parent_id"] == region_id].to_dict("records")
    collected.extend(subregions)

    # recurse
    for sub in subregions:
        get_subregions(dataframe, sub["id"], collected, return_original=False)

    if return_original:
        original = dataframe[dataframe["id"] == region_id].to_dict("records")
        collected.extend(original)

    return pd.DataFrame(collected)


def create_mask_for_region(
    atlas_id: int,
    atlasimg: np.ndarray,
    atlas_df: pd.DataFrame,
    include_subregions: bool = False,
) -> np.ndarray:
    """
    Make a boolean mask for `atlas_id` in `atlasimg` (Z,H,W or H,W). If include_subregions=True,
    include all descendants of `atlas_id` as defined in `atlas_df` (columns: id, parent_id).
    """
    tmpmask = np.zeros(np.shape(atlasimg), dtype=bool)
    try:
        if include_subregions:
            subset_df = get_subregions(atlas_df, atlas_id, return_original=True)
            ids = subset_df["id"].values if len(subset_df) else np.array([atlas_id])
            tmpmask[np.where(np.isin(atlasimg, ids))] = True
        else:
            tmpmask[np.where(atlasimg == atlas_id)] = True
    except Exception:
        print("No atlas_id in the atlas dataframe")
    return tmpmask
