# brain_vis/regions.py
from typing import Optional, List
import numpy as np
import pandas as pd

# subset to target region
# make the slice move along the z position

def get_slice(z,target_site_subids,hemi = 'left',window = 60):
    ys = np.array([])
    xs = np.array([])
    if hemi == 'left':
        hemi_slice = slice(0,atlas_img.shape[2]//2)
    else:
        hemi_slice = slice(atlas_img.shape[2]//2,atlas_img.shape[2])
    for ID in target_site_subids:
        y_,x_ = np.where(atlas_img[z,:,hemi_slice] == ID)
        xs = np.concatenate([xs,x_])
        ys = np.concatenate([ys,y_])
    # find the center of mass of the OFC
    y_center = int(np.mean(ys))
    x_center = int(np.mean(xs))
    # set the slice to the center of mass of the target site
    if hemi == 'left':
        yslice = slice(y_center-window,y_center+window)
        xslice = slice(x_center-window,x_center+window)
    elif hemi == 'right':
        yslice = slice(y_center-window,y_center+window)
        xslice = slice(x_center-window + atlas_img.shape[2]//2,x_center+window + atlas_img.shape[2]//2)        
    elif hemi == 'center':
        yslice = slice(y_center-window,y_center+window)
        xslice = slice(atlas_img.shape[2]//2-window,window + atlas_img.shape[2]//2)

    return yslice,xslice

def get_subregions(
    dataframe: pd.DataFrame,
    region_id: int,
    collected: Optional[List[dict]] = None,
    return_original: bool = False,
    id_col: str = "id",
    parent_id_col: str = "parent_id",
) -> pd.DataFrame:
    """
    Recursively collect all descendants of a given region from a tree-structured dataframe.

    Args:
        dataframe: DataFrame containing tree structure defined by id and parent_id columns.
        region_id: ID of the region to get subregions for.
        collected: Internal list to store collected subregions during recursion. Defaults to None.
        return_original: If True, include the original region in the result. Defaults to False.
        id_col: Name of the column containing region IDs. Defaults to "id".
        parent_id_col: Name of the column containing parent region IDs. Defaults to "parent_id".

    Returns:
        DataFrame containing rows for all subregions (descendants) of the given region_id.
        If return_original=True, also includes the original region itself.
    """
    if collected is None:
        collected = []

    # Find direct children of the given region_id
    subregions = dataframe[dataframe[parent_id_col] == region_id].to_dict("records")
    collected.extend(subregions)

    # Recursively find subregions of each child
    for subregion in subregions:
        get_subregions(
            dataframe,
            subregion[id_col],
            collected,
            return_original=False,
            id_col=id_col,
            parent_id_col=parent_id_col,
        )

    # Include the original region if requested
    if return_original:
        original = dataframe[dataframe[id_col] == region_id].to_dict("records")
        collected.extend(original)

    return pd.DataFrame(collected)


def create_mask_for_region(
    atlas_id: int,
    atlasimg: np.ndarray,
    atlas_df: pd.DataFrame,
    include_subregions: bool = False,
    id_col: str = "id",
    parent_id_col: str = "parent_id",
) -> np.ndarray:
    """
    Create a boolean mask for a given region ID in an atlas image.

    Args:
        atlas_id: ID of the region to create a mask for.
        atlasimg: Image array (2D or 3D) storing brain region labels for each pixel.
        atlas_df: DataFrame containing tree structure of brain regions.
        include_subregions: If True, include all descendant regions in the mask. Defaults to False.
        id_col: Name of the column containing region IDs. Defaults to "id".
        parent_id_col: Name of the column containing parent region IDs. Defaults to "parent_id".

    Returns:
        Boolean mask array of the same shape as atlasimg, with True values for pixels
        belonging to the specified region (and its subregions if include_subregions=True).
    """
    mask = np.zeros(np.shape(atlasimg), dtype=bool)

    try:
        if include_subregions:
            subset_df = get_subregions(
                atlas_df, atlas_id, return_original=True, id_col=id_col, parent_id_col=parent_id_col
            )
            if len(subset_df) == 0:
                mask[np.where(atlasimg == atlas_id)] = True
            else:
                ids = subset_df[id_col].values
                mask[np.where(np.isin(atlasimg, np.append(ids, atlas_id)))] = True
        else:
            mask[np.where(atlasimg == atlas_id)] = True
    except Exception:
        print(f"Warning: Could not create mask for atlas_id {atlas_id} in the atlas dataframe")

    return mask
