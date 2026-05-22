import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import tifffile
from brain_vis.rgba import set_transparency
from typing import Union, Optional

def adjust_intensity(image: np.ndarray) -> np.ndarray:
    min_val, max_val = np.min(image), np.max(image)
    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    return (image - min_val) / denom

def convert_to_cmap(image: np.ndarray, cmin: float=-2, cmax: float=2, colormap=plt.cm.coolwarm) -> np.ndarray:
    """Map a 2D image to RGB in [0,1] using a matplotlib colormap."""
    norm = Normalize(vmin=cmin, vmax=cmax)
    return colormap(norm(image))

def convert_overlap_image(image: np.ndarray) -> np.ndarray:
    """Map overlap image to RGBA with Greys: 0→black (background), 255→white (overlay)."""
    norm_image = adjust_intensity(image)
    colormap = plt.cm.Greys
    return colormap(norm_image)

def overlay_images(base_color: np.ndarray, overlap_color: np.ndarray, alpha: float=0.5, colormap=None, overlap_black: bool = False) -> np.ndarray:
    base = base_color.copy()
    
    # Determine the value range of base_color (could be [0,1] or [0,255])
    max_val = np.max(base)
    value_range = 255 if max_val > 1.0 else 1.0
    
    # Background color where overlap is 0: 0 (black) or value_range (255/white) based on overlap_black
    if overlap_black:
        background_value = 0
    elif colormap is not None:
        # Get the color that the colormap returns for zero (normalized to 0.0)
        zero_color = colormap(0.0)
        is_white_at_zero = np.sum(zero_color[:3]) > 2.5 if len(zero_color) >= 3 else False
        background_value = 0 if is_white_at_zero else value_range
    else:
        background_value = value_range
    
    # Set background where overlap is zero
    # overlap_color is RGBA from Greys colormap, so minimum values are black (0,0,0,1)
    # We need to check where RGB channels are all 0 (or very close to 0)
    if overlap_color.ndim == 3 and overlap_color.shape[2] >= 3:
        # For RGBA arrays, check where RGB channels are all 0 (or very close to 0)
        # Use a small threshold to account for floating point precision
        threshold = 0.01 if max_val <= 1.0 else 2.55
        mask = np.all(overlap_color[:, :, :3] <= threshold, axis=2)
    else:
        # Fallback: check where overlap_color is 0 (or very close to 0)
        threshold = 0.01 if max_val <= 1.0 else 2.55
        mask = overlap_color <= threshold
    
    # Set background where overlap is zero
    if base.ndim == 3:
        if base.shape[2] >= 3:
            base[mask, :3] = background_value
        else:
            base[mask] = background_value
    else:
        base[mask] = background_value
    
    return base

def overlay_images_for_2_color(base_color1: np.ndarray, base_color2: np.ndarray, alpha: float=0.5) -> np.ndarray:
    return base_color1 * alpha + base_color2 * (1 - alpha)

def overlap_contour(
    base_image: np.ndarray,
    overlap_image: np.ndarray,
    cmin: float = -100,
    cmax: float = 100,
    outputpath: Union[str, bool] = False,
    colormap=plt.cm.coolwarm,
    base_image2: Optional[np.ndarray] = None,
    cmin2: float = -100,
    cmax2: float = 100,
    colormap2=plt.cm.viridis,
    overlap_black: bool = False,
    ):  
    """Create a heatmap volume with optional second base image and overlay a contour volume.

    Returns
    -------
    base_image_adj, overlayed_image : (ndarray, ndarray)
        Adjusted base image and final RGBA overlay volume.
    """
    base_image_adj = base_image
    overlap_image_adj = adjust_intensity(overlap_image)
    base_image_color = convert_to_cmap(base_image_adj, cmin, cmax, colormap=colormap)

    if base_image2 is not None:
        base_image2_adj = adjust_intensity(base_image2)
        base_image2_color = convert_to_cmap(base_image2_adj, cmin2, cmax2, colormap=colormap2)
        base_image_color = overlay_images_for_2_color(base_image_color, base_image2_color, alpha=0.5)

    overlap_image_color = convert_overlap_image(overlap_image_adj)
    overlayed_image = overlay_images(base_image_color, overlap_image_color, alpha=1.0, colormap=colormap, overlap_black=overlap_black)

    if outputpath:
        tifffile.imwrite(outputpath, overlayed_image)
    return base_image_adj, overlayed_image
