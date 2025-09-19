import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import tifffile
from brain_vis.rgba import set_transparency

def adjust_intensity(image: np.ndarray) -> np.ndarray:
    min_val, max_val = np.min(image), np.max(image)
    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    return (image - min_val) / denom

def convert_to_cmap(image: np.ndarray, cmin: float=-2, cmax: float=2, colormap=plt.cm.coolwarm) -> np.ndarray:
    """Map a 2D image to RGB in [0,1] using a matplotlib colormap."""
    norm = Normalize(vmin=cmin, vmax=cmax)
    return colormap(norm(image))

def convert_overlap_image(image: np.ndarray) -> np.ndarray:
    norm_image = adjust_intensity(image)
    colormap = plt.cm.Greys
    return colormap(norm_image)

def overlay_images(base_color: np.ndarray, overlap_color: np.ndarray, alpha: float=0.5) -> np.ndarray:
    base = base_color.copy()
    base[overlap_color == 0] = 255
    return base

def overlay_images_for_2_color(base_color1: np.ndarray, base_color2: np.ndarray, alpha: float=0.5) -> np.ndarray:
    return base_color1 * alpha + base_color2 * (1 - alpha)

def overlap_contour(base_image: np.ndarray,
                    overlap_image: np.ndarray,
                    cmin: float=-100, cmax: float=100,
                    outputpath: str|bool=False,
                    colormap=plt.cm.coolwarm,
                    base_image2: np.ndarray|None=None,
                    cmin2: float=-100, cmax2: float=100,
                    colormap2=plt.cm.viridis):
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
    overlayed_image = overlay_images(base_image_color, overlap_image_color, alpha=1.0)

    if outputpath:
        tifffile.imwrite(outputpath, overlayed_image)
    return base_image_adj, overlayed_image
