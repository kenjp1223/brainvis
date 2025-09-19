import numpy as np

def set_transparency(rgba_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a transparency mask to an RGBA image.

    Parameters
    ----------
    rgba_img : np.ndarray (H, W, 4), dtype=uint8
        RGBA image. Alpha channel will be modified.
    mask : np.ndarray (H, W), dtype=bool
        True -> pixel becomes transparent (alpha=0).

    Returns
    -------
    np.ndarray
        New RGBA image with alpha channel updated.
    """
    if rgba_img.ndim != 3 or rgba_img.shape[-1] != 4:
        raise ValueError("Input image must be RGBA (H x W x 4).")
    if rgba_img.shape[:2] != mask.shape:
        raise ValueError("Mask shape must match image height and width.")

    result = rgba_img.copy()
    result[mask, 3] = 0  # alpha -> 0 where mask is True
    return result
