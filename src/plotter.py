import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


def shiftedColorMap(cmap: mpl.colors.Colormap, start: float = 0, midpoint: float = 0.5, stop: float = 1.0, name: str = "shiftedcmap") -> mpl.colors.Colormap:
    """
    Function to offset the "center" of a colormap. Useful for data with a negative min and positive max and you want the middle of the colormap's dynamic range to be at zero.

    Args:
        cmap (mpl.colors.Colormap): The matplotlib colormap to be altered
        start (float, optional): Offset from lowest point in the colormap's range. Defaults to 0.0.
        midpoint (float, optional): The new center of the colormap. Defaults to 0.5.
        stop (float, optional): Offset from highest point in the colormap's range. Defaults to 1.0.
        name (str, optional): Name of the new colormap. Defaults to "shiftedcmap".

    Returns:
        mpl.colors.Colormap: The shifted colormap
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin: float, vmax: float, midpoint: float = 0, clip: bool = False):
        """
        Normalize a given value to the 0-1 range at a midpoint

        Args:
            vmin (float): Minimum data value
            vmax (float): Maximum data value
            midpoint (float, optional): Midpoint for normalization. Defaults to 0.
            clip (bool, optional): Whether to clip values outside the range. Defaults to False.
        """
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value: np.ndarray, clip: bool = None) -> np.ma.masked_array:
        """
        Normalize the value

        Args:
            value (np.ndarray): Data value to normalize
            clip (bool, optional): Whether to clip values outside the range. Defaults to None.

        Returns:
            np.ma.masked_array: Normalized data value
        """
        normalized_min = max(
            0,
            1
            / 2
            * (
                1
                - abs(
                    (self.midpoint - self.vmin) / (self.midpoint - self.vmax)
                )
            ),
        )
        normalized_max = min(
            1,
            1
            / 2
            * (
                1
                + abs(
                    (self.vmax - self.midpoint) / (self.midpoint - self.vmin)
                )
            ),
        )
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [
            normalized_min,
            normalized_mid,
            normalized_max,
        ]
        return np.ma.masked_array(np.interp(value, x, y))
