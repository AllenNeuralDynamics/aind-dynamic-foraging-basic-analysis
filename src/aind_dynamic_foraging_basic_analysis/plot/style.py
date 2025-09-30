"""
    Defines a dictionary of styles
"""

import matplotlib.pyplot as plt
import numpy as np

# General plotting style
STYLE = {
    "axis_ticks_fontsize": 12,
    "axis_fontsize": 16,
    "data_color_all": "blue",
    "data_alpha": 1,
    "axline_color": "k",
    "axline_linestyle": "-",
    "axline_alpha": 0.5,
}

# Colorscheme for photostim
PHOTOSTIM_EPOCH_MAPPING = {
    "after iti start": "cyan",
    "before go cue": "cyan",
    "after go cue": "green",
    "whole trial": "blue",
}

# Colorscheme for FIP channels
FIP_COLORS = {
    "G": "g",
    "R": "r",
    "Iso": "gray",
    "goCue_start_time": "b",
    "left_lick_time": "m",
    "right_lick_time": "r",
    "left_reward_delivery_time": "b",
    "right_reward_delivery_time": "r",
}


def get_colors(labels, cmap_name="hsv", offset=None):
    """
    Returns a dictionary of colors for each label.
    Colors are equally spaced from a matplotlib colormap

    Args:
        labels (list of strings): keys for dictionary of colors
        cmap_name (str): The name of matplotlib colormapt to use
            (e.g., 'viridis', 'plasma', 'coolwarm').
        offset (mixed): None, 'random', or float
            None, equivalent to 0
            'random', draw a random offset from 0 to 1
            used to get a random mix of colors

    Returns:
        dictionary: a list of label/RGB-alpha tuple key/value pairs
    """
    colors = get_n_colors(len(labels), cmap_name, offset)
    return {labels[i]: colors[i] for i in range(len(labels))}


def get_n_colors(n, cmap_name="hsv", offset=None):
    """
    Returns n equally spaced colors from a matplotlib colormap.

    Args:
        n (int): The number of colors to generate.
        cmap_name (str): The name of the matplotlib colormap to use
            (e.g., 'viridis', 'plasma', 'coolwarm').
        offset (mixed): None, 'random', or float
            None, equivalent to 0
            'random', draw a random offset from 0 to 1
            used to get a random mix of colors

    Returns:
        list: A list of RGB tuples representing the equally spaced colors.
    """
    cmap = plt.get_cmap(cmap_name)

    if cmap_name in ["twilight", "twilight_shifted", "hsv"]:
        # cyclical color maps, so we need one extra spacing point for unique colors
        n_spacing = n
    else:
        n_spacing = n - 1
    if n_spacing <= 0:
        n_spacing = 1

    # determine offset for mixing up colors
    if offset is None:
        offset = 0
    elif offset == "random":
        offset = np.random.rand()

    colors = [cmap(np.mod(i / n_spacing + offset, 1)) for i in range(n)]
    return colors
