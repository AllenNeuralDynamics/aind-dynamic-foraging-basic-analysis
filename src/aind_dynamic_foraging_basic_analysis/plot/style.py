"""
    Defines a dictionary of styles
"""
import matplotlib.pyplot as plt

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

def get_colors(labels,method='random'):
    if method == 'even':
        colors = get_n_colors(len(labels))
    elif method == 'random':
        colors = get_n_random_colors(len(labels))
    return {labels[i]:colors[i] for i in range(len(labels))}

def get_n_random_colors(n,cmap_name='hsv'):
    cmap = plt.get_cmap(cmap_name)
    offset = np.random.rand()
    colors = [cmap(np.mod(i / (n)+offset,1)) for i in range(n)]
    return colors 

def get_n_colors(n, cmap_name='plasma'):
    """
    Returns n equally spaced colors from a matplotlib colormap.

    Args:
        n (int): The number of colors to generate.
        cmap_name (str): The name of the matplotlib colormap to use (e.g., 'viridis', 'plasma', 'coolwarm').

    Returns:
        list: A list of RGB tuples representing the equally spaced colors.
    """
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i / (n - 1)) for i in range(n)]
    return colors 



