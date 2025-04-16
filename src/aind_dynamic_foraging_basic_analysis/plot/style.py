"""
    Defines a dictionary of styles
"""

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
