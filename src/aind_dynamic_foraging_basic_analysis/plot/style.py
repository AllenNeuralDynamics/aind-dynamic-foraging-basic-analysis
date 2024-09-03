"""
    Defines a dictionary of styles
"""

# General plotting style
STYLE = {"axis_ticks_fontsize": 12, "axis_fontsize": 16}

# Colorscheme for photostim
PHOTOSTIM_EPOCH_MAPPING = {
    "after iti start": "cyan",
    "before go cue": "cyan",
    "after go cue": "green",
    "whole trial": "blue",
}

# Colorscheme for FIP channels
FIP_COLORS = {
    "G_1": "g",
    "G_1_preprocessed": "g",
    "G_2": "darkgreen",
    "G_2_preprocessed": "darkgreen",
    "R_1": "r",
    "R_1_preprocessed": "r",
    "R_2": "darkred",
    "R_2_preprocessed": "darkred",
    "Iso_1": "gray",
    "Iso_1_preprocessed": "gray",
    "Iso_2": "k",
    "Iso_2_preprocessed": "k",
    "goCue_start_time": "b",
    "left_lick_time": "m",
    "right_lick_time": "r",
    "left_reward_delivery_time": "b",
    "right_reward_delivery_time": "r",
}
