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
    "G_1": "g",
    "G_1_dff-bright": "g",
    "G_1_dff-exp": "g",
    "G_1_dff-poly": "g",
    "G_2": "darkgreen",
    "G_2_dff-bright": "darkgreen",
    "G_2_dff-exp": "darkgreen",
    "G_2_dff-poly": "darkgreen",
    "R_1": "r",
    "R_1_dff-bright": "r",
    "R_1_dff-exp": "r",
    "R_1_dff-poly": "r",
    "R_2": "darkred",
    "R_2_dff-bright": "darkred",
    "R_2_dff-exp": "darkred",
    "R_2_dff-poly": "darkred",
    "Iso_1": "gray",
    "Iso_1_dff-bright": "gray",
    "Iso_1_dff-exp": "gray",
    "Iso_1_dff-poly": "gray",
    "Iso_2": "k",
    "Iso_2_dff-bright": "k",
    "Iso_2_dff-exp": "k",
    "Iso_2_dff-poly": "k",
    "goCue_start_time": "b",
    "left_lick_time": "m",
    "right_lick_time": "r",
    "left_reward_delivery_time": "b",
    "right_reward_delivery_time": "r",
}
