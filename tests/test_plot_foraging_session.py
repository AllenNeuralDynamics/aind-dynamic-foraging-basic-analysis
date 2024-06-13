"""Test plot foraging session"""


import unittest
import numpy as np
import os

from aind_dynamic_foraging_basic_analysis import plot_foraging_session
from nwb_io import get_history_from_nwb


class TestPlotSession(unittest.TestCase):
    """Test plot session"""
    
    def test_plot_session(self):
        
        nwb_file = os.path.dirname(__file__) + f"/data/697929_2024-02-22_08-38-30.nwb"
        (
            _,
            choice_history,
            reward_history,
            p_reward,
            autowater_offered,
            _,
        ) = get_history_from_nwb(nwb_file)
        
        # Plot session
        fig, axes = plot_foraging_session(
            choice_history=choice_history,
            reward_history=reward_history,
            p_reward=p_reward,
            autowater_offered=autowater_offered,
            fitted_data=None, 
            photostim=None,    # trial, power, s_type
            valid_range=None,
            smooth_factor=5, 
            base_color='y', 
            ax=None, 
            vertical=False
        )
        
        # Save fig
        fig.savefig(
            os.path.dirname(__file__) + "/data/test_plot_session.png",
            bbox_inches='tight',
        )
    
    
if __name__ == "__main__":
    # Run the tests in the current file only
    suite = unittest.defaultTestLoader.loadTestsFromName(__name__)
    unittest.TextTestRunner().run(suite)