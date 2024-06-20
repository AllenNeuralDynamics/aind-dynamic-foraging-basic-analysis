"""Test plot foraging session

To run the test, execute "python -m unittest tests/test_plot_foraging_session.py".

"""


import unittest
import numpy as np
import os

from aind_dynamic_foraging_basic_analysis import plot_foraging_session
from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import moving_average
from tests.nwb_io import get_history_from_nwb


class TestPlotSession(unittest.TestCase):
    """Test plot session"""
    
    @classmethod
    def setUpClass(cls):
        nwb_file = os.path.dirname(__file__) + f"/data/697929_2024-02-22_08-38-30.nwb"
        (
            _,
            cls.choice_history,
            cls.reward_history,
            cls.p_reward,
            cls.autowater_offered,
            _,
        ) = get_history_from_nwb(nwb_file)
    
    def test_plot_session(self):
        # Add some fake data for testing
        fitted_data = np.ones(len(self.choice_history)) * 0.5
        photostim = {
            'trial': [10, 20, 30], 
            'power': np.array([3.0, 3.0, 3.0]), 
            'stim_epoch': ['before go cue', 'after iti start', 'after go cue'],
        }
        valid_range = [0, 400]
        
        # Plot session
        fig, _ = plot_foraging_session(
            choice_history=self.choice_history,
            reward_history=self.reward_history,
            p_reward=self.p_reward,
            autowater_offered=self.autowater_offered,
            fitted_data=fitted_data, 
            photostim=photostim,    # trial, power, s_type
            valid_range=valid_range,
            smooth_factor=5, 
            base_color='y', 
            ax=None, 
            vertical=False,
        )
        
        # Save fig
        fig.savefig(
            os.path.dirname(__file__) + "/data/test_plot_session.png",
            bbox_inches='tight',
        )
    
    def test_plot_session_vertical(self):        
        # Plot session
        fig, _ = plot_foraging_session(
            choice_history=self.choice_history,
            reward_history=self.reward_history,
            p_reward=self.p_reward,
            autowater_offered=None,
            fitted_data=None, 
            photostim=None,    # trial, power, s_type
            valid_range=None,
            smooth_factor=5, 
            base_color='y', 
            ax=None, 
            vertical=True,
        )
        
        # Save fig
        fig.savefig(
            os.path.dirname(__file__) + "/data/test_plot_session_vertical.png",
            bbox_inches='tight',
        )