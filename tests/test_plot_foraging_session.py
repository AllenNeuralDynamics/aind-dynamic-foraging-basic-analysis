"""Test plot foraging session"""


import unittest
import numpy as np
import os

from aind_dynamic_foraging_basic_analysis import plot_foraging_session
from tests.nwb_io import get_history_from_nwb


class TestPlotSession(unittest.TestCase):
    """Test plot session"""
    
    @classmethod
    def setUp(cls):
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
        # Plot session
        fig, _ = plot_foraging_session(
            choice_history=self.choice_history,
            reward_history=self.reward_history,
            p_reward=self.p_reward,
            autowater_offered=self.autowater_offered,
            fitted_data=None, 
            photostim=None,    # trial, power, s_type
            valid_range=None,
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
            autowater_offered=self.autowater_offered,
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
    