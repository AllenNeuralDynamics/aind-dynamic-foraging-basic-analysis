"""Test plot foraging session

To run the test, execute "python -m unittest tests/test_plot_foraging_session.py".

"""

import os
import unittest

import numpy as np
import pandas as pd

from aind_dynamic_foraging_basic_analysis import plot_foraging_session
import aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session as pfs
from tests.nwb_io import get_history_from_nwb


class EmptyNWB:
    """
    Just an empty class for saving attributes to
    """

    pass


class TestPlotSession(unittest.TestCase):
    """Test plot session"""

    @classmethod
    def setUpClass(cls):
        """Load example session"""
        nwb_file = os.path.dirname(__file__) + "/data/697929_2024-02-22_08-38-30.nwb"
        (
            _,
            cls.choice_history,
            cls.reward_history,
            cls.p_reward,
            cls.autowater_offered,
            _,
        ) = get_history_from_nwb(nwb_file)

    def test_nwb_wrapper(self):
        """
        Test wrapper function that plots foraging session from nwb file
        """
        # Test we have df_trials
        nwb = EmptyNWB()
        pfs.plot_foraging_session_nwb(nwb)

        # Test without bias column
        choices = np.array([0, 0, 1, 1, 2, 2])
        rewards = np.array([True, False, True, False, False, False])
        pL = [0.1] * 6
        pR = [0.8] * 6
        df = pd.DataFrame()
        df["animal_response"] = choices
        df["earned_reward"] = rewards
        df["reward_probabilityL"] = pL
        df["reward_probabilityR"] = pR
        df["auto_waterL"] = [0] * 6
        df["auto_waterR"] = [0] * 6
        nwb.df_trials = df
        nwb.session_id = "test"
        pfs.plot_foraging_session_nwb(nwb)

        # Test with bias column
        nwb.df_trials["bias"] = np.array([0, 0, 0.1, 0.1, 0.05, 0.05])
        nwb.df_trials["bias_ci_lower"] = np.array([0] * 6)
        nwb.df_trials["bias_ci_upper"] = np.array([0.2] * 6)
        pfs.plot_foraging_session_nwb(nwb)

    def test_plot_session(self):
        """Test plot real session"""
        # Add some fake data for testing
        fitted_data = np.ones(len(self.choice_history)) * 0.5
        valid_range = [0, 400]

        # Plot session
        fig, _ = plot_foraging_session(
            choice_history=list(self.choice_history),
            reward_history=list(self.reward_history),
            p_reward=self.p_reward,
            autowater_offered=self.autowater_offered,
            fitted_data=fitted_data,
            photostim={
                "trial": [10, 20, 30],
                "power": np.array([3.0, 3.0, 3.0]),
                "stim_epoch": ["before go cue", "after iti start", "after go cue"],
            },
            valid_range=valid_range,
            smooth_factor=5,
            base_color="y",
            ax=None,
            vertical=False,
        )

        # Plot session
        fig, _ = plot_foraging_session(
            choice_history=list(self.choice_history),
            reward_history=list(self.reward_history),
            p_reward=self.p_reward,
            autowater_offered=self.autowater_offered,
            fitted_data=fitted_data,
            photostim={
                "trial": [10, 20, 30],
                "power": np.array([3.0, 3.0, 3.0]),  # No stim_epoch
            },
            valid_range=valid_range,
            smooth_factor=5,
            base_color="y",
            ax=None,
            vertical=False,
        )

        # Save fig
        fig.savefig(
            os.path.dirname(__file__) + "/data/test_plot_session.png",
            bbox_inches="tight",
        )

    def test_plot_session_vertical(self):
        """Test plotting the same session vertically"""
        # Plot session
        fig, _ = plot_foraging_session(
            choice_history=self.choice_history,
            reward_history=self.reward_history,
            p_reward=self.p_reward,
            autowater_offered=None,
            fitted_data=None,
            photostim=None,  # trial, power, s_type
            valid_range=None,
            smooth_factor=5,
            base_color="y",
            ax=None,
            vertical=True,
        )

        # Save fig
        fig.savefig(
            os.path.dirname(__file__) + "/data/test_plot_session_vertical.png",
            bbox_inches="tight",
        )

        fig, _ = plot_foraging_session(
            choice_history=self.choice_history,
            reward_history=self.reward_history,
            p_reward=self.p_reward,
            autowater_offered=np.array([0] * len(self.choice_history)),
            fitted_data=None,
            photostim=None,  # trial, power, s_type
            valid_range=None,
            smooth_factor=5,
            base_color="y",
            ax=None,
            vertical=True,
        )

    def test_plot_session_wrong_format(self):
        """Some wrong input format"""
        with self.assertRaises(ValueError):
            plot_foraging_session(
                choice_history=[0, 1, np.nan],
                reward_history=[0, 1, 1],
                p_reward=[[0.5, 0.5, 0.4], [0.5, 0.5, 0.4]],
                fitted_data=[0, 1],  # Wrong length
            )

        with self.assertRaises(ValueError):
            plot_foraging_session(
                choice_history=[0, 1, np.nan],
                reward_history=[0, 1, 1],
                p_reward=[[0.5, 0.5, 0.4], [0.5, 0.5, 0.4]],
                photostim={"trial": [1, 2, 3], "power": [3.0, 3.0]},  # Wrong length
            )

        with self.assertRaises(ValueError):
            plot_foraging_session(
                choice_history=[0, 1, np.nan],
                reward_history=[0, 1, 1],
                p_reward=[[0.5, 0.5, 0.4], [0.5, 0.5, 0.4]],
                photostim={
                    "trial": [1, 2, 3],
                    "power": [3.0, 3.0, 5.0],
                    "stim_epoch": ["after iti start"],
                },  # Wrong length
            )


if __name__ == "__main__":
    unittest.main()
