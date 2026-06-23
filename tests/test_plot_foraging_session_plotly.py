"""Test the plotly foraging-session plots.

To run the test, execute "python -m unittest tests/test_plot_foraging_session_plotly.py".
"""

import os
import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from aind_dynamic_foraging_basic_analysis import (
    plot_foraging_session_plotly,
    plot_session_in_time_plotly,
)
from tests.nwb_io import get_history_from_nwb


class TestPlotForagingSessionPlotly(unittest.TestCase):
    """Test the trial-based plotly plot against a real session."""

    @classmethod
    def setUpClass(cls):
        """Load example session history from the bundled NWB."""
        nwb_file = os.path.dirname(__file__) + "/data/697929_2024-02-22_08-38-30.nwb"
        (
            _,
            cls.choice_history,
            cls.reward_history,
            cls.p_reward,
            cls.autowater_offered,
            _,
        ) = get_history_from_nwb(nwb_file)

    def test_returns_figure(self):
        """A plotly Figure is returned with both panels populated."""
        fig = plot_foraging_session_plotly(
            choice_history=self.choice_history,
            reward_history=self.reward_history,
            p_reward=self.p_reward,
            autowater_offered=self.autowater_offered,
        )
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)

    def test_optional_traces(self):
        """Bias band and photostim markers are accepted without error."""
        n = len(self.choice_history)
        fig = plot_foraging_session_plotly(
            choice_history=self.choice_history,
            reward_history=self.reward_history,
            p_reward=self.p_reward,
            bias=np.zeros(n),
            bias_lower=-np.ones(n) * 0.2,
            bias_upper=np.ones(n) * 0.2,
            photostim={"trial": [10, 20], "power": [3.0, 3.0]},
            plot_list=["choice", "finished", "reward_prob", "bias"],
        )
        self.assertIsInstance(fig, go.Figure)

    def test_multi_session(self):
        """A per-trial session_id concatenates sessions with a boundary line."""
        n = len(self.choice_history)
        session_id = np.array(["a"] * n + ["b"] * n)
        fig = plot_foraging_session_plotly(
            np.concatenate([self.choice_history, self.choice_history]),
            np.concatenate([self.reward_history, self.reward_history]),
            np.concatenate([self.p_reward, self.p_reward], axis=1),
            session_id=session_id,
        )
        # One boundary, drawn as a vertical line (shape) in each of the two rows.
        self.assertEqual(len(fig.layout.shapes), 2)


class TestPlotSessionInTimePlotly(unittest.TestCase):
    """Test the time-based plotly plot with a synthetic events / trials frame."""

    def setUp(self):
        """Build a small tidy events frame and matching trials frame."""
        go_cues = np.arange(5, 25, 2.0)  # 10 trials
        events = []
        for t in go_cues:
            events.append((t, "goCue_start_time"))
            events.append((t + 0.3, "left_lick_time"))
            events.append((t + 0.4, "right_lick_time"))
            events.append((t + 0.5, "left_reward_delivery_time"))
        self.df_events = pd.DataFrame(events, columns=["timestamps", "event"]).sort_values(
            "timestamps"
        )
        self.df_trials = pd.DataFrame(
            {
                "goCue_start_time": go_cues,
                "reward_probabilityL": np.linspace(0.1, 0.8, len(go_cues)),
                "reward_probabilityR": np.linspace(0.8, 0.1, len(go_cues)),
            }
        )

    def test_events_only(self):
        """Works with just an events frame (no probability band)."""
        fig = plot_session_in_time_plotly(self.df_events)
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)

    def test_with_trials(self):
        """Supplying df_trials adds the reward-probability band traces."""
        fig = plot_session_in_time_plotly(
            self.df_events, df_trials=self.df_trials, title="unit_test"
        )
        names = [tr.name for tr in fig.data]
        self.assertIn("pR", names)
        self.assertIn("pL", names)

    def test_multi_session(self):
        """A session_id column concatenates sessions end-to-end with a boundary line."""
        e1 = self.df_events.assign(session_id="s1")
        e2 = self.df_events.assign(session_id="s2", timestamps=self.df_events["timestamps"] + 100)
        t1 = self.df_trials.assign(session_id="s1")
        t2 = self.df_trials.assign(session_id="s2",
                                   goCue_start_time=self.df_trials["goCue_start_time"] + 100)
        fig = plot_session_in_time_plotly(
            pd.concat([e1, e2], ignore_index=True),
            df_trials=pd.concat([t1, t2], ignore_index=True),
        )
        self.assertEqual(len(fig.layout.shapes), 2)  # one boundary, drawn in both panels


if __name__ == "__main__":
    unittest.main()
