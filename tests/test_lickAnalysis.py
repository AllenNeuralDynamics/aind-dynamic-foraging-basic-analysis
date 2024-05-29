""" Import all packages."""
import unittest
from aind_dynamic_foraging_basic_analysis.lick_analysis import (
    plot_lick_analysis,
    load_nwb,
    cal_metrics,
    plot_met,
    load_data,
)

import matplotlib.pyplot as plt
import os
from pathlib import Path


class testLickPlot(unittest.TestCase):
    """Test lickAnalysis module."""

    def test_loadnwb_happy_case(self):
        """Test loading of nwb."""
        data_dir = Path(os.path.dirname(__file__))
        nwbfile = os.path.join(data_dir, "data/689514_2024-02-01_18-06-43.nwb")
        nwb = load_nwb(nwbfile)
        fig, session_id = plot_lick_analysis(nwb)
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(session_id, str)

    def test_output_is_nwb(self):
        """Test the plotLickAnalysis."""
        data_dir = Path(os.path.dirname(__file__))
        nwbfile = os.path.join(data_dir, "data/689514_2024-02-01_18-06-43.nwb")
        nwb = load_nwb(nwbfile)
        self.assertIsNotNone(nwb)

    def test_lickMetrics(self):
        """Test lickMetrics."""
        data_dir = Path(os.path.dirname(__file__))
        nwbfile = os.path.join(data_dir, "data/689514_2024-02-01_18-06-43.nwb")
        nwb = load_nwb(nwbfile)
        data = load_data(nwb)
        lick_sum = cal_metrics(data)
        fig, session_id = plot_met(data, lick_sum)
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(session_id, str)
