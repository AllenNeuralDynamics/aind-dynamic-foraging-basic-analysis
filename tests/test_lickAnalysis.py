""" Import all packages."""
import unittest
from aind_dynamic_foraging_basic_analysis.lickAnalysis import (
    plotLickAnalysis,
    loadnwb,
    lickMetrics,
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
        nwb = loadnwb(nwbfile)
        fig, sessionID = plotLickAnalysis(nwb)
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(sessionID, str)

    def test_output_is_figure(self):
        """Test the plotLickAnalysis."""
        data_dir = Path(os.path.dirname(__file__))
        nwbfile = os.path.join(data_dir, "data/689514_2024-02-01_18-06-43.nwb")
        nwb = loadnwb(nwbfile)
        self.assertIsNotNone(nwb)

    def test_lickMetrics(self):
        """Test lickMetrics."""
        data_dir = Path(os.path.dirname(__file__))
        nwbfile = os.path.join(data_dir, "data/689514_2024-02-01_18-06-43.nwb")
        nwb = loadnwb(nwbfile)
        lickSum = lickMetrics(nwb)
        lickSum.calMetrics()
        fig, sessionID = lickSum.plot()
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(sessionID, str)
