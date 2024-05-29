"""Example of how to test the truth of a statement."""
import unittest
from aind_dynamic_foraging_basic_analysis.lickAnalysis import (
    plotLickAnalysis,
    loadnwb,
)
import matplotlib.pyplot as plt
import os
from pathlib import Path


class testLickPlot(unittest.TestCase):
    """Example of how to test the truth of a statement."""
    def test_loadnwb_happy_case(self):
        """Example of how to test the truth of a statement."""
        data_dir = Path(os.path.dirname(__file__))
        nwbfile = os.path.join(
            data_dir, "data/689514_2024-02-01_18-06-43.nwb"
        )
        nwb = loadnwb(nwbfile)
        fig, sessionID = plotLickAnalysis(nwb)
        """Example of how to test the truth of a statement."""
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(sessionID, str)

    def test_output_is_figure(self):
        """Example of how to test the truth of a statement."""
        data_dir = Path(os.path.dirname(__file__))
        nwbfile = os.path.join(
            data_dir, "data/689514_2024-02-01_18-06-43.nwb"
        )
        nwb = loadnwb(nwbfile)
        self.assertIsNotNone(nwb)
