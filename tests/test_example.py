"""Example test template."""

import unittest
import matplotlib.pyplot as plt
from src.aind_dynamic_foraging_basic_analysis.lickAnalysis import (
    plotLickAnalysis,
)
from src.aind_dynamic_foraging_basic_analysis.lickAnalysis import loadnwb


class testLickPlot(unittest.TestCase):
    """Example Test Class"""

    def test_output_is_figure(self):
        """Example of how to test the truth of a statement."""
        nwb = loadnwb("tests\689514_2024-02-01_18-06-43.nwb")
        fig, sessionID = plotLickAnalysis(nwb)
        self.assertIsInstance(fig, plt.Figure)


if __name__ == "__main__":
    unittest.main()
