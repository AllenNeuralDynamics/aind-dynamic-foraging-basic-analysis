"""Example test template."""
import unittest
from aind_dynamic_foraging_basic_analysis.lickAnalysis import (
    plotLickAnalysis,
    loadnwb,
)
import matplotlib.pyplot as plt


class testLickPlot(unittest.TestCase):
    """Example Test Class"""

    def test_output_is_figure(self):
        """Example of how to test the truth of a statement."""
        nwb = loadnwb("tests\689514_2024-02-01_18-06-43.nwb")
        fig, sessionID = plotLickAnalysis(nwb)
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(sessionID, str)


if __name__ == "__main__":
    unittest.main()
