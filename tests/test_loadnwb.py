"""Example test template."""
import unittest
from aind_dynamic_foraging_basic_analysis.lickAnalysis import (
    loadnwb,
)
import os
from pathlib import Path


class testNwb(unittest.TestCase):
    """Example Test Class"""

    def test_output_is_figure(self):
        """Example of how to test the truth of a statement."""
        data_dir = Path(os.path.dirname(__file__))
        nwbfile = os.path.join(
            data_dir, "data/689514_2024-02-01_18-06-43.nwb"
        )
        nwb = loadnwb(nwbfile)
        self.assertIsNotNone(nwb)


if __name__ == "__main__":
    unittest.main()
