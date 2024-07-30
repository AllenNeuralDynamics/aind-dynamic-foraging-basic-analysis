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

    def test_lick_happy_case(self):
        """Test loading of nwb."""
        data_dir = Path(os.path.dirname(__file__))
        nwbfile = os.path.join(data_dir, "data/689514_2024-02-01_18-06-43.nwb")
        nwb = load_nwb(nwbfile)
        fig, session_id = plot_lick_analysis(nwb)
        fig.savefig(os.path.join(data_dir, "data/689514_2024-02-01_18-06-43qc.png"))
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(session_id, str)

    def test_lick_short_session(self):
        """Test loading of nwb."""
        data_dir = Path(os.path.dirname(__file__))
        nwbfile = os.path.join(data_dir, "data/705599_2024-05-31_14-06-54.nwb")
        nwb = load_nwb(nwbfile)
        fig, session_id = plot_lick_analysis(nwb)
        fig.savefig(os.path.join(data_dir, "data/705599_2024-05-31_14-06-54.png"))
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(session_id, str)

    def test_output_is_nwb_file(self):
        """Test the nwb file load."""
        data_dir = Path(os.path.dirname(__file__))
        nwbfile = os.path.join(data_dir, "data/689514_2024-02-01_18-06-43.nwb")
        nwb = load_nwb(nwbfile)
        self.assertIsNotNone(nwb)

    def test_output_is_nwb_folder(self):
        """Test the nwb file load."""
        data_dir = Path(os.path.dirname(__file__))
        nwbfile = os.path.join(data_dir, "data/717121_2024-06-15_10-58-01.nwb")
        nwb = load_nwb(nwbfile)
        self.assertIsNotNone(nwb)

    def test_output_is_none(self):
        """Test the nwb file load."""
        data_dir = Path(os.path.dirname(__file__))
        nwbfile = os.path.join(data_dir, "data/717121_2024-06-15_10-58-00.nwb")
        nwb = load_nwb(nwbfile)
        self.assertIsNone(nwb)

    def test_lickMetrics_long(self):
        """Test lickMetrics."""
        data_dir = Path(os.path.dirname(__file__))
        nwbfile = os.path.join(data_dir, "data/689514_2024-02-01_18-06-43.nwb")
        nwb = load_nwb(nwbfile)
        data = load_data(nwb)
        lick_sum = cal_metrics(data)
        fig, session_id = plot_met(data, lick_sum)
        fig.savefig(os.path.join(data_dir, "data/689514_2024-02-01_18-06-43qc.png"))
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(session_id, str)

    def test_lickMetrics_short(self):
        """Test lickMetrics."""
        data_dir = Path(os.path.dirname(__file__))
        nwbfile = os.path.join(data_dir, "data/705599_2024-05-31_14-06-54.nwb")
        nwb = load_nwb(nwbfile)
        data = load_data(nwb)
        lick_sum = cal_metrics(data)
        fig, session_id = plot_met(data, lick_sum)
        fig.savefig(os.path.join(data_dir, "data/705599_2024-05-31_14-06-54qc.png"))
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(session_id, str)
