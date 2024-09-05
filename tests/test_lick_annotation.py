"""Test lick annotation

To run the test, execute "python -m unittest tests/test_lick_annotation.py".

"""

import unittest

import pandas as pd

import aind_dynamic_foraging_basic_analysis.licks.annotation as a


class EmptyNWB:
    """
    Just an empty class for saving attributes to
    """

    pass


class TestLickAnnotation(unittest.TestCase):
    """Test annotating licks"""

    def test_lick_annotation(self):
        """
        Test annotating licks
        """

        # Generate some simple data
        nwb = EmptyNWB()
        times = [1, 1.2, 1.4, 5, 5.2, 10]
        df = pd.DataFrame(
            {
                "timestamps": times,
                "data": [1.0] * len(times),
                "event": ["left_lick_time"] * len(times),
                "trial": [1] * len(times),
            }
        )

        # Ensure the annotations run
        nwb.df_events = df
        nwb.df_licks = a.annotate_lick_bouts(nwb)
        nwb.df_licks = a.annotate_rewards(nwb)


if __name__ == "__main__":
    unittest.main()
