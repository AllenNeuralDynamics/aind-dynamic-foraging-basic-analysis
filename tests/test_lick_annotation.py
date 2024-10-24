"""Test lick annotation

To run the test, execute "python -m unittest tests/test_lick_annotation.py".

"""

import unittest
import numpy as np
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
        times = [1, 1.2, 1.4, 5, 5.2, 10, 20, 20.2, 20.4, 50, 50.1, 50.2]
        expected_bout_start = [
            True,
            False,
            False,
            True,
            False,
            True,
            True,
            False,
            False,
            True,
            False,
            False,
        ]
        expected_bout_end = [
            False,
            False,
            True,
            False,
            True,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
        ]
        expected_bout_number = [1, 1, 1, 2, 2, 3, 4, 4, 4, 5, 5, 5]
        expected_rewarded = [
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            True,
            False,
        ]
        expected_bout_rewarded = [
            True,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
        ]
        expected_cue_response = [
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
        expected_bout_cue_response = [
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
        expected_intertrial_choice = [
            False,
            False,
            False,
            True,
            False,
            True,
            True,
            False,
            False,
            True,
            False,
            False,
        ]
        expected_bout_intertrial_choice = [
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ]
        df = pd.DataFrame(
            {
                "timestamps": times + [1.1, 20.1, 30, 40, 50.15, 0.9],
                "data": [1.0] * (len(times) + 6),
                "event": ["left_lick_time"] * 6
                + ["right_lick_time"] * 3
                + ["right_lick_time", "left_lick_time", "left_lick_time"]
                + ["left_reward_delivery_time", "right_reward_delivery_time"]
                + ["left_reward_delivery_time", "right_reward_delivery_time"]
                + ["left_reward_delivery_time"]
                + ["goCue_start_time"],
                "trial": [1] * 6 + [2] * 3 + [5, 5, 5, 1, 2, 3, 4, 5, 1],
            }
        )
        df = df.sort_values(by="timestamps")

        # Verify that we check for the existence of df_events
        assert a.annotate_lick_bouts(nwb) is None
        assert a.annotate_rewards(nwb) is None
        assert a.annotate_cue_response(nwb) is None
        assert a.annotate_intertrial_choices(nwb) is None
        assert a.annotate_switches(nwb) is None
        assert a.annotate_within_session(nwb) is None

        # Verify that we check for the prerequisite columns from other annotations
        nwb.df_events = df
        nwb.df_licks = a.annotate_lick_bouts(nwb)
        del nwb.df_licks
        nwb.df_licks = a.annotate_rewards(nwb)
        del nwb.df_licks
        nwb.df_licks = a.annotate_cue_response(nwb)
        del nwb.df_licks
        nwb.df_licks = a.annotate_licks(nwb)
        del nwb.df_licks
        nwb.df_licks = a.annotate_intertrial_choices(nwb)
        del nwb.df_licks
        nwb.df_licks = a.annotate_lick_bouts(nwb)
        nwb.df_licks = a.annotate_intertrial_choices(nwb)
        del nwb.df_licks
        nwb.df_licks = a.annotate_switches(nwb)
        del nwb.df_licks
        nwb.df_licks = a.annotate_lick_bouts(nwb)
        nwb.df_licks = a.annotate_switches(nwb)
        del nwb.df_licks
        nwb.df_licks = a.annotate_lick_bouts(nwb)
        nwb.df_licks = a.annotate_cue_response(nwb)
        nwb.df_licks = a.annotate_switches(nwb)
        del nwb.df_licks
        nwb.df_licks = a.annotate_within_session(nwb)

        # Test annotations are correct from annotate_lick_bouts
        del nwb.df_licks
        nwb.df_licks = a.annotate_licks(nwb)
        assert np.all(nwb.df_licks["bout_start"].values == expected_bout_start)
        assert np.all(nwb.df_licks["bout_end"].values == expected_bout_end)
        assert np.all(nwb.df_licks["bout_number"].values == expected_bout_number)

        # Test annotations are correct from annotate_rewards
        # Checks for rewards being triggered by start and middle of bout
        assert np.all(nwb.df_licks["rewarded"].values == expected_rewarded)
        assert np.all(nwb.df_licks["bout_rewarded"].values == expected_bout_rewarded)

        # Checking this attributes as well, but isolated unit tests below
        assert np.all(nwb.df_licks["cue_response"] == expected_cue_response)
        assert np.all(nwb.df_licks["bout_cue_response"] == expected_bout_cue_response)
        assert np.all(nwb.df_licks["intertrial_choice"] == expected_intertrial_choice)
        assert np.all(nwb.df_licks["bout_intertrial_choice"] == expected_bout_intertrial_choice)

    def test_cue_response_annotation(self):
        """
        Test annotation of cue response, simple case of a go cue followed by a response
        """

        # Generate some simple data
        nwb = EmptyNWB()
        times = [1, 1.1, 1.2]
        expected_cue_response = [True, False, False]
        expected_bout_cue_response = [True, True, True]
        expected_intertrial_choice = [False, False, False]
        expected_bout_intertrial_choice = [False, False, False]
        cue_times = [0.95]
        df = pd.DataFrame(
            {
                "timestamps": times + cue_times,
                "data": [1.0] * (len(times) + len(cue_times)),
                "event": ["left_lick_time", "left_lick_time", "left_lick_time", "goCue_start_time"],
                "trial": [1] * (len(times) + len(cue_times)),
            }
        )
        df = df.sort_values(by="timestamps")

        # Verify that we check for the prerequisite columns from other annotations
        nwb.df_events = df
        nwb.df_licks = a.annotate_licks(nwb)
        assert np.all(nwb.df_licks["cue_response"] == expected_cue_response)
        assert np.all(nwb.df_licks["bout_cue_response"] == expected_bout_cue_response)
        assert np.all(nwb.df_licks["intertrial_choice"] == expected_intertrial_choice)
        assert np.all(nwb.df_licks["bout_intertrial_choice"] == expected_bout_intertrial_choice)

    def test_cue_response_annotation_1(self):
        """
        Test annotation of cue response, simple case of a go cue in the middle of a licking bout
        """

        # Generate some simple data
        nwb = EmptyNWB()
        times = [1, 1.1, 1.2]
        expected_cue_response = [False, False, False]
        expected_bout_cue_response = [False, False, False]
        expected_intertrial_choice = [True, False, False]
        expected_bout_intertrial_choice = [True, True, True]
        cue_times = [1.05]
        df = pd.DataFrame(
            {
                "timestamps": times + cue_times,
                "data": [1.0] * (len(times) + len(cue_times)),
                "event": ["left_lick_time", "left_lick_time", "left_lick_time", "goCue_start_time"],
                "trial": [1] * (len(times) + len(cue_times)),
            }
        )
        df = df.sort_values(by="timestamps")

        # Verify that we check for the prerequisite columns from other annotations
        nwb.df_events = df
        nwb.df_licks = a.annotate_licks(nwb)
        assert np.all(nwb.df_licks["cue_response"] == expected_cue_response)
        assert np.all(nwb.df_licks["bout_cue_response"] == expected_bout_cue_response)
        assert np.all(nwb.df_licks["intertrial_choice"] == expected_intertrial_choice)
        assert np.all(nwb.df_licks["bout_intertrial_choice"] == expected_bout_intertrial_choice)

    def test_cue_response_annotation_2(self):
        """
        Test annotation of cue response, simple case of a licking bout far after a go cue
        """

        # Generate some simple data
        nwb = EmptyNWB()
        times = [1, 3.1, 3.2, 3.3]
        expected_cue_response = [False, False, False, False]
        expected_bout_cue_response = [False, False, False, False]
        expected_intertrial_choice = [True, True, False, False]
        expected_bout_intertrial_choice = [True, True, True, True]
        cue_times = [2.0]
        df = pd.DataFrame(
            {
                "timestamps": times + cue_times,
                "data": [1.0] * (len(times) + len(cue_times)),
                "event": [
                    "left_lick_time",
                    "left_lick_time",
                    "left_lick_time",
                    "left_lick_time",
                    "goCue_start_time",
                ],
                "trial": [1] * (len(times) + len(cue_times)),
            }
        )
        df = df.sort_values(by="timestamps")

        # Verify that we check for the prerequisite columns from other annotations
        nwb.df_events = df
        nwb.df_licks = a.annotate_licks(nwb)
        assert np.all(nwb.df_licks["cue_response"] == expected_cue_response)
        assert np.all(nwb.df_licks["bout_cue_response"] == expected_bout_cue_response)
        assert np.all(nwb.df_licks["intertrial_choice"] == expected_intertrial_choice)
        assert np.all(nwb.df_licks["bout_intertrial_choice"] == expected_bout_intertrial_choice)

    def test_switch_annotation(self):
        """
        Test annotation of switches, two cue responses
        """

        # Generate some simple data
        nwb = EmptyNWB()
        times = [1.1, 3.1, 5.1, 7.1]
        expected_cue_switch = [False, False, True, False]
        expected_iti_switch = [False, False, False, True]
        events = ["left_lick_time", "left_lick_time", "right_lick_time", "left_lick_time"]
        cue_times = [1.0, 3.0, 5.0]
        events += ["goCue_start_time"] * len(cue_times)
        df = pd.DataFrame(
            {
                "timestamps": times + cue_times,
                "data": [1.0] * (len(times) + len(cue_times)),
                "event": events,
                "trial": [1] * (len(times) + len(cue_times)),
            }
        )
        df = df.sort_values(by="timestamps")

        # Verify that we check for the prerequisite columns from other annotations
        nwb.df_events = df
        nwb.df_licks = a.annotate_licks(nwb)
        assert np.all(nwb.df_licks["cue_switch"] == expected_cue_switch)
        assert np.all(nwb.df_licks["iti_switch"] == expected_iti_switch)

    def test_within_session_annotation(self):
        """
        Test annotation of within session labels
        """
        nwb = EmptyNWB()
        times = [0.1, 1.1, 1.2, 10]
        expected_within_session = [False, True, True, False]
        events = ["left_lick_time"] * len(times)
        cue_times = [1]
        events += ["goCue_start_time"] * len(cue_times)
        df = pd.DataFrame(
            {
                "timestamps": times + cue_times,
                "data": [1.0] * (len(times) + len(cue_times)),
                "event": events,
                "trial": [1] * (len(times) + len(cue_times)),
            }
        )
        df = df.sort_values(by="timestamps")

        # Verify that we check for the prerequisite columns from other annotations
        nwb.df_events = df
        nwb.df_licks = a.annotate_licks(nwb)
        assert np.all(nwb.df_licks["within_session"] == expected_within_session)

    def test_within_session_annotation_1(self):
        """
        Test annotation of within session labels with no go cues
        """
        nwb = EmptyNWB()
        times = [0.1, 1.1, 1.2, 10]
        expected_within_session = [False, False, False, False]
        events = ["left_lick_time"] * len(times)
        cue_times = []
        df = pd.DataFrame(
            {
                "timestamps": times + cue_times,
                "data": [1.0] * (len(times) + len(cue_times)),
                "event": events,
                "trial": [1] * (len(times) + len(cue_times)),
            }
        )
        df = df.sort_values(by="timestamps")

        # Verify that we check for the prerequisite columns from other annotations
        nwb.df_events = df
        nwb.df_licks = a.annotate_licks(nwb)
        assert np.all(nwb.df_licks["within_session"] == expected_within_session)


if __name__ == "__main__":
    unittest.main()
