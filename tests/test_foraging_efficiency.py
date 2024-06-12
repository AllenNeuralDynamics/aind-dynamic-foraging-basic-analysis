import unittest
import os
import numpy as np

from pynwb import NWBHDF5IO
from aind_dynamic_foraging_basic_analysis import compute_foraging_efficiency


def get_history_from_nwb(nwb_file):
    """Get choice and reward history from nwb file"""

    io = NWBHDF5IO(nwb_file, mode="r")
    nwb = io.read()
    df_trial = nwb.trials.to_dataframe()

    # Exclude autowater
    df_trial["non_autowater_trial"] = False
    df_trial.loc[
        (df_trial.auto_waterL == 0) & (df_trial.auto_waterR == 0),
        "non_autowater_trial",
    ] = True
    non_autowater = df_trial.non_autowater_trial

    choice_history = df_trial.animal_response[non_autowater]
    choice_history[choice_history == 2] = np.nan  # Recoding
    reward_history = (
        df_trial.rewarded_historyL[non_autowater]
        | df_trial.rewarded_historyR[non_autowater]
    )
    reward_probability = [
        df_trial.reward_probabilityL[non_autowater].values,
        df_trial.reward_probabilityR[non_autowater].values,
    ]
    random_number = [
        df_trial.reward_random_number_left[non_autowater].values,
        df_trial.reward_random_number_right[non_autowater].values,
    ]

    return choice_history, reward_history, reward_probability, random_number


class TestForagingEfficiency(unittest.TestCase):
    """Test foraging efficiency"""

    correct_answers = [
        {
            "nwb_file": "697929_2024-02-22_08-38-30.nwb",  # coupled baiting example session
            "foraging_efficiency": 0.6944444444444443,
            "foraging_efficiency_random_seed": 0.7499999999999999,
        },
        {
            "nwb_file": "727456_2024-06-12_11-10-53.nwb",  # uncoupled no baiting example session
            "foraging_efficiency": 0.5896149584561394,
            "foraging_efficiency_random_seed": 0.638655462184874,
        },
    ]

    def test_example_sessions(self):
        """Test foraging efficiency on example sessions"""

        for correct_answer in self.correct_answers:
            nwb_file = (
                os.path.dirname(__file__)
                + f"/data/{correct_answer['nwb_file']}"
            )
            (
                choice_history,
                reward_history,
                reward_probability,
                random_number,
            ) = get_history_from_nwb(nwb_file)
            foraging_efficiency, foraging_efficiency_random_seed = (
                compute_foraging_efficiency(
                    choice_history,
                    reward_history,
                    reward_probability,
                    random_number,
                )
            )
            self.assertAlmostEqual(
                foraging_efficiency, correct_answer["foraging_efficiency"]
            )
            self.assertAlmostEqual(
                foraging_efficiency_random_seed,
                correct_answer["foraging_efficiency_random_seed"],
            )

    def test_wrong_format(self):
        """Test wrong input format"""
        choice_history = [0, 1, 2]
        reward_history = [0, 1, 1]
        reward_probability = [[0.5, 0.5, 0.4], [0.5, 0.5, 0.4]]
        random_number = [[0.1, 0.2], [0.3, 0.4]]

        with self.assertRaises(ValueError):
            compute_foraging_efficiency(
                choice_history,
                reward_history,
                reward_probability,
                random_number,
            )

        compute_foraging_efficiency(
            choice_history, reward_history, reward_probability
        )  # This should work


if __name__ == "__main__":
    unittest.main()
