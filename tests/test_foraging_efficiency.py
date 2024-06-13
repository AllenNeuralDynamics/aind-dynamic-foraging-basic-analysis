"""Test foraging efficiency computation"""

import os
import unittest

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
    
    baiting = False if 'without baiting' in nwb.protocol.lower() else True

    return choice_history, reward_history, reward_probability, random_number, baiting


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
            "foraging_efficiency": 0.6946983546617915,
            "foraging_efficiency_random_seed": 0.7378640776699029,
        },
        {
            "nwb_file": "703548_2024-03-01_08-51-32.nwb",  # well trained uncoupled baiting
            "foraging_efficiency": 0.8048681814442915,
            "foraging_efficiency_random_seed": 0.8390092879256966,
        }
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
                baiting
            ) = get_history_from_nwb(nwb_file)
            foraging_efficiency, foraging_efficiency_random_seed = (
                compute_foraging_efficiency(
                    choice_history,
                    reward_history,
                    reward_probability,
                    random_number,
                    baited=baiting,
                )
            )
            self.assertAlmostEqual(
                foraging_efficiency, correct_answer["foraging_efficiency"]
            )
            self.assertAlmostEqual(
                foraging_efficiency_random_seed,
                correct_answer["foraging_efficiency_random_seed"],
            )
            
        # Test returning np.nan if random_number is None
        for baited in [True, False]:
            foraging_efficiency, foraging_efficiency_random_seed = compute_foraging_efficiency(
                choice_history,
                reward_history,
                reward_probability,
                baited=baited,
            )
            self.assertTrue(np.isnan(foraging_efficiency_random_seed))
        

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
        with self.assertRaises(ValueError):
            compute_foraging_efficiency(
                choice_history,
                reward_history[:2],
                reward_probability,
                random_number,
            )
        with self.assertRaises(ValueError):
            compute_foraging_efficiency(
                choice_history,
                reward_history,
                reward_probability[1],
            )
            

if __name__ == "__main__":
    unittest.main()
