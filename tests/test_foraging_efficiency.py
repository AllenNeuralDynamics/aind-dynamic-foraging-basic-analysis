"""Test foraging efficiency computation

To run the test, execute "python -m unittest tests/test_foraging_efficiency.py".
"""

import os
import unittest

import numpy as np

from aind_dynamic_foraging_basic_analysis import compute_foraging_efficiency
from tests.nwb_io import get_history_from_nwb


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
        },
    ]

    def test_example_sessions(self):
        """Test foraging efficiency on example sessions"""

        for correct_answer in self.correct_answers:
            nwb_file = os.path.dirname(__file__) + f"/data/{correct_answer['nwb_file']}"
            (
                baiting,
                choice_history,
                reward_history,
                p_reward,
                autowater_offered,
                random_number,
            ) = get_history_from_nwb(nwb_file)
            foraging_efficiency, foraging_efficiency_random_seed = compute_foraging_efficiency(
                baited=baiting,
                choice_history=choice_history,
                reward_history=reward_history,
                p_reward=p_reward,
                autowater_offered=autowater_offered,
                random_number=random_number,
            )
            self.assertAlmostEqual(foraging_efficiency, correct_answer["foraging_efficiency"])
            self.assertAlmostEqual(
                foraging_efficiency_random_seed,
                correct_answer["foraging_efficiency_random_seed"],
            )

        # Test returning np.nan if random_number and autowater_offered is None
        for baited in [True, False]:
            foraging_efficiency, foraging_efficiency_random_seed = compute_foraging_efficiency(
                baited=baited,
                choice_history=choice_history,
                reward_history=reward_history,
                autowater_offered=autowater_offered,
                p_reward=p_reward,
            )
            self.assertTrue(np.isnan(foraging_efficiency_random_seed))

    def test_wrong_format(self):
        """Test wrong input format"""
        choice_history = [0, 1, np.nan]
        reward_history = [0, 1, 1]
        p_reward = [[0.5, 0.5, 0.4], [0.5, 0.5, 0.4]]
        random_number = [[0.1, 0.2], [0.3, 0.4]]

        # Wrong random_number shape
        with self.assertRaises(ValueError):
            compute_foraging_efficiency(
                baited=True,
                choice_history=choice_history,
                reward_history=reward_history,
                p_reward=p_reward,
                random_number=random_number,
            )

        # Wrong reward_history shape
        with self.assertRaises(ValueError):
            compute_foraging_efficiency(
                baited=True,
                choice_history=choice_history,
                reward_history=reward_history[:2],
                p_reward=p_reward,
            )

        # Wrong p_reward shape
        with self.assertRaises(ValueError):
            compute_foraging_efficiency(
                baited=True,
                choice_history=choice_history,
                reward_history=reward_history,
                p_reward=p_reward[:1],
                random_number=None,
            )

        # Wrong choice_history values
        with self.assertRaises(ValueError):
            compute_foraging_efficiency(
                baited=True,
                choice_history=[0, 1, 2],
                reward_history=reward_history,
                p_reward=p_reward,
                random_number=None,
            )

        # Wrong autowater_offered shape
        with self.assertRaises(ValueError):
            compute_foraging_efficiency(
                baited=True,
                choice_history=choice_history,
                reward_history=reward_history,
                p_reward=p_reward,
                autowater_offered=np.array([True, False]),
                random_number=None,
            )


if __name__ == "__main__":
    unittest.main()
