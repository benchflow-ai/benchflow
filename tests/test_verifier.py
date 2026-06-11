import unittest
from benchflow.verifier import Verifier

class TestVerifier(unittest.TestCase):
    def test_default_reward_range(self):
        task_config = {}
        verifier = Verifier(task_config)
        self.assertEqual(verifier.reward_range, [0.0, 1.0])

    def test_custom_reward_range(self):
        task_config = {'reward_range': [-1.0, 1.0]}
        verifier = Verifier(task_config)
        self.assertEqual(verifier.reward_range, [-1.0, 1.0])

    def test_validate_reward(self):
        task_config = {'reward_range': [-1.0, 1.0]}
        verifier = Verifier(task_config)
        self.assertTrue(verifier.validate_reward(-1.0))
        self.assertTrue(verifier.validate_reward(0.0))
        self.assertTrue(verifier.validate_reward(1.0))
        self.assertFalse(verifier.validate_reward(-2.0))
        self.assertFalse(verifier.validate_reward(2.0))