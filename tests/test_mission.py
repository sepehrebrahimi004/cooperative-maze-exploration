"""End-to-end test: the full two-robot mission in the offline simulator.

Takes about a minute.
"""

import unittest

from sim.run_simulation import run


class TestMission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        *_, cls.metrics, _ = run(quiet=True, seed=0)

    def test_all_red_tiles_found(self):
        self.assertEqual(self.metrics["leader"]["red_tiles"], "3/3")
        self.assertEqual(self.metrics["follower"]["red_tiles"], "3/3")

    def test_no_collisions(self):
        self.assertEqual(self.metrics["leader"]["collisions"], 0)
        self.assertEqual(self.metrics["follower"]["collisions"], 0)

    def test_border_never_crossed(self):
        self.assertFalse(self.metrics["leader"]["crossed_border"])
        self.assertFalse(self.metrics["follower"]["crossed_border"])

    def test_mission_completes_and_leader_returns(self):
        self.assertIsNotNone(self.metrics["leader"]["finished_at_s"])
        self.assertIsNotNone(self.metrics["follower"]["finished_at_s"])

    def test_map_coverage(self):
        self.assertGreater(self.metrics["final_map_coverage"], 0.85)


if __name__ == "__main__":
    unittest.main()
