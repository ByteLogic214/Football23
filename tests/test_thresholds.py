import unittest
from engine.model_advanced import AdvancedPredictionModel

class TestThresholds(unittest.TestCase):
    def setUp(self):
        self.model = AdvancedPredictionModel()

    def test_goals_safe_lines_order(self):
        features = {
            'xg_home': 1.8,
            'xg_away': 1.5,
            'consistency_home_off': 0.7,
            'consistency_away_off': 0.6,
            'sample_size_home': 10,
            'sample_size_away': 10
        }
        preds = self.model.predict_market(features)
        goals = preds['goals']
        proj = goals['projection_total']
        over = goals['safe_over_line']
        under = goals['safe_under_line']
        self.assertTrue(over < proj < under, f"Safe lines not ordered: over={over}, proj={proj}, under={under}")

    def test_shots_safe_lines_order(self):
        features = {
            'shots_on_target_total': 5,
            'shots_on_target_home': 3,
            'shots_on_target_away': 2,
            'consistency_home_off': 0.6,
            'consistency_away_off': 0.6
        }
        preds = self.model.predict_market(features)
        shots = preds['shots_on_target']
        proj = shots['projection_total']
        over = shots['safe_over_line']
        under = shots['safe_under_line']
        self.assertTrue(over < proj < under, f"Safe lines not ordered: over={over}, proj={proj}, under={under}")

if __name__ == '__main__':
    unittest.main()
