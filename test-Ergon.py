import unittest
import numpy as np

# Import functions from Ergon.py
from Ergon import calculate_angle, calculate_risk_score

class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class TestErgonFunctions(unittest.TestCase):
    def test_calculate_angle_straight_line(self):
        """
        Angle should be 180 degrees for a straight line.
        """
        a = [0, 0]
        b = [1, 0]
        c = [2, 0]
        angle = calculate_angle(a, b, c)
        self.assertAlmostEqual(angle, 180)

    def test_calculate_angle_right_angle(self):
        """
        Angle should be 90 degrees for a right angle.
        """
        a = [0, 0]
        b = [0, 1]
        c = [1, 1]
        angle = calculate_angle(a, b, c)
        self.assertAlmostEqual(angle, 90)

    def test_calculate_risk_score_low(self):
        """
        Should return 'Low Risk' with 0 score for ideal posture.
        """
        # Mock landmarks for ideal posture
        landmarks = [None] * 33  # MediaPipe has 33 pose landmarks
        # Shoulders, hips, knees, ankles: all aligned
        landmarks[11] = MockLandmark(0.5, 0.5)  # LEFT_SHOULDER
        landmarks[12] = MockLandmark(0.5, 0.5)  # RIGHT_SHOULDER
        landmarks[23] = MockLandmark(0.5, 0.7)  # LEFT_HIP
        landmarks[24] = MockLandmark(0.5, 0.7)  # RIGHT_HIP
        landmarks[25] = MockLandmark(0.5, 1.0)  # LEFT_KNEE
        landmarks[26] = MockLandmark(0.5, 1.0)  # RIGHT_KNEE
        landmarks[27] = MockLandmark(0.5, 1.3)  # LEFT_ANKLE
        landmarks[28] = MockLandmark(0.5, 1.3)  # RIGHT_ANKLE

        risk_level, risk_score = calculate_risk_score(landmarks)
        self.assertEqual(risk_level, "Low Risk")
        self.assertEqual(risk_score, 0)

    def test_calculate_risk_score_high(self):
        """
        Should return 'High Risk' for poor alignment and knee posture.
        """
        # Mock landmarks for poor posture
        landmarks = [None] * 33
        landmarks[11] = MockLandmark(0.5, 0.5)  # LEFT_SHOULDER
        landmarks[12] = MockLandmark(0.5, 0.8)  # RIGHT_SHOULDER (large y diff)
        landmarks[23] = MockLandmark(0.5, 0.7)  # LEFT_HIP
        landmarks[24] = MockLandmark(0.5, 1.0)  # RIGHT_HIP (large y diff)
        landmarks[25] = MockLandmark(0.5, 1.0)  # LEFT_KNEE
        landmarks[26] = MockLandmark(0.5, 1.2)  # RIGHT_KNEE (bad angle)
        landmarks[27] = MockLandmark(0.5, 1.3)  # LEFT_ANKLE
        landmarks[28] = MockLandmark(0.5, 1.5)  # RIGHT_ANKLE

        risk_level, risk_score = calculate_risk_score(landmarks)
        self.assertEqual(risk_level, "High Risk")
        self.assertTrue(risk_score >= 2)

if __name__ == "__main__":
    unittest.main()
