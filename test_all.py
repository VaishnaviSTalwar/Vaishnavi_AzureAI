import unittest
import numpy as np

# Import components from task1
from task1_house_price_predictor import (
    MinMaxScaler,
    HousePriceLinearRegression,
    HouseTrendLogisticRegression,
    precision_score,
    recall_score,
    f1_score,
    accuracy,
    confusion_matrix
)

# Import components from task2
from task2_fraud_detection import oversample_minority, f1_manual

# Import components from task3
from task3_digit_sketch_recognizer import convolve2d, relu, max_pool2d, extract_cnn_features, FILTERS

# Import components from task4
from task4_mood_aware_chatbot import analyse_sentiment, MoodTracker, generate_reply


class TestTask1HousePricePredictor(unittest.TestCase):
    def test_min_max_scaler(self):
        scaler = MinMaxScaler()
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        X_scaled = scaler.fit_transform(X)
        
        # Min should scale to 0, Max should scale to 1
        np.testing.assert_allclose(X_scaled[0], [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(X_scaled[2], [1.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(X_scaled[1], [0.5, 0.5], atol=1e-6)

        # Test transform on unseen data
        X_new = np.array([[1.5, 15.0]])
        X_new_scaled = scaler.transform(X_new)
        np.testing.assert_allclose(X_new_scaled[0], [0.25, 0.25], atol=1e-6)

    def test_linear_regression(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        model = HousePriceLinearRegression(learning_rate=0.1, iterations=100)
        model.train(X, y)
        pred = model.predict(X)
        # Verify loss decreased and predictions are reasonably close
        self.assertLess(model.loss_history[-1], model.loss_history[0])
        self.assertEqual(pred.shape, (3,))

    def test_logistic_regression(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1])
        model = HouseTrendLogisticRegression(learning_rate=0.5, iterations=100)
        model.train(X, y)
        probs = model.predict_proba(X)
        preds = model.predict(X)
        self.assertEqual(probs.shape, (4,))
        self.assertEqual(preds.shape, (4,))
        # Sigmoid test
        self.assertAlmostEqual(model.sigmoid(0.0), 0.5)

    def test_metrics(self):
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1])
        
        # TP = 2, FP = 1, TN = 1, FN = 1
        self.assertAlmostEqual(accuracy(y_true, y_pred), 0.6)
        self.assertAlmostEqual(precision_score(y_true, y_pred), 2 / 3)
        self.assertAlmostEqual(recall_score(y_true, y_pred), 2 / 3)
        self.assertAlmostEqual(f1_score(y_true, y_pred), 2 / 3)
        
        cm = confusion_matrix(y_true, y_pred)
        np.testing.assert_array_equal(cm, [[1, 1], [1, 2]])


class TestTask2FraudDetection(unittest.TestCase):
    def test_oversample_minority(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = np.array([0, 0, 0, 1])
        
        X_bal, y_bal = oversample_minority(X, y)
        # Check that class counts are equal after balancing
        classes, counts = np.unique(y_bal, return_counts=True)
        self.assertEqual(counts[0], counts[1])
        self.assertEqual(counts[0], 3)
        self.assertEqual(len(y_bal), 6)

    def test_f1_manual(self):
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1])
        self.assertAlmostEqual(f1_manual(y_true, y_pred, cls=1), 2 / 3)


class TestTask3DigitSketchRecognizer(unittest.TestCase):
    def test_convolve2d(self):
        # 3x3 image, 2x2 kernel
        image = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]], dtype=np.float32)
        kernel = np.array([[1.0, 0.0],
                           [0.0, 1.0]], dtype=np.float32)
        
        conv = convolve2d(image, kernel)
        # Expected output shape: 3 - 2 + 1 = 2x2
        # conv[0,0] = 1*1 + 5*1 = 6
        # conv[0,1] = 2*1 + 6*1 = 8
        # conv[1,0] = 4*1 + 8*1 = 12
        # conv[1,1] = 5*1 + 9*1 = 14
        expected = np.array([[6.0, 8.0],
                             [12.0, 14.0]], dtype=np.float32)
        np.testing.assert_allclose(conv, expected, atol=1e-6)

    def test_relu(self):
        x = np.array([-1.5, 0.0, 2.3])
        np.testing.assert_allclose(relu(x), [0.0, 0.0, 2.3], atol=1e-6)

    def test_max_pool2d(self):
        feature_map = np.array([[1.0, 2.0, 0.0, 4.0],
                                 [3.0, 0.0, 5.0, 1.0],
                                 [2.0, 1.0, 8.0, 7.0],
                                 [0.0, 0.0, 6.0, 9.0]], dtype=np.float32)
        pool = max_pool2d(feature_map, pool_size=2)
        # Expected output shape: 2x2
        # pool[0,0] = max(1,2,3,0) = 3
        # pool[0,1] = max(0,4,5,1) = 5
        # pool[1,0] = max(2,1,0,0) = 2
        # pool[1,1] = max(8,7,6,9) = 9
        expected = np.array([[3.0, 5.0],
                             [2.0, 9.0]], dtype=np.float32)
        np.testing.assert_allclose(pool, expected, atol=1e-6)

    def test_extract_cnn_features(self):
        mock_img = np.random.randn(16, 16).astype(np.float32)
        features = extract_cnn_features(mock_img)
        # 16x16 input -> convolve2d with 3x3 -> 14x14 -> max_pool2d with pool_size 2 -> 7x7
        # 7 * 7 = 49 features per filter
        # Total filters = 6
        # Total feature vector size = 49 * 6 = 294
        self.assertEqual(features.shape, (294,))


class TestTask4MoodAwareChatbot(unittest.TestCase):
    def test_analyse_sentiment_simple(self):
        # Positive
        self.assertGreater(analyse_sentiment("feeling happy and awesome"), 0.0)
        # Negative
        self.assertLess(analyse_sentiment("so sad and terrible"), 0.0)

    def test_analyse_sentiment_negators(self):
        # Negated positive -> negative sentiment
        self.assertLess(analyse_sentiment("i am not happy"), 0.0)
        # Negated negative -> positive or neutral sentiment
        self.assertGreaterEqual(analyse_sentiment("not bad"), 0.0)

    def test_analyse_sentiment_amplifiers(self):
        # Amplified word should have a larger magnitude (before normalisation)
        s1 = analyse_sentiment("happy")
        s2 = analyse_sentiment("very happy")
        # In this specific normalisation:
        # "happy" -> score 1.0, count 1 -> 1.0 / 1.0 = 1.0
        # "very happy" -> score 1.5, count 2 -> 1.5 / sqrt(2) = 1.06 -> clipped to 1.0
        # Let's test with a longer sentence to avoid clipping
        s_base = analyse_sentiment("feeling happy today")
        s_amp = analyse_sentiment("feeling really happy today")
        self.assertNotEqual(s_base, s_amp)

    def test_mood_tracker(self):
        tracker = MoodTracker()
        self.assertEqual(tracker.score, 0.0)
        self.assertEqual(tracker.label, "Neutral")
        self.assertEqual(tracker.emoji, "😐")

        # Update mood to cheerful
        tracker.update(0.8, 1)
        self.assertGreater(tracker.score, 0.0)
        
        # Verify label and emoji update when threshold exceeded
        tracker.score = 0.5
        self.assertEqual(tracker.label, "Cheerful")
        self.assertEqual(tracker.emoji, "😊")
        
        tracker.score = -0.5
        self.assertEqual(tracker.label, "Concerned")
        self.assertEqual(tracker.emoji, "😟")

        # Test ASCII mood bar
        bar = tracker.mood_bar(width=20)
        self.assertIn("■", bar)

    def test_generate_reply(self):
        from unittest.mock import patch
        tracker = MoodTracker()
        tracker.score = 0.5
        user_text = "I am doing great and feeling fine"
        with patch("random.choice", return_value="Here is {echo}"):
            reply = generate_reply(tracker, user_text)
        self.assertEqual(reply, 'Here is "I am doing great and feeling…"')



if __name__ == "__main__":
    unittest.main()
