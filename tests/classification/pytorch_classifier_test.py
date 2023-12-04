import unittest

from classification.pytorch_classifier import TorchClassifier
from tests.classification.run_classifier_test import get_test_files


class TestRunClassifier(unittest.TestCase):
    def test_pytorch_classifier(self):
        training_data, test_data, test_dir = get_test_files()
        class_label = "lang"
        text_labels = ["text"]

        pytorch_classifier = TorchClassifier()
        pytorch_classifier.train(training_data, text_labels, class_label)
        prediction = pytorch_classifier.classify(
            {"text": "Dies ist ein Brief von Friedrich an Wolfgang. Welche Sprache?"},
            "text",
        )
        self.assertTrue(prediction == "de")
        prediction = pytorch_classifier.classify(
            {"text": "Kære herr Ibsen. Det er et brev jeg nylig sendte til mig"}, "text"
        )
        self.assertTrue(prediction == "da")


if __name__ == "__main__":
    unittest.main()
