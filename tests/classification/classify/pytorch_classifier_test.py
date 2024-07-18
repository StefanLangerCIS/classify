import unittest

from classification.classify.pytorch_classifier import TorchClassifier
from classification.classify.text_classifier import get_data_records_from_file
from tests.classification.classify.run_classifier_test import get_test_files


class TestRunClassifier(unittest.TestCase):
    def test_pytorch_classifier(self):
        training_file, test_file, test_dir = get_test_files()
        class_label = "lang"
        text_labels = ["text"]
        pytorch_classifier = TorchClassifier()
        training_data = get_data_records_from_file(training_file, text_label=text_labels, class_label=class_label)
        pytorch_classifier.train(training_data)
        prediction = pytorch_classifier.classify(
            {"text": "Dies ist ein Brief von Friedrich an Wolfgang. Welche Sprache?"})
        self.assertTrue(prediction[0].class_name == "de")
        prediction = pytorch_classifier.classify(
            {"text": "Kære herr Ibsen. Det er et brev jeg nylig sendte til mig"})
        self.assertTrue(prediction[0].class_name == "da")


if __name__ == "__main__":
    unittest.main()
