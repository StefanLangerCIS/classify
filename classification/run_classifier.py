"""
Evaluate any of the classifiers, print a confusion matrix and create further evalution metrics
"""
import argparse
import json
import os
import time
import warnings
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import sklearn.exceptions
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from classification.sklearn_classifiers import SklearnClassifier

# Data directory. Follow structure with default or
# set the CLS_SRCH_DATA_DIR environment directory
_this_dir = os.path.abspath(os.path.dirname(__file__))
_default_data_dir = os.path.abspath(os.path.join(_this_dir, "../../../", "data"))
DATA_DIR = os.getenv("CLS_SRCH_DATA_DIR", _default_data_dir)


def plot_and_store_confusion_matrix(
    y_true: list,
    y_pred: list,
    file_name: str,
    normalize=True,
    cmap=plt.cm.Blues,
    show=False,
) -> Dict:
    """
    This function prints and plots the confusion matrix, and saves it to a file
    :param y_true: The true classes
    :param y_pred: The predicted classes
    :param file_name: The file name to store the image of the confusion matrix
    :param normalize: normalize numbers (counts to relative counts)
    :param cmap: Layout
    :param show: Display the matrix. If false, only store it
    :return: Nothing
    """
    np.set_printoptions(precision=2)
    if normalize:
        title = "Normalized confusion matrix"
    else:
        title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=[20, 27])
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.savefig(file_name)
    if show:
        plt.show()


def count_lines(file: str) -> int:
    # Determine the number of training lines (for the record)
    n_lines = 0
    if file is not None:
        with open(file, encoding="utf-8") as f:
            for _ in f:
                n_lines += 1
    return n_lines


def run_classifier(
    classifier_type: str,
    training_data: str,
    test_data: str,
    class_label: str,
    text_labels: List[str],
    dense: bool,
    lsa: bool,
    output: str,
    max_train: int = 0,
    verbose: bool = False,
) -> Dict:
    """
    Run classifier of classifier_type

    :param classifier_type: The classifier algorithm
    :param training_data: jsonl training data
    :param test_data: jsonl data to classify
    :param class_label: The class label in the training data
    :param text_labels: The keys in the json which point to the text
    :param dense: Use dense vectors (embeddings)
    :param lsa: Use lsa vector compression
    :param output: Output folder for classification report
    :param max_train: Maximum number of lines to use for training
    :param verbose: Verbose output
    :return: The classification report as dictionary
    """
    classifier = SklearnClassifier(classifier_type, dense=dense, lsa=lsa)
    print(f"INFO: Evaluating classification of classifier {classifier.name()}")
    training_time = 0
    n_training_lines = count_lines(training_data)
    if training_data is not None:
        print(f"INFO: Reading training data from {training_data}")
        training_time = time.time()
        print(
            f"INFO: Training classifier {classifier_type} with text fields {text_labels} for label {class_label}"
        )
        classifier.train(training_data, text_labels, class_label, max_train)
        training_time = int(time.time() - training_time)
        print(f"INFO: Training completed in {training_time} seconds")
    else:
        print("INFO: Using pre-trained classifier")

    classifier.verbose = verbose

    print(
        f"INFO: Starting classification of data in {test_data} with classifier {classifier.name()}"
    )
    predicted_classes = []
    expected_classes = []
    # Keep track of time used
    classification_time = time.time()
    with open(test_data, encoding="utf-8") as infile:
        for line in infile:
            json_data = json.loads(line)
            res = classifier.classify(json_data, text_labels)
            class_name = "none"
            if len(res) > 0:
                class_name = res[0].class_name
            predicted_classes.append(class_name)
            expected_classes.append(json_data[class_label])

    classification_time = int(time.time() - classification_time)
    print(
        f"INFO: Classification completed for classifier {classifier.name()} in {classification_time} s"
    )

    outfile_identifier = f"results_{classifier.name()}_{class_label}"
    outfile_name = os.path.join(output, f"{outfile_identifier}.txt")
    print(f"INFO: Writing results of classifier {classifier.name()} to {outfile_name}")
    with open(outfile_name, "w", encoding="utf-8") as outfile:
        outfile.write("#Info:\n")
        outfile.write(f"Classifier: {classifier.name()}\n")
        outfile.write(f"Parameters: {json.dumps(classifier.info())}\n")
        outfile.write(f"Label: {class_label}\n")
        outfile.write(f"Text labels: {text_labels}\n")
        outfile.write(f"Dense|LSA: {dense}|{lsa}\n")
        outfile.write("\n#Counts:\n")
        outfile.write(f"Number of training data_records: {n_training_lines}\n")
        outfile.write(f"Number of classified data_records: {len(expected_classes)}\n")
        outfile.write(
            f"Number of unique classes in data_records: {len(set(expected_classes))}\n"
        )
        outfile.write(
            f"Number of unique classes found: {len(set(predicted_classes))}\n"
        )
        outfile.write("\n#Performance:\n")
        outfile.write(f"Seconds used for training: {training_time}\n")
        outfile.write(f"Seconds used for classification: {classification_time}\n")

        warnings.filterwarnings(
            "ignore", category=sklearn.exceptions.UndefinedMetricWarning
        )
        classification_report = sklearn.metrics.classification_report(
            expected_classes, predicted_classes, digits=3
        )
        outfile.write(f"\n#Classification report:\n{classification_report}\n")

        # Print the entire confusion matrix, not truncated
        np.set_printoptions(threshold=np.inf, linewidth=200)
        outfile.write(
            f"\n#Confusion matrix:\n{sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)}\n"
        )
    classification_report_dict = sklearn.metrics.classification_report(
        expected_classes, predicted_classes, output_dict=True
    )
    outfile_json = os.path.join(output, f"{outfile_identifier}.json")
    with open(outfile_json, "w", encoding="utf-8") as f:
        json.dump(classification_report_dict, f, indent=2, ensure_ascii=False)
    # Also store confusion matrix as image
    imagefile_name = os.path.join(output, f"{outfile_identifier}.jpg")
    plot_and_store_confusion_matrix(expected_classes, predicted_classes, imagefile_name)
    print(f"INFO: Processing completed for classifier {classifier.name()}")
    return classification_report_dict


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one or several text classifiers"
    )

    # All available classifier types
    classifier_types = SklearnClassifier.supported_classifiers

    parser.add_argument(
        "--training",
        default=os.path.join(
            DATA_DIR, "letters/classification/classifier_data_train.jsonl"
        ),
        help='The training data for the classifier. If "None", the existing model is loaded (if it exists)',
    )

    parser.add_argument(
        "--input",
        default=os.path.join(
            DATA_DIR, "letters/classification/classifier_data_test.jsonl"
        ),
        help="The text data to use for evaluation (one json per line)",
    )

    parser.add_argument(
        "--output",
        default=os.path.join(DATA_DIR, "letters/classification/results"),
        help="Folder where to write the classifier evaluation results",
    )

    parser.add_argument(
        "--classifier",
        choices=classifier_types + ["all"],
        default="MLPClassifier",
        help="The classifier to use. If 'all' iterate through all available classifiers",
    )

    parser.add_argument(
        "--dense",
        action="store_true",
        help="Use transformer dense vector for classification",
    )

    parser.add_argument(
        "--lsa",
        action="store_true",
        help="Use latent semantic analysis dense vector for classification",
    )

    parser.add_argument(
        "--text_label",
        default="text",
        help="Label/field in the json data which contains the text to classify."
        "Can also be multiple labels separated by comma (,)",
    )

    parser.add_argument(
        "--label",
        default="author",
        help="Label/field to use for training and classification",
    )

    parser.add_argument(
        "--max_train",
        type=int,
        default=0,
        help="Maximum number of data points to use for training",
    )

    parser.add_argument("--verbose", action="store_true", help="Provide verbose output")

    args = parser.parse_args()

    # Run all classifiers
    if args.classifier == "all":
        classifiers = classifier_types
    else:
        classifiers = [args.classifier]

    print(f"INFO: Evaluating classifier(s) {classifiers}")

    # Iterate over the classifiers
    text_labels = args.text_label.split(",")
    for classifier_type in classifiers:
        run_classifier(
            classifier_type,
            args.training,
            args.input,
            args.label,
            text_labels,
            args.dense,
            args.lsa,
            args.output,
            args.max_train,
            args.verbose,
        )


if __name__ == "__main__":
    main()
