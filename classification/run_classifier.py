"""
Evaluate any of the classifiers, print a confusion matrix and create further evalution metrics
"""
import argparse
import json
import os
from sklearn_classifiers import SklearnClassifier
import sklearn.metrics
import sklearn.exceptions
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import time
import numpy as np
import matplotlib.pyplot as plt

import warnings

# Data directory. Follow structure with default or
# set the CLS_SRCH_DATA_DIR environment directory
_this_dir = os.path.abspath(os.path.dirname(__file__))
_default_data_dir = os.path.abspath(os.path.join(_this_dir, "../../../", "data"))
DATA_DIR = os.getenv("CLS_SRCH_DATA_DIR", _default_data_dir)


def plot_and_store_confusion_matrix(y_true: list,
                                    y_pred:list,
                                    file_name: str,
                                    normalize=True,
                                    cmap=plt.cm.Blues,
                                    show=False):
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
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=[20,27])
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(file_name)
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate one or several text classifiers')

    # All available classifier types
    classifier_types = SklearnClassifier.supported_classifiers

    parser.add_argument('--training',
                    default=os.path.join(DATA_DIR, "letters/classifier/classifier_data_train.json"),
                    help='The training data for the classifier. If "None", the existing model is loaded (if it exists)')
    
    parser.add_argument('--input',
                    default=os.path.join(DATA_DIR, "letters/classifier/classifier_data_eval.json"),
                    help='The text data to use for evaluation (one json per line)')

    parser.add_argument('--output',
                        default=os.path.join(DATA_DIR, "results/classification-results"),
                        help='Folder where to write the classifier evaluation results')

    parser.add_argument('--classifier',
                    choices=classifier_types + ["all"],
                    default="MLPClassifier",
                    help="The classifier to use. If 'all' iterate through all available classifiers" )

    parser.add_argument('--dense',
                        action='store_true',
                        help='Use transformer dense vector for classification')

    parser.add_argument('--lsa',
                        action='store_true',
                        help='Use latent semantic analysis dense vector for classification')

    parser.add_argument('--text_label',
                        default="text",
                        help='Label/field in the json data contains the text to classify')

    parser.add_argument('--label',
                        default="author",
                        help='Label/field to use for training and classification')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Provide verbose output')

    args = parser.parse_args()

    # Run all classifiers
    if args.classifier == "all":
        classifiers = classifier_types
    else:
        classifiers = [args.classifier]

    print("INFO: Evaluating classifier(s) {0}".format(classifiers))

    # Determine the number of training lines (for the record)
    n_training_lines = 0
    if args.training is not None:
        with open(args.training, encoding="utf-8") as training:
            for line in training:
                n_training_lines += 1

    # Iterate over the classifiers
    for classifier_type in classifiers:
        classifier = SklearnClassifier(classifier_type, dense=args.dense, lsa=args.lsa)
        print("INFO: Evaluating classification of classifier {0}".format(classifier.name()))
        training_time = 0
        if args.training is not None:
            training_time = time.time()
            print("INFO: Training classifier")
            classifier.train(args.training, args.text_label, args.label)
            training_time = int(time.time() - training_time)
            print("INFO: Training completed in {} seconds".format(training_time))

        else:
            print("INFO: Using pre-trained classifier")

        classifier.verbose = args.verbose

        print("INFO: Starting classification of data in {0} with classifier {1}".format(args.input, classifier.name()))
        predicted_classes = []
        expected_classes = []
        # Keep track of time used
        classification_time = time.time()
        with open(args.input, encoding="utf-8") as infile:
            for line in infile:
                json_data = json.loads(line)
                res = classifier.classify(json_data, args.text_label)
                class_name = "none"
                if len(res) > 0:
                    class_name = res[0].class_name
                predicted_classes.append(class_name)
                expected_classes.append(json_data[args.label])

        classification_time = int(time.time()-classification_time)
        print("INFO: Classification completed for classifier {0} in {1} s".format(classifier.name(), classification_time))
        print("INFO: Writing results of classifier {0}".format(classifier.name()))

        outfile_name = os.path.join(args.output, "results_{0}.txt".format(classifier.name()))
        with open(outfile_name, "w", encoding="utf-8") as outfile:
            outfile.write("#Info:\n")
            outfile.write("Classifier: {0}\n".format(classifier.name()))
            outfile.write("Label: {0}\n".format(args.label))
            outfile.write("\n#Counts:\n")
            outfile.write("Number of training data_records: {0}\n".format(n_training_lines))
            outfile.write("Number of classified data_records: {0}\n".format(len(expected_classes)))
            outfile.write("Number of unique classes in data_records: {0}\n".format(len(set(expected_classes))))
            outfile.write("Number of unique classes found: {0}\n".format(len(set(predicted_classes))))
            outfile.write("\n#Performance:\n")
            outfile.write("Seconds used for training: {0}\n".format(training_time))
            outfile.write("Seconds used for classification: {0}\n".format(classification_time))

            warnings.filterwarnings("ignore", category = sklearn.exceptions.UndefinedMetricWarning)
            outfile.write("\n#Classification report:\n{0}\n".format(sklearn.metrics.classification_report(expected_classes, predicted_classes)))

            # Print the entire confusion matrix, not truncated
            np.set_printoptions(threshold=np.inf, linewidth=200)
            outfile.write("\n#Confusion matrix:\n{0}\n".format(
                sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)))

        # Also store confusion matrix as image
        imagefile_name = os.path.join(args.output, "results_{0}.jpg".format(classifier.name()))
        plot_and_store_confusion_matrix(expected_classes, predicted_classes, imagefile_name)
        print("INFO: Processing completed for classifier {0}".format(classifier.name()))

if __name__ == "__main__":
    main()